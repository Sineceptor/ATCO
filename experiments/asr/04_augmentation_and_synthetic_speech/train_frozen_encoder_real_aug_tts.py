# ---------------------------------------------------------------
# Archived experiment (late 2025). Original file: ASR/train_atc_augment_v2.py
# What it does: Whisper-small with the encoder frozen, trained on real + augmented + radio-simulated TTS data (Adafactor, lr 1e-4, 3 epochs).
# Known problems:
#   - Labels already start with <|startoftranscript|> and the collator's bos check never fires for Whisper, so the decoder sees the start token twice.
#   - Validation is a random 5% of the pool, which contains augmented twins of training clips.
# Not maintained: paths and dependencies are as they were at the time.
# ---------------------------------------------------------------
import json
import os
import glob
import torch
import multiprocessing
from dataclasses import dataclass
from typing import Any, Dict, List, Union
from datasets import load_dataset, Audio, concatenate_datasets
import evaluate

from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)

# 极速配置 (Speed Demon)
MODEL_NAME = "openai/whisper-small"
OUTPUT_DIR = "./whisper-atco2-speed"

REAL_DATA_PATH = "processed_data/train.jsonl"
AUGMENTED_DATA_PATH = "processed_data/train_augmented.jsonl"
SYNTHETIC_DATA_PATH = "processed_data/train_synthetic_real_pro.jsonl"

# 激进加速策略
# 因为冻结了 Encoder 且用了 Adafactor，显存占用极低，可以直接开大 Batch
BATCH_SIZE = 32
GRADIENT_ACCUMULATION = 1
LEARNING_RATE = 1e-4  # 只训练 Decoder，学习率可以稍微大一点点
NUM_EPOCHS = 3

# 辅助函数
FOUND_AUDIO_DIRS = set()


def scan_for_audio_files(base_dir):
    print(f"正在扫描音频文件夹...")
    wav_files = glob.glob(os.path.join(base_dir, "**", "*.wav"), recursive=True)
    dirs = set()
    for f in wav_files:
        dirs.add(os.path.dirname(f))
    return dirs


def fix_audio_path(batch):
    path = batch["audio_file"]
    if os.path.exists(path):
        return batch
    filename = os.path.basename(path)
    for d in FOUND_AUDIO_DIRS:
        potential_path = os.path.join(d, filename)
        if os.path.exists(potential_path):
            batch["audio_file"] = potential_path
            return batch
    batch["audio_file"] = os.path.abspath(path)
    return batch


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(
        self, features: List[Dict[str, Union[List[int], torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch


wer_metric = evaluate.load("wer")


def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
    pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)
    wer = 100 * wer_metric.compute(predictions=pred_str, references=label_str)
    return {"wer": wer}


def prepare_dataset(batch, processor):
    audio = batch["audio_file"]
    batch["input_features"] = processor.feature_extractor(
        audio["array"], sampling_rate=audio["sampling_rate"]
    ).input_features[0]
    batch["labels"] = processor.tokenizer(batch["ASR_transcript_clean"]).input_ids
    return batch


if __name__ == "__main__":
    multiprocessing.freeze_support()
    print("启动【光速训练】模式 (Target: < 3 Hours)")

    project_root = os.getcwd()
    FOUND_AUDIO_DIRS = scan_for_audio_files(project_root)
    processor = WhisperProcessor.from_pretrained(MODEL_NAME, language="English", task="transcribe")

    # 加载数据
    files = {"real": REAL_DATA_PATH, "aug": AUGMENTED_DATA_PATH, "syn": SYNTHETIC_DATA_PATH}
    data_files = {k: v for k, v in files.items() if os.path.exists(v)}
    raw_datasets = load_dataset("json", data_files=data_files)
    combined_dataset = concatenate_datasets([raw_datasets[k] for k in raw_datasets.keys()])

    # 修复路径 & 过滤
    combined_dataset = combined_dataset.map(fix_audio_path, load_from_cache_file=False)
    combined_dataset = combined_dataset.filter(lambda x: os.path.exists(x["audio_file"]))
    combined_dataset = combined_dataset.cast_column("audio_file", Audio(sampling_rate=16000))

    # 预处理 (存入 RAM)
    print("预处理并加载至内存 (In-Memory)...")
    tokenized_dataset = combined_dataset.map(
        prepare_dataset,
        remove_columns=combined_dataset.column_names,
        num_proc=1,
        fn_kwargs={"processor": processor},
    )

    # 关键点：设置格式为 torch，这样数据在 RAM 里已经是 Tensor，取用极快
    tokenized_dataset = tokenized_dataset.with_format("torch")

    split_dataset = tokenized_dataset.train_test_split(test_size=0.05)
    train_ds = split_dataset["train"]
    val_ds = split_dataset["test"]

    # 加载模型
    print(f"\n加载模型: {MODEL_NAME}")
    model = WhisperForConditionalGeneration.from_pretrained(MODEL_NAME)
    model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(
        language="English", task="transcribe"
    )

    # 核心加速 1：冻结 Encoder (听力部分)
    # 只训练 Decoder (翻译/理解部分) 和 投影层
    # 这一步能减少 ~50% 的计算量和显存占用
    print("冻结 Encoder 参数...")
    model.freeze_encoder()

    # 确保梯度检查点关闭 (加速)
    model.gradient_checkpointing_disable()

    # 训练参数
    training_args = Seq2SeqTrainingArguments(
        output_dir=OUTPUT_DIR,
        # 核心加速 2：增大 Batch Size (因为冻结了 encoder)
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION,
        learning_rate=LEARNING_RATE,
        num_train_epochs=NUM_EPOCHS,  # 改用 Epoch 计数，更直观
        gradient_checkpointing=False,  # 坚决关闭
        fp16=True,
        # 核心加速 3：使用 Adafactor 优化器 (比 AdamW 省显存)
        optim="adafactor",
        eval_strategy="steps",
        per_device_eval_batch_size=16,  # 验证时 Batch 也可以大一点
        predict_with_generate=True,
        generation_max_length=225,
        save_steps=500,
        eval_steps=500,
        logging_steps=50,
        report_to=["tensorboard"],
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        save_total_limit=2,
        dataloader_num_workers=0,
        remove_unused_columns=False,
    )

    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=DataCollatorSpeechSeq2SeqWithPadding(processor=processor),
        compute_metrics=compute_metrics,
        processing_class=processor.feature_extractor,
    )

    print(f"\n引擎全开！Batch Size: {BATCH_SIZE}, Encoder: Frozen")
    print("   预计训练时间将大幅缩短。")
    trainer.train()

    print(f"\n保存模型到: {OUTPUT_DIR}")
    model.save_pretrained(OUTPUT_DIR)
    processor.save_pretrained(OUTPUT_DIR)
    print("训练完成！")
