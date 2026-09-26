# ---------------------------------------------------------------
# Archived experiment (late 2025). Original file: ASR/train_asr_mixed_v5.py
# What it does: Full fine-tune of Whisper-small on raw + augmented clips for 3 epochs.
# Known problems:
#   - Labels already start with <|startoftranscript|> and the collator's bos check never fires for Whisper, so the decoder sees the start token twice.
#   - The test set is used as the Trainer's eval set and selects the best checkpoint, so it is not an untouched test set.
#   - Reads an augmented_data/ folder that no surviving script creates.
# Not maintained: paths and dependencies are as they were at the time.
# ---------------------------------------------------------------
import os

# 屏蔽干扰日志
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import json
import torch
import torchaudio
import torchaudio.transforms as T
from dataclasses import dataclass
from typing import Any, List, Dict
from torch.utils.data import Dataset
from transformers import (
    WhisperForConditionalGeneration,
    WhisperProcessor,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)

# Configuration
TRAIN_FILES = ["processed_data/train.jsonl", "augmented_data/train.jsonl"]
AUDIO_ROOTS = ["./processed_data/audio", "./augmented_data/audio"]

TEST_FILE = "processed_data/test.jsonl"
OUTPUT_DIR = "whisper_atco_TURBO_v6"  # 新保存路径
MODEL_ID = "openai/whisper-small"


# 数据集加载器
class MixedATCDataset(Dataset):
    def __init__(self, jsonl_paths, processor, audio_roots):
        self.data = []
        self.processor = processor
        self.target_sr = 16000

        print(f'准备加载 {len(jsonl_paths)} 个数据源...')
        for json_path, root in zip(jsonl_paths, audio_roots):
            if not os.path.exists(json_path):
                print(f'跳过: {json_path} (请确保已运行数据增强)')
                continue

            count = 0
            with open(json_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        item = json.loads(line)
                        item["_root"] = root
                        self.data.append(item)
                        count += 1
            print(f'   -> {os.path.basename(json_path)}: {count} 条')
        print(f'数据集就绪! 总计: {len(self.data)} 条')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        audio_root = item["_root"]
        raw_path = item["audio_file"]
        wav_path = (
            raw_path
            if os.path.exists(raw_path)
            else os.path.join(audio_root, os.path.basename(raw_path))
        )

        try:
            waveform, sample_rate = torchaudio.load(wav_path)
            if sample_rate != self.target_sr:
                resampler = T.Resample(sample_rate, self.target_sr)
                waveform = resampler(waveform)
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0)
            else:
                waveform = waveform.squeeze(0)

            input_features = self.processor.feature_extractor(
                waveform.numpy(), sampling_rate=16000
            ).input_features[0]

            txt = item.get("ASR_transcript_clean") or item.get("text")
            labels = self.processor.tokenizer(txt).input_ids
            return {"input_features": input_features, "labels": labels}
        except Exception:
            return self.__getitem__(0)


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(self, features):
        input_features = [{"input_features": f["input_features"]} for f in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")
        label_features = [{"input_ids": f["labels"]} for f in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]
        batch["labels"] = labels
        return batch


# 训练流程
def main():
    if not torch.cuda.is_available():
        exit("需要 GPU")
    torch.cuda.empty_cache()

    print(f"加载模型: {MODEL_ID}")
    processor = WhisperProcessor.from_pretrained(MODEL_ID, task="transcribe", language="en")
    model = WhisperForConditionalGeneration.from_pretrained(MODEL_ID)

    # 训练设置
    model.config.use_cache = False

    # Optional: freeze the encoder to train only the decoder.
    # print(" 冻结 Encoder 参数，只训练 Decoder...")
    # model.model.encoder.gradient_checkpointing = False
    # model.freeze_encoder()

    train_dataset = MixedATCDataset(TRAIN_FILES, processor, AUDIO_ROOTS)
    test_dataset = MixedATCDataset([TEST_FILE], processor, [AUDIO_ROOTS[0]])
    collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

    # 计算步数
    # 数据量 ~2100, Batch=8 => 每 Epoch 约 260 步
    # 3 Epochs = 780 步 (远少于之前的 4000 步)

    training_args = Seq2SeqTrainingArguments(
        output_dir=OUTPUT_DIR,
        # 速度优化核心参数
        per_device_train_batch_size=8,  # 增大 Batch (如果显存爆了改成 4)
        gradient_accumulation_steps=2,  # 累积减少，加快更新频率
        # 训练时长控制
        num_train_epochs=3,  # Three training epochs in this experiment.
        # max_steps=4000,               # 删除这个强制步数
        # 学习率
        learning_rate=2e-5,  # 稍微调大一点点加速收敛
        warmup_ratio=0.1,  # 用比例代替固定步数
        # 评估策略优化
        eval_strategy="epoch",  # 每个 Epoch 结束才评估一次，别太频繁
        save_strategy="epoch",  # 每个 Epoch 保存一次
        logging_steps=20,  # Log loss during training.
        fp16=True,  # 必开
        predict_with_generate=True,  # 评估时生成文本计算 WER
        generation_max_length=128,
        dataloader_num_workers=0,  # Windows 保持 0，Linux 可改为 4
        report_to="none",
        load_best_model_at_end=True,
        metric_for_best_model="loss",
    )

    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        data_collator=collator,
        processing_class=processor.feature_extractor,
    )

    print("\n开始训练 (V6)...")
    print(f"训练轮数: 3")
    trainer.train()

    print(f"保存模型到: {os.path.abspath(OUTPUT_DIR)}")
    model.save_pretrained(OUTPUT_DIR)
    processor.save_pretrained(OUTPUT_DIR)
    print("训练完成！")


if __name__ == "__main__":
    main()
