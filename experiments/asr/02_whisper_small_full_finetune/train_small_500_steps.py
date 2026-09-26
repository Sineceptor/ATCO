# ---------------------------------------------------------------
# Archived experiment (late 2025). Original file: ASR/train_atc.py
# What it does: First attempt: full fine-tune of Whisper-small on the raw training clips, lr 1e-5, 500 steps.
# Known problems:
#   - Labels already start with <|startoftranscript|> and the collator's bos check never fires for Whisper, so the decoder sees the start token twice.
#   - The test set is used as the Trainer's eval set and selects the best checkpoint, so it is not an untouched test set.
# Not maintained: paths and dependencies are as they were at the time.
# ---------------------------------------------------------------
import os

os.environ["HF_HOME"] = "./hf_cache"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import json
import torch
import jiwer
import soundfile as sf
import numpy as np
from dataclasses import dataclass
from typing import Any, Dict, List, Union
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import Dataset
from transformers import (
    WhisperForConditionalGeneration,
    WhisperProcessor,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)

# Configuration
MODEL_NAME = "openai/whisper-small"
OUTPUT_DIR = "atc_whisper_model_v3_fast"  # 改个名，避免混淆
TRAIN_FILE = "processed_data/train.jsonl"
TEST_FILE = "processed_data/test.jsonl"
AUDIO_ROOT = "./processed_data/audio"


def check_gpu():
    if not torch.cuda.is_available():
        print("致命错误：未检测到 GPU！")
        exit(1)
    print(f"检测到显卡: {torch.cuda.get_device_name(0)}")
    # 打印显存信息
    prop = torch.cuda.get_device_properties(0)
    print(f'   显存总量: {prop.total_memory / 1024 ** 3:.2f} GB')


# 内存缓存 Dataset (保持不变，因为读取很快)
class ATCInMemoryDataset(Dataset):
    def __init__(self, jsonl_path, processor):
        self.processor = processor
        self.audio_root = AUDIO_ROOT
        self.cached_features = []

        print(f"正在加载: {jsonl_path}")
        if not os.path.exists(jsonl_path):
            return

        lines = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    lines.append(json.loads(line))

        print(f'预加载 {len(lines)} 条音频...')
        for item in tqdm(lines, unit="条"):
            filename = os.path.basename(item["audio_file"])
            audio_path = os.path.join(self.audio_root, filename)
            text = item["ASR_transcript_clean"]

            if not os.path.exists(audio_path):
                continue

            try:
                audio, sr = sf.read(audio_path, dtype="float32")
                # 截断过长的音频 (防止显存爆掉) - 限制为 30秒
                if len(audio) > 16000 * 30:
                    audio = audio[: 16000 * 30]

                input_features = self.processor.feature_extractor(
                    audio, sampling_rate=16000
                ).input_features[0]

                labels = self.processor.tokenizer(text).input_ids
                self.cached_features.append({"input_features": input_features, "labels": labels})
            except Exception:
                continue
        print(f'缓存 {len(self.cached_features)} 条')

    def __len__(self):
        return len(self.cached_features)

    def __getitem__(self, idx):
        return self.cached_features[idx]


def main():
    check_gpu()

    print(f"加载模型...")
    processor = WhisperProcessor.from_pretrained(MODEL_NAME, language="English", task="transcribe")
    model = WhisperForConditionalGeneration.from_pretrained(MODEL_NAME)
    model.config.use_cache = False

    train_dataset = ATCInMemoryDataset(TRAIN_FILE, processor)
    test_dataset = ATCInMemoryDataset(TEST_FILE, processor)

    if len(train_dataset) == 0:
        return

    @dataclass
    class DataCollatorSpeechSeq2SeqWithPadding:
        processor: Any

        def __call__(
            self, features: List[Dict[str, Union[List[int], torch.Tensor]]]
        ) -> Dict[str, torch.Tensor]:
            input_features = [{"input_features": feature["input_features"]} for feature in features]
            batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")
            label_features = [{"input_ids": feature["labels"]} for feature in features]
            labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
            labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
            if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
                labels = labels[:, 1:]
            batch["labels"] = labels
            return batch

    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

    # V15 关键参数修改
    training_args = Seq2SeqTrainingArguments(
        output_dir=OUTPUT_DIR,
        # 显存保护策略
        per_device_train_batch_size=2,  # Smaller batch size to reduce memory use.
        gradient_accumulation_steps=8,  # 2 * 8 = 16，保持之前的训练强度
        learning_rate=1e-5,
        warmup_steps=50,
        # 训练步数 (700条数据，batch=16，1个epoch约44步)
        # About 11 epochs for 700 examples and effective batch size 16.
        max_steps=500,
        gradient_checkpointing=False,  # 显存够就关，不够就开
        fp16=True,
        # 提速：训练过程中不生成文本，只看 Loss
        eval_strategy="steps",
        predict_with_generate=False,  # Disable generated-text evaluation during training.
        save_steps=100,
        eval_steps=100,
        logging_steps=10,  # Log training progress.
        load_best_model_at_end=True,
        metric_for_best_model="loss",  # Select checkpoints by loss because generation is disabled.
        greater_is_better=False,
        dataloader_num_workers=0,
        remove_unused_columns=False,
    )

    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        data_collator=data_collator,
        tokenizer=processor.feature_extractor,
    )

    print("\n开始训练 (V15 显存优化版)...")
    print("现在的策略是：小 Batch (2) + 多累积 (8) + 不生成文本")
    print("训练期间记录 loss。")

    trainer.train()

    print(f"\n训练完成！保存中...")
    model.save_pretrained(OUTPUT_DIR)
    processor.save_pretrained(OUTPUT_DIR)


if __name__ == "__main__":
    main()
