# ---------------------------------------------------------------
# Archived experiment (late 2025). Original file: ASR/train_atc_v17.py
# What it does: Same line, 'v17': label cleaning, LoRA r=64 on q/k/v/o, lr lowered to 5e-4 because 1e-3 was too aggressive, 800 steps.
# Known problems:
#   - Labels already start with <|startoftranscript|> and the collator's bos check never fires for Whisper, so the decoder sees the start token twice.
#   - The test set is used as the Trainer's eval set and selects the best checkpoint, so it is not an untouched test set.
#   - The label-cleaning regex \bNE[A-Z]+\b also deletes real words such as NEXT and NEGATIVE.
# Not maintained: paths and dependencies are as they were at the time.
# ---------------------------------------------------------------
import os

os.environ["HF_HOME"] = "./hf_cache"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import json
import torch
import re
import soundfile as sf
import numpy as np
from dataclasses import dataclass
from typing import Any, Dict, List, Union
from tqdm import tqdm
from torch.utils.data import Dataset

from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    WhisperForConditionalGeneration,
    WhisperProcessor,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    BitsAndBytesConfig,
)

# Configuration
MODEL_NAME = "openai/whisper-large-v3"
OUTPUT_DIR = "atc_whisper_lora_v17_clean"  # 新目录
TRAIN_FILE = "processed_data/train.jsonl"
TEST_FILE = "processed_data/test.jsonl"
AUDIO_ROOT = "./processed_data/audio"


def check_gpu():
    if not torch.cuda.is_available():
        print("致命错误：未检测到 GPU！")
        exit(1)
    print(f"检测到显卡: {torch.cuda.get_device_name(0)}")


# 文本清洗函数
def clean_ground_truth(text):
    text = text.upper()
    # 去除元数据标签 (如 NEGERMAN, NEFRENCH, HES, NOISE)
    # 正则逻辑：去除所有以 NE 开头的单词，去除 HES, NOISE
    text = re.sub(r"\bNE[A-Z]+\b", "", text)  # 去除 NEGERMANSCHONE, NEFRENCHMERCI 等
    text = re.sub(r"\bHES\b", "", text)  # 去除犹豫音标记
    text = re.sub(r"\bNOISE\b", "", text)  # 去除噪音标记

    # 修复常见拼写不一致 (标准化)
    text = text.replace("RYAN AIR", "RYANAIR")
    text = text.replace("AIR PORTUGAL", "AIRPORTUGAL")

    # 去除多余空格
    text = re.sub(r"\s+", " ", text).strip()
    return text


class ATCInMemoryDataset(Dataset):
    def __init__(self, jsonl_path, processor, augment=False):
        self.processor = processor
        self.audio_root = AUDIO_ROOT
        self.cached_features = []
        self.augment = augment

        print(f"加载并清洗: {jsonl_path}")
        if not os.path.exists(jsonl_path):
            return

        lines = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    lines.append(json.loads(line))

        print(f"处理音频与文本...")
        for item in tqdm(lines):
            filename = os.path.basename(item["audio_file"])
            audio_path = os.path.join(self.audio_root, filename)

            # 调用清洗函数
            raw_text = item["ASR_transcript_clean"]
            text = clean_ground_truth(raw_text)

            # 如果清洗后文本为空 (比如本来只有 "NOISE")，则跳过该样本
            # 否则模型会因为学不到东西而产生幻觉
            if not text or len(text) < 2:
                continue

            if not os.path.exists(audio_path):
                continue

            self.cached_features.append({"path": audio_path, "text": text})

        print(f'有效样本数: {len(self.cached_features)} (已过滤无效数据)')

    def __len__(self):
        return len(self.cached_features)

    def __getitem__(self, idx):
        item = self.cached_features[idx]
        audio, sr = sf.read(item["path"], dtype="float32")

        if self.augment:
            if np.random.rand() < 0.3:
                noise = np.random.randn(len(audio))
                audio = audio + 0.003 * noise  # 降低一点噪音强度，防止太难学

        if len(audio) > 16000 * 30:
            audio = audio[: 16000 * 30]

        input_features = self.processor.feature_extractor(
            audio, sampling_rate=16000
        ).input_features[0]

        labels = self.processor.tokenizer(item["text"]).input_ids
        return {"input_features": input_features, "labels": labels}


def main():
    check_gpu()

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )

    print(f"加载 Whisper-Large-v3...")
    processor = WhisperProcessor.from_pretrained(MODEL_NAME, language="en", task="transcribe")
    model = WhisperForConditionalGeneration.from_pretrained(
        MODEL_NAME, quantization_config=bnb_config, device_map="auto"
    )

    # 强制设置生成配置：抑制幻觉
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []  # No token suppression during generation.
    # 强制英语转录 (防止它自动变成翻译模式)
    model.generation_config.language = "en"
    model.generation_config.task = "transcribe"

    model = prepare_model_for_kbit_training(model)

    config = LoraConfig(
        r=64,
        lora_alpha=128,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],  # 增加微调模块，提升拟合能力
        lora_dropout=0.05,
        bias="none",
    )

    model = get_peft_model(model, config)
    model.print_trainable_parameters()

    train_dataset = ATCInMemoryDataset(TRAIN_FILE, processor, augment=True)
    test_dataset = ATCInMemoryDataset(TEST_FILE, processor, augment=False)

    @dataclass
    class DataCollatorSpeechSeq2SeqWithPadding:
        processor: Any

        def __call__(self, features):
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

    training_args = Seq2SeqTrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=8,
        gradient_accumulation_steps=2,
        # 调低学习率：1e-3 有点太激进了，导致学乱了，改成 5e-4
        learning_rate=5e-4,
        warmup_steps=50,
        max_steps=800,
        fp16=True,
        eval_strategy="steps",
        predict_with_generate=True,
        generation_max_length=225,
        save_steps=200,
        eval_steps=200,
        logging_steps=25,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        remove_unused_columns=False,
        report_to=["tensorboard"],
    )

    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        data_collator=data_collator,
        tokenizer=processor.feature_extractor,
    )

    print("\n开始 V17 清洗版微调...")
    trainer.train()

    print(f"\n训练完成！")
    model.save_pretrained(OUTPUT_DIR)
    processor.save_pretrained(OUTPUT_DIR)


if __name__ == "__main__":
    main()
