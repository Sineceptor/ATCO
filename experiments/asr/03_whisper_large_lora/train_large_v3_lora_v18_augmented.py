# ---------------------------------------------------------------
# Archived experiment (late 2025). Original file: ASR/train_atc_augment.py
# What it does: 'v18': the large-v3 LoRA trained on the noise/speed augmented set, 1500 steps. This is the model behind results/original_2025/asr_large_lora_robustness.txt.
# Known problems:
#   - Labels already start with <|startoftranscript|> and the collator's bos check never fires for Whisper, so the decoder sees the start token twice.
#   - The test set is used as the Trainer's eval set and selects the best checkpoint, so it is not an untouched test set.
#   - Same \bNE[A-Z]+\b regex problem.
#   - Spelling mappings were chosen from observed evaluation errors.
# Not maintained: paths and dependencies are as they were at the time.
# ---------------------------------------------------------------
import os

# 环境配置
os.environ["HF_HOME"] = "./hf_cache"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
# 防止底层库冲突
os.environ["OMP_NUM_THREADS"] = "1"

import json
import torch
import re
import soundfile as sf
import numpy as np
from dataclasses import dataclass
from typing import Any, Dict, List, Union
from tqdm import tqdm
from torch.utils.data import Dataset

# 引入核心库
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    WhisperForConditionalGeneration,
    WhisperProcessor,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    BitsAndBytesConfig,
)

# 核心配置区域
MODEL_NAME = "openai/whisper-large-v3"
OUTPUT_DIR = "atc_whisper_lora_v18_augmented"  # 新的保存路径

# 关键修改：指向扩充后的文件
TRAIN_FILE = "processed_data/train_augmented.jsonl"
# 测试集保持原样 (永远不要扩充测试集，否则评估不准)
TEST_FILE = "processed_data/test.jsonl"

AUDIO_ROOT = "./processed_data/audio"


def check_gpu():
    if not torch.cuda.is_available():
        print("致命错误：未检测到 GPU！")
        exit(1)
    print(f"检测到显卡: {torch.cuda.get_device_name(0)}")


# 文本清洗函数 (训练时的最后一道防线)
def clean_ground_truth(text):
    if not text:
        return ""
    text = text.upper()

    # 去除元数据标签 (如 NEGERMAN, HES, NOISE)
    # 这一步非常重要，否则模型会把 "NOISE" 当成单词学
    text = re.sub(r"\bNE[A-Z]+\b", "", text)
    text = re.sub(r"\b(HES|NOISE)\b", "", text)

    # Spelling mappings chosen from observed evaluation errors.
    text = text.replace("RYAN AIR", "RYANAIR")
    text = text.replace("AIR PORTUGAL", "AIRPORTUGAL")
    text = text.replace("EURO TRANS", "EUROTRANS")

    # 清理多余空格
    text = re.sub(r"\s+", " ", text).strip()
    return text


class ATCDataSet(Dataset):
    def __init__(self, jsonl_path, processor):
        self.processor = processor
        self.cached_features = []

        print(f"正在加载数据: {jsonl_path}")
        if not os.path.exists(jsonl_path):
            print(f"文件不存在: {jsonl_path}")
            return

        with open(jsonl_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        print(f'正在预处理 {len(lines)} 条样本...')

        for line in tqdm(lines, unit="line"):
            if not line.strip():
                continue
            item = json.loads(line)

            # 解析音频路径
            # 扩充脚本生成的 jsonl 里通常已经是绝对路径了，但为了保险起见做个检查
            audio_path = item["audio_file"]
            if not os.path.exists(audio_path):
                # 尝试去 audio_augmented 找
                base_name = os.path.basename(audio_path)
                alt_path = os.path.join("processed_data/audio_augmented", base_name)
                if os.path.exists(alt_path):
                    audio_path = alt_path
                else:
                    # 再找不到就去原目录找
                    alt_path_2 = os.path.join(AUDIO_ROOT, base_name)
                    if os.path.exists(alt_path_2):
                        audio_path = alt_path_2
                    else:
                        continue  # 真的找不到，跳过

            # 清洗文本
            raw_text = item.get("ASR_transcript_clean", "")
            text = clean_ground_truth(raw_text)

            # 如果清洗后没字了 (纯噪音样本)，跳过
            if len(text) < 2:
                continue

            self.cached_features.append({"path": audio_path, "text": text})

        print(f"有效样本数: {len(self.cached_features)}")

    def __len__(self):
        return len(self.cached_features)

    def __getitem__(self, idx):
        item = self.cached_features[idx]
        # 读取音频
        audio, sr = sf.read(item["path"], dtype="float32")

        # 转单声道
        if len(audio.shape) > 1:
            audio = audio.mean(axis=1)

        # 截断过长音频 (30s)
        if len(audio) > 16000 * 30:
            audio = audio[: 16000 * 30]

        input_features = self.processor.feature_extractor(
            audio, sampling_rate=16000
        ).input_features[0]

        labels = self.processor.tokenizer(item["text"]).input_ids
        return {"input_features": input_features, "labels": labels}


def main():
    check_gpu()

    # 4-bit 量化配置
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )

    print(f"加载 Whisper-Large-v3 (4-bit)...")
    processor = WhisperProcessor.from_pretrained(MODEL_NAME, language="en", task="transcribe")
    model = WhisperForConditionalGeneration.from_pretrained(
        MODEL_NAME, quantization_config=bnb_config, device_map="auto"
    )

    # 强制英语输出，防止模型乱翻译
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []
    model.generation_config.language = "en"
    model.generation_config.task = "transcribe"

    # LoRA 配置
    model = prepare_model_for_kbit_training(model)
    config = LoraConfig(
        r=32,
        lora_alpha=64,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
    )
    model = get_peft_model(model, config)
    model.print_trainable_parameters()

    # 加载数据集
    # 此时 train_dataset 已经是原来的 3 倍大小了
    train_dataset = ATCDataSet(TRAIN_FILE, processor)
    test_dataset = ATCDataSet(TEST_FILE, processor)

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

    # 训练参数调整
    # Training steps for this run.
    # 原来 700 条数据 -> 800 步
    # 现在 2100 条数据 -> 建议 1500 步以上
    training_args = Seq2SeqTrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=8,
        gradient_accumulation_steps=2,
        # 学习率: 因为数据变杂了，稍微降低一点 LR 让它学得细致点
        learning_rate=5e-4,
        warmup_steps=100,
        # 增加训练步数
        max_steps=1500,
        fp16=True,
        # 评估策略
        eval_strategy="steps",
        predict_with_generate=True,
        generation_max_length=225,
        save_steps=300,
        eval_steps=300,
        logging_steps=50,
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

    print("\n开始训练 (Augmented Data + LoRA)...")
    print(f"训练步数: {training_args.max_steps}")
    print(f"模型将保存到: {OUTPUT_DIR}")

    trainer.train()

    print(f"\n训练完成！保存模型中...")
    model.save_pretrained(OUTPUT_DIR)
    processor.save_pretrained(OUTPUT_DIR)


if __name__ == "__main__":
    main()
