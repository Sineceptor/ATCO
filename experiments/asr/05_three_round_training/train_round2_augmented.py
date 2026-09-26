# ---------------------------------------------------------------
# Archived experiment (late 2025). Original file: ASR/train_finale_v2_round2.py
# What it does: Round 2: continues from round 1 on the augmented set, lr 1e-5, 3000 steps.
# Known problems:
#   - The collator prepends <|startoftranscript|> to labels that already start with it, and labels carry no language/task tokens, so training sequences do not match the inference prompt.
#   - Every path is rebuilt as audio_augmented/<name>; the original clips live elsewhere, so unless they were copied by hand about a third of the samples were silence with real labels.
# Not maintained: paths and dependencies are as they were at the time.
# ---------------------------------------------------------------
import os
import json
import torch
import soundfile as sf
import scipy.signal
import gc
import numpy as np
from tqdm import tqdm
from dataclasses import dataclass
from typing import Any, Dict, List, Union
from torch.utils.data import Dataset
from transformers import (
    WhisperForConditionalGeneration,
    WhisperProcessor,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)

# 系统设置
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
torch.set_num_threads(4)

# 关键修改区域

# 指向上一轮训练好的模型路径 (Raw Model)
# Previous checkpoint folder.
PREV_MODEL_PATH = r"whisper_atco_raw_memory_final"

# 切换到增强数据集
ACTIVE_CONFIG = "aug"

# 自动检查路径
if not os.path.exists(PREV_MODEL_PATH):
    print(f"错误：找不到上一轮的模型文件夹: {PREV_MODEL_PATH}")
    print("Set PREV_MODEL_PATH to the saved checkpoint folder.")
    exit(1)

LOCAL_MODEL_PATH = PREV_MODEL_PATH

CONFIG_POOL = {
    "raw": {"train_file": "processed_data/train.jsonl", "audio_root": "./processed_data/audio"},
    "aug": {
        "train_file": "processed_data/train_augmented.jsonl",
        "audio_root": "./processed_data/audio_augmented",
    },
    "tts_generated": {
        "train_file": "processed_data/train_synthetic_real_pro.jsonl",
        "audio_root": "./processed_data/audio_synthetic_real_pro",
    },
}

TRAIN_FILE = CONFIG_POOL[ACTIVE_CONFIG]["train_file"]
AUDIO_ROOT = CONFIG_POOL[ACTIVE_CONFIG]["audio_root"]
OUTPUT_DIR = f"whisper_atco_{ACTIVE_CONFIG}_finetuned"  # 新的输出目录


# 音频读取
def load_audio_fast(file_path):
    try:
        audio, orig_sr = sf.read(file_path)
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)
        if orig_sr != 16000:
            num_samples = int(len(audio) * 16000 / orig_sr)
            audio = scipy.signal.resample(audio, num_samples)
        return audio.astype(np.float32)
    except Exception as e:
        return np.zeros(16000, dtype=np.float32)


# 内存数据集
class ATCDataset(Dataset):
    def __init__(self, jsonl_path, processor, audio_root):
        self.processor = processor
        self.audio_root = audio_root
        self.cached_features = []

        print(f"读取数据配置: {ACTIVE_CONFIG}")
        data_list = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    data_list.append(json.loads(line))

        print(f'正在预加载 {len(data_list)} 条音频到内存 (Stage 2)...')

        for item in tqdm(data_list, desc="Loading Aug Data"):
            wav_path = os.path.join(self.audio_root, os.path.basename(item["audio_file"]))
            try:
                audio = load_audio_fast(wav_path)
                input_features = self.processor.feature_extractor(
                    audio, sampling_rate=16000
                ).input_features[0]
                labels = self.processor.tokenizer(item["ASR_transcript_clean"]).input_ids

                self.cached_features.append({"input_features": input_features, "labels": labels})
            except:
                continue
        print(f"加载完成，准备开始第二轮微调！")

    def __len__(self):
        return len(self.cached_features)

    def __getitem__(self, idx):
        return self.cached_features[idx]


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(self, features):
        input_features = [{"input_features": f["input_features"]} for f in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")
        label_features = [{"input_ids": f["labels"]} for f in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        batch["labels"] = labels
        decoder_start_token_id = self.processor.tokenizer.convert_tokens_to_ids(
            "<|startoftranscript|>"
        )
        decoder_input_ids = labels.new_zeros(labels.shape)
        decoder_input_ids[:, 1:] = labels[:, :-1].clone()
        decoder_input_ids[:, 0] = decoder_start_token_id
        batch["decoder_input_ids"] = decoder_input_ids.masked_fill(
            decoder_input_ids == -100, self.processor.tokenizer.pad_token_id
        )
        return batch


# 训练主流程
def main():
    if not torch.cuda.is_available():
        exit(1)
    gc.collect()
    torch.cuda.empty_cache()

    print(f"加载上一轮的模型: {LOCAL_MODEL_PATH}")
    # Load the previous raw-audio checkpoint.
    processor = WhisperProcessor.from_pretrained(LOCAL_MODEL_PATH)
    model = WhisperForConditionalGeneration.from_pretrained(LOCAL_MODEL_PATH)

    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []
    model.config.use_cache = False
    model.gradient_checkpointing_disable()

    train_dataset = ATCDataset(TRAIN_FILE, processor, AUDIO_ROOT)
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

    training_args = Seq2SeqTrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        # 关键修改：降低学习率 (因为是在已有模型上微调)
        learning_rate=1e-5,
        # 步数控制：如果Aug数据量大，可以适当跑久一点
        max_steps=3000,
        fp16=True,
        dataloader_num_workers=0,
        eval_strategy="no",
        save_strategy="steps",
        save_steps=500,
        logging_steps=10,
        report_to="none",
    )

    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=train_dataset,
        data_collator=data_collator,
        processing_class=processor.feature_extractor,
    )

    print(f"\n开始第二轮增强训练 (Augmented Training)...")
    trainer.train()

    model.save_pretrained(OUTPUT_DIR)
    processor.save_pretrained(OUTPUT_DIR)
    print(f"第二轮训练完成！模型保存在: {os.path.abspath(OUTPUT_DIR)}")


if __name__ == "__main__":
    main()
