# ---------------------------------------------------------------
# Archived experiment (late 2025). Original file: ASR/train_finale_v2.py
# What it does: Round 1: full fine-tune of Whisper-small on raw clips, lr 2e-5, 3500 steps.
# Known problems:
#   - The collator prepends <|startoftranscript|> to labels that already start with it, and labels carry no language/task tokens, so training sequences do not match the inference prompt.
#   - A clip that fails to load becomes one second of silence paired with a real transcript.
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

# 系统防死锁设置
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
torch.set_num_threads(4)

# 路径与配置
BASE_CACHE_PATH = (
    r"C:\path\to\ATCO\ASR\hf_cache\hub\models--openai--whisper-small"
)


def get_model_path(base_path):
    # 自动寻找 hash 文件夹
    snap = os.path.join(base_path, "snapshots")
    if os.path.exists(snap):
        subdirs = [d for d in os.listdir(snap) if os.path.isdir(os.path.join(snap, d))]
        if subdirs:
            return os.path.join(snap, subdirs[0])
    return base_path


LOCAL_MODEL_PATH = get_model_path(BASE_CACHE_PATH)
ACTIVE_CONFIG = "raw"

CONFIG_POOL = {
    "raw": {"train_file": "processed_data/train.jsonl", "audio_root": "./processed_data/audio"},
    "aug": {"train_file": "augmented_data/train.jsonl", "audio_root": "./augmented_data/audio"},
    "gan": {"train_file": "generate_data/train.jsonl", "audio_root": "./generate_data/audio"},
}

TRAIN_FILE = CONFIG_POOL[ACTIVE_CONFIG]["train_file"]
AUDIO_ROOT = CONFIG_POOL[ACTIVE_CONFIG]["audio_root"]
OUTPUT_DIR = f"whisper_atco_{ACTIVE_CONFIG}_memory_final"


# 音频读取工具
def load_audio_fast(file_path):
    """Load mono audio and resample to 16 kHz."""
    try:
        audio, orig_sr = sf.read(file_path)
        # 转单声道
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)
        # 重采样到 16000
        if orig_sr != 16000:
            num_samples = int(len(audio) * 16000 / orig_sr)
            audio = scipy.signal.resample(audio, num_samples)
        return audio.astype(np.float32)
    except Exception as e:
        print(f"读取失败: {file_path}")
        return np.zeros(16000, dtype=np.float32)


# Dataset cached in memory.


class ATCDataset(Dataset):
    def __init__(self, jsonl_path, processor, audio_root):
        self.processor = processor
        self.audio_root = audio_root
        self.data = []

        # 读取列表
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    self.data.append(json.loads(line))

        print(f'正在预加载所有音频到内存，以加速训练 (共 {len(self.data)} 条)...')
        print(f"这一步可能需要 1-2 分钟，请耐心等待进度条跑完。")

        # 预加载所有音频特征 (Feature Cache)
        self.cached_features = []
        success_count = 0

        for item in tqdm(self.data, desc="Loading Audio into RAM"):
            wav_path = os.path.join(self.audio_root, os.path.basename(item["audio_file"]))

            try:
                # 使用之前的 fast loader
                audio = load_audio_fast(wav_path)

                # 提取特征
                # 注意：这里保持 float32，让 Trainer 在 forward 时自动转 fp16，防止类型冲突
                input_features = self.processor.feature_extractor(
                    audio, sampling_rate=16000
                ).input_features[0]

                # 预先处理文本
                labels = self.processor.tokenizer(item["ASR_transcript_clean"]).input_ids

                self.cached_features.append({"input_features": input_features, "labels": labels})
                success_count += 1
            except Exception as e:
                print(f"跳过坏文件: {wav_path}")
                continue

        print(f'预加载完成！有效数据: {len(self.cached_features)} 条。')
        print(f"接下来的训练将完全在内存中进行，不受硬盘速度影响。")

    def __len__(self):
        return len(self.cached_features)

    def __getitem__(self, idx):
        # 直接从内存返回，无需磁盘IO，无需CPU解码
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

        # 修正 decoder_input_ids
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

    print(f"加载模型: {LOCAL_MODEL_PATH}")
    processor = WhisperProcessor.from_pretrained(LOCAL_MODEL_PATH)
    model = WhisperForConditionalGeneration.from_pretrained(LOCAL_MODEL_PATH)

    # 关键设置
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []
    model.config.use_cache = False

    # 关闭梯度检查点：这是为了防止之前那个 RuntimeError 报错
    # This setting uses more memory; reduce batch size if needed.
    model.gradient_checkpointing_disable()

    # 初始化内存数据集
    train_dataset = ATCDataset(TRAIN_FILE, processor, AUDIO_ROOT)
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

    training_args = Seq2SeqTrainingArguments(
        output_dir=OUTPUT_DIR,
        # 参数配置
        per_device_train_batch_size=2,  # 设为 2 比较稳，如果显存还爆就改 1
        gradient_accumulation_steps=8,  # 累积梯度，保持总 Batch = 16
        learning_rate=2e-5,
        max_steps=3500,
        fp16=True,  # 开启混合精度加速
        dataloader_num_workers=0,  # Windows 必须为 0
        eval_strategy="no",
        save_strategy="steps",
        save_steps=500,  # 每 500 步保存一次
        logging_steps=10,  # 每 10 步打印一次日志
        report_to="none",
    )

    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=train_dataset,
        data_collator=data_collator,
        processing_class=processor.feature_extractor,
    )

    print(f"\n开始训练 (In-Memory 模式)...")
    trainer.train()

    model.save_pretrained(OUTPUT_DIR)
    processor.save_pretrained(OUTPUT_DIR)
    print(f"训练完成！模型保存在: {os.path.abspath(OUTPUT_DIR)}")


if __name__ == "__main__":
    main()
