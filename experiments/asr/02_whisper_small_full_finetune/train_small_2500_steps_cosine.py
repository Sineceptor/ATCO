# ---------------------------------------------------------------
# Archived experiment (late 2025). Original file: ASR/train_atc_v4_test.py
# What it does: Full fine-tune of Whisper-small with cached features, lr 5e-5 cosine, 2500 steps. Direct ancestor of the final training script in training/train_whisper.py.
# Known problems:
#   - Labels already start with <|startoftranscript|> and the collator's bos check never fires for Whisper, so the decoder sees the start token twice.
#   - The test set is used as the Trainer's eval set and selects the best checkpoint, so it is not an untouched test set.
# Not maintained: paths and dependencies are as they were at the time.
# ---------------------------------------------------------------
import os

# 屏蔽 TensorFlow 和 oneDNN 的烦人日志
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import json
import torch
import torchaudio
import torchaudio.transforms as T
import gc
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

# 自动寻路配置
CACHE_ROOT = r"C:\path\to\ATCO\ASR\hf_cache"
MODEL_NAME_FOLDER = "models--openai--whisper-small"


def find_local_model_path(cache_root, model_folder):
    snapshots_dir = os.path.join(cache_root, "hub", model_folder, "snapshots")
    if not os.path.exists(snapshots_dir):
        snapshots_dir = os.path.join(cache_root, model_folder, "snapshots")
        if not os.path.exists(snapshots_dir):
            return "openai/whisper-small"
    subfolders = [f.path for f in os.scandir(snapshots_dir) if f.is_dir()]
    return subfolders[0] if subfolders else "openai/whisper-small"


try:
    LOCAL_MODEL_PATH = find_local_model_path(CACHE_ROOT, MODEL_NAME_FOLDER)
except:
    LOCAL_MODEL_PATH = "openai/whisper-small"

ACTIVE_CONFIG = "raw"

CONFIG_POOL = {
    "raw": {
        "train_file": "processed_data/train.jsonl",
        "test_file": "processed_data/test.jsonl",
        "audio_root": "./processed_data/audio",
    },
    "aug": {
        "train_file": "augmented_data/train.jsonl",
        "test_file": "augmented_data/test.jsonl",
        "audio_root": "./augmented_data/audio",
    },
    "gan": {
        "train_file": "generate_data/train.jsonl",
        "test_file": "generate_data/test.jsonl",
        "audio_root": "./generate_data/audio",
    },
}

FIELD_MAPPING = {"text": "ASR_transcript_clean", "audio": "audio_file"}

TRAIN_FILE = CONFIG_POOL[ACTIVE_CONFIG]["train_file"]
TEST_FILE = CONFIG_POOL[ACTIVE_CONFIG]["test_file"]
AUDIO_ROOT = CONFIG_POOL[ACTIVE_CONFIG]["audio_root"]
OUTPUT_DIR = f"whisper_atco_{ACTIVE_CONFIG}_V2"


# 环境检查
def check_gpu_and_clear():
    if not torch.cuda.is_available():
        print("未检测到 GPU！")
        exit(1)
    gc.collect()
    torch.cuda.empty_cache()
    print(f"GPU 环境已就绪。配置: {ACTIVE_CONFIG}")


# 内存预加载数据集 (核心提速)


class InMemoryATCDataset(Dataset):
    def __init__(self, jsonl_path, processor, audio_root, split="train"):
        self.data = []  # 这里存放处理好的 Tensor
        self.processor = processor
        self.target_sr = 16000

        if not os.path.exists(jsonl_path):
            raise FileNotFoundError(f"找不到数据文件: {jsonl_path}")

        # 读取原始列表
        raw_items = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    raw_items.append(json.loads(line))

        print(f'[{split}] 正在预处理 {len(raw_items)} 条数据到内存 (只需一次，请耐心等待)...')

        # 预处理循环
        text_key = FIELD_MAPPING["text"]
        audio_key = FIELD_MAPPING["audio"]

        for item in tqdm(raw_items, desc=f"Loading {split}"):
            # 音频处理
            raw_path = item[audio_key]
            if os.path.exists(raw_path):
                wav_path = raw_path
            else:
                wav_path = os.path.join(audio_root, os.path.basename(raw_path))

            try:
                waveform, sample_rate = torchaudio.load(wav_path)
                if sample_rate != self.target_sr:
                    resampler = T.Resample(sample_rate, self.target_sr)
                    waveform = resampler(waveform)

                if waveform.shape[0] > 1:
                    waveform = torch.mean(waveform, dim=0)
                else:
                    waveform = waveform.squeeze(0)

                # 计算 Log-Mel 特征
                input_features = self.processor.feature_extractor(
                    waveform.numpy(), sampling_rate=16000
                ).input_features[0]

            except Exception as e:
                continue

                # 文本处理
            labels = self.processor.tokenizer(item[text_key]).input_ids

            # 存入内存
            self.data.append({"input_features": input_features, "labels": labels})

        print(f'[{split}] 内存加载完毕！有效数据: {len(self.data)} 条')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(
        self, features: List[Dict[str, Union[List[int], torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        input_features = [{"input_features": f["input_features"]} for f in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")
        label_features = [{"input_ids": f["labels"]} for f in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]
        batch["labels"] = labels
        return batch


# 训练主流程


def main():
    check_gpu_and_clear()

    print(f"正在加载模型: {LOCAL_MODEL_PATH}")
    processor = WhisperProcessor.from_pretrained(LOCAL_MODEL_PATH, task="transcribe")
    model = WhisperForConditionalGeneration.from_pretrained(LOCAL_MODEL_PATH)

    # 模型精简配置
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []
    model.config.use_cache = False

    # 必须保持注释状态，否则会报错 RuntimeError
    # model.gradient_checkpointing_enable()

    # 使用内存数据集
    train_dataset = InMemoryATCDataset(TRAIN_FILE, processor, AUDIO_ROOT, split="Train")
    test_dataset = InMemoryATCDataset(TEST_FILE, processor, AUDIO_ROOT, split="Test")
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

    # 显存优化参数配置
    training_args = Seq2SeqTrainingArguments(
        output_dir=OUTPUT_DIR,
        # ⬇⬇⬇ 降批次，防爆显存 ⬇⬇⬇
        per_device_train_batch_size=2,  # 从 8 改为 2 (确保全部吃进显存，不走内存交换)
        gradient_accumulation_steps=8,  # 从 2 改为 8 (2*8=16，保持总训练量不变)
        # ⬆⬆⬆ 核心修改结束 ⬆⬆⬆
        learning_rate=5e-5,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        max_steps=2500,
        fp16=True,
        dataloader_num_workers=0,  # 内存数据集不需要多进程
        eval_strategy="steps",
        eval_steps=500,
        save_steps=500,
        logging_steps=50,
        predict_with_generate=True,
        load_best_model_at_end=True,
        metric_for_best_model="loss",
        report_to="none",
    )

    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        data_collator=data_collator,
        processing_class=processor.feature_extractor,
    )

    print(f"开始极速训练 (Batch=2, Accum=8)...")
    trainer.train()

    print(f"保存模型到: {os.path.abspath(OUTPUT_DIR)}")
    model.save_pretrained(OUTPUT_DIR)
    processor.save_pretrained(OUTPUT_DIR)
    print("完成！")


if __name__ == "__main__":
    main()
