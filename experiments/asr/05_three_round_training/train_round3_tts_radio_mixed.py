# ---------------------------------------------------------------
# Archived experiment (late 2025). Original file: ASR/train_finale_v2_round3_gan.py
# What it does: Round 3: continues from round 2 on radio-simulated TTS speech mixed with 20% real clips, lr 5e-6, 2500 steps.
# Known problems:
#   - The collator prepends <|startoftranscript|> to labels that already start with it, and labels carry no language/task tokens, so training sequences do not match the inference prompt.
#   - The code calls the synthetic data 'GAN'. There is no GAN: it is TTS speech passed through simulate_radio_channel.py.
# Not maintained: paths and dependencies are as they were at the time.
# ---------------------------------------------------------------
import os
import json
import torch
import soundfile as sf
import scipy.signal
import gc
import numpy as np
import random  # 用于随机抽取真实数据
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

# 路径与配置

# 指向 Round 2 (Augmented) 训练好的模型
PREV_MODEL_PATH = r"whisper_atco_aug_finetuned"

# 当前主要任务配置
ACTIVE_CONFIG = "gan"

# 自动检查模型路径
if not os.path.exists(PREV_MODEL_PATH):
    print(f"错误：找不到 Round 2 模型文件夹: {PREV_MODEL_PATH}")
    exit(1)

LOCAL_MODEL_PATH = PREV_MODEL_PATH

# 数据集路径池
CONFIG_POOL = {
    "raw": {"train_file": "processed_data/train.jsonl", "audio_root": "./processed_data/audio"},
    "aug": {
        "train_file": "processed_data/train_augmented.jsonl",
        "audio_root": "./processed_data/audio_augmented",
    },
    "gan": {
        "train_file": "processed_data/train_synthetic_real_pro.jsonl",
        "audio_root": "./processed_data/audio_synthetic_real_pro",
    },
}

# 主要训练数据 (GAN)
TRAIN_FILE = CONFIG_POOL[ACTIVE_CONFIG]["train_file"]
AUDIO_ROOT = CONFIG_POOL[ACTIVE_CONFIG]["audio_root"]
OUTPUT_DIR = f"whisper_atco_{ACTIVE_CONFIG}_mixed_final"  # 输出目录改名，标明是 mixed 版本


# 音频读取工具
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


# 混合数据集类 (核心修改)
class MixedATCDataset(Dataset):
    def __init__(self, gan_jsonl, processor, gan_audio_root, mix_ratio=0.2):
        """
        mix_ratio=0.2 表示混入 20% 的真实数据
        """
        self.processor = processor
        self.cached_features = []

        # 第一步：加载 GAN 数据
        print(f"[Stage 3] 正在准备混合数据集...")
        data_items = []

        if os.path.exists(gan_jsonl):
            with open(gan_jsonl, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        item = json.loads(line)
                        # 标记：这是 GAN 数据，使用 gan_audio_root
                        item["root_path"] = gan_audio_root
                        data_items.append(item)
            print(f"   - 已加载合成数据(GAN): {len(data_items)}条")
        else:
            print(f"找不到合成数据文件: {gan_jsonl}")
            exit(1)

        # 第二步：混入 Raw (真实) 数据
        raw_config = CONFIG_POOL["raw"]
        raw_jsonl = raw_config["train_file"]
        raw_root = raw_config["audio_root"]

        if os.path.exists(raw_jsonl) and mix_ratio > 0:
            raw_items = []
            with open(raw_jsonl, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        raw_items.append(json.loads(line))

            # 随机打乱并抽取
            random.seed(42)  # Fix the random seed for this sampling step.
            random.shuffle(raw_items)

            # 计算需要混入的数量 (例如 GAN有2000条，则混入 2000*0.2 = 400条)
            mix_count = int(len(data_items) * mix_ratio)
            # The raw sample count is a proportion of the current dataset size.
            # 但如果 raw 数据本身很少，就全用上
            mix_count = min(mix_count, len(raw_items))

            print(f'   - 正在混入真实数据(Raw): 抽取 {mix_count} 条 (Mix Ratio={mix_ratio})')

            for i in range(mix_count):
                item = raw_items[i]
                # 标记：这是 Raw 数据，使用 raw_root
                item["root_path"] = raw_root
                data_items.append(item)
        else:
            print("未找到 Raw 数据或 mix_ratio=0，将仅使用 GAN 数据。")

        # 打乱最终列表，让真实数据和合成数据穿插在一起
        random.shuffle(data_items)

        # 第三步：预处理并加载到内存
        print(f"正在将 {len(data_items)}条混合音频加载到内存...")

        success_count = 0
        for item in tqdm(data_items, desc="Caching Mixed Audio"):
            # 动态获取该条数据对应的音频文件夹路径
            current_root = item["root_path"]
            filename = os.path.basename(item["audio_file"])
            wav_path = os.path.join(current_root, filename)

            try:
                if not os.path.exists(wav_path):
                    continue

                audio = load_audio_fast(wav_path)

                # 提取特征
                input_features = self.processor.feature_extractor(
                    audio, sampling_rate=16000
                ).input_features[0]

                # 提取文本
                labels = self.processor.tokenizer(item["ASR_transcript_clean"]).input_ids

                self.cached_features.append({"input_features": input_features, "labels": labels})
                success_count += 1
            except Exception as e:
                continue

        print(f'混合加载完成！有效数据: {success_count} 条。')

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

    print(f"加载 Round 2 模型: {LOCAL_MODEL_PATH}")
    processor = WhisperProcessor.from_pretrained(LOCAL_MODEL_PATH)
    model = WhisperForConditionalGeneration.from_pretrained(LOCAL_MODEL_PATH)

    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []
    model.config.use_cache = False
    model.gradient_checkpointing_disable()

    # 初始化混合数据集 (混入 20% 真实数据)
    train_dataset = MixedATCDataset(TRAIN_FILE, processor, AUDIO_ROOT, mix_ratio=0.2)
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

    training_args = Seq2SeqTrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        # 学习率保持极低，防止带偏
        learning_rate=5e-6,
        # 步数控制：因为混入了数据，总数据量变大了，可以稍微多跑一点点
        max_steps=2500,
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

    print(f"\n开始 Round 3 (GAN + Real Mixed Training)...")
    trainer.train()

    model.save_pretrained(OUTPUT_DIR)
    processor.save_pretrained(OUTPUT_DIR)
    print(f"最终混合模型训练完成！保存在: {os.path.abspath(OUTPUT_DIR)}")


if __name__ == "__main__":
    main()
