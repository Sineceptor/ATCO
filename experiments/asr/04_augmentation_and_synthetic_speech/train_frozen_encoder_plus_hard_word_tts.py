# ---------------------------------------------------------------
# Archived experiment (late 2025). Original file: ASR/train_atc_augment_v3.py
# What it does: Adds 2,000 extra TTS sentences built around words the model kept getting wrong (STEFANIK, SION, NAV CHECKER...), weighted 2x.
# Known problems:
#   - Labels already start with <|startoftranscript|> and the collator's bos check never fires for Whisper, so the decoder sees the start token twice.
#   - The hard words were taken from test-set errors. Vocabulary or prompt words were chosen after looking at test-set errors, so the test score is optimistic.
# Not maintained: paths and dependencies are as they were at the time.
# ---------------------------------------------------------------
import json
import os
import glob
import torch
import random
import asyncio
import edge_tts
import subprocess
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

# 核心配置区域
MODEL_NAME = "openai/whisper-small"
OUTPUT_DIR = "./whisper-atco2-sota-fixed"

# 现有数据路径 (请确保这些文件存在)
REAL_DATA_PATH = "processed_data/train.jsonl"  # 真实数据 (700条)
AUGMENTED_DATA_PATH = "processed_data/train_augmented.jsonl"  # 增强数据 (1402条) - 关键！
SYNTHETIC_DATA_PATH = "processed_data/train_synthetic_real_pro.jsonl"  # 合成数据 (5000条)

# 自动生成的“纠错补丁”数据路径
HARD_DATA_DIR = "processed_data/audio_synthetic_hard"
HARD_DATA_JSONL = "processed_data/train_synthetic_hard.jsonl"

# 训练参数 (8GB 显存光速版)
BATCH_SIZE = 32
GRADIENT_ACCUMULATION = 1
LEARNING_RATE = 1e-4
NUM_EPOCHS = 3

# 第一部分：自动造血系统 (Auto Data Gen)

# Synthetic examples based on observed test-set errors.
HARD_VOCAB = {
    "LOCATIONS": ["STEFANIK", "SION", "RUZYNE", "BERN", "ZURICH", "BRNO", "PRAGUE", "BRATISLAVA"],
    "CALLSIGNS": [
        "MOSQUITO",
        "NAV CHECKER",
        "TATRA",
        "BAIR",
        "TIME AIR",
        "SLOVAK GOVERNMENT",
        "TRANS EUROPE",
        "SKY TRAVEL",
        "ELITE JET",
    ],
    "GREETINGS": ["DOBRY DEN", "DOBRE RANO", "AHOJ", "BONJOUR", "BONNE SOIREE"],
}

VOICES = [
    "en-GB-RyanNeural",
    "en-GB-SoniaNeural",
    "en-IE-EmilyNeural",
    "en-US-GuyNeural",
    "en-AU-WilliamNeural",
]


async def convert_to_wav(input_path, output_path):
    # 自动转码
    cmd = [
        "ffmpeg",
        "-i",
        input_path,
        "-ar",
        "16000",
        "-ac",
        "1",
        "-y",
        "-loglevel",
        "error",
        output_path,
    ]
    proc = await asyncio.create_subprocess_exec(*cmd)
    await proc.wait()


async def generate_hard_sample(i, sem, results):
    async with sem:
        # 随机组合针对性句子，强行纠正 Sydney 偏见
        loc = random.choice(HARD_VOCAB["LOCATIONS"])
        cs = f"{random.choice(HARD_VOCAB['CALLSIGNS'])} {random.choice(['ONE', 'TWO', 'FIVE'])} {random.choice(['ALPHA', 'BRAVO'])}"
        greet = random.choice(HARD_VOCAB["GREETINGS"])

        templates = [
            f"{loc} TOWER {cs} {greet}",
            f"{cs} CONTACT {loc} GROUND",
            f"REPORT DOWNWIND RUNWAY TWO FIVE {cs}",
            f"{loc} RADAR {cs} PASSING FLIGHT LEVEL ONE ZERO ZERO",
            f"CLEARED TO LAND {loc} {cs}",
        ]
        text = random.choice(templates)

        # 生成
        voice = random.choice(VOICES)
        filename_base = f"HARD_{i:04d}"
        mp3_path = os.path.join(HARD_DATA_DIR, f"{filename_base}.mp3")
        wav_path = os.path.join(HARD_DATA_DIR, f"{filename_base}.wav")

        try:
            comm = edge_tts.Communicate(text, voice, rate="+0%")
            await comm.save(mp3_path)
            await convert_to_wav(mp3_path, wav_path)

            if os.path.exists(wav_path):
                results.append(
                    {"audio_file": os.path.abspath(wav_path), "ASR_transcript_clean": text}
                )
            if os.path.exists(mp3_path):
                os.remove(mp3_path)
        except:
            pass


async def auto_generate_hard_data():
    if os.path.exists(HARD_DATA_JSONL):
        print("检测到针对性训练数据已存在，跳过生成。")
        return

    print("正在自动生成 2000 条【纠错数据】(修复 Sydney/Mosquito 问题)...")
    if not os.path.exists(HARD_DATA_DIR):
        os.makedirs(HARD_DATA_DIR)

    sem = asyncio.Semaphore(15)  # 并发数
    results = []
    tasks = [generate_hard_sample(i, sem, results) for i in range(2000)]

    done_count = 0
    for f in asyncio.as_completed(tasks):
        await f
        done_count += 1
        if done_count % 200 == 0:
            print(f"   已生成: {done_count}/2000...")

    with open(HARD_DATA_JSONL, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    print(f"纠错数据生成完毕！")


# 第二部分：极速训练系统

# 自动寻路函数
FOUND_AUDIO_DIRS = set()


def scan_for_audio_files(base_dir):
    print(f"正在全盘扫描音频文件 (为了修复路径错误)...")
    wav_files = glob.glob(os.path.join(base_dir, "**", "*.wav"), recursive=True)
    dirs = set()
    for f in wav_files:
        dirs.add(os.path.dirname(f))
    print(f'找到了 {len(dirs)} 个音频目录')
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

    def __call__(self, features):
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
        audio["array"], sampling_rate=16000
    ).input_features[0]
    batch["labels"] = processor.tokenizer(batch["ASR_transcript_clean"]).input_ids
    return batch


# 主程序
if __name__ == "__main__":
    multiprocessing.freeze_support()

    # 启动前先生成纠错数据
    try:
        asyncio.run(auto_generate_hard_data())
    except Exception as e:
        print(f'生成部分数据警告: {e} (将尝试继续)')

    print("启动全自动修复训练 (Auto-Fix & Train)...")

    # 扫描文件
    project_root = os.getcwd()
    FOUND_AUDIO_DIRS = scan_for_audio_files(project_root)

    processor = WhisperProcessor.from_pretrained(MODEL_NAME, language="English", task="transcribe")

    # 加载数据 (Real + Aug + Syn + Hard)
    files = {
        "real": REAL_DATA_PATH,
        "aug": AUGMENTED_DATA_PATH,  # 1402条增强数据在这里！
        "syn": SYNTHETIC_DATA_PATH,
        "hard": HARD_DATA_JSONL,  # 2000条纠错数据在这里！
    }

    # 自动过滤不存在的文件
    data_files = {k: v for k, v in files.items() if os.path.exists(v)}
    if not data_files:
        print("严重错误：找不到任何数据文件！")
        exit(1)

    raw_datasets = load_dataset("json", data_files=data_files)

    # 混合策略
    datasets_list = []
    if "real" in raw_datasets:
        datasets_list.append(raw_datasets["real"])
    if "aug" in raw_datasets:
        datasets_list.append(raw_datasets["aug"])  # 增强数据加入
    if "syn" in raw_datasets:
        datasets_list.append(raw_datasets["syn"])

    # 关键策略：给 Hard 数据加 2 倍权重，强迫模型记住
    if "hard" in raw_datasets:
        print("正在注入【纠错数据】(2x 权重)...")
        datasets_list.append(raw_datasets["hard"])
        datasets_list.append(raw_datasets["hard"])

    combined_dataset = concatenate_datasets(datasets_list)
    print(f'最终训练集规模: {len(combined_dataset)} 条')

    # 修复路径 & 过滤 & 转换
    # 这一步会自动修复 aug 数据里的相对路径问题
    combined_dataset = combined_dataset.map(fix_audio_path, load_from_cache_file=False)
    combined_dataset = combined_dataset.filter(lambda x: os.path.exists(x["audio_file"]))
    combined_dataset = combined_dataset.cast_column("audio_file", Audio(sampling_rate=16000))

    print("正在预处理数据至内存 (In-Memory)...")
    tokenized_dataset = combined_dataset.map(
        prepare_dataset,
        remove_columns=combined_dataset.column_names,
        num_proc=1,  # Windows 必须为 1
        fn_kwargs={"processor": processor},
    ).with_format("torch")  # 关键加速点

    split = tokenized_dataset.train_test_split(test_size=0.05)
    train_ds = split["train"]
    val_ds = split["test"]

    # 加载模型 (冻结加速版)
    print(f"\n加载模型并应用加速策略...")
    model = WhisperForConditionalGeneration.from_pretrained(MODEL_NAME)
    model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(
        language="English", task="transcribe"
    )

    # 冻结 Encoder：省显存，速度快，不影响纠错效果
    model.freeze_encoder()
    model.gradient_checkpointing_disable()

    # 训练配置
    args = Seq2SeqTrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=BATCH_SIZE,  # 32 (得益于冻结和Adafactor)
        gradient_accumulation_steps=GRADIENT_ACCUMULATION,
        learning_rate=LEARNING_RATE,
        num_train_epochs=NUM_EPOCHS,
        gradient_checkpointing=False,
        fp16=True,
        optim="adafactor",  # 省显存神器
        eval_strategy="steps",
        per_device_eval_batch_size=16,
        predict_with_generate=True,
        generation_max_length=225,
        save_steps=500,
        eval_steps=500,
        logging_steps=50,
        save_total_limit=2,
        dataloader_num_workers=0,  # Windows 必须为 0
        remove_unused_columns=False,
    )

    trainer = Seq2SeqTrainer(
        args=args,
        model=model,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=DataCollatorSpeechSeq2SeqWithPadding(processor=processor),
        compute_metrics=compute_metrics,
        processing_class=processor.feature_extractor,
    )

    print("\n引擎全开！开始训练...")
    trainer.train()

    print(f"\n保存最终模型到: {OUTPUT_DIR}")
    model.save_pretrained(OUTPUT_DIR)
    processor.save_pretrained(OUTPUT_DIR)
    print("训练完成！现在的模型应该能分清 Stefanik 和 Sydney 了。")
