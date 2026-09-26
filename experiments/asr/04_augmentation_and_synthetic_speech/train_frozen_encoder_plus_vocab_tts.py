# ---------------------------------------------------------------
# Archived experiment (late 2025). Original file: ASR/train_atc_with_vocab.py
# What it does: Generates TTS clips for every 'critical' location and callsign in vocab_master.json and trains on them at 2x weight.
# Known problems:
#   - Labels already start with <|startoftranscript|> and the collator's bos check never fires for Whisper, so the decoder sees the start token twice.
#   - Vocabulary or prompt words were chosen after looking at test-set errors, so the test score is optimistic.
# Not maintained: paths and dependencies are as they were at the time.
# ---------------------------------------------------------------
import json
import os
import glob
import torch
import random
import asyncio
import edge_tts
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

# Configuration
MODEL_NAME = "openai/whisper-small"
OUTPUT_DIR = "./whisper-atco2-vocab-aware"
VOCAB_FILE = "processed_data/vocab_master.json"

# 数据路径
REAL_DATA_PATH = "processed_data/train.jsonl"
AUGMENTED_DATA_PATH = "processed_data/train_augmented.jsonl"
VOCAB_FIX_DATA_DIR = "processed_data/audio_vocab_fix"
VOCAB_FIX_JSONL = "processed_data/train_vocab_fix.jsonl"

# 训练参数
BATCH_SIZE = 32
GRADIENT_ACCUMULATION = 1
LEARNING_RATE = 1e-4
NUM_EPOCHS = 3

# 语音合成角色
VOICES = ["en-GB-RyanNeural", "en-GB-SoniaNeural", "en-US-GuyNeural", "en-US-AriaNeural"]


# 第一部分：基于词库的生成系统


async def convert_to_wav(input_path, output_path):
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


async def generate_vocab_sample(word, category, i, sem, results):
    async with sem:
        if category == "LOCATIONS":
            templates = [
                f"{word} TOWER GOOD DAY",
                f"CONTACT {word} GROUND ONE TWO ONE DECIMAL NINE",
                f"APPROACHING {word} RADAR",
                f"DEPARTURE FROM {word} RUNWAY TWO FOUR",
                f"CLEARED TO LAND {word}",
            ]
        elif category == "CALLSIGNS":
            templates = [
                f"{word} ONE FIVE ALPHA REQUEST TAXI",
                f"STEFANIK TOWER {word} TWO ZERO NINE",
                f"REPORT DOWNWIND {word} ONE ONE",
                f"HOLD POSITION {word} SIX SIX",
                f"{word} CONTACT TOWER BYE BYE",
            ]
        else:
            templates = [f"SAY AGAIN {word}", f"CONFIRM {word}"]

        text = random.choice(templates)
        voice = random.choice(VOICES)
        filename_base = f"VOCAB_{word}_{i:02d}"
        filename_base = "".join([c for c in filename_base if c.isalnum() or c == "_"])

        mp3_path = os.path.join(VOCAB_FIX_DATA_DIR, f"{filename_base}.mp3")
        wav_path = os.path.join(VOCAB_FIX_DATA_DIR, f"{filename_base}.wav")

        try:
            comm = edge_tts.Communicate(text, voice, rate="+10%")
            await comm.save(mp3_path)
            await convert_to_wav(mp3_path, wav_path)
            if os.path.exists(wav_path):
                results.append(
                    {"audio_file": os.path.abspath(wav_path), "ASR_transcript_clean": text}
                )
            if os.path.exists(mp3_path):
                os.remove(mp3_path)
        except Exception:
            pass


async def auto_generate_vocab_data():
    if not os.path.exists(VOCAB_FILE):
        print(f"找不到词库文件: {VOCAB_FILE}")
        return
    if os.path.exists(VOCAB_FIX_JSONL):
        print("检测到词库增强数据已存在，跳过生成步骤。")
        return

    print("读取词库，准备进行针对性补全...")
    with open(VOCAB_FILE, "r", encoding="utf-8") as f:
        vocab = json.load(f)

    tasks = []
    results = []
    sem = asyncio.Semaphore(20)
    if not os.path.exists(VOCAB_FIX_DATA_DIR):
        os.makedirs(VOCAB_FIX_DATA_DIR)

    locs = vocab.get("CRITICAL_LOCATIONS", {})
    print(f'    正在为 {len(locs)} 个关键地点生成训练数据...')
    for word in locs.keys():
        for i in range(20):
            tasks.append(generate_vocab_sample(word, "LOCATIONS", i, sem, results))

    calls = vocab.get("CRITICAL_CALLSIGNS", {})
    print(f'    正在为 {len(calls)} 个关键呼号生成训练数据...')
    for word in calls.keys():
        for i in range(15):
            tasks.append(generate_vocab_sample(word, "CALLSIGNS", i, sem, results))

    from tqdm import tqdm

    print(f'    开始生成约 {len(tasks)} 条音频...')
    for f in tqdm(asyncio.as_completed(tasks), total=len(tasks)):
        await f

    with open(VOCAB_FIX_JSONL, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    print(f'词库增强数据生成完毕: {len(results)} 条')


# 第二部分：训练配置

FOUND_AUDIO_DIRS = set()


def scan_for_audio_files(base_dir):
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
        if os.path.exists(os.path.join(d, filename)):
            batch["audio_file"] = os.path.join(d, filename)
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
    asyncio.run(auto_generate_vocab_data())

    print("启动基于词库的训练 (Vocab-Aware Training)...")

    project_root = os.getcwd()
    FOUND_AUDIO_DIRS = scan_for_audio_files(project_root)

    processor = WhisperProcessor.from_pretrained(MODEL_NAME, language="English", task="transcribe")

    data_files = {}
    if os.path.exists(REAL_DATA_PATH):
        data_files["real"] = REAL_DATA_PATH
    if os.path.exists(AUGMENTED_DATA_PATH):
        data_files["aug"] = AUGMENTED_DATA_PATH
    if os.path.exists(VOCAB_FIX_JSONL):
        data_files["vocab"] = VOCAB_FIX_JSONL

    raw_datasets = load_dataset("json", data_files=data_files)
    datasets_list = []
    if "real" in raw_datasets:
        datasets_list.append(raw_datasets["real"])
    if "aug" in raw_datasets:
        datasets_list.append(raw_datasets["aug"])
    if "vocab" in raw_datasets:
        print("注入词库补全数据 (权重 x2)...")
        datasets_list.append(raw_datasets["vocab"])
        datasets_list.append(raw_datasets["vocab"])

    combined_dataset = concatenate_datasets(datasets_list)
    print(f'最终训练集规模: {len(combined_dataset)} 条')

    combined_dataset = combined_dataset.map(fix_audio_path).filter(
        lambda x: os.path.exists(x["audio_file"])
    )
    combined_dataset = combined_dataset.cast_column("audio_file", Audio(sampling_rate=16000))

    tokenized_dataset = combined_dataset.map(
        prepare_dataset,
        remove_columns=combined_dataset.column_names,
        num_proc=1,
        fn_kwargs={"processor": processor},
    ).with_format("torch")

    split = tokenized_dataset.train_test_split(test_size=0.05)

    model = WhisperForConditionalGeneration.from_pretrained(MODEL_NAME)
    model.freeze_encoder()
    model.gradient_checkpointing_disable()
    model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(
        language="English", task="transcribe"
    )

    args = Seq2SeqTrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION,
        learning_rate=LEARNING_RATE,
        num_train_epochs=NUM_EPOCHS,
        fp16=True,
        optim="adafactor",
        # 改用 eval_strategy
        eval_strategy="steps",
        eval_steps=1000,
        save_steps=1000,
        logging_steps=50,
        save_total_limit=2,
        dataloader_num_workers=0,
        remove_unused_columns=False,
    )

    trainer = Seq2SeqTrainer(
        args=args,
        model=model,
        train_dataset=split["train"],
        eval_dataset=split["test"],
        data_collator=DataCollatorSpeechSeq2SeqWithPadding(processor=processor),
        processing_class=processor.feature_extractor,
    )

    print("\n开始训练...")
    trainer.train()

    model.save_pretrained(OUTPUT_DIR)
    processor.save_pretrained(OUTPUT_DIR)
    print("训练完成！")
