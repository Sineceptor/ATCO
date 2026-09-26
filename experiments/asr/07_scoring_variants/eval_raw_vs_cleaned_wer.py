# ---------------------------------------------------------------
# Archived experiment (late 2025). Original file: ASR/eval_normalizer.py
# What it does: Reports WER twice: plainly, and after removing noise tags and foreign greetings from both sides.
# Known problems:
#   - The scoring cleaner deletes tag strings such as HES and UNK wherever they appear, in both reference and hypothesis, so it can also cut into real words (THESE -> TE).
#   - Changing the reference changes what is being measured.
# Not maintained: paths and dependencies are as they were at the time.
# ---------------------------------------------------------------
import torch
import json
import re
import os
import glob
from tqdm import tqdm
from datasets import load_dataset, Audio
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from transformers.models.whisper.english_normalizer import BasicTextNormalizer
import evaluate

# Configuration
MODEL_PATH = "./whisper-atco2-speed"
TEST_DATA_PATH = "processed_data/test.jsonl"

BATCH_SIZE = 16
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 自动寻路系统 (移植自训练脚本)
FOUND_AUDIO_DIRS = set()


def scan_for_audio_files(base_dir):
    """Find directories containing WAV files."""
    print(f'正在扫描音频文件夹 (在 {base_dir} 下)...')
    # 查找所有 .wav 文件
    wav_files = glob.glob(os.path.join(base_dir, "**", "*.wav"), recursive=True)
    dirs = set()
    for f in wav_files:
        dirs.add(os.path.dirname(f))
    print(f'找到了 {len(dirs)} 个包含音频的文件夹')
    return dirs


def fix_audio_path(batch):
    """Look for the audio file in the available recording directories."""
    path = batch["audio_file"]
    # 如果已经存在，直接返回
    if os.path.exists(path):
        return batch

    filename = os.path.basename(path)
    # 在扫描到的文件夹里找
    for d in FOUND_AUDIO_DIRS:
        potential_path = os.path.join(d, filename)
        if os.path.exists(potential_path):
            batch["audio_file"] = potential_path
            return batch

    # 实在找不到，尝试绝对路径
    batch["audio_file"] = os.path.abspath(path)
    return batch


# ATC 专用清洗器
std_normalizer = BasicTextNormalizer()


def clean_atc_text(text):
    if not text:
        return ""
    text = text.upper()

    # 移除噪音标签
    noise_tags = ["HES", "UNK", "NOISE", "NE CZECH", "NE FRENCH", "NE GERMAN", "NE SLOVAK"]
    for tag in noise_tags:
        text = text.replace(tag, "")

    # Remove greetings before scoring. This changes the evaluation target.
    foreign_greetings = [
        "DOBRY DEN",
        "DOBRE RANO",
        "DOBRE POPOLUDNIE",
        "DOBRY VECER",
        "HEZKY VECER",
        "DEKUJEM",
        "NA SLYSENOU",
        "AHOJ",
        "BONJOUR",
        "BONNE SOIREE",
        "A TOUTE",
        "AU REVOIR",
        "MERCI",
        "GRUEZI",
        "SERVUS",
        "TSCHUSS",
        "BIS SPATER",
        "DANKE",
        "GUTEN TAG",
    ]
    for g in foreign_greetings:
        text = text.replace(g, "")

    text = text.replace(".", " DECIMAL ")
    return std_normalizer(text)


# 主逻辑
def main():
    # 先扫描路径
    global FOUND_AUDIO_DIRS
    FOUND_AUDIO_DIRS = scan_for_audio_files(os.getcwd())

    print(f"加载模型: {MODEL_PATH}")
    try:
        processor = WhisperProcessor.from_pretrained(MODEL_PATH)
        model = WhisperForConditionalGeneration.from_pretrained(MODEL_PATH).to(DEVICE)
    except Exception as e:
        print(f"模型加载失败: {e}")
        return

    model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(
        language="English", task="transcribe"
    )
    wer_metric = evaluate.load("wer")

    print(f"加载测试数据: {TEST_DATA_PATH}")
    if not os.path.exists(TEST_DATA_PATH):
        print("找不到测试文件!")
        exit(1)

    dataset = load_dataset("json", data_files=TEST_DATA_PATH, split="train")

    # 修复路径 (关键修复步骤)
    print("正在修复音频路径...")
    dataset = dataset.map(fix_audio_path)

    # 过滤掉仍然找不到文件的
    dataset = dataset.filter(lambda x: os.path.exists(x["audio_file"]))
    print(f"有效测试样本: {len(dataset)}")

    # 预处理音频
    print("预处理音频...")
    dataset = dataset.cast_column("audio_file", Audio(sampling_rate=16000))

    def map_to_pred(batch):
        audio = batch["audio_file"]
        input_features = processor(
            audio["array"], sampling_rate=16000, return_tensors="pt"
        ).input_features

        with torch.no_grad():
            predicted_ids = model.generate(input_features.to(DEVICE))

        transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]

        return {"reference": batch["ASR_transcript_clean"], "prediction": transcription}

    print("开始推理...")
    # Recompute the mapped dataset instead of using cached features.
    results = dataset.map(
        map_to_pred, remove_columns=dataset.column_names, load_from_cache_file=False
    )

    refs_raw = results["reference"]
    preds_raw = results["prediction"]

    wer_raw = 100 * wer_metric.compute(predictions=preds_raw, references=refs_raw)

    refs_clean = [clean_atc_text(r) for r in refs_raw]
    preds_clean = [clean_atc_text(p) for p in preds_raw]

    clean_pairs = [(p, r) for p, r in zip(preds_clean, refs_clean) if r.strip() != ""]
    if clean_pairs:
        preds_clean, refs_clean = zip(*clean_pairs)
        wer_clean = 100 * wer_metric.compute(predictions=preds_clean, references=refs_clean)
    else:
        wer_clean = 0.0

    print("\n" + "=" * 40)
    print(f"评估报告 (样本数: {len(results)})")
    print("=" * 40)
    print(f"原始 WER: {wer_raw:.2f}%")
    print(f"清洗 WER: {wer_clean:.2f}%")
    print("=" * 40)

    print("\n差异示例:")
    for i in range(min(5, len(results))):
        print(f"\n[样本 {i + 1}]")
        print(f"Ref (原): {refs_raw[i]}")
        print(f"Hyp (预): {preds_raw[i]}")
        print(f"-> Ref (洗): {clean_atc_text(refs_raw[i])}")
        print(f"-> Hyp (洗): {clean_atc_text(preds_raw[i])}")


if __name__ == "__main__":
    main()
