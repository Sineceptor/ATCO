# ---------------------------------------------------------------
# Archived experiment (late 2025). Original file: ASR/augment_test_robustness.py
# What it does: Builds two stress-test copies of the test set: added Gaussian noise, and 1.1x speed.
# Known problems:
#   - No random seed.
# Not maintained: paths and dependencies are as they were at the time.
# ---------------------------------------------------------------
import json
import os
import numpy as np
import soundfile as sf
import scipy.signal
import time
from tqdm import tqdm

# Configuration
INPUT_JSONL = "processed_data/test.jsonl"
OUTPUT_AUDIO_DIR = "processed_data/audio_test_augmented"

# Keep separate test variants for clean, noisy and faster speech.
OUTPUT_JSONL_NOISY = "processed_data/test_noisy.jsonl"
OUTPUT_JSONL_SPEED = "processed_data/test_speed.jsonl"

# 开关
ENABLE_NOISE = True
ENABLE_SPEED = True


def load_audio(path):
    audio, sr = sf.read(path)
    if len(audio.shape) > 1:
        audio = audio.mean(axis=1)
    return audio, sr


def add_noise(audio, noise_level=0.005):
    noise = np.random.randn(len(audio))
    return audio + noise_level * noise


def change_speed_resample(audio, rate):
    new_len = int(len(audio) / rate)
    return scipy.signal.resample_poly(audio, new_len, len(audio))


def main():
    print(f"启动测试集扩充 (用于鲁棒性评估)...")

    if not os.path.exists(OUTPUT_AUDIO_DIR):
        os.makedirs(OUTPUT_AUDIO_DIR, exist_ok=True)

    with open(INPUT_JSONL, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    print(f"原始测试样本: {len(lines)}")

    noisy_entries = []
    speed_entries = []

    start_time = time.time()

    for line in tqdm(lines, unit="file"):
        item = json.loads(line)

        # 路径容错
        original_filename = os.path.basename(item["audio_file"])
        possible_paths = [
            item["audio_file"],
            os.path.join("processed_data/audio", original_filename),
            os.path.join(os.getcwd(), "processed_data/audio", original_filename),
        ]
        original_path = next((p for p in possible_paths if os.path.exists(p)), None)

        if not original_path:
            continue

        try:
            audio, sr = load_audio(original_path)
            base_name = os.path.splitext(original_filename)[0]
            ext = os.path.splitext(original_filename)[1]

            # 生成 Noisy Test Set
            if ENABLE_NOISE:
                audio_noise = add_noise(audio, noise_level=0.008)
                fname = f"{base_name}_noise{ext}"
                fpath = os.path.join(OUTPUT_AUDIO_DIR, fname)
                sf.write(fpath, audio_noise, sr)

                entry = item.copy()
                entry["audio_file"] = os.path.abspath(fpath)
                entry["utt_id"] = f"{item['utt_id']}_noise"
                noisy_entries.append(entry)

            # 生成 Speed Test Set (只做变快 1.1x 用于压力测试)
            if ENABLE_SPEED:
                # 为了评估的一致性，这里固定倍率 1.1 (模拟快语速)，不随机
                rate = 1.1
                audio_speed = change_speed_resample(audio, rate)

                fname = f"{base_name}_speed{rate}{ext}"
                fpath = os.path.join(OUTPUT_AUDIO_DIR, fname)
                sf.write(fpath, audio_speed, sr)

                entry = item.copy()
                entry["audio_file"] = os.path.abspath(fpath)
                entry["utt_id"] = f"{item['utt_id']}_speed_{rate}"
                entry["duration_sec"] = len(audio_speed) / sr
                speed_entries.append(entry)

        except Exception as e:
            print(f"跳过 {original_filename}: {e}")

    # 保存两个独立的文件
    print(f'\n保存加噪测试集: {OUTPUT_JSONL_NOISY} ({len(noisy_entries)}条)')
    with open(OUTPUT_JSONL_NOISY, "w", encoding="utf-8") as f:
        for entry in noisy_entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f'保存变速测试集: {OUTPUT_JSONL_SPEED} ({len(speed_entries)}条)')
    with open(OUTPUT_JSONL_SPEED, "w", encoding="utf-8") as f:
        for entry in speed_entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print("Test variants saved. Run evaluation separately to measure WER.")


if __name__ == "__main__":
    main()
