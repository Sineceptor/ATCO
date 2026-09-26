# ---------------------------------------------------------------
# Archived experiment (late 2025). Original file: ASR/augment_data.py
# What it does: Triples the training set: original + Gaussian-noise copy + 0.9x/1.1x speed copy.
# Known problems:
#   - No random seed.
#   - Originals are included in the output, so scripts that load both files count them twice.
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
INPUT_JSONL = "processed_data/train.jsonl"
OUTPUT_AUDIO_DIR = "processed_data/audio_augmented"
OUTPUT_JSONL = "processed_data/train_augmented.jsonl"

# 扩充倍数
ENABLE_NOISE = True  # 加噪
ENABLE_SPEED = True  # 变速变调 (核心数据扩充)


def load_audio(path):
    # 使用 soundfile 直接读取，避开 librosa 的依赖
    audio, sr = sf.read(path)
    # 如果是立体声，转单声道
    if len(audio.shape) > 1:
        audio = audio.mean(axis=1)
    return audio, sr


def add_noise(audio, noise_level=0.005):
    """Add Gaussian noise to the waveform."""
    noise = np.random.randn(len(audio))
    return audio + noise_level * noise


def change_speed_resample(audio, rate):
    """Resample to change speed and pitch. A rate above 1 makes the clip faster."""
    # 计算目标长度
    new_len = int(len(audio) / rate)
    # 使用多相滤波重采样 (比 FFT 重采样更安全且快)
    return scipy.signal.resample_poly(audio, new_len, len(audio))


def main():
    print(f"启动音频扩充...")
    print(f"策略变更: 使用 'Resampling' 代替 'TimeStretch'")
    print(f"处理速度取决于文件长度和硬件。")

    if not os.path.exists(OUTPUT_AUDIO_DIR):
        os.makedirs(OUTPUT_AUDIO_DIR, exist_ok=True)

    # 自动纠正 JSONL 路径 (防止路径错误)
    with open(INPUT_JSONL, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    print(f"待处理样本: {len(lines)}")

    new_entries = []
    start_time = time.time()

    # 开始处理
    for line in tqdm(lines, unit="file"):
        item = json.loads(line)

        # 路径容错逻辑
        original_filename = os.path.basename(item["audio_file"])
        possible_paths = [
            item["audio_file"],
            os.path.join("processed_data/audio", original_filename),
            os.path.join(os.getcwd(), "processed_data/audio", original_filename),
        ]

        original_path = None
        for p in possible_paths:
            if os.path.exists(p):
                original_path = p
                break

        if not original_path:
            # 文件找不到，跳过
            continue

        try:
            # 加载
            audio, sr = load_audio(original_path)

            # A. 保留原版
            # 更新绝对路径，确保训练时能找到
            item["audio_file"] = os.path.abspath(original_path)
            new_entries.append(item)

            base_name = os.path.splitext(original_filename)[0]
            ext = os.path.splitext(original_filename)[1]

            # B. 加噪版本
            if ENABLE_NOISE:
                audio_noise = add_noise(audio, noise_level=0.008)
                fname = f"{base_name}_noise{ext}"
                fpath = os.path.join(OUTPUT_AUDIO_DIR, fname)
                sf.write(fpath, audio_noise, sr)

                entry = item.copy()
                entry["audio_file"] = os.path.abspath(fpath)
                entry["utt_id"] = f"{item['utt_id']}_noise"
                new_entries.append(entry)

            # C. 变速版本 (随机快/慢)
            if ENABLE_SPEED:
                # 随机选择一个倍率: 0.9 (慢) 或 1.1 (快)
                rate = np.random.choice([0.9, 1.1])

                # Resample with SciPy.
                audio_speed = change_speed_resample(audio, rate)

                fname = f"{base_name}_speed{rate}{ext}"
                fpath = os.path.join(OUTPUT_AUDIO_DIR, fname)
                sf.write(fpath, audio_speed, sr)

                entry = item.copy()
                entry["audio_file"] = os.path.abspath(fpath)
                entry["utt_id"] = f"{item['utt_id']}_speed_{rate}"
                # 关键: 更新时长
                entry["duration_sec"] = len(audio_speed) / sr
                new_entries.append(entry)

        except Exception as e:
            print(f"跳过 {original_filename}: {e}")

    # 保存
    print(f"\n正在保存新列表到: {OUTPUT_JSONL}")
    with open(OUTPUT_JSONL, "w", encoding="utf-8") as f:
        for entry in new_entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print("=" * 40)
    print(f"扩充完成！")
    print(f'总耗时: {time.time() - start_time:.2f} 秒')
    print(f'数据量变化: {len(lines)} -> {len(new_entries)}')
    print("=" * 40)
    print("请在 train_atc_improve.py 中修改:")
    print(f"   TRAIN_FILE = '{OUTPUT_JSONL}'")


if __name__ == "__main__":
    main()
