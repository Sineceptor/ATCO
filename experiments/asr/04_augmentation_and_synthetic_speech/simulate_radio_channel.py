# ---------------------------------------------------------------
# Archived experiment (late 2025). Original file: ASR/make_it_real.py
# What it does: Makes the clean TTS speech sound like VHF radio: 8 kHz band-limiting, brown (integrated white) engine noise, white static, mu-law companding, 300-3400 Hz Butterworth band-pass, gain + clipping, and a squelch burst on half the clips.
# Known problems:
#   - This is signal processing, not a GAN, although later scripts call its output 'GAN' data.
#   - Real airband AM radio does not use mu-law; that stage imitates telephone-style distortion.
#   - No random seed, and its effect on WER was never isolated.
# Not maintained: paths and dependencies are as they were at the time.
# ---------------------------------------------------------------
import os
import json
import random
import numpy as np
import soundfile as sf
import scipy.signal
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

# Configuration
# Input: synthetic speech from the earlier generation step.
INPUT_JSONL = "processed_data/train_synthetic_ultimate_wav_final.jsonl"

# 输出：这一轮生成的“音质真实”的数据
OUTPUT_DIR = "processed_data/audio_synthetic_real_pro"
OUTPUT_JSONL = "processed_data/train_synthetic_real_pro.jsonl"

NUM_WORKERS = 8  # 并行核数


# 黑科技：真实无线电模拟算法


def float2pcm(sig, dtype="int16"):
    """Convert floating-point audio to integer PCM."""
    if dtype == "int16":
        return (sig * 32767).astype(np.int16)
    return sig


def mulaw_compression(signal, quantization_channels=256):
    """Apply mu-law companding to approximate a compressed radio sound."""
    mu = quantization_channels - 1
    # 压缩 (Encoding)
    signal = np.sign(signal) * np.log(1 + mu * np.abs(signal)) / np.log(1 + mu)
    # 量化 (Quantization - 降低位深)
    signal = np.round(signal * 128) / 128
    # 解压 (Decoding)
    signal = np.sign(signal) * (1 / mu) * ((1 + mu) ** np.abs(signal) - 1)
    return signal


def generate_brown_noise(N):
    """Generate low-frequency noise for the audio augmentation."""
    # 积分白噪声得到布朗噪声
    white = np.random.normal(0, 1, N)
    brown = np.cumsum(white)
    # 归一化
    if np.max(np.abs(brown)) > 0:
        brown /= np.max(np.abs(brown))
    return brown


def generate_squelch_burst(sr, duration_ms=150):
    """Generate a short noise burst resembling a radio switch release."""
    samples = int(sr * (duration_ms / 1000.0))
    # 爆破音是高强度的粉红噪声
    burst = np.random.normal(0, 0.5, samples)
    # 加一个淡出效果
    envelope = np.linspace(1, 0, samples) ** 2
    return burst * envelope


def apply_radio_chain(audio, sr):
    """Apply the radio-effect augmentation steps."""
    # 降采样模拟 (Downsampling)
    # 真实无线电带宽很窄，先把高清音频“劣化”
    # 先降采样到 8000Hz (电话音质)，再插值回 16000Hz
    orig_len = len(audio)
    resampled = scipy.signal.resample(audio, int(orig_len * 8000 / sr))
    audio = scipy.signal.resample(resampled, orig_len)

    # 引擎背景音 (Engine Drone - Brown Noise)
    # 这种噪音是低频的，一直存在
    noise_level = random.uniform(0.01, 0.05)  # 随机引擎音量
    engine_noise = generate_brown_noise(len(audio)) * noise_level
    audio = audio + engine_noise

    # 信号底噪 (Static Hiss - White Noise)
    # 这种噪音是高频的
    static_level = random.uniform(0.005, 0.02)
    white_noise = np.random.normal(0, 1, len(audio)) * static_level
    audio = audio + white_noise

    # G.711 µ-law 压缩 (The Digital Grit)
    # 这是灵魂步骤！让声音听起来像从电话里传出来的
    audio = mulaw_compression(audio)

    # 带通滤波 (300-3400Hz)
    # 切掉不该有的高低频
    nyquist = 0.5 * sr
    b, a = scipy.signal.butter(4, [300 / nyquist, 3400 / nyquist], btype="band")
    audio = scipy.signal.lfilter(b, a, audio)

    # 麦克风过载 (Clipping)
    # 模拟飞行员嘴巴贴着麦克风喊话
    gain = random.uniform(1.5, 3.0)
    audio = np.clip(audio * gain, -0.95, 0.95)

    # (可选) 添加首尾的“咔嚓”声 (Squelch)
    # 50% 的概率在结尾出现爆破音
    if random.random() < 0.5:
        burst = generate_squelch_burst(sr) * random.uniform(0.1, 0.3)
        # 拼接到结尾
        audio = np.concatenate([audio, burst])

    return audio


def process_single_file(item):
    try:
        src_path = item["audio_file"]

        # 读取
        audio, sr = sf.read(src_path)

        # 应用无线电链路
        audio_processed = apply_radio_chain(audio, sr)

        # 保存
        filename = os.path.basename(src_path)
        dst_path = os.path.join(OUTPUT_DIR, filename)

        sf.write(dst_path, audio_processed, sr, subtype="PCM_16")

        # 更新条目
        new_item = item.copy()
        new_item["audio_file"] = os.path.abspath(dst_path)
        return new_item

    except Exception as e:
        print(f"Error {item['utt_id']}: {e}")
        return None


def main():
    print(f"启动无线电效果处理...")
    print(f"特效链路: 降采样(8k) -> 引擎红噪 -> G.711压缩 -> 带通滤波 -> PTT爆破音")

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR, exist_ok=True)

    with open(INPUT_JSONL, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f if line.strip()]

    print(f'处理数量: {len(data)} 条')

    processed_data = []
    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        results = list(tqdm(executor.map(process_single_file, data), total=len(data), unit="file"))
        for res in results:
            if res is not None:
                processed_data.append(res)

    with open(OUTPUT_JSONL, "w", encoding="utf-8") as f:
        for entry in processed_data:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print("=" * 40)
    print("音频处理完成！")
    print("现在的听感差异：")
    print("   1. 声音变‘闷’了 (模拟 8k 采样)")
    print("   2. 背景有‘轰隆隆’的低频声 (模拟引擎)")
    print("   3. 声音有‘沙沙’的数码味 (模拟 G.711 压缩)")
    print(f"新数据列表: {OUTPUT_JSONL}")
    print("=" * 40)


if __name__ == "__main__":
    main()
