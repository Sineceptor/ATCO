# ---------------------------------------------------------------
# Archived experiment (late 2025). Original file: ASR/evaluate_all.py
# What it does: Scores the v18 large-v3 LoRA on the clean, noisy and fast test sets. Saved result: 27.86% / 34.42% / 28.78% WER.
# Known problems:
#   - Scoring deletes every word starting with NE (NEXT, NEGATIVE...) and maps EURO SCAN -> EUROTRANS, which repairs a model error.
#   - Missing clips are skipped silently.
# Not maintained: paths and dependencies are as they were at the time.
# ---------------------------------------------------------------
import os

# 环境配置
os.environ["HF_HOME"] = "./hf_cache"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
# 限制评估时的线程数，防止底层库冲突
os.environ["OMP_NUM_THREADS"] = "1"

import json
import torch
import re
import soundfile as sf
import numpy as np
from tqdm import tqdm
from jiwer import wer, cer
from peft import PeftModel
from transformers import WhisperForConditionalGeneration, WhisperProcessor, BitsAndBytesConfig

# Configuration
# Saved V18 model.
MODEL_DIR = "atc_whisper_lora_v18_augmented"
BASE_MODEL_NAME = "openai/whisper-large-v3"

# 测试集列表 (键名 : 路径)
# Test variants produced by augment_test_robustness.py.
TEST_SETS = {
    "Original (Clean)": "processed_data/test.jsonl",
    "Noisy (+Noise)  ": "processed_data/test_noisy.jsonl",
    "Fast (Speed 1.1x)": "processed_data/test_speed.jsonl",
}

# 输出报告路径
REPORT_FILE = "final_evaluation_report.txt"

# 音频目录 (用于回退查找)
AUDIO_DIRS = ["processed_data/audio", "processed_data/audio_test_augmented"]


def clean_text(text):
    """Normalise transcript text before scoring."""
    if not text:
        return ""
    text = text.upper()
    # 去除元数据标签
    text = re.sub(r"\bNE[A-Z]+\b", "", text)
    text = re.sub(r"\b(HES|NOISE)\b", "", text)
    # 统一常见拼写 (与训练时保持一致)
    text = text.replace("RYAN AIR", "RYANAIR")
    text = text.replace("AIR PORTUGAL", "AIRPORTUGAL")
    text = text.replace("EURO TRANS", "EUROTRANS")
    text = text.replace("EURO SCAN", "EUROTRANS")
    # 去除标点和多余空格
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_model():
    print(f"正在加载基座模型 ({BASE_MODEL_NAME})...")

    # 加载 4-bit 量化的基座模型
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )

    model = WhisperForConditionalGeneration.from_pretrained(
        BASE_MODEL_NAME, quantization_config=bnb_config, device_map="auto"
    )

    # 加载 LoRA 适配器
    if os.path.exists(MODEL_DIR):
        print(f"正在挂载 LoRA 适配器: {MODEL_DIR}")
        model = PeftModel.from_pretrained(model, MODEL_DIR)
        print("LoRA 挂载成功")
    else:
        print(f"警告: 找不到 {MODEL_DIR}，正在使用原始 Whisper 模型进行评估!")

    # 加载处理器
    processor = WhisperProcessor.from_pretrained(BASE_MODEL_NAME)

    return model, processor


def evaluate_dataset(model, processor, dataset_name, file_path):
    print(f"\n正在评估数据集: {dataset_name}")
    print(f"   路径: {file_path}")

    if not os.path.exists(file_path):
        print(f"文件不存在: {file_path}(跳过)")
        return None

    with open(file_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    if not lines:
        print("数据集为空")
        return None

    refs = []
    hyps = []

    # 进度条
    pbar = tqdm(lines, unit="sample")

    for line in pbar:
        item = json.loads(line)

        # 获取并清洗真值 (Reference)
        original_ref = item.get("ASR_transcript_clean", "")
        clean_ref = clean_text(original_ref)

        # 如果真值为空 (纯噪音样本)，跳过评估，避免干扰 WER
        if len(clean_ref) < 2:
            continue

        # 寻找音频文件
        audio_path = item["audio_file"]
        if not os.path.exists(audio_path):
            # 尝试去备选目录找
            filename = os.path.basename(audio_path)
            found = False
            for d in AUDIO_DIRS:
                candidate = os.path.join(d, filename)
                if os.path.exists(candidate):
                    audio_path = candidate
                    found = True
                    break
            if not found:
                # 真的找不到，跳过
                continue

        # 读取音频
        try:
            audio, sr = sf.read(audio_path)
            if len(audio.shape) > 1:
                audio = audio.mean(axis=1)  # 转单声道
        except Exception as e:
            print(f"读取音频失败: {e}")
            continue

        # 推理 (Inference)
        # 添加 .half() 将输入转为 FP16，解决 RuntimeError
        input_features = (
            processor(audio, sampling_rate=16000, return_tensors="pt")
            .input_features.to("cuda")
            .half()
        )

        with torch.no_grad():
            # 强制使用英语解码
            generated_ids = model.generate(
                input_features, language="en", task="transcribe", max_new_tokens=255
            )

        transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        clean_hyp = clean_text(transcription)

        refs.append(clean_ref)
        hyps.append(clean_hyp)

        # 实时显示当前的 WER 估算
        if len(refs) > 0 and len(refs) % 10 == 0:
            current_wer = wer(refs, hyps)
            pbar.set_description(f"WER: {current_wer:.2%}")

    if len(refs) == 0:
        return {"wer": 0.0, "cer": 0.0, "samples": 0}

    # 计算最终指标
    final_wer = wer(refs, hyps)
    final_cer = cer(refs, hyps)

    return {"wer": final_wer, "cer": final_cer, "samples": len(refs)}


def main():
    model, processor = load_model()

    results = {}

    print("\n" + "=" * 50)
    print("开始多场景鲁棒性评估")
    print("=" * 50)

    for name, path in TEST_SETS.items():
        res = evaluate_dataset(model, processor, name, path)
        if res:
            results[name] = res

    # 生成最终报告 (Paper Ready Format)
    print("\n\n" + "=" * 65)
    print("FINAL ROBUSTNESS REPORT (For Your Paper)")
    print("=" * 65)

    header = f"{'Test Set':<20} | {'WER':<10} | {'CER':<10} | {'Samples':<8}"
    print(header)
    print("-" * len(header))

    report_content = [header, "-" * len(header)]

    for name, res in results.items():
        # 格式化输出
        line = f"{name:<20} | {res['wer']:.2%}   | {res['cer']:.2%}   | {res['samples']:<8}"
        print(line)
        report_content.append(line)

    # 保存到文件
    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(report_content))
        f.write("\n\n=== Configuration ===\n")
        f.write(f"Model: {MODEL_DIR}\n")
        f.write(f"Base: {BASE_MODEL_NAME}\n")

    print("-" * len(header))
    print(f"\n详细报告已保存至: {REPORT_FILE}")
    print("   (请将表格数据填入论文的 Experiment 章节)")


if __name__ == "__main__":
    main()
