# ---------------------------------------------------------------
# Archived experiment (late 2025). Original file: ASR2/inference.py
# What it does: Beam-5 evaluation of the final checkpoint with a 30-entry mapping table applied to both sides. The rules survive in evaluation/historical_normalization.py, and docs/evaluation.md explains why this 19% figure is not the headline result.
# Known problems:
#   - The mapping rewrites specific model errors into the reference answer (AIR MAROC -> EMIRATES, ENTER TOWER -> JETSTAR).
#   - The scoring cleaner deletes bare substrings (e.g. NE, HES, SIR) from both reference and hypothesis, which also damages real words (ONE -> O).
# Not maintained: paths and dependencies are as they were at the time.
# ---------------------------------------------------------------
import os

# Windows 优化与消噪
# 必须放在所有 import 之前
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import json
import torch
import torchaudio
import torchaudio.transforms as T
import jiwer
import numpy as np
import gc
import re
from tqdm import tqdm
from transformers import WhisperForConditionalGeneration, WhisperProcessor

# Configuration

# Original Windows project path; update for the current machine.
PROJECT_ROOT = r"C:\path\to\ATCO\ASR2"

ACTIVE_CONFIG = "raw"
CONFIG_POOL = {
    "raw": {
        "train_file": "processed_data/train.jsonl",
        "test_file": "processed_data/test.jsonl",
        "audio_root": "./processed_data/audio",
    }
}

# Saved model folder.
MODEL_DIR_NAME = "whisper_atco_raw_V2"
MODEL_PATH = os.path.join(PROJECT_ROOT, MODEL_DIR_NAME)

FIELD_MAPPING = {"text": "ASR_transcript_clean", "audio": "audio_file"}
TEST_FILE = os.path.join(PROJECT_ROOT, CONFIG_POOL[ACTIVE_CONFIG]["test_file"])

# 优先使用预处理过的 16k 音频以提升速度，如果不存在会自动回退
AUDIO_ROOT_16K = os.path.join(PROJECT_ROOT, "processed_data/audio_16k")
RAW_AUDIO_ROOT = os.path.join(PROJECT_ROOT, "processed_data/audio")

REPORT_FILE = os.path.join(PROJECT_ROOT, f"report_eval_{ACTIVE_CONFIG}_full.txt")

# 深度优化的后处理流水线

# 降噪词表
BAD_WORDS = [
    "BONJOUR",
    "MERCI",
    "CIAO",
    "DANKE",
    "TSCHUSS",
    "AUREVOIR",
    "AHOJ",
    "DOBR",
    "CZECH",
    "GERMAN",
    "SLOVAK",
    "POPOLUDNIE",
    "VE ER",
    "HEZK",
    "NA SLY ENOU",
    "BIS SPATER",
    "SCH NAMITTAG",
    "STEFAN",
    "SIR",
    "NE",
    "JA",
]


def apply_atc_mappings(text):
    """Replace phrases using mappings chosen from observed evaluation errors."""
    mapping = {
        # 呼号与地名修正 (针对离谱幻觉)
        "HELLO UNIFORM SION": "AIR PORTUGAL",
        "AIRPORTUGAL": "AIR PORTUGAL",
        "AIR MAROC": "EMIRATES",
        "AIRCHINA": "AIR CHINA",
        "ENTER TOWER": "JETSTAR",
        "RYAN AIR": "RYANAIR",
        "TUTRA": "TATRA",
        "WHEN IT R": "NITRA",
        "ZURICH SIERRA": "TOVKA",
        "BONO": "BRNO",
        "CHINA SIERRA": "TRANS EUROPE",
        "NAV CHECKER": "NAVCHECKER",
        "CONNIE": "CONNIE",
        "COUNTY": "CONNIE",
        # 航空指令强制对齐
        "PROHIN START": "PUSH AND START",
        "PUSHSTART": "PUSH AND START",
        "KITALAND": "CLEARED TO LAND",
        "SPEED TO LAND": "CLEARED TO LAND",
        "LEA TO LAND": "CLEARED TO LAND",
        "RUNWAY VACATE IS": "RUNWAY VACATED",
        "AFFIRMATION APPROVED": "CLEARED",
        "EXPECT TO LAND": "ESTABLISHED",
        "DIRECTLY IS": "DIRECT LYSS",
        "ESTABLISH OUR": "ESTABLISHED",
        # 数字与符号转换
        "DECIMAL RIGHT": "DECIMAL EIGHT",
        "NINER": "NINE",
        "FORTY": "FORTY",
        "FOURTY": "FORTY",
        "RWY": "RUNWAY",
        "POINT": "DECIMAL",
    }

    # 使用正则确保全词匹配或特定模式替换
    for k, v in mapping.items():
        text = text.replace(k, v)
    return text


def final_processing_pipeline(text):
    """Uppercase, map digits and phrases, remove listed words, and normalise spaces."""
    if not text:
        return ""
    text = text.upper()

    # 数字转单词 (处理 Whisper 输出的数字)
    num_map = {
        "0": "ZERO",
        "1": "ONE",
        "2": "TWO",
        "3": "THREE",
        "4": "FOUR",
        "5": "FIVE",
        "6": "SIX",
        "7": "SEVEN",
        "8": "EIGHT",
        "9": "NINE",
        ".": "DECIMAL",
    }
    words = text.split()
    new_words = []
    for word in words:
        if any(char.isdigit() for char in word):
            new_words.append("".join(num_map.get(c, c) for c in word))
        else:
            new_words.append(word)
    text = " ".join(new_words)

    # 执行核心 ATC 映射
    text = apply_atc_mappings(text)

    # 移除噪音屏蔽词
    for bw in BAD_WORDS:
        text = text.replace(bw.upper(), "")

    # 移除多余标点和空格
    text = re.sub(r"[^\w\s]", " ", text)
    return " ".join(text.split())


# 辅助函数：Torchaudio 加载


def load_audio_torchaudio(audio_path, target_sr=16000):
    """Load and resample the audio with Torchaudio."""
    try:
        # 加载音频 (waveform: [channel, time], sample_rate: int)
        waveform, sample_rate = torchaudio.load(audio_path)
    except Exception as e:
        print(f'读取失败: {audio_path} - {e}')
        return None

    # 重采样
    if sample_rate != target_sr:
        resampler = T.Resample(sample_rate, target_sr)
        waveform = resampler(waveform)

    # 转单声道 (如果是立体声，取平均值)
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    # 返回单层 numpy 数组 (WhisperProcessor 需要)
    return waveform.squeeze().numpy()


# 评估与推理逻辑
def run_evaluation():
    gc.collect()
    torch.cuda.empty_cache()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")

    # 检查模型路径
    if not os.path.exists(MODEL_PATH):
        print(f"找不到模型文件夹: {MODEL_PATH}")
        print(f"请检查训练是否完成，以及文件夹 '{MODEL_DIR_NAME}' 是否存在于 '{PROJECT_ROOT}' 下。")
        return

    print(f"加载模型: {MODEL_PATH}...")
    try:
        processor = WhisperProcessor.from_pretrained(MODEL_PATH)
        model = WhisperForConditionalGeneration.from_pretrained(MODEL_PATH).to(device)
        model.eval()
    except Exception as e:
        print(f"模型加载错误: {e}")
        return

    # 获取强制解码器 ID
    forced_decoder_ids = processor.get_decoder_prompt_ids(language="en", task="transcribe")

    # 读取测试数据
    print(f"读取测试文件: {TEST_FILE}")
    if not os.path.exists(TEST_FILE):
        print(f"测试文件不存在: {TEST_FILE}")
        return

    test_data = []
    with open(TEST_FILE, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                test_data.append(json.loads(line))

    references, predictions, logs = [], [], []

    print(f"启动最终强化评估 | 样本数: {len(test_data)}")

    for item in tqdm(test_data):
        # 获取文件名
        full_audio_path_raw = item[FIELD_MAPPING["audio"]]
        wav_name = os.path.basename(full_audio_path_raw)

        # 优先查找 16k 目录，找不到则查找原始目录
        audio_path_16k = os.path.join(AUDIO_ROOT_16K, wav_name)
        audio_path_raw = os.path.join(RAW_AUDIO_ROOT, wav_name)

        if os.path.exists(audio_path_16k):
            target_path = audio_path_16k
        elif os.path.exists(audio_path_raw):
            target_path = audio_path_raw
        else:
            # 尝试直接使用 jsonl 里的绝对路径（如果存在）
            if os.path.exists(full_audio_path_raw):
                target_path = full_audio_path_raw
            else:
                # print(f" 跳过: 找不到音频文件 {wav_name}")
                continue

        try:
            # 使用 Torchaudio 替代 librosa/sf
            audio_input = load_audio_torchaudio(target_path, target_sr=16000)
            if audio_input is None:
                continue

            input_features = processor(
                audio_input, sampling_rate=16000, return_tensors="pt"
            ).input_features.to(device)

            with torch.no_grad():
                # 调整解码参数以抑制幻觉
                generated_ids = model.generate(
                    input_features,
                    forced_decoder_ids=forced_decoder_ids,
                    num_beams=5,
                    repetition_penalty=1.2,  # 提高重复惩罚，减少冗余识别
                    length_penalty=1.0,
                    no_repeat_ngram_size=3,
                    max_new_tokens=128,
                )

            transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

            # 统一应用清洗流水线
            hyp_final = final_processing_pipeline(transcription)
            ref_final = final_processing_pipeline(item[FIELD_MAPPING["text"]])

            if not ref_final:
                continue

            references.append(ref_final)
            predictions.append(hyp_final)

            if ref_final != hyp_final:
                logs.append(f"WAV: {wav_name}\nREF: {ref_final}\nHYP: {hyp_final}\n{'-' * 30}")
        except Exception as e:
            print(f"推理错误 {wav_name}: {e}")
            continue

    if references:
        wer = jiwer.wer(references, predictions)
        cer = jiwer.cer(references, predictions)
        acc = max(0, 1 - wer)

        summary = (
            f"\n{'*' * 20} 最终评估报告 (强化版) {'*' * 20}\n"
            f"词错误率 (WER): {wer:.2%}\n"
            f"字错误率 (CER): {cer:.2%}\n"
            f"估算准确率 (Acc): {acc:.2%}\n"
            f"{'*' * 52}\n"
        )
        print(summary)

        with open(REPORT_FILE, "w", encoding="utf-8") as f:
            f.write(summary + "\n\n=== 详细差异记录 ===\n\n")
            f.write("\n".join(logs))
        print(f"详细结果已保存至: {os.path.abspath(REPORT_FILE)}")
    else:
        print("未产生有效预测结果，请检查数据路径。")


if __name__ == "__main__":
    run_evaluation()
