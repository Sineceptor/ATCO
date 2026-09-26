# ---------------------------------------------------------------
# Archived experiment (late 2025). Original file: ASR/v2_raw_inference_v2.py
# What it does: Same evaluation with a structured prompt (format hint + longest locations/callsigns), temperature 0 and no conditioning on previous tokens.
# Known problems:
#   - The scoring cleaner deletes tag strings such as HES and UNK wherever they appear, in both reference and hypothesis, so it can also cut into real words (THESE -> TE).
#   - Vocabulary or prompt words were chosen after looking at test-set errors, so the test score is optimistic.
# Not maintained: paths and dependencies are as they were at the time.
# ---------------------------------------------------------------
import torch
import json
import os
import jiwer
import torchaudio
import torchaudio.transforms as T
from tqdm import tqdm
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from transformers.models.whisper.english_normalizer import BasicTextNormalizer

# Configuration
MODEL_PATH = "whisper_atco_raw_v2"
TEST_DATA_PATH = "processed_data/test.jsonl"
VOCAB_FILE = "processed_data/vocab_master.json"
AUDIO_ROOT = "processed_data/audio"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# 高级 Prompt 构造器
def generate_structured_prompt(vocab_path):
    """Build a prompt from a format example and the vocabulary lists."""
    if not os.path.exists(vocab_path):
        return "Air Traffic Control transcript. Callsigns and Waypoints follow."

    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab = json.load(f)

    # 提取地点 (Waypoints)
    locs = []
    if "CRITICAL_LOCATIONS" in vocab:
        # 按长度排序，优先把长的难词放前面
        locs = sorted(vocab["CRITICAL_LOCATIONS"].keys(), key=len, reverse=True)[:30]

    # 提取呼号 (Callsigns)
    calls = []
    if "CRITICAL_CALLSIGNS" in vocab:
        calls = sorted(vocab["CRITICAL_CALLSIGNS"].keys(), key=len, reverse=True)[:30]

    # 构造 Prompt 句子
    # 技巧：在 Prompt 里直接写几个数字单词，暗示模型不要输 "123"
    prompt_parts = [
        "Transcribe Air Traffic Control communications.",
        "Format: UPPERCASE. Numbers: ONE TWO THREE.",  # 暗示格式
        "Waypoints: " + ", ".join(locs) + ".",  # 分类 1
        "Callsigns: " + ", ".join(calls) + ".",  # 分类 2
    ]

    # 组合，并截断以防超过 Whisper 上限 (约224 tokens)
    full_prompt = " ".join(prompt_parts)
    print(f"\n[优化 Prompt Preview]: {full_prompt[:150]}...")
    return full_prompt


# 清洗与评估工具
std_normalizer = BasicTextNormalizer()


def atc_cleaner(text):
    if not text:
        return ""
    text = text.upper()
    # 移除 Whisper 常见的无关标点
    text = text.replace(".", " ").replace(",", "").replace("?", "").replace("!", "")
    # 移除噪音标记
    noise = ["HES", "UNK", "NOISE", "[", "]"]
    for t in noise:
        text = text.replace(t, "")
    return " ".join(text.split())


def calc_metrics(r, p):
    # 过滤空数据
    valid = [(ref, pred) for ref, pred in zip(r, p) if ref.strip()]
    if not valid:
        return 0, 0, 0, 0, 0

    # zip(*valid) 返回的是 tuple，必须转成 list 才能给 jiwer 用
    r_valid_tuple, p_valid_tuple = zip(*valid)
    r_valid = list(r_valid_tuple)
    p_valid = list(p_valid_tuple)

    # 使用 jiwer 计算
    out = jiwer.process_words(r_valid, p_valid)

    # 计算总单词数作为分母 (避免百分比溢出)
    total_words = sum(len(s.split()) for s in r_valid)
    if total_words == 0:
        total_words = 1

    wer = out.wer * 100
    acc = 100 - wer

    sub = out.substitutions / total_words * 100
    dele = out.deletions / total_words * 100
    ins = out.insertions / total_words * 100

    return acc, wer, sub, dele, ins


# 主逻辑
def main():
    print(f"加载 Raw 模型: {MODEL_PATH}")
    try:
        processor = WhisperProcessor.from_pretrained(MODEL_PATH)
        model = WhisperForConditionalGeneration.from_pretrained(MODEL_PATH).to(DEVICE)
    except Exception as e:
        print(f"加载失败: {e}")
        return

    # 生成优化后的 Prompt
    prompt_text = generate_structured_prompt(VOCAB_FILE)
    prompt_ids = processor.get_prompt_ids(prompt_text)
    prompt_tensor = torch.tensor(prompt_ids).to(DEVICE)

    # 加载数据
    data = []
    with open(TEST_DATA_PATH, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))

    refs = []
    preds = []

    print(f'开始推理 {len(data)} 条 (Beam=5, No-Prev-Context)...')

    for item in tqdm(data):
        # 音频加载
        raw_path = item["audio_file"]
        if os.path.exists(raw_path):
            wav_path = raw_path
        else:
            wav_path = os.path.join(AUDIO_ROOT, os.path.basename(raw_path))

        try:
            waveform, sr = torchaudio.load(wav_path)
            if sr != 16000:
                waveform = T.Resample(sr, 16000)(waveform)
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0)
            else:
                waveform = waveform.squeeze(0)

            input_features = processor(
                waveform.numpy(), sampling_rate=16000, return_tensors="pt"
            ).input_features.to(DEVICE)

            # 推理 (关键参数调优)
            with torch.no_grad():
                generated_ids = model.generate(
                    input_features,
                    prompt_ids=prompt_tensor,  # 注入结构化 Prompt
                    max_new_tokens=128,
                    num_beams=5,  # 开启 Beam Search 提升质量
                    condition_on_prev_tokens=False,  # 禁止依赖前文 (防止幻觉扩散)
                    temperature=0.0,  # 贪婪采样 (最稳)
                    repetition_penalty=1.0,  # 不惩罚重复 (ATC需要重复)
                    return_dict_in_generate=False,
                )

            text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

            refs.append(atc_cleaner(item["ASR_transcript_clean"]))
            preds.append(atc_cleaner(text))

        except Exception as e:
            # print(f"Error: {e}")
            continue

    # 评估
    acc, wer, sub, dele, ins = calc_metrics(refs, preds)

    print("\n" + "=" * 50)
    print("最终评估报告 (Optimized Inference)")
    print("=" * 50)
    print(f"Acc (准确率): {acc:.2f}%")
    print(f"WER (词错误率): {wer:.2f}%")
    print("-" * 30)
    print(f"    Sub (替换): {sub:.2f}%")
    print(f"    Del (删除): {dele:.2f}%")
    print(f"    Ins (插入): {ins:.2f}%")
    print("=" * 50)

    # 抽样展示
    print("\n结果抽样:")
    for i in range(min(5, len(refs))):
        print(f"\n[Ref]:  {refs[i]}")
        print(f"[Pred]: {preds[i]}")


if __name__ == "__main__":
    main()
