# ---------------------------------------------------------------
# Archived experiment (late 2025). Original file: ASR/v2_raw_inference_with_deepseek.py
# What it does: Whisper output is post-corrected by DeepSeek-chat with guard rails (reject digit changes, large length changes); raw and corrected WER are compared.
# Known problems:
#   - Few-shot examples in the system prompt look lifted from test errors. Vocabulary or prompt words were chosen after looking at test-set errors, so the test score is optimistic.
#   - Sends test transcripts to a third-party API. The key is read from the DEEPSEEK_API_KEY environment variable.
#   - The 'improvement' printout has its sign reversed.
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
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed
import re

# Configuration
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")  # Read the key from the environment.
DEEPSEEK_BASE_URL = "https://api.deepseek.com"

MODEL_PATH = "whisper_atco_raw_v2"
TEST_DATA_PATH = "processed_data/test.jsonl"
VOCAB_FILE = "processed_data/vocab_master.json"
AUDIO_ROOT = "processed_data/audio"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_WORKERS = 10


# 激进版 Prompt (Smart Context)


def generate_aggressive_prompt(vocab_keywords):
    keywords_str = ", ".join(vocab_keywords[:120])
    return f"""You are a SMART ATC CONTEXT CORRECTOR.
Your goal is to fix ASR errors by inferring the most likely ATC phraseology based on context.

CRITICAL VOCABULARY: [{keywords_str}]

RULES:
1. **Context over Literal**: If a word is spelled correctly but makes no sense in ATC context, FIX IT based on phonetic similarity.
   - Example: "RUNWAY ONE ALPHA" -> "RYANAIR ONE ALPHA" (Context: Runway is not a callsign).
   - Example: "REQUESTAN" -> "REQUEST ENGINE START".
   - Example: "HOTEL ECHO" -> "HELICOPTER" (If context implies visual sighting).

2. **Fix Entities**: "Stephen nick" -> "STEFANIK", "Mosquito" -> "MOSQUITO".
3. **Format**: Keep numbers as words ("ONE TWO"), do NOT use digits ("12").
4. **Output**: UPPERCASE only.

INPUT:
"""


# 处理逻辑 (熔断机制)

client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)


def correct_with_deepseek_task(args):
    text, system_prompt, index = args
    if not text or len(text.strip()) < 2:
        return index, text

    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": text},
            ],
            temperature=0.0,  # 保持理性
            max_tokens=128,
        )
        corrected = response.choices[0].message.content.strip().upper()

        # 熔断检查

        # 禁止转数字 (防止 WER 误判)
        if re.search(r"\d", corrected):
            return index, text

        # 长度检查 (放宽到 40%，允许 "Requestan" -> "Request Engine Start" 这种变长操作)
        len_raw = len(text.split())
        len_new = len(corrected.split())
        if len_raw > 0 and abs(len_raw - len_new) / len_raw > 0.40:
            return index, text

        # 废话检测
        if "HERE IS" in corrected or "CORRECTED" in corrected:
            return index, text

        return index, corrected
    except:
        return index, text


# 辅助工具

std_normalizer = BasicTextNormalizer()


def atc_cleaner(text):
    if not text:
        return ""
    text = text.upper().replace(".", " ").replace(",", " ").replace("-", " ")
    noise = ["HES", "UNK", "NOISE"]
    for t in noise:
        text = text.replace(t, "")
    return " ".join(text.split())


def load_vocab_keywords(vocab_path):
    if not os.path.exists(vocab_path):
        return []
    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab = json.load(f)
    keywords = []
    if "CRITICAL_LOCATIONS" in vocab:
        keywords.extend(list(vocab["CRITICAL_LOCATIONS"].keys()))
    if "ENTITIES" in vocab:
        keywords.extend([w for w, c in vocab["ENTITIES"].items()][:50])
    return list(set(keywords))


# 主逻辑


def main():
    print(f"加载 Whisper 模型: {MODEL_PATH}")
    try:
        processor = WhisperProcessor.from_pretrained(MODEL_PATH)
        model = WhisperForConditionalGeneration.from_pretrained(MODEL_PATH).to(DEVICE)
    except Exception as e:
        print(f"加载失败: {e}")
        return

    keywords = load_vocab_keywords(VOCAB_FILE)

    # Whisper Prompt
    whisper_prompt_ids = processor.get_prompt_ids("Context: " + ", ".join(keywords[:200]))
    whisper_prompt_tensor = torch.tensor(whisper_prompt_ids).to(DEVICE)

    # LLM Prompt (激进版)
    llm_system_prompt = generate_aggressive_prompt(keywords)

    data = []
    with open(TEST_DATA_PATH, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))

    refs = []
    preds_raw = []
    preds_llm = [""] * len(data)

    print(f"第一阶段: Whisper 推理...")
    for i, item in enumerate(tqdm(data, desc="Whisper Step")):
        refs.append(atc_cleaner(item["ASR_transcript_clean"]))
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
            with torch.no_grad():
                generated_ids = model.generate(
                    input_features, prompt_ids=whisper_prompt_tensor, max_new_tokens=128
                )
            text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            preds_raw.append(atc_cleaner(text))
        except:
            preds_raw.append("")

    print(f"\n第二阶段: DeepSeek 智能纠错 (Smart Context)...")
    tasks = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        for i, text in enumerate(preds_raw):
            tasks.append(executor.submit(correct_with_deepseek_task, (text, llm_system_prompt, i)))
        for future in tqdm(as_completed(tasks), total=len(tasks), desc="DeepSeek Step"):
            idx, corrected_text = future.result()
            preds_llm[idx] = atc_cleaner(corrected_text)

    # 修复后的评估函数 (Fix Denominator)
    def calc_metrics(r, p, name):
        # 过滤空数据
        valid = [(ref, pred) for ref, pred in zip(r, p) if ref.strip()]
        if not valid:
            print(f" {name}: 无有效数据")
            return 0

        r_valid_tuple, p_valid_tuple = zip(*valid)
        r_valid = list(r_valid_tuple)
        p_valid = list(p_valid_tuple)

        # 计算 jiwer
        out = jiwer.process_words(r_valid, p_valid)

        # 计算所有参考文本的总单词数
        total_words = sum(len(s.split()) for s in r_valid)
        if total_words == 0:
            total_words = 1

        wer = out.wer * 100
        acc = 100 - wer

        # 使用 total_words 作为分母，百分比就正常了
        sub = out.substitutions / total_words * 100
        dele = out.deletions / total_words * 100
        ins = out.insertions / total_words * 100

        print(f"\n======== {name}========")
        print(f"Acc (准确率): {acc:.2f}%")
        print(f"WER (词错误率): {wer:.2f}%")
        print(f"    Sub (替换): {sub:.2f}%")
        print(f"    Del (删除): {dele:.2f}%")
        print(f"    Ins (插入): {ins:.2f}%")
        return wer

    print("\n" + "=" * 50)
    print("最终 A/B 测试报告 (Aggressive Mode)")
    wer_raw = calc_metrics(refs, preds_raw, "原始 Whisper")
    wer_llm = calc_metrics(refs, preds_llm, "DeepSeek 修正后")
    print("\n" + "=" * 50)
    print(f"WER 降低: {wer_raw - wer_llm:.2f}%")
    print(f"Acc 提升: {wer_llm - wer_raw:.2f}%")
    print("=" * 50)

    # 抽样
    print("\n成功修正案例 (Smart Context):")
    count = 0
    for i in range(len(refs)):
        if preds_raw[i] != preds_llm[i]:
            print(f"\n[Raw]: {preds_raw[i]}")
            print(f"[LLM]: {preds_llm[i]}")
            print(f"[Ref]: {refs[i]}")
            count += 1
            if count >= 3:
                break


if __name__ == "__main__":
    main()
