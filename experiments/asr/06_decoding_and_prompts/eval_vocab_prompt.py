# ---------------------------------------------------------------
# Archived experiment (late 2025). Original file: ASR/eval_vocab_constrained.py
# What it does: Evaluates a checkpoint with a vocabulary prompt and beam 5. A comment records "acc" for three models: raw 76.18, augmented 76.23, TTS-mixed 76.67. The recorded scores moved by less than half a point; which test file and settings produced them cannot now be confirmed.
# Known problems:
#   - The scoring cleaner deletes tag strings such as HES and UNK wherever they appear, in both reference and hypothesis, so it can also cut into real words (THESE -> TE).
#   - Vocabulary or prompt words were chosen after looking at test-set errors, so the test score is optimistic.
#   - Substitution/deletion/insertion 'rates' are divided by the number of clips, not words.
#   - The prompt loader reads a key that the vocab file never contains, so callsigns are not loaded.
# Not maintained: paths and dependencies are as they were at the time.
# ---------------------------------------------------------------
import torch
import json
import os
import glob
import jiwer  # 用于详细计算 S/D/I
from datasets import load_dataset, Audio
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from transformers.models.whisper.english_normalizer import BasicTextNormalizer
import evaluate

# Configuration
MODEL_PATH = "atc_whisper_model_v3_fast"  # whisper_atco_raw_v2:acc-76.18%， whisper_atco_aug_finetuned: acc-76.23%，whisper_atco_tts_mixed_final: acc-76.67%
TEST_DATA_PATH = "processed_data/test.jsonl"
VOCAB_FILE = "processed_data/vocab_master.json"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# 词库加载器
def load_dynamic_prompt(vocab_path):
    if not os.path.exists(vocab_path):
        print(f"警告：找不到词库文件 {vocab_path}，将使用默认 Prompt。")
        return "Air traffic control communications."

    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab = json.load(f)

    critical_words = []
    # 提取地点和呼号
    if "CRITICAL_LOCATIONS" in vocab:
        critical_words.extend(list(vocab["CRITICAL_LOCATIONS"].keys()))
    if "CRITICAL_CALLSIGNS" in vocab:
        critical_words.extend(list(vocab["CRITICAL_CALLSIGNS"].keys()))
    if "ENTITIES" in vocab:
        top_entities = sorted(vocab["ENTITIES"].items(), key=lambda x: x[1], reverse=True)[:20]
        critical_words.extend([w for w, c in top_entities])

    # 截取前 200 个词构建 Context
    final_list = list(set(critical_words))[:200]
    prompt_str = "Context: " + ", ".join(final_list) + "."
    print(f'已加载 {len(final_list)} 个核心词汇用于 Prompt 引导')
    return prompt_str


# 自动寻路
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


# 清洗器
std_normalizer = BasicTextNormalizer()


def clean_atc_text(text):
    if not text:
        return ""
    text = text.upper()
    noise_tags = ["HES", "UNK", "NOISE", "NE CZECH", "NE FRENCH", "NE GERMAN"]
    foreign = ["DOBRY DEN", "AHOJ", "BONJOUR", "MERCI", "DANKE", "TSCHUSS"]
    for t in noise_tags + foreign:
        text = text.replace(t, "")
    text = text.replace(".", " DECIMAL ")
    return std_normalizer(text)


# 主逻辑
def main():
    global FOUND_AUDIO_DIRS
    FOUND_AUDIO_DIRS = scan_for_audio_files(os.getcwd())

    print(f"加载模型: {MODEL_PATH}")
    processor = WhisperProcessor.from_pretrained(MODEL_PATH)
    model = WhisperForConditionalGeneration.from_pretrained(MODEL_PATH).to(DEVICE)
    model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(
        language="English", task="transcribe"
    )

    # 加载指标
    wer_metric = evaluate.load("wer")
    cer_metric = evaluate.load("cer")

    # 准备 Prompt
    prompt_text = load_dynamic_prompt(VOCAB_FILE)
    prompt_ids = torch.tensor(processor.tokenizer.get_prompt_ids(prompt_text)).to(DEVICE)

    # 加载数据
    dataset = load_dataset("json", data_files=TEST_DATA_PATH, split="train")
    dataset = dataset.map(fix_audio_path).filter(lambda x: os.path.exists(x["audio_file"]))
    dataset = dataset.cast_column("audio_file", Audio(sampling_rate=16000))

    def map_to_pred(batch):
        audio = batch["audio_file"]
        input_features = processor(
            audio["array"], sampling_rate=16000, return_tensors="pt"
        ).input_features.to(DEVICE)

        with torch.no_grad():
            predicted_ids = model.generate(
                input_features, prompt_ids=prompt_ids, num_beams=5, num_return_sequences=1
            )

        transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
        return {"reference": batch["ASR_transcript_clean"], "prediction": transcription}

    print("开始评估 (包含详细指标)...")
    results = dataset.map(
        map_to_pred, remove_columns=dataset.column_names, load_from_cache_file=False
    )

    refs = [clean_atc_text(r) for r in results["reference"]]
    preds = [clean_atc_text(p) for p in results["prediction"]]

    # 过滤空值
    valid_pairs = [(p, r) for p, r in zip(preds, refs) if r.strip()]

    if valid_pairs:
        # zip(*valid_pairs) 返回的是 tuple，jiwer 需要 list
        p_tuple, r_tuple = zip(*valid_pairs)
        p_list = list(p_tuple)
        r_list = list(r_tuple)

        # 基础 WER / CER
        wer = 100 * wer_metric.compute(predictions=p_list, references=r_list)
        cer = 100 * cer_metric.compute(predictions=p_list, references=r_list)
        accuracy = 100 - wer

        # 详细 S/D/I 分析 (Sub-Error Rates)
        out = jiwer.process_words(r_list, p_list)
        sub_rate = out.substitutions / len(r_list) * 100  # 替换率 (听错了)
        del_rate = out.deletions / len(r_list) * 100  # 删除率 (没听见)
        ins_rate = out.insertions / len(r_list) * 100  # 插入率 (幻觉/多嘴)

        print("\n" + "=" * 50)
        print("评估报告")
        print("=" * 50)
        print(f"词错误率 (WER):      {wer:.2f}%")
        print(f"100 - WER (not accuracy):    {accuracy:.2f}%")
        print(f"字错误率 (CER):      {cer:.2f}%")
        print("-" * 50)
        print("错误类型细分 (Sub-Error Rates):")
        print(f"    替换错误 (Sub):   {sub_rate:.2f}%  (词读错了)")
        print(f"    删除错误 (Del):   {del_rate:.2f}%  (词漏掉了)")
        print(f"    插入错误 (Ins):   {ins_rate:.2f}%  (词加多了)")
        print("=" * 50)


if __name__ == "__main__":
    main()
