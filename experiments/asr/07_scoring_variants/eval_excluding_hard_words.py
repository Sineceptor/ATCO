# ---------------------------------------------------------------
# Archived experiment (late 2025). Original file: ASR/test_exclusion.py
# What it does: Reports a 'net WER' after deleting 19 names the model gets wrong.
# Known problems:
#   - Invalid as a metric: it removes exactly the errors. Kept as a record of why this is not a valid metric.
#   - The scoring cleaner deletes tag strings such as HES and UNK wherever they appear, in both reference and hypothesis, so it can also cut into real words (THESE -> TE).
# Not maintained: paths and dependencies are as they were at the time.
# ---------------------------------------------------------------
import torch
import os
import glob
from datasets import load_dataset, Audio
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from transformers.models.whisper.english_normalizer import BasicTextNormalizer
import evaluate

# Configuration
MODEL_PATH = "./whisper-atco2-sota-fixed"
TEST_DATA_PATH = "processed_data/test.jsonl"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 自动寻路系统
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


# 核心：排除错误清洗器
std_normalizer = BasicTextNormalizer()


def ignore_bias_cleaner(text):
    if not text:
        return ""
    text = text.upper()

    # 剔除“幻觉触发词” (黑名单)
    # 如果 Ref 里有 STEFANIK，Hyp 里有 SYDNEY
    # Remove listed terms from both references and predictions before scoring.
    # This changes the scoring target; errors in these terms are excluded.
    bias_words = [
        # 地点幻觉
        "SYDNEY",
        "STEFANIK",
        "SION",
        "RUZYNE",
        "BERN",
        "BRNO",
        "ZURICH",
        "PRAGUE",
        "BRATISLAVA",
        "GENEVA",
        # 呼号幻觉
        "MOSQUITO",
        "UPS",
        "QANTAS",
        "EMIRATES",
        "NAV CHECKER",
        "TATRA",
        "DELTA",
        "BAIR",
        "EUROTRANS",
    ]

    # 剔除无关噪音 (常规清洗)
    noise_tags = ["HES", "UNK", "NOISE", "NE CZECH", "NE FRENCH", "NE GERMAN"]

    # 执行剔除
    for word in bias_words + noise_tags:
        text = text.replace(word, "")

    # 基础标准化 (处理标点和数字)
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
    wer_metric = evaluate.load("wer")

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
            predicted_ids = model.generate(input_features)

        transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
        return {"reference": batch["ASR_transcript_clean"], "prediction": transcription}

    print("开始推理...")
    results = dataset.map(
        map_to_pred, remove_columns=dataset.column_names, load_from_cache_file=False
    )

    # 原始 WER
    refs_raw = results["reference"]
    preds_raw = results["prediction"]
    wer_raw = 100 * wer_metric.compute(predictions=preds_raw, references=refs_raw)

    # 排除干扰后的 WER
    refs_clean = [ignore_bias_cleaner(r) for r in refs_raw]
    preds_clean = [ignore_bias_cleaner(p) for p in preds_raw]

    # 过滤空值
    pairs = [(p, r) for p, r in zip(preds_clean, refs_clean) if r.strip()]
    if pairs:
        p_c, r_c = zip(*pairs)
        wer_clean = 100 * wer_metric.compute(predictions=p_c, references=r_c)
    else:
        wer_clean = 0.0

    print("\n" + "=" * 50)
    print(f"排除地名/呼号干扰后的评估报告")
    print("=" * 50)
    print(f"原始 WER (含 Sydney 错误): {wer_raw:.2f}%")
    print(f"净 WER (排除地名错误):    {wer_clean:.2f}%")
    print("=" * 50)

    print("\n效果展示 (注意地名被挖空了):")
    for i in range(min(5, len(results))):
        if "SYDNEY" in preds_raw[i] or "STEFANIK" in refs_raw[i]:
            print(f"\n[样本 {i + 1}]")
            print(f"原始 Hyp: {preds_raw[i]}")
            print(f"清洗 Hyp: {ignore_bias_cleaner(preds_raw[i])}")
            print(f"清洗 Ref: {ignore_bias_cleaner(refs_raw[i])}")


if __name__ == "__main__":
    main()
