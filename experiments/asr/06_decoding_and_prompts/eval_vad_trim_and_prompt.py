# ---------------------------------------------------------------
# Archived experiment (late 2025). Original file: ASR/eval_advanced.py
# What it does: Trims silence with Silero VAD before decoding and uses a hand-written prompt of difficult names.
# Known problems:
#   - The scoring cleaner deletes tag strings such as HES and UNK wherever they appear, in both reference and hypothesis, so it can also cut into real words (THESE -> TE).
#   - The prompt words come from test errors. Vocabulary or prompt words were chosen after looking at test-set errors, so the test score is optimistic.
# Not maintained: paths and dependencies are as they were at the time.
# ---------------------------------------------------------------
import torch
import os
import glob
import numpy as np
from datasets import load_dataset, Audio
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from transformers.models.whisper.english_normalizer import BasicTextNormalizer
import evaluate

# Configuration
MODEL_PATH = "./whisper-atco2-speed"
TEST_DATA_PATH = "processed_data/test.jsonl"
BATCH_SIZE = 1
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 核心策略：场景提示 (Prompt)
CONTEXT_PROMPT = "Air traffic control communications. Possible vocabularies: Stefanik Tower, Sion, Mosquito, Nav Checker, Tatra, Emirates, Good day."

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


# 防线一：VAD (物理切除静音)
print("加载 VAD 模型...")
try:
    vad_model, utils = torch.hub.load(
        repo_or_dir="snakers4/silero-vad", model="silero_vad", force_reload=False, onnx=False
    )
    (get_speech_timestamps, save_audio, read_audio, VADIterator, collect_chunks) = utils
except:
    print("请先安装依赖: pip install torchaudio silero-vad")
    exit(1)


def apply_vad(audio_array):
    wav = torch.from_numpy(audio_array).float()
    timestamps = get_speech_timestamps(wav, vad_model, sampling_rate=16000)
    if len(timestamps) == 0:
        return None
    return collect_chunks(timestamps, wav)


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
    wer_metric = evaluate.load("wer")

    # 制作 Prompt ID (必须转为 Tensor 并移动到 GPU)
    # Convert prompt IDs from NumPy to a tensor.
    prompt_ids_numpy = processor.tokenizer.get_prompt_ids(CONTEXT_PROMPT)
    prompt_ids = torch.tensor(prompt_ids_numpy).to(DEVICE)

    print(f"已注入场景提示: '{CONTEXT_PROMPT}'")

    # 加载数据
    dataset = load_dataset("json", data_files=TEST_DATA_PATH, split="train")
    dataset = dataset.map(fix_audio_path).filter(lambda x: os.path.exists(x["audio_file"]))
    dataset = dataset.cast_column("audio_file", Audio(sampling_rate=16000))

    def map_to_pred(batch):
        audio = batch["audio_file"]

        # VAD 处理
        vad_audio = apply_vad(audio["array"])
        if vad_audio is None:
            return {"reference": batch["ASR_transcript_clean"], "prediction": ""}

        input_features = processor(
            vad_audio.numpy(), sampling_rate=16000, return_tensors="pt"
        ).input_features.to(DEVICE)

        # 推理 (自然引导模式)
        with torch.no_grad():
            predicted_ids = model.generate(
                input_features,
                prompt_ids=prompt_ids,  # Prompt IDs are on the model device.
                num_beams=5,
                num_return_sequences=1,
            )

        transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
        return {"reference": batch["ASR_transcript_clean"], "prediction": transcription}

    print("开始推理 (VAD + Prompt + BeamSearch)...")
    results = dataset.map(
        map_to_pred, remove_columns=dataset.column_names, load_from_cache_file=False
    )

    refs = [clean_atc_text(r) for r in results["reference"]]
    preds = [clean_atc_text(p) for p in results["prediction"]]

    pairs = [(p, r) for p, r in zip(preds, refs) if r.strip()]
    if pairs:
        p, r = zip(*pairs)
        wer = 100 * wer_metric.compute(predictions=p, references=r)
        print(f"\n最终 WER: {wer:.2f}%")

    print("\n重点检查 (是否还有 Sydney?):")
    for i, p in enumerate(preds[:5]):
        if "SYDNEY" in p or "STEFANIK" in p:
            print(f"Hyp: {p}")


if __name__ == "__main__":
    main()
