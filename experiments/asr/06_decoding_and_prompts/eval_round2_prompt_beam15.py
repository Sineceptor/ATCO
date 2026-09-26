# ---------------------------------------------------------------
# Archived experiment (late 2025). Original file: ASR/1.py
# What it does: 'V37': evaluates the round-2 model with beam 15, length penalty 0.1 and a 50-word prompt.
# Known problems:
#   - The scoring cleaner deletes tag strings such as HES and UNK wherever they appear, in both reference and hypothesis, so it can also cut into real words (THESE -> TE).
#   - Vocabulary or prompt words were chosen after looking at test-set errors, so the test score is optimistic.
#   - Decoding settings were tuned on the test set over many iterations.
# Not maintained: paths and dependencies are as they were at the time.
# ---------------------------------------------------------------
import torch
import json
import os
import glob
import jiwer
import random
from datasets import load_dataset, Audio
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from transformers.models.whisper.english_normalizer import BasicTextNormalizer
import evaluate

# Configuration
MODEL_PATH = "whisper_atco_aug_finetuned"
TEST_DATA_PATH = "processed_data/test.jsonl"
VOCAB_FILE = "processed_data/vocab_master.json"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42


# V37 Standard Prompt
def load_v37_prompt(vocab_path, tokenizer):
    if not os.path.exists(vocab_path):
        return tokenizer.get_prompt_ids("Transcribe ATC communication.")

    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab = json.load(f)

    # Harvest Callsigns (40 - Cleanest Set)
    callsigns = []
    if "CRITICAL_CALLSIGNS" in vocab:
        callsigns = list(vocab["CRITICAL_CALLSIGNS"].keys())
    random.seed(SEED)
    random.shuffle(callsigns)
    callsigns_list = [c.upper() for c in callsigns[:40]]

    # Harvest Waypoints (10 - Cleanest Set)
    waypoints = []
    if "CRITICAL_LOCATIONS" in vocab:
        waypoints = list(vocab["CRITICAL_LOCATIONS"].keys())
    random.seed(SEED)
    random.shuffle(waypoints)
    waypoints_list = [w.upper() for w in waypoints[:10]]

    # Construct Standard V26 Prompt
    # "Transcribe these ATC communications."
    prompt_parts = ["Transcribe these ATC communications."]

    if callsigns_list:
        prompt_parts.append("Callsigns: " + ", ".join(callsigns_list) + ".")

    if waypoints_list:
        prompt_parts.append("Waypoints: " + ", ".join(waypoints_list) + ".")

    full_prompt = " ".join(prompt_parts)
    print(
        f'V37 Prompt: Deep Positive Style ({len(callsigns_list)} Calls, {len(waypoints_list)} Wpts).'
    )
    return torch.tensor(tokenizer.get_prompt_ids(full_prompt)).to(DEVICE)


# Path Handling
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


# Cleaner
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


# Main Logic
def main():
    global FOUND_AUDIO_DIRS
    FOUND_AUDIO_DIRS = scan_for_audio_files(os.getcwd())

    print(f"Loading Model: {MODEL_PATH}")
    processor = WhisperProcessor.from_pretrained(MODEL_PATH)
    model = WhisperForConditionalGeneration.from_pretrained(MODEL_PATH).to(DEVICE)

    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []

    wer_metric = evaluate.load("wer")
    cer_metric = evaluate.load("cer")

    prompt_ids = load_v37_prompt(VOCAB_FILE, processor.tokenizer)

    dataset = load_dataset("json", data_files=TEST_DATA_PATH, split="train")
    dataset = dataset.map(fix_audio_path).filter(lambda x: os.path.exists(x["audio_file"]))
    dataset = dataset.cast_column("audio_file", Audio(sampling_rate=16000))

    def map_to_pred(batch):
        audio = batch["audio_file"]
        input_features = processor(
            audio["array"], sampling_rate=16000, return_tensors="pt"
        ).input_features.to(DEVICE)

        with torch.no_grad():
            # V37 CONFIG: DEEP BEAM + POSITIVE PENALTY
            predicted_ids = model.generate(
                input_features,
                prompt_ids=prompt_ids,
                language="en",
                task="transcribe",
                num_beams=15,  # Deep Search (Low Subs)
                num_return_sequences=1,
                repetition_penalty=1.0,  # Disabled
                length_penalty=0.1,  # Positive length penalty used in this run.
                max_new_tokens=128,
            )

        transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
        return {"reference": batch["ASR_transcript_clean"], "prediction": transcription}

    print("Starting V37 'Deep-Positive' Evaluation...")
    results = dataset.map(
        map_to_pred, remove_columns=dataset.column_names, load_from_cache_file=False
    )

    refs = [clean_atc_text(r) for r in results["reference"]]
    preds = [clean_atc_text(p) for p in results["prediction"]]

    valid_pairs = [(p, r) for p, r in zip(preds, refs) if r.strip()]

    if valid_pairs:
        p_tuple, r_tuple = zip(*valid_pairs)
        p_list = list(p_tuple)
        r_list = list(r_tuple)

        # Standard WER / CER
        wer = 100 * wer_metric.compute(predictions=p_list, references=r_list)
        cer = 100 * cer_metric.compute(predictions=p_list, references=r_list)
        accuracy = 100 - wer

        # Detailed S/D/I Analysis
        out = jiwer.process_words(r_list, p_list)
        total_words = out.hits + out.substitutions + out.deletions

        sub_rate = out.substitutions / total_words * 100
        del_rate = out.deletions / total_words * 100
        ins_rate = out.insertions / total_words * 100

        print("\n" + "=" * 50)
        print("V37 EVALUATION")
        print("=" * 50)
        print(f"Word Error Rate (WER):      {wer:.2f}%")
        print(f"100 - WER (not accuracy):         {accuracy:.2f}%")
        print(f"Character Error Rate (CER): {cer:.2f}%")
        print("-" * 50)
        print("True Error Breakdown:")
        print(f"    Substitution (Sub):      {sub_rate:.2f}%")
        print(f"    Deletion (Del):          {del_rate:.2f}%")
        print(f"    Insertion (Ins):         {ins_rate:.2f}%")
        print("=" * 50)
        print("Config: Beam=15, Prompt=V26(40/10), LenPen=0.1")


if __name__ == "__main__":
    main()
