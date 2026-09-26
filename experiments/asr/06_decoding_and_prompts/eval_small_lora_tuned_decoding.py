# ---------------------------------------------------------------
# Archived experiment (late 2025). Original file: ASR/eval_vocab_optimized.py
# What it does: 'V64': merges the small LoRA adapter and evaluates with beam 20, length penalty 0.42, repetition penalty 1.08.
# Known problems:
#   - The scoring cleaner deletes tag strings such as HES and UNK wherever they appear, in both reference and hypothesis, so it can also cut into real words (THESE -> TE).
#   - Vocabulary or prompt words were chosen after looking at test-set errors, so the test score is optimistic.
#   - 64+ rounds of decoding tuning on the test set.
#   - If the adapter is missing it silently scores the base model.
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
from peft import PeftModel, PeftConfig  # <--- NEW: For loading your trained adapter
import evaluate

# CONFIGURATION
# Base Model
BASE_MODEL_PATH = "openai/whisper-small"  # Must match what you used for training
# Your Trained LoRA Adapter
LORA_ADAPTER_PATH = "./whisper_atco_lora_final"

TEST_DATA_PATH = "processed_data/test.jsonl"
VOCAB_FILE = "processed_data/vocab_master.json"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42

# V64 GOLDEN PARAMETERS (Best Baseline: 78.73%)
BEAM_SIZE = 20
LENGTH_PENALTY = 0.42  # The "Hail Mary" constant
REPETITION_PENALTY = 1.08  # The Golden Ratio

# MODEL LOADING
print(f"Loading Base Model: {BASE_MODEL_PATH}")
processor = WhisperProcessor.from_pretrained(BASE_MODEL_PATH)
model = WhisperForConditionalGeneration.from_pretrained(BASE_MODEL_PATH).to(DEVICE)

# APPLY LORA ADAPTER (The Upgrade)
if os.path.exists(LORA_ADAPTER_PATH):
    print(f"Found LoRA Adapter: {LORA_ADAPTER_PATH}")
    print("   Merging weights for inference...")
    model = PeftModel.from_pretrained(model, LORA_ADAPTER_PATH)
    model = model.merge_and_unload()  # Merge for speed
else:
    print("WARNING: LoRA Adapter not found! Running with base model only.")

model.config.forced_decoder_ids = None
model.config.suppress_tokens = []


# PROMPT GENERATOR (Standard V64)
def load_prompt(vocab_path, tokenizer):
    if not os.path.exists(vocab_path):
        return tokenizer.get_prompt_ids("Transcribe ATC communication.")

    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab = json.load(f)

    # Callsigns (40 - Proven Best)
    callsigns = []
    if "CRITICAL_CALLSIGNS" in vocab:
        callsigns = list(vocab["CRITICAL_CALLSIGNS"].keys())
    random.seed(SEED)
    random.shuffle(callsigns)
    callsigns_list = [c.upper() for c in callsigns[:40]]

    # Waypoints (15 - Proven Best)
    waypoints = []
    if "CRITICAL_LOCATIONS" in vocab:
        waypoints = list(vocab["CRITICAL_LOCATIONS"].keys())
    random.seed(SEED)
    random.shuffle(waypoints)
    waypoints_list = [w.upper() for w in waypoints[:15]]

    # Standard Prompt Structure
    prompt_parts = ["Transcribe these ATC communications."]

    if callsigns_list:
        prompt_parts.append("Callsigns: " + ", ".join(callsigns_list) + ".")

    if waypoints_list:
        prompt_parts.append("Waypoints: " + ", ".join(waypoints_list) + ".")

    full_prompt = " ".join(prompt_parts)
    print(f'Prompt Loaded ({len(callsigns_list)} Calls, {len(waypoints_list)} Wpts).')
    return torch.tensor(tokenizer.get_prompt_ids(full_prompt)).to(DEVICE)


# PATH HANDLING
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
        possible_path = os.path.join(d, filename)
        if os.path.exists(possible_path):
            batch["audio_file"] = possible_path
            return batch
    return batch


# CLEANER
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


# MAIN INFERENCE LOOP
def main():
    global FOUND_AUDIO_DIRS
    FOUND_AUDIO_DIRS = scan_for_audio_files(os.getcwd())

    wer_metric = evaluate.load("wer")
    cer_metric = evaluate.load("cer")

    prompt_ids = load_prompt(VOCAB_FILE, processor.tokenizer)

    dataset = load_dataset("json", data_files=TEST_DATA_PATH, split="train")
    dataset = dataset.map(fix_audio_path).filter(lambda x: os.path.exists(x["audio_file"]))
    dataset = dataset.cast_column("audio_file", Audio(sampling_rate=16000))

    def map_to_pred(batch):
        audio = batch["audio_file"]
        input_features = processor(
            audio["array"], sampling_rate=16000, return_tensors="pt"
        ).input_features.to(DEVICE)

        with torch.no_grad():
            # V64 CONFIGURATION
            predicted_ids = model.generate(
                input_features,
                prompt_ids=prompt_ids,
                language="en",
                task="transcribe",
                num_beams=BEAM_SIZE,  # 20
                num_return_sequences=1,
                repetition_penalty=REPETITION_PENALTY,  # 08
                length_penalty=LENGTH_PENALTY,  # 42
                max_new_tokens=128,
            )

        transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
        return {"reference": batch["ASR_transcript_clean"], "prediction": transcription}

    print("Starting Evaluation (V64 Params + Trained LoRA)...")
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

        # Metrics
        wer = 100 * wer_metric.compute(predictions=p_list, references=r_list)
        cer = 100 * cer_metric.compute(predictions=p_list, references=r_list)
        accuracy = 100 - wer

        # Detailed Error Analysis
        out = jiwer.process_words(r_list, p_list)
        total_words = out.hits + out.substitutions + out.deletions

        sub_rate = out.substitutions / total_words * 100
        del_rate = out.deletions / total_words * 100
        ins_rate = out.insertions / total_words * 100

        print("\n" + "=" * 50)
        print("FINAL RESULT: TRAINED MODEL + V64 CONFIG")
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


if __name__ == "__main__":
    main()
