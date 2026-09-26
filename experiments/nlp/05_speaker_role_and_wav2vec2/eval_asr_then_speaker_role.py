# ---------------------------------------------------------------
# Archived experiment (late 2025). Original file: NLP/test_asr_srd.py
# What it does: End to end: wav2vec2 transcript -> speaker-role tagger, scored on correctly recognised words.
# Known problems:
#   - Scores only the first word of each matching chunk.
#   - Uses a shorter keyword heuristic than training did.
# Not maintained: paths and dependencies are as they were at the time.
# ---------------------------------------------------------------
import os
import json
import torch
import torchaudio
import torchaudio.transforms as T
import jiwer
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from transformers import (
    Wav2Vec2ForCTC,
    Wav2Vec2Processor,
    BertForTokenClassification,
    BertTokenizerFast,
)

# Configuration
ASR_MODEL_PATH = "comparison_pure_asr_model"
SRD_MODEL_PATH = "comparison_bert_srd_model"
TEST_JSON = "processed_data/ner_dataset_raw_split/test_raw.json"
AUDIO_BASE_DIR = "processed_data/audio"

# Set to True if your GPU is causing crashes
FORCE_CPU = False

ATCO_KEYWORDS = [
    "WIND",
    "DEGREES",
    "KNOTS",
    "QNH",
    "SQUAWK",
    "IDENT",
    "RADAR",
    "CLEARED",
    "CONTACT",
]
LABEL_MAP = {0: "PILOT", 1: "ATCO"}


def guess_speaker_role(text):
    text_upper = text.upper()
    if any(k in text_upper for k in ATCO_KEYWORDS):
        return "ATCO"
    if len(text_upper.split()) > 12:
        return "ATCO"
    return "PILOT"


def load_test_data(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    dataset = []
    for session in data:
        for turn in session.get("dialogue", []):
            audio_file = turn.get("audio", "")
            if not audio_file:
                continue
            full_path = os.path.join(AUDIO_BASE_DIR, audio_file)
            if not os.path.exists(full_path):
                continue

            text = turn.get("text", "").strip().upper()
            if len(text) < 2:
                continue

            role = turn.get("speaker", turn.get("role", "Unknown")).upper()
            if role == "UNKNOWN":
                role = guess_speaker_role(text)
            label = "ATCO" if any(x in role for x in ["ATC", "TOWER", "CONTROLLER"]) else "PILOT"

            dataset.append({"path": full_path, "ref_text": text, "ref_label": label})
    return dataset


def main():
    if FORCE_CPU:
        device = "cpu"
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Running on Device: {device}")

    print("Loading Models...")
    asr_processor = Wav2Vec2Processor.from_pretrained(ASR_MODEL_PATH)
    asr_model = Wav2Vec2ForCTC.from_pretrained(ASR_MODEL_PATH).to(device)
    srd_tokenizer = BertTokenizerFast.from_pretrained(SRD_MODEL_PATH)
    srd_model = BertForTokenClassification.from_pretrained(SRD_MODEL_PATH).to(device)

    print("Loading Test Data...")
    test_data = load_test_data(TEST_JSON)
    print(f'Found {len(test_data)} test samples.')

    all_ref_words = []
    all_hyp_words = []
    y_true_labels = []
    y_pred_labels = []

    print("Starting Inference Loop...")

    for i, sample in enumerate(test_data):
        print(f"\rProcessing {i + 1}/{len(test_data)}: {sample['path']}", end="")

        try:
            # Step 1: Load Audio with Torchaudio
            waveform, sample_rate = torchaudio.load(sample["path"])

            # Resample if not 16k
            if sample_rate != 16000:
                resampler = T.Resample(sample_rate, 16000)
                waveform = resampler(waveform)

            # Convert to mono if stereo
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)

            # Squeeze to get 1D array for processor
            input_values = waveform.squeeze().numpy()

            # Step 2: ASR Inference
            inputs = asr_processor(
                input_values, sampling_rate=16000, return_tensors="pt", padding=True
            )
            inputs = inputs.to(device)

            with torch.no_grad():
                logits = asr_model(inputs.input_values).logits

            pred_ids = torch.argmax(logits, dim=-1)
            pred_text = asr_processor.batch_decode(pred_ids)[0].upper()

            # Step 3: SRD Inference
            words = pred_text.split()
            if not words:
                all_ref_words.append(sample["ref_text"])
                all_hyp_words.append("")
                continue

            tokenized_inputs = srd_tokenizer(
                words, is_split_into_words=True, return_tensors="pt", truncation=True
            )
            tokenized_inputs = tokenized_inputs.to(device)

            with torch.no_grad():
                outputs = srd_model(**tokenized_inputs)

            predictions = torch.argmax(outputs.logits, dim=2)

            # Extract labels for words
            word_ids = tokenized_inputs.word_ids()
            word_labels = []
            seen_indices = set()

            for idx, w_id in enumerate(word_ids):
                if w_id is not None and w_id not in seen_indices:
                    label_id = predictions[0][idx].item()
                    word_labels.append(LABEL_MAP.get(label_id, "PILOT"))
                    seen_indices.add(w_id)

            # Step 4: Alignment & Scoring
            alignment = jiwer.process_words(sample["ref_text"], pred_text)
            all_ref_words.append(sample["ref_text"])
            all_hyp_words.append(pred_text)

            for chunk in alignment.alignments[0]:
                if chunk.type == "equal":
                    # Only score if ASR got the word right
                    if chunk.hyp_start_idx < len(word_labels):
                        y_true_labels.append(sample["ref_label"])
                        y_pred_labels.append(word_labels[chunk.hyp_start_idx])

        except Exception as e:
            print(f"\nError on file {sample['path']}: {e}")
            continue

    # Final Report
    print("\n\n" + "=" * 40)
    print("      ASR-SRD PIPELINE RESULTS      ")
    print("=" * 40)

    if len(all_ref_words) > 0:
        wer = jiwer.wer(all_ref_words, all_hyp_words)
        print(f"\n  ASR Performance:")
        print(f'   - Word Error Rate (WER): {wer:.4f} ({wer * 100:.2f}%)')

        print(f"\nSRD Performance (on correctly transcribed words):")
        if y_true_labels:
            print(classification_report(y_true_labels, y_pred_labels, labels=["ATCO", "PILOT"]))

            cm = confusion_matrix(y_true_labels, y_pred_labels, labels=["ATCO", "PILOT"])
            print("\n   Confusion Matrix:")
            print("           Pred:ATCO  Pred:PILOT")
            print(f"   True:ATCO   {cm[0][0]:<10} {cm[0][1]}")
            print(f"   True:PILOT  {cm[1][0]:<10} {cm[1][1]}")
        else:
            print("    No correctly aligned words found to evaluate SRD.")
    else:
        print("No data processed.")

    print("\n" + "=" * 40)


if __name__ == "__main__":
    main()
