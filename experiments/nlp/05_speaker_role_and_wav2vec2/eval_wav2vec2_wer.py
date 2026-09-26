# ---------------------------------------------------------------
# Archived experiment (late 2025). Original file: NLP/test_asr_comparison.py
# What it does: WER/CER of the wav2vec2 baseline.
# Not maintained: paths and dependencies are as they were at the time.
# ---------------------------------------------------------------
import os
import json
import torch
import torchaudio
import torchaudio.transforms as T
import jiwer
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from tqdm import tqdm

# Configuration
# Path to your trained ASR model (Part 1 output)
ASR_MODEL_PATH = "comparison_pure_asr_model"

# Path to your test data
TEST_JSON = "processed_data/ner_dataset_raw_split/test_raw.json"
AUDIO_BASE_DIR = "processed_data/audio"

# Hardware Setup
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_test_data(json_path):
    """
    Parses your JSON to get audio paths and reference text.
    """
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Cannot find {json_path}")

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    dataset = []
    for session in data:
        for turn in session.get("dialogue", []):
            audio_file = turn.get("audio", "")
            if not audio_file:
                continue

            full_path = os.path.join(AUDIO_BASE_DIR, audio_file)

            # Skip if file doesn't exist
            if not os.path.exists(full_path):
                continue

            text = turn.get("text", "").strip().upper()
            if len(text) < 1:
                continue

            dataset.append({"path": full_path, "ref_text": text})
    return dataset


def main():
    print(f"Running CER Test on: {DEVICE}")

    # Load Model and Processor
    print("Loading ASR Model...")
    try:
        processor = Wav2Vec2Processor.from_pretrained(ASR_MODEL_PATH)
        model = Wav2Vec2ForCTC.from_pretrained(ASR_MODEL_PATH).to(DEVICE)
    except Exception as e:
        print(f"Error loading model: {e}")
        print(f"   (Did you run the 'train_pure_asr.py' script first?)")
        return

    # Load Data
    print("Loading Test Data...")
    test_samples = load_test_data(TEST_JSON)
    print(f'Found {len(test_samples)} samples.')

    all_references = []
    all_hypotheses = []

    # Inference Loop
    print("Starting Inference...")
    for sample in tqdm(test_samples):
        try:
            # Load Audio (using torchaudio to avoid librosa issues)
            waveform, sample_rate = torchaudio.load(sample["path"])

            # Resample to 16kHz if necessary
            if sample_rate != 16000:
                resampler = T.Resample(sample_rate, 16000)
                waveform = resampler(waveform)

            # Convert to Mono (if stereo)
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)

            # Prepare Input Tensor
            input_values = processor(
                waveform.squeeze().numpy(), sampling_rate=16000, return_tensors="pt"
            ).input_values.to(DEVICE)

            # Forward Pass (No Gradients needed)
            with torch.no_grad():
                logits = model(input_values).logits

            # Decode (Greedy Search)
            predicted_ids = torch.argmax(logits, dim=-1)
            transcription = processor.batch_decode(predicted_ids)[0]

            # Store results
            all_references.append(sample["ref_text"])
            all_hypotheses.append(transcription)

        except Exception as e:
            print(f"Error processing {sample['path']}: {e}")
            continue

    # Calculate Metrics
    if len(all_references) > 0:
        print("\n" + "=" * 30)
        print("      RESULTS      ")
        print("=" * 30)

        # Calculate CER
        cer = jiwer.cer(all_references, all_hypotheses)

        # Calculate WER (for context)
        wer = jiwer.wer(all_references, all_hypotheses)

        print(f"Total Samples: {len(all_references)}")
        print(f'Character Error Rate (CER): {cer:.4f} ({cer * 100:.2f}%)')
        print(f'Word Error Rate (WER):      {wer:.4f} ({wer * 100:.2f}%)')
        print("=" * 30)
    else:
        print("No data processed.")


if __name__ == "__main__":
    main()
