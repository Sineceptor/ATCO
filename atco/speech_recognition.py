"""Whisper transcription for one recording of up to 30 seconds."""
import torch
import soundfile as sf
import scipy.signal
import numpy as np
from transformers import WhisperProcessor, WhisperForConditionalGeneration


def load_audio_fast(file_path):
    """Load mono audio and resample to 16 kHz."""
    try:
        audio, orig_sr = sf.read(file_path)

        # Average stereo channels to mono.
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)

        # Whisper expects 16 kHz audio.
        if orig_sr != 16000:
            num_samples = int(len(audio) * 16000 / orig_sr)
            audio = scipy.signal.resample(audio, num_samples)

        return audio.astype(np.float32)
    except Exception as e:
        print(f"Could not read audio: {e}")
        return None


def transcribe(model_path, audio_path, local_files_only=False):
    print(f"Loading Whisper: {model_path}")

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"  # Apple silicon GPU
    else:
        device = "cpu"
    print(f"Device: {device}")

    try:
        processor = WhisperProcessor.from_pretrained(model_path, local_files_only=local_files_only)
        model = (
            WhisperForConditionalGeneration.from_pretrained(
                model_path, local_files_only=local_files_only
            )
            .to(device)
            .eval()
        )
    except Exception as e:
        print(f"Could not load the model: {e}")
        raise

    print(f"Reading audio: {audio_path}")
    audio_input = load_audio_fast(audio_path)

    if audio_input is None:
        raise ValueError("Could not read input audio")
    if not 0 < len(audio_input) <= 30 * 16000:
        raise ValueError("Use one non-empty audio clip of at most 30 seconds")

    inputs = processor(
        audio_input, sampling_rate=16000, return_tensors="pt", return_attention_mask=True
    ).to(device)
    input_features = inputs.input_features

    print("Transcribing...")

    # English beam search with repetition guards. Plain greedy decoding sometimes
    # gets stuck repeating a phrase on noisy or non-English audio; on the
    # validation clips these settings gave the lowest error rate for both the 2025
    # and the retrained model.
    with torch.inference_mode():
        predicted_ids = model.generate(
            input_features,
            attention_mask=inputs.attention_mask,
            language="english",
            task="transcribe",
            max_new_tokens=128,
            num_beams=5,
            repetition_penalty=1.2,
            no_repeat_ngram_size=3,
        )

    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]

    print(f"Transcript: {transcription}")
    return transcription
