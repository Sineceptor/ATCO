"""Measure word error rate against signal-to-noise ratio.

White Gaussian noise is added to each evaluation clip at a chosen SNR and the
clip is transcribed again. Optionally the clip is first passed through a
300-3400 Hz band-pass, the voice band of an AM radio channel. Running this for
the fine-tuned checkpoint and for unmodified Whisper-small shows whether
fine-tuning on real radio audio bought any robustness to noise.

    python -m evaluation.noise_sweep --snr 30 20 15 10 5 0
    python -m evaluation.noise_sweep --model openai/whisper-small --allow-downloads --label zero_shot

SNR is defined on the clip as a whole: 10 log10(mean signal power / mean noise
power). ATCO2 clips already contain radio noise, so the stated SNR is an upper
bound on the true one. Noise is seeded per clip, so runs are repeatable.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def add_noise(audio, snr_db, rng):
    """Return audio plus white Gaussian noise at the requested SNR in dB."""
    signal_power = float(np.mean(np.square(audio, dtype=np.float64)))
    if signal_power == 0:
        return audio
    noise_power = signal_power / (10 ** (snr_db / 10))
    noise = rng.normal(0.0, np.sqrt(noise_power), size=audio.shape)
    return (audio + noise).astype(np.float32)


def voice_band(audio, rate=16000, low=300.0, high=3400.0):
    """Fourth-order Butterworth band-pass, applied forwards and backwards."""
    from scipy.signal import butter, sosfiltfilt

    sos = butter(4, [low, high], btype="band", fs=rate, output="sos")
    return sosfiltfilt(sos, audio).astype(np.float32)


def clip_seed(name, snr_db):
    text = f"{name}|{snr_db}".encode()
    return int.from_bytes(hashlib.sha256(text).digest()[:8], "big")


def main():
    cli = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    cli.add_argument("--snr", type=float, nargs="+", default=[30, 20, 15, 10, 5, 0])
    cli.add_argument("--band-pass", action="store_true", help="Apply the 300-3400 Hz filter first")
    cli.add_argument("--model", help="Checkpoint folder or hub name; default models/whisper-small")
    cli.add_argument("--allow-downloads", action="store_true")
    cli.add_argument("--label", default="finetuned")
    cli.add_argument("--data-dir", type=Path, default=ROOT / "datasets")
    cli.add_argument("--models-dir", type=Path, default=ROOT / "models")
    cli.add_argument("--output", type=Path, default=ROOT / "outputs/evaluation")
    cli.add_argument("--manifest", type=Path, help="Score these clips instead of speech/test.jsonl")
    cli.add_argument("--audio-dir", type=Path, help="Folder holding the manifest's audio")
    cli.add_argument("--limit", type=int, help="Smoke test on the first N clips")
    cli.add_argument("--threads", type=int, default=4)
    cli.add_argument("--device", default="cpu", help='"cpu", or "mps" for an Apple GPU')
    cli.add_argument("--guards", action="store_true",
                     help="Add the repetition guards (penalty 1.2, no repeated 3-grams) so noise cannot trap "
                     "greedy decoding in a loop")
    args = cli.parse_args()

    import jiwer
    import soundfile as sf
    import torch
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    from .evaluate_models import error_rates, spell_digits

    torch.set_num_threads(args.threads)
    model_path = args.model or args.models_dir / "whisper-small"
    local_only = not args.allow_downloads
    processor = WhisperProcessor.from_pretrained(model_path, local_files_only=local_only)
    model = WhisperForConditionalGeneration.from_pretrained(
        model_path, local_files_only=local_only
    ).eval().to(args.device)
    plain = jiwer.Compose(
        [jiwer.ToUpperCase(), jiwer.RemovePunctuation(), jiwer.RemoveMultipleSpaces(), jiwer.Strip()]
    )

    def normalise(text):
        return plain(spell_digits(text))

    manifest = args.manifest or args.data_dir / "speech/test.jsonl"
    audio_dir = args.audio_dir or args.data_dir / "speech/audio"
    rows = [json.loads(s) for s in manifest.read_text().splitlines() if s.strip()]
    if args.limit:
        rows = rows[: args.limit]
    clips = []
    for row in rows:
        name = row["audio_file"].replace("\\", "/").split("/")[-1]
        audio, rate = sf.read(audio_dir / name, dtype="float32")
        if rate != 16000 or audio.ndim != 1:
            raise ValueError(f"{name}: expected mono 16 kHz audio")
        if args.band_pass:
            audio = voice_band(audio)
        clips.append((name, audio, normalise(row["ASR_transcript_clean"])))

    guards = dict(repetition_penalty=1.2, no_repeat_ngram_size=3) if args.guards else {}

    def transcribe(audio):
        inputs = processor(audio, sampling_rate=16000, return_tensors="pt", return_attention_mask=True).to(args.device)
        with torch.inference_mode():
            ids = model.generate(**inputs, language="en", task="transcribe", max_new_tokens=128, **guards)
        return processor.batch_decode(ids, skip_special_tokens=True)[0]

    results = []
    for snr in [None, *args.snr]:  # None = the clip as recorded
        predictions = []
        for name, audio, _ in clips:
            if snr is not None:
                audio = add_noise(audio, snr, np.random.default_rng(clip_seed(name, snr)))
            predictions.append(normalise(transcribe(audio)))
        kept = [(ref, hyp) for (_, _, ref), hyp in zip(clips, predictions) if ref]
        rates = error_rates([r for r, _ in kept], [h for _, h in kept])
        results.append({"added_noise_snr_db": snr, "clips": len(kept), **rates})
        print(f"SNR {'as recorded' if snr is None else f'{snr:g} dB':>12}: WER {rates['wer']:.2%}", flush=True)

    args.output.mkdir(parents=True, exist_ok=True)
    suffix = args.label + ("_bandpass" if args.band_pass else "")
    path = args.output / f"noise_sweep_{suffix}.json"
    path.write_text(json.dumps({
        "model": str(model_path),
        "manifest": str(manifest.relative_to(ROOT) if manifest.is_relative_to(ROOT) else manifest),
        "band_pass_300_3400_hz": args.band_pass,
        "decoding": "greedy, English, max_new_tokens=128"
        + (", repetition_penalty=1.2, no_repeat_ngram_size=3" if args.guards else ""),
        "normalisation": "digits spelled out, uppercase, punctuation removed",
        "results": results,
    }, indent=2) + "\n")
    print(path)


if __name__ == "__main__":
    main()
