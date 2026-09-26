"""Turn the UWB-ATCC corpus into clips and a manifest this project can train on.

    python -m training.prepare_uwb_atcc --source datasets/uwb_atcc_raw

UWB-ATCC (University of West Bohemia) is about 13 hours of real, hand-transcribed
controller and pilot speech from Czech airspace, published under CC BY-NC-SA 4.0
at https://huggingface.co/datasets/Jzuluaga/uwb_atcc. Only its training split is
used. The audio is not redistributed by this repository.

Its transcripts already follow nearly the same rules as the ATCO2 ones (lower
case, digits spelled out, no punctuation). Four spellings differ and are mapped
to the ATCO2 form, decided by looking only at the ATCO2 training and validation
transcripts: decimal -> point, xray -> x ray, alfa -> alpha, ok -> okay.
"""

import argparse
import io
import json
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
SPELLING = {"decimal": "point", "xray": "x ray", "alfa": "alpha", "ok": "okay"}


def harmonise(text):
    return " ".join(SPELLING.get(w, w) for w in text.lower().split())


def to_16k_mono(data, rate):
    if data.ndim > 1:
        data = data.mean(axis=1)
    if rate != 16000:
        from scipy.signal import resample_poly

        data = resample_poly(data, 16000, rate)
    return data.astype(np.float32)


def main():
    cli = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    cli.add_argument("--source", type=Path, default=ROOT / "datasets/uwb_atcc_raw")
    cli.add_argument("--output", type=Path, default=ROOT / "datasets/uwb_atcc")
    cli.add_argument("--max-seconds", type=float, default=30.0)
    args = cli.parse_args()

    import pyarrow.parquet as pq

    audio_dir = args.output / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    kept = skipped = 0
    seconds = 0.0
    with (args.output / "train.jsonl").open("w", encoding="utf-8") as manifest:
        for shard in sorted((args.source / "data").glob("train-*.parquet")):
            table = pq.read_table(shard, columns=["id", "audio", "text"])
            for utt_id, audio, text in zip(*(table.column(c).to_pylist() for c in ["id", "audio", "text"])):
                text = harmonise(text or "")
                data, rate = sf.read(io.BytesIO(audio["bytes"]), dtype="float32")
                data = to_16k_mono(data, rate)
                duration = len(data) / 16000
                if not text or not 0.3 <= duration <= args.max_seconds:
                    skipped += 1
                    continue
                name = f"UWB_{utt_id}.wav"
                sf.write(audio_dir / name, data, 16000, subtype="PCM_16")
                manifest.write(json.dumps(dict(audio_file=name, duration_sec=round(duration, 2),
                                               ASR_transcript_clean=text, utt_id=utt_id)) + "\n")
                kept += 1
                seconds += duration
    print(f"{kept} clips, {seconds / 3600:.1f} hours kept; {skipped} skipped (empty or outside 0.3 to "
          f"{args.max_seconds:g} s)")
    print(args.output / "train.jsonl")


if __name__ == "__main__":
    main()
