"""Publish the split and each model's errors clip by clip, without any transcript.

Anyone with the ATCO2 one-hour set can rebuild the exact split from the clip
IDs, and check every interval in results/retraining.json from the per-clip word
error counts. No audio or transcript text is written, only file names and counts.

    python -m evaluation.export_clips

Writes results/clips/split.csv (all 874 clips: recording, airport, split) and
results/clips/errors_validation.csv and errors_test.csv (word errors per model,
under the decoding each model was scored with, and the clip's reference words).
"""

import csv
import json
from pathlib import Path

from .compare_models import clip_counts
from .session_split import airport_id, recording_id

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "clips"
FOLDERS = {"checkpoint_2025": "2025"}


def rows(path):
    return [json.loads(s) for s in Path(path).read_text().splitlines() if s.strip()]


def clip_id(audio_file):
    return Path(audio_file.replace("\\", "/")).stem


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "split.csv").open("w", newline="") as out:
        writer = csv.writer(out)
        writer.writerow(["clip", "recording", "airport", "split"])
        for part in ["train", "validation", "test"]:
            for row in rows(ROOT / f"datasets/speech_split/{part}.jsonl"):
                writer.writerow([clip_id(row["audio_file"]), recording_id(row["audio_file"]),
                                 airport_id(row["audio_file"]), part])

    report = json.loads((ROOT / "results/retraining.json").read_text())
    models = {**report["runs"], **report.get("ablations", {})}
    for part in ["validation", "test"]:
        columns, clips, words = {}, None, None
        for name, entry in models.items():
            mode = entry["headline_test"]["decoding"]
            found = sorted(rows(ROOT / f"outputs/evaluation/{FOLDERS.get(name, name)}_{part}/asr_predictions.jsonl"),
                           key=lambda r: r["file"])
            edits, counts = clip_counts(found, mode, spelled=True)
            names = [clip_id(r["file"]) for r in found]
            if clips is None:
                clips, words = names, counts
            assert names == clips and list(counts) == list(words), name
            columns[f"{name}_{mode}"] = [int(e) for e in edits]
        with (OUT / f"errors_{part}.csv").open("w", newline="") as out:
            writer = csv.writer(out)
            writer.writerow(["clip", "reference_words", *columns])
            for i, clip in enumerate(clips):
                writer.writerow([clip, int(words[i]), *(column[i] for column in columns.values())])
        total = {k: sum(v) for k, v in columns.items()}
        print(part, len(clips), "clips;", int(sum(words)), "words;", total)


if __name__ == "__main__":
    main()
