"""Publish the split and each model's errors clip by clip, without any transcript.

Anyone with the ATCO2 one-hour set can rebuild the exact split from the clip
IDs, and check every interval in results/retraining.json from the per-clip word
error counts. No audio or transcript text is written, only file names and counts.

    python -m evaluation.export_clips

Writes results/clips/split.csv (all 874 clips: recording, airport, split) and
results/clips/errors_validation.csv and errors_test.csv (word errors per model,
under the decoding each model was scored with, and the clip's reference words),
and results/clips/errors_all_874.csv (every clip's errors in the cross-validation
by recording, the leave-one-airport-out test and unmodified Whisper-small).
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


def export_folds():
    """Per-clip word errors for the cross-validation and the airport test (beam, as scored)."""
    from .session_split import airport_id

    by_recording, by_airport, zero = {}, {}, {}
    for k in range(5):
        for r in rows(ROOT / f"outputs/evaluation/cv_fold{k}_test/asr_predictions.jsonl"):
            by_recording[r["file"]] = (k, r)
    for folder in sorted((ROOT / "outputs/evaluation").glob("airport_*_test")):
        for r in rows(folder / "asr_predictions.jsonl"):
            by_airport[r["file"]] = r
    for name in ["zero_shot", "zero_shot_train"]:
        for r in rows(ROOT / f"outputs/evaluation/{name}/asr_predictions.jsonl"):
            zero[r["file"]] = r
    files = sorted(by_recording)
    count = lambda r: tuple(int(x[0]) for x in clip_counts([r], "beam", spelled=True))
    with (OUT / "errors_all_874.csv").open("w", newline="") as out:
        writer = csv.writer(out)
        writer.writerow(["clip", "airport", "cv_fold", "reference_words", "cv_by_recording_beam",
                         "leave_one_airport_out_beam", "unmodified_whisper_small_beam"])
        totals = [0, 0, 0, 0]
        for f in files:
            fold, r = by_recording[f]
            e_rec, words = count(r)
            e_air, w2 = count(by_airport[f]) if f in by_airport else (None, None)
            e_zero, w3 = count(zero[f])
            assert w2 in (None, words) and w3 == words, f
            writer.writerow([clip_id(f), airport_id(f), fold, words, e_rec, e_air, e_zero])
            totals = [totals[0] + words, totals[1] + e_rec, totals[2] + (e_air or 0), totals[3] + e_zero]
    print("all 874:", len(files), "clips; words, errors by recording, by airport, unmodified:", totals)


if __name__ == "__main__":
    main()
    export_folds()
