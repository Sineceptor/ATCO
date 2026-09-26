"""Make a train/test split in which no recording appears on both sides.

The original split was made per clip, so 101 of 175 evaluation clips share a
recording (same controller, same frequency, same minutes) with training clips.
This script pools both manifests and splits by recording instead.

    python -m evaluation.session_split
    # then retrain with training/train_whisper.py pointed at the new train.jsonl

A recording ID is the filename without its final clip number.
"""

import argparse
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def recording_id(audio_file):
    name = audio_file.replace("\\", "/").split("/")[-1]
    return name.rsplit(".", 1)[0].rsplit("_", 1)[0]


def split_by_recording(rows, test_fraction=0.2, seed=42):
    """Assign whole recordings to test until it holds about test_fraction of clips."""
    groups = {}
    for row in rows:
        groups.setdefault(recording_id(row["audio_file"]), []).append(row)
    order = sorted(groups)
    random.Random(seed).shuffle(order)
    target = test_fraction * len(rows)
    train, test = [], []
    for key in order:
        (test if len(test) < target else train).extend(groups[key])
    return train, test


def make_folds(rows, folds=5, validation_fraction=0.1, seed=42):
    """Cross-validation by recording: every clip is tested exactly once.

    Recordings are shuffled and dealt to whichever fold has the fewest clips, so
    the folds come out close in size. For each fold, whole recordings from the
    remaining data are set aside as validation for choosing checkpoints.
    """
    groups = {}
    for row in rows:
        groups.setdefault(recording_id(row["audio_file"]), []).append(row)
    order = sorted(groups)
    random.Random(seed).shuffle(order)
    buckets = [[] for _ in range(folds)]
    for key in order:
        min(buckets, key=lambda b: sum(len(groups[k]) for k in b)).append(key)
    result = []
    for k, bucket in enumerate(buckets):
        test = [row for key in bucket for row in groups[key]]
        rest = [row for key in order if key not in set(bucket) for row in groups[key]]
        train, validation = split_by_recording(rest, validation_fraction, seed + k)
        result.append(dict(train=train, validation=validation, test=test))
    return result


def airport_id(audio_file):
    """The ICAO code that starts every ATCO2 file name, for example LKPR for Prague."""
    return recording_id(audio_file).split("_", 1)[0]


def make_airport_folds(rows, validation_fraction=0.1, seed=42):
    """Leave one airport out: each fold tests on every clip from one airport.

    The model for that fold trains on the other airports, with whole recordings
    set aside as validation for choosing checkpoints. This is a stricter meaning
    of unseen than a new recording: new controllers, frequencies and place names.
    """
    airports = sorted({airport_id(r["audio_file"]) for r in rows})
    result = {}
    for k, airport in enumerate(airports):
        test = [r for r in rows if airport_id(r["audio_file"]) == airport]
        rest = [r for r in rows if airport_id(r["audio_file"]) != airport]
        train, validation = split_by_recording(rest, validation_fraction, seed + k)
        result[airport] = dict(train=train, validation=validation, test=test)
    return result


def split_original_test(train_rows, test_rows):
    """Keep the 2025 training clips, and divide the 2025 test clips in two.

    Test clips from recordings that also gave training clips become validation
    data, for choosing checkpoints and decoding settings. The rest, from
    recordings no model was trained on, become the test set. Because the 2025
    checkpoint was trained on exactly the same clips, it can be scored on this
    test set too, which makes old and new models directly comparable.
    """
    seen = {recording_id(r["audio_file"]) for r in train_rows}
    validation = [r for r in test_rows if recording_id(r["audio_file"]) in seen]
    test = [r for r in test_rows if recording_id(r["audio_file"]) not in seen]
    return train_rows, validation, test


def main():
    cli = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    cli.add_argument("--data-dir", type=Path, default=ROOT / "datasets")
    cli.add_argument("--output", type=Path, default=ROOT / "datasets/speech_by_recording")
    cli.add_argument("--test-fraction", type=float, default=0.2)
    cli.add_argument("--seed", type=int, default=42)
    cli.add_argument("--folds", type=int, help="Write this many cross-validation folds instead")
    cli.add_argument("--by-airport", action="store_true", help="Write one leave-one-airport-out fold per airport")
    cli.add_argument(
        "--from-original",
        action="store_true",
        help="Keep the 2025 training clips; split the 2025 test clips into validation and unseen test",
    )
    args = cli.parse_args()
    if args.from_original:
        return write_from_original(args)
    rows = []
    for split in ["train", "test"]:
        text = (args.data_dir / f"speech/{split}.jsonl").read_text(encoding="utf-8")
        rows += [json.loads(s) for s in text.splitlines() if s.strip()]
    if args.by_airport:
        output = ROOT / "datasets/speech_airports" if args.output == ROOT / "datasets/speech_by_recording" else args.output
        for airport, fold in make_airport_folds(rows, seed=args.seed).items():
            places = {name: {airport_id(r["audio_file"]) for r in part} for name, part in fold.items()}
            assert places["test"] == {airport} and airport not in places["train"] | places["validation"]
            (output / airport).mkdir(parents=True, exist_ok=True)
            for name, part in fold.items():
                with (output / f"{airport}/{name}.jsonl").open("w", encoding="utf-8") as out:
                    for row in part:
                        out.write(json.dumps(row, ensure_ascii=False) + "\n")
            print(f"{airport}: {len(fold['train'])} train, {len(fold['validation'])} validation, {len(fold['test'])} test")
        return
    if args.folds:
        output = ROOT / "datasets/speech_folds" if args.output == ROOT / "datasets/speech_by_recording" else args.output
        for k, fold in enumerate(make_folds(rows, args.folds, seed=args.seed)):
            ids = {name: {recording_id(r["audio_file"]) for r in part} for name, part in fold.items()}
            assert not (ids["train"] & ids["test"] or ids["validation"] & ids["test"] or ids["train"] & ids["validation"])
            (output / f"fold{k}").mkdir(parents=True, exist_ok=True)
            for name, part in fold.items():
                with (output / f"fold{k}/{name}.jsonl").open("w", encoding="utf-8") as out:
                    for row in part:
                        out.write(json.dumps(row, ensure_ascii=False) + "\n")
            print(f"fold {k}: {len(fold['train'])} train, {len(fold['validation'])} validation, {len(fold['test'])} test")
        return
    train, test = split_by_recording(rows, args.test_fraction, args.seed)
    shared = {recording_id(r["audio_file"]) for r in train} & {recording_id(r["audio_file"]) for r in test}
    assert not shared
    args.output.mkdir(parents=True, exist_ok=True)
    for name, part in [("train", train), ("test", test)]:
        with (args.output / f"{name}.jsonl").open("w", encoding="utf-8") as out:
            for row in part:
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"{len(train)} training clips, {len(test)} evaluation clips, 0 shared recordings")
    print(args.output)


def write_from_original(args):
    output = args.output if args.output != ROOT / "datasets/speech_by_recording" else ROOT / "datasets/speech_split"
    parts = {}
    for split in ["train", "test"]:
        text = (args.data_dir / f"speech/{split}.jsonl").read_text(encoding="utf-8")
        parts[split] = [json.loads(s) for s in text.splitlines() if s.strip()]
    train, validation, test = split_original_test(parts["train"], parts["test"])
    assert not {recording_id(r["audio_file"]) for r in train} & {recording_id(r["audio_file"]) for r in test}
    output.mkdir(parents=True, exist_ok=True)
    for name, part in [("train", train), ("validation", validation), ("test", test)]:
        with (output / f"{name}.jsonl").open("w", encoding="utf-8") as out:
            for row in part:
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"{len(train)} training, {len(validation)} validation, {len(test)} unseen-recording test clips")
    print(output)


if __name__ == "__main__":
    main()
