"""Describe the saved split boundaries and changes in NER reference labels."""

import argparse
import json
from pathlib import Path

from .evaluate_models import ROOT, digest
from . import reference_labels as labels


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--data-dir", type=Path, default=ROOT / "datasets")
    cli.add_argument("--output", type=Path, default=ROOT / "outputs/evaluation")
    args = cli.parse_args()
    root = args.data_dir
    asr = {
        split: [
            json.loads(s)
            for s in (root / f"speech/{split}.jsonl")
            .read_text()
            .splitlines()
            if s.strip()
        ]
        for split in ["train", "test"]
    }
    files = {split: {row["audio_file"] for row in rows} for split, rows in asr.items()}
    sessions = {
        split: {name.rsplit("_", 1)[0] for name in names}
        for split, names in files.items()
    }
    nlp = {
        split: json.loads(
            (
                root / f"entities/raw/{split}.json"
            ).read_text()
        )
        for split in ["train", "test"]
    }
    nlp_sessions = {
        split: {row["session_id"] for row in rows} for split, rows in nlp.items()
    }
    regenerated = labels.load_raw_and_tag(
        str(root / "entities/raw/test.json")
    )
    stored_path = root / "entities/test.jsonl"
    stored = [json.loads(s) for s in stored_path.read_text().splitlines() if s.strip()]
    assert len(stored) == len(regenerated)
    differences = 0
    for previous, current in zip(stored, regenerated, strict=True):
        assert previous["tokens"] == current["tokens"]
        differences += sum(
            labels.convert_bio_to_entity(a) != labels.convert_bio_to_entity(b)
            for a, b in zip(previous["ner_tags"], current["ner_tags"], strict=True)
        )
    summary = dict(
        asr_train_rows=len(asr["train"]),
        asr_test_rows=len(asr["test"]),
        asr_duplicate_train_filenames=len(asr["train"]) - len(files["train"]),
        asr_duplicate_test_filenames=len(asr["test"]) - len(files["test"]),
        asr_shared_filenames=len(files["train"] & files["test"]),
        asr_shared_sessions=len(sessions["train"] & sessions["test"]),
        asr_test_sessions=len(sessions["test"]),
        asr_test_clips_in_shared_sessions=sum(
            name.rsplit("_", 1)[0] in sessions["train"] for name in files["test"]
        ),
        nlp_train_sessions=len(nlp["train"]),
        nlp_test_sessions=len(nlp["test"]),
        nlp_shared_sessions=len(nlp_sessions["train"] & nlp_sessions["test"]),
        nlp_matching_test_sequences=len(stored),
        nlp_total_test_words=sum(len(row["tokens"]) for row in stored),
        nlp_changed_collapsed_word_labels=differences,
        note="ASR session IDs inferred by removing the final clip-number suffix. "
        "NER stored labels are used for DistilBERT; regenerated labels are used by the old TensorFlow BERT evaluator.",
        hashes={
            str(path.relative_to(root)): digest(path)
            for path in [
                root / "speech/train.jsonl",
                root / "speech/test.jsonl",
                root / "entities/raw/train.json",
                root / "entities/raw/test.json",
                stored_path,
            ]
        },
    )
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "data_splits.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
