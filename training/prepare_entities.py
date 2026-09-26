"""Apply the saved labelling rules to an existing session split."""
import argparse
import json
from pathlib import Path
from .entity_labels import tag_entities

ROOT = Path(__file__).resolve().parents[1]


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--input", type=Path, default=ROOT / "datasets/entities/raw")
    cli.add_argument("--output", type=Path, default=ROOT / "outputs/training/entity_labels")
    args = cli.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    for split in ["train", "test"]:
        sessions = json.loads((args.input / f"{split}.json").read_text())
        count = 0
        with (args.output / f"{split}.jsonl").open("w") as output:
            for session in sessions:
                for turn in session.get("dialogue", []):
                    text = turn.get("text", "")
                    if not text:
                        continue
                    tokens, tags = tag_entities(text, session.get("context", {}))
                    if any(tag != "O" for tag in tags):
                        output.write(json.dumps({"tokens": tokens, "ner_tags": tags}) + "\n")
                        count += 1
        print(f"{split}: {count} labelled turns")
    (args.output / "label_map.json").write_bytes((ROOT / "training/label_map.json").read_bytes())


if __name__ == "__main__":
    main()
