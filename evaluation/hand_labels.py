"""Score the taggers, and the labelling rules, against labels written by hand.

The stored labels come from word lists, so the published F1 measures agreement
with those lists. This tool replaces them with a human reference for the 89
evaluation turns (about 1,000 words, an hour or two of work).

    python -m evaluation.hand_labels export
    # fill in the 'label' column of outputs/evaluation/hand_labels.tsv
    # using O, CALLSIGN, COMMAND, VALUE or WAYPOINT, or ? where unsure
    # (rules for the awkward cases: docs/hand_labelling_guide.md)
    python -m evaluation.evaluate_models distilbert
    python -m evaluation.hand_labels score

The export does not show the rule labels or the model's predictions, so they
cannot influence the person labelling. The score compares the word-list rules,
the tagger and the app's actual output (the tagger after its function-word
clean-up) with the labels: per-label precision and recall, exact whole-entity
matches, and a few examples of each kind of mistake.
"""

import argparse
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LABELS = ["O", "CALLSIGN", "COMMAND", "VALUE", "WAYPOINT"]
UNSURE = "?"  # the labeller could not decide; left out of every score, and counted


def collapse(tag):
    return tag if tag == "O" else tag.split("-", 1)[1]


def read_jsonl(path):
    return [json.loads(s) for s in path.read_text(encoding="utf-8").splitlines() if s.strip()]


def export(test_file, sheet):
    sheet.parent.mkdir(parents=True, exist_ok=True)
    if sheet.exists():
        raise SystemExit(f"{sheet} already exists; move it first so no work is overwritten.")
    with sheet.open("w", newline="", encoding="utf-8") as out:
        writer = csv.writer(out, delimiter="\t")
        writer.writerow(["turn", "position", "word", "label", "whole_turn"])
        for turn, row in enumerate(read_jsonl(test_file)):
            for position, word in enumerate(row["tokens"]):
                writer.writerow([turn, position, word, "", " ".join(row["tokens"]) if position == 0 else ""])
    print(sheet)


def read_sheet(sheet):
    human = {}
    with sheet.open(newline="", encoding="utf-8") as source:
        for line in csv.DictReader(source, delimiter="\t"):
            label = line["label"].strip().upper() or None
            if label is not None and label not in LABELS + [UNSURE]:
                raise SystemExit(f"Turn {line['turn']} word {line['word']!r}: unknown label {label!r}")
            human[int(line["turn"]), int(line["position"])] = label
    return human


def compare(reference, candidate):
    """Accuracy and per-label precision, recall and F1 over aligned label lists."""
    report = {}
    for label in LABELS:
        tp = sum(r == c == label for r, c in zip(reference, candidate))
        fp = sum(r != label and c == label for r, c in zip(reference, candidate))
        fn = sum(r == label and c != label for r, c in zip(reference, candidate))
        report[label] = {
            "precision": tp / (tp + fp) if tp + fp else 0.0,
            "recall": tp / (tp + fn) if tp + fn else 0.0,
            "f1": 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else 0.0,
            "support": tp + fn,
        }
    correct = sum(r == c for r, c in zip(reference, candidate))
    return {"accuracy": correct / len(reference), "words": len(reference), "labels": report}


def spans(labels):
    """Runs of the same entity label in one sentence: (label, first word, last word)."""
    out, start = [], None
    for i, label in enumerate(labels + ["O"]):
        if start is not None and label != labels[start]:
            if labels[start] not in ("O", UNSURE):
                out.append((labels[start], start, i - 1))
            start = None
        if start is None and label != "O":
            start = i
    return out


def span_scores(reference_by_turn, candidate_by_turn):
    """Exact entity-span matches: same label, same first and last word.

    A candidate span touching a word the labeller marked unsure is left out.
    """
    report = {}
    for label in LABELS[1:]:
        tp = fp = fn = 0
        for turn, ref in reference_by_turn.items():
            cand = candidate_by_turn[turn]
            unsure = {i for i, x in enumerate(ref) if x == UNSURE}
            r = {s for s in spans(ref) if s[0] == label}
            c = {s for s in spans(cand) if s[0] == label and not unsure & set(range(s[1], s[2] + 1))}
            tp += len(r & c)
            fp += len(c - r)
            fn += len(r - c)
        report[label] = {"exact_span_precision": tp / (tp + fp) if tp + fp else 0.0,
                         "exact_span_recall": tp / (tp + fn) if tp + fn else 0.0,
                         "reference_spans": tp + fn}
    return report


def mistakes(rows, keys, reference, candidate, per_kind=3):
    """A few examples of each kind of disagreement, with the sentence."""
    found = {}
    for (turn, position), r, c in zip(keys, reference, candidate):
        if r != c and len(found.setdefault(f"{r} -> {c}", [])) < per_kind:
            words = rows[turn]["tokens"]
            found[f"{r} -> {c}"].append(f"{words[position]} | {' '.join(words)}")
    return dict(sorted(found.items(), key=lambda item: -len(item[1])))


def score(test_file, sheet, predictions_file, output):
    human = read_sheet(sheet)
    missing = [key for key, label in human.items() if label is None]
    if missing:
        raise SystemExit(f"{len(missing)} words are still unlabelled, first at turn {missing[0][0]}.")
    rows = read_jsonl(test_file)
    keys = [(t, p) for t, row in enumerate(rows) for p in range(len(row["tokens"]))]
    if set(keys) != set(human):
        raise SystemExit("The sheet does not match the evaluation file; export it again.")
    unsure = [key for key in keys if human[key] == UNSURE]
    keys = [key for key in keys if human[key] != UNSURE]
    reference = [human[key] for key in keys]
    rules = [collapse(rows[t]["ner_tags"][p]) for t, p in keys]
    summary = {"words_left_out_as_unsure": len(unsure), "rules_vs_human": compare(reference, rules)}
    summary["rules_vs_human"]["spans"] = span_scores(
        {t: [human[(t, p)] for p in range(len(row["tokens"]))] for t, row in enumerate(rows)},
        {t: [collapse(tag) for tag in row["ner_tags"]] for t, row in enumerate(rows)})
    if predictions_file.is_file():
        predicted = {(r["index"], p): label for r in read_jsonl(predictions_file) for p, label in enumerate(r["prediction"])}
        if set(keys) <= set(predicted) and len(predicted) == len(keys) + len(unsure):
            from atco.entity_extraction import FUNCTION_WORDS

            # The app drops waypoint labels from function words; score that output too.
            app = {k: ("O" if v == "WAYPOINT" and rows[k[0]]["tokens"][k[1]].lower() in FUNCTION_WORDS else v)
                   for k, v in predicted.items()}
            reference_by_turn = {t: [human[(t, p)] for p in range(len(row["tokens"]))] for t, row in enumerate(rows)}
            for name, labels in [("distilbert", predicted), ("app_output", app)]:
                result = compare(reference, [labels[key] for key in keys])
                result["spans"] = span_scores(
                    reference_by_turn, {t: [labels[(t, p)] for p in range(len(row["tokens"]))] for t, row in enumerate(rows)})
                result["examples_of_mistakes"] = mistakes(rows, keys, reference, [labels[key] for key in keys])
                summary[f"{name}_vs_human"] = result
        else:
            print("Model predictions cover different words (was --limit used?); skipping them.")
    else:
        print(f"No {predictions_file.name}; run evaluate_models distilbert to score the model too.")
    output.write_text(json.dumps(summary, indent=2) + "\n")
    for name, result in summary.items():
        if isinstance(result, dict):
            print(f"{name}: accuracy {result['accuracy']:.2%} over {result['words']} words")
    print(f"left out as unsure: {len(unsure)} words")
    print(output)


def main():
    cli = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    cli.add_argument("action", choices=["export", "score"])
    cli.add_argument("--data-dir", type=Path, default=ROOT / "datasets")
    cli.add_argument("--output", type=Path, default=ROOT / "outputs/evaluation")
    cli.add_argument("--sheet", default="hand_labels.tsv",
                     help="Sheet name in --output; the default is the one for labelling by hand")
    cli.add_argument("--scores", default="hand_label_scores.json", help="Where to write the scores, in --output")
    args = cli.parse_args()
    test_file = args.data_dir / "entities/test.jsonl"
    sheet = args.output / args.sheet
    if args.action == "export":
        export(test_file, sheet)
    else:
        score(test_file, sheet, args.output / "distilbert_predictions.jsonl", args.output / args.scores)


if __name__ == "__main__":
    main()
