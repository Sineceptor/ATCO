"""Compare speech models on the same clips, with bootstrap intervals.

Each input is an asr_predictions.jsonl written by evaluation.evaluate_models.
Every model must have transcribed exactly the same clips. The script reports
word error rate for each decoding mode, a 95% interval from resampling clips,
and a paired interval for the difference from the first (reference) model.

    python -m evaluation.compare_models \\
        --model "2025 checkpoint=outputs/evaluation/unseen_2025/asr_predictions.jsonl" \\
        --model "Retrained=outputs/evaluation/unseen_fixed/asr_predictions.jsonl" \\
        --output outputs/evaluation/comparison.json

With 74 clips a difference of a point or two can easily be chance, which is
why the paired interval matters more than the headline numbers.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from .evaluate_models import edit_distance, spell_digits

ROOT = Path(__file__).resolve().parents[1]


def normalise(text, spelled):
    import jiwer

    plain = jiwer.Compose(
        [jiwer.ToUpperCase(), jiwer.RemovePunctuation(), jiwer.RemoveMultipleSpaces(), jiwer.Strip()]
    )
    return plain(spell_digits(text)) if spelled else plain(text)


def clip_counts(records, mode, spelled):
    """Word edits and reference words for each clip, in file order."""
    edits, words = [], []
    for r in records:
        reference = normalise(r["reference"], spelled).split()
        if not reference:
            continue
        edits.append(edit_distance(reference, normalise(r[mode], spelled).split()))
        words.append(len(reference))
    return np.array(edits), np.array(words)


def interval(values, low=2.5, high=97.5):
    return [float(np.percentile(values, low)), float(np.percentile(values, high))]


def main():
    cli = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    cli.add_argument("--model", action="append", required=True, help='"Name=path/to/asr_predictions.jsonl"')
    cli.add_argument("--modes", nargs="+", default=["greedy", "beam", "beamplain"])
    cli.add_argument("--spelled", action="store_true", help="Spell out digits before scoring")
    cli.add_argument("--resamples", type=int, default=10000)
    cli.add_argument("--seed", type=int, default=0)
    cli.add_argument("--output", type=Path)
    args = cli.parse_args()

    systems = {}
    for spec in args.model:
        name, path = spec.split("=", 1)
        rows = [json.loads(s) for s in Path(path).read_text().splitlines() if s.strip()]
        systems[name] = {r["file"]: r for r in rows}
    files = sorted(next(iter(systems.values())))
    for name, rows in systems.items():
        if sorted(rows) != files:
            raise SystemExit(f"{name} did not transcribe the same clips as the first model")
        for f in files:
            if rows[f]["reference"] != systems[next(iter(systems))][f]["reference"]:
                raise SystemExit(f"{name} has a different reference for {f}")

    rng = np.random.default_rng(args.seed)
    picks = rng.integers(0, len(files), size=(args.resamples, len(files)))
    reference_name = next(iter(systems))
    summary = dict(
        clips=len(files),
        resamples=args.resamples,
        normalisation="uppercase, punctuation removed" + (", digits spelled out" if args.spelled else ""),
        method="95% intervals from resampling whole clips with replacement; "
        "differences are paired (the same resampled clips for both models).",
        models={},
    )
    counts = {}
    for name, rows in systems.items():
        records = [rows[f] for f in files]
        summary["models"][name] = {}
        for mode in args.modes:
            if mode not in records[0]:
                continue
            edits, words = clip_counts(records, mode, args.spelled)
            counts[name, mode] = (edits, words)
            boot = edits[picks].sum(axis=1) / words[picks].sum(axis=1)
            summary["models"][name][mode] = dict(
                wer=float(edits.sum() / words.sum()),
                word_errors=int(edits.sum()),
                reference_words=int(words.sum()),
                interval_95=interval(boot),
            )
    for name in systems:
        if name == reference_name:
            continue
        for mode in args.modes:
            if (name, mode) not in counts or (reference_name, mode) not in counts:
                continue
            e1, w = counts[reference_name, mode]
            e2, _ = counts[name, mode]
            diff = (e2[picks].sum(axis=1) - e1[picks].sum(axis=1)) / w[picks].sum(axis=1)
            summary["models"][name][mode]["change_from_" + reference_name] = dict(
                points=float(100 * (e2.sum() - e1.sum()) / w.sum()),
                interval_95_points=[100 * x for x in interval(diff)],
                share_of_resamples_better=float(np.mean(diff < 0)),
            )

    text = json.dumps(summary, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
    for name, modes in summary["models"].items():
        for mode, m in modes.items():
            change = next((v for k, v in m.items() if k.startswith("change_from_")), None)
            extra = (f"  change {change['points']:+.2f} pts "
                     f"[{change['interval_95_points'][0]:+.2f}, {change['interval_95_points'][1]:+.2f}]") if change else ""
            print(f"{name:28} {mode:10} WER {100 * m['wer']:6.2f}%  "
                  f"[{100 * m['interval_95'][0]:.2f}, {100 * m['interval_95'][1]:.2f}]{extra}")


if __name__ == "__main__":
    main()
