"""Collect the five-fold cross-validation into results/cross_validation.json.

    python -m evaluation.session_split --folds 5       # writes datasets/speech_folds/
    # train and score one model per fold (see evaluation/README.md), then:
    python -m evaluation.summarise_cv

Every one of the 874 clips is transcribed by a model that never trained on its
recording, so the pooled word error rate describes new recordings far better
than 74 clips can. The unmodified Whisper-small is scored on the same 874 clips
for comparison. Decoding is fixed in advance: beam search with the repetition
guards, the setting chosen on validation for the fixes-only model.
"""

import json
from pathlib import Path

import numpy as np

from .compare_models import clip_counts, interval

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "evaluation"
MODE = "beam"


def rows(path):
    return [json.loads(s) for s in Path(path).read_text().splitlines() if s.strip()]


def main():
    folds, retrained = [], []
    for k in range(5):
        fold = rows(OUT / f"cv_fold{k}_test" / "asr_predictions.jsonl")
        log = json.loads((ROOT / "outputs" / "training" / f"cv_fold{k}" / "training_log.json").read_text())
        edits, words = clip_counts(fold, MODE, spelled=True)
        folds.append(dict(fold=k, clips=len(fold), wer=round(float(edits.sum() / words.sum()), 4),
                          best_epoch=log["best_epoch"], best_validation_wer=round(log["best_validation_wer"], 4)))
        retrained += fold
    by_file = {r["file"]: r for r in retrained}
    zero = {r["file"]: r for name in ["zero_shot", "zero_shot_train"]
            for r in rows(OUT / name / "asr_predictions.jsonl")}
    files = sorted(by_file)
    if sorted(zero) != files:
        raise SystemExit("the unmodified model was not scored on the same clips")

    rng = np.random.default_rng(0)
    picks = rng.integers(0, len(files), size=(10000, len(files)))
    report = dict(
        method="Five folds by recording over all 874 ATCO2 clips; each fold's model trained on the other four "
        "folds minus a validation tenth, which chose its checkpoint. Word error rate pooled over all clips.",
        decoding="beam search, 5 beams, repetition_penalty 1.2, no repeated 3-grams (fixed in advance)",
        normalisation="uppercase, punctuation removed, digits spelled out",
        clips=len(files),
        folds=folds,
    )
    edits = {}
    for name, source in [("retrained", by_file), ("zero_shot", zero)]:
        e, w = clip_counts([source[f] for f in files], MODE, spelled=True)
        edits[name] = (e, w)
        boot = e[picks].sum(axis=1) / w[picks].sum(axis=1)
        report[name] = dict(wer=round(float(e.sum() / w.sum()), 4), word_errors=int(e.sum()),
                            reference_words=int(w.sum()), interval_95=[round(x, 4) for x in interval(boot)])
    path = ROOT / "results" / "cross_validation.json"
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ["retrained", "zero_shot"]}, indent=2))
    for f in folds:
        print(f)
    print(path)


if __name__ == "__main__":
    main()
