"""Collect the leave-one-airport-out test into results/leave_one_airport_out.json.

    python -m evaluation.session_split --by-airport   # writes datasets/speech_airports/
    # train and score one model per airport (outputs/airports_and_long.sh), then:
    python -m evaluation.summarise_airports

Each airport's clips are transcribed by a model that never heard that airport.
The same 874 clips were also transcribed in the five-fold cross-validation by
recording, where the model had heard other recordings from the same airport, and
by unmodified Whisper-small. Comparing the three on the same clips shows what a
new airport costs on top of a new recording. Decoding is fixed in advance, as
in the cross-validation: beam search with the repetition guards.
"""

import json
from pathlib import Path

import numpy as np

from .compare_models import clip_counts, interval
from .session_split import airport_id

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "evaluation"
MODE = "beam"
NAMES = {"LKPR": "Prague", "LKTB": "Brno", "LSGS": "Sion", "LSZB": "Bern", "LSZH": "Zurich",
         "LZIB": "Bratislava", "YSSY": "Sydney"}


def rows(path):
    return [json.loads(s) for s in Path(path).read_text().splitlines() if s.strip()]


def main():
    airport_model, by_recording, zero = {}, {}, {}
    for code in NAMES:
        for r in rows(OUT / f"airport_{code}_test" / "asr_predictions.jsonl"):
            airport_model[r["file"]] = r
    for k in range(5):
        for r in rows(OUT / f"cv_fold{k}_test" / "asr_predictions.jsonl"):
            by_recording[r["file"]] = r
    for name in ["zero_shot", "zero_shot_train"]:
        for r in rows(OUT / name / "asr_predictions.jsonl"):
            zero[r["file"]] = r
    files = sorted(airport_model)
    if sorted(by_recording) != files or sorted(zero) != files:
        raise SystemExit("the three systems were not scored on the same clips")

    systems = {"new_airport": airport_model, "new_recording": by_recording, "zero_shot": zero}
    counts = {name: clip_counts([source[f] for f in files], MODE, spelled=True) for name, source in systems.items()}
    rng = np.random.default_rng(0)
    picks = rng.integers(0, len(files), size=(10000, len(files)))
    pooled = {}
    for name, (e, w) in counts.items():
        boot = e[picks].sum(axis=1) / w[picks].sum(axis=1)
        pooled[name] = dict(wer=round(float(e.sum() / w.sum()), 4), interval_95=[round(x, 4) for x in interval(boot)])
    (e_air, w), (e_rec, _) = counts["new_airport"], counts["new_recording"]
    diff = (e_air[picks].sum(axis=1) - e_rec[picks].sum(axis=1)) / w[picks].sum(axis=1)
    pooled["new_airport_minus_new_recording_points"] = dict(
        change=round(float(100 * (e_air.sum() - e_rec.sum()) / w.sum()), 2),
        interval_95=[round(100 * x, 2) for x in interval(diff)])

    airports = []
    codes = np.array([airport_id(f) for f in files])
    for code, place in NAMES.items():
        mask = codes == code
        entry = dict(airport=code, place=place, clips=int(mask.sum()), reference_words=int(w[mask].sum()))
        for name, (e, words) in counts.items():
            entry[name] = round(float(e[mask].sum() / words[mask].sum()), 4)
        log = json.loads((ROOT / "outputs" / "training" / f"airport_{code}" / "training_log.json").read_text())
        entry["training"] = dict(epochs_run=len(log["epochs"]), best_epoch=log["best_epoch"],
                                 best_validation_wer=round(log["best_validation_wer"], 4))
        airports.append(entry)

    report = dict(
        method="Recipe A (fixes only, Whisper-small, 8 planned epochs, patience 3) trained once per ATCO2 airport "
               "without that airport, a tenth of the other airports' recordings held out to choose the checkpoint, "
               "then scored on every clip from the airport it never heard. Compared on the same 874 clips with the "
               "five-fold cross-validation by recording and with unmodified Whisper-small.",
        decoding="beam search, 5 beams, repetition_penalty 1.2, no repeated 3-grams (fixed in advance)",
        normalisation="uppercase, punctuation removed, digits spelled out",
        clips=len(files),
        pooled=pooled,
        airports=airports,
    )
    path = ROOT / "results" / "leave_one_airport_out.json"
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(pooled, indent=2))
    for a in airports:
        print(f"{a['place']:11} {a['clips']:4} clips  new airport {a['new_airport']:.4f}  "
              f"new recording {a['new_recording']:.4f}  unmodified {a['zero_shot']:.4f}")
    print(path)


if __name__ == "__main__":
    main()
