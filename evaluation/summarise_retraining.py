"""Collect the 2026 retraining runs into results/retraining.json.

    python -m evaluation.summarise_retraining

Reads the training logs in outputs/training/, the scores written by
evaluation.evaluate_models for each run on the validation and unseen-test clips,
and the noise sweeps. The model and its decoding setting are chosen on the
validation clips only; the unseen test clips are reported for every run, but
play no part in the choice.
"""

import json
from pathlib import Path

import numpy as np

from .compare_models import clip_counts, interval

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
RUNS = {
    "fixed": "Start-token fix, validation-based checkpoint choice, SpecAugment; real clips only",
    "synthetic": "As fixed, plus 700 of my synthetic radio clips mixed in each epoch",
    "augmented": "As fixed, plus 700 noise and speed copies of the training clips each epoch",
    "continued": "The 2025 checkpoint trained further with the start-token fix, learning rate 5e-6",
    "real_data": "As fixed, plus 1,400 clips per epoch of real ATC speech from UWB-ATCC (10.5 hours, Czech airspace)",
    "soup_fixed_real": "Weight average of the fixed and real_data models (a model soup)",
    "medium_real": "Whisper-medium with LoRA adapters (rank 32), real clips plus 1,000 UWB-ATCC clips per epoch",
    "soup_medium": "Weight average of medium_real and the ATCO2-only Whisper-medium",
    "medium_longer": "medium_real trained for up to three more epochs with fresh rank-32 adapters, learning rate 2e-4",
}
# Declared in advance as ablations: reported under "ablations", never eligible to be chosen.
ABLATIONS = {
    "medium_atco2_only": "Whisper-medium with LoRA adapters (rank 32) on the ATCO2 clips only, to separate model "
                         "size from the extra UWB-ATCC speech",
}
DESCRIPTIONS = {
    "zero_shot": "Unmodified openai/whisper-small, no fine-tuning",
    "checkpoint_2025": "The 2025 checkpoint used by the app, unchanged",
    **RUNS,
    **ABLATIONS,
}
MODES = ["greedy", "beam", "beamplain"]


def predictions(folder):
    path = OUT / "evaluation" / folder / "asr_predictions.jsonl"
    rows = [json.loads(s) for s in path.read_text().splitlines() if s.strip()]
    return {r["file"]: r for r in sorted(rows, key=lambda r: r["file"])}


def scores(rows, mode):
    edits, words = clip_counts(list(rows.values()), mode, spelled=True)
    return edits, words


def main():
    rng = np.random.default_rng(0)
    systems = {"zero_shot": ("zero_shot_validation", "zero_shot_test"),
               "checkpoint_2025": ("2025_validation", "2025_test")}
    for name in [*RUNS, *ABLATIONS]:
        if (OUT / "evaluation" / f"{name}_test" / "asr_predictions.jsonl").exists():
            systems[name] = (f"{name}_validation", f"{name}_test")

    report = dict(
        split=dict(
            train="the 699 clips the 2025 model was trained on",
            validation="the 101 old test clips from recordings that also gave training clips; "
            "used to choose checkpoints and decoding",
            test="the 74 old test clips from recordings that no model was trained on; never used for a choice",
        ),
        normalisation="uppercase, punctuation removed, digits spelled out",
        interval_method="95% interval from 10,000 resamples of whole clips; differences are paired",
        runs={},
    )
    test_rows = {}
    for name, (val_folder, test_folder) in systems.items():
        val, test = predictions(val_folder), predictions(test_folder)
        test_rows[name] = test
        entry = dict(description=DESCRIPTIONS[name])
        log_path = OUT / "training" / name / "training_log.json"
        if name in DESCRIPTIONS and log_path.exists():
            log = json.loads(log_path.read_text())
            entry["training"] = dict(
                epochs_run=len(log["epochs"]),
                best_epoch=log["best_epoch"],
                extra_clips_per_epoch=int(log["settings"].get("extra_per_epoch", 0)),
                learning_rate=float(log["settings"]["learning_rate"]),
                effective_batch=int(log["settings"]["batch_size"]) * int(log["settings"]["accumulate"]),
                validation_wer_by_epoch=[round(e["validation_wer"], 4) for e in log["epochs"]],
            )
        entry["validation"], entry["test"] = {}, {}
        modes = [m for m in MODES if m in next(iter(val.values()))]
        for mode in modes:
            for part, rows in (("validation", val), ("test", test)):
                edits, words = scores(rows, mode)
                entry[part][mode] = dict(
                    wer=round(float(edits.sum() / words.sum()), 4),
                    word_errors=int(edits.sum()),
                    reference_words=int(words.sum()),
                )
        entry["decoding_chosen_on_validation"] = min(modes, key=lambda m: entry["validation"][m]["wer"])
        report["runs"][name] = entry

    new_runs = [n for n in report["runs"] if n in RUNS]
    best = min(new_runs, key=lambda n: report["runs"][n]["validation"][report["runs"][n]["decoding_chosen_on_validation"]]["wer"])
    report["chosen_on_validation"] = best

    files = sorted(test_rows["checkpoint_2025"])
    picks = rng.integers(0, len(files), size=(10000, len(files)))
    old_mode = report["runs"]["checkpoint_2025"]["decoding_chosen_on_validation"]
    e_old, w = scores(test_rows["checkpoint_2025"], old_mode)
    for name in report["runs"]:
        mode = report["runs"][name]["decoding_chosen_on_validation"]
        edits, words = scores(test_rows[name], mode)
        boot = edits[picks].sum(axis=1) / words[picks].sum(axis=1)
        result = dict(decoding=mode, wer=round(float(edits.sum() / words.sum()), 4),
                      interval_95=[round(x, 4) for x in interval(boot)])
        if name not in ("checkpoint_2025", "zero_shot"):
            diff = (edits[picks].sum(axis=1) - e_old[picks].sum(axis=1)) / w[picks].sum(axis=1)
            result["change_from_2025_points"] = round(float(100 * (edits.sum() - e_old.sum()) / w.sum()), 2)
            result["change_interval_95_points"] = [round(100 * x, 2) for x in interval(diff)]
            result["share_of_resamples_better"] = round(float(np.mean(diff < 0)), 4)
        report["runs"][name]["headline_test"] = result

    ablations = {name: report["runs"].pop(name) for name in list(report["runs"]) if name in ABLATIONS}
    if ablations:
        report["ablations"] = ablations

    sweeps = {}
    for label in ["zero_shot", "checkpoint_2025", "retrained", "zero_shot_bandpass", "retrained_bandpass"]:
        path = OUT / "evaluation" / "noise" / f"noise_sweep_{label}.json"
        if path.exists():
            data = json.loads(path.read_text())
            sweeps[label] = [dict(added_noise_snr_db=r["added_noise_snr_db"], wer=round(r["wer"], 4))
                             for r in data["results"]]
    if sweeps:
        report["noise_sweep"] = dict(
            clips="the 74 unseen test clips",
            noise="white Gaussian noise added at the stated signal-to-noise ratio",
            decoding=json.loads((OUT / "evaluation" / "noise" / "noise_sweep_retrained.json").read_text())["decoding"],
            retrained_model=json.loads((OUT / "evaluation" / "noise" / "noise_sweep_retrained.json").read_text())["model"],
            results=sweeps,
        )

    path = ROOT / "results" / "retraining.json"
    path.write_text(json.dumps(report, indent=2) + "\n")
    for name, entry in report["runs"].items():
        h = entry["headline_test"]
        extra = f"  change {h['change_from_2025_points']:+.2f} [{h['change_interval_95_points'][0]:+.2f}, {h['change_interval_95_points'][1]:+.2f}]" if "change_from_2025_points" in h else ""
        print(f"{name:16} val {entry['validation'][h['decoding']]['wer']:.4f}  test {h['wer']:.4f} ({h['decoding']}){extra}")
    print(f"chosen on validation: {best}")
    print(path)


if __name__ == "__main__":
    main()
