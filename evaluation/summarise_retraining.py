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
    "soup_medium_long": "Weight average of medium_long (medium_real's settings with 12 planned epochs) and the "
                        "ATCO2-only Whisper-medium",
}
# Trained and scored on validation only: the rule written before the run compared it
# with its soup, and only the better of the two could be scored on test.
VALIDATION_ONLY = {
    "medium_long": "medium_real's settings with 12 planned epochs instead of 6 (patience 3); run from the internal "
                   "disk after two attempts were lost to the external drive being unplugged",
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


def training_summary(log_path):
    """Epoch budget, when and why the run stopped, and validation WER after each epoch."""
    log = json.loads(log_path.read_text())
    settings = log["settings"]
    budget, run = int(settings["epochs"]), len(log["epochs"])
    return dict(
        epoch_budget=budget,
        patience=int(settings["patience"]),
        epochs_run=run,
        stopped="early, no improvement" if run < budget else "at the epoch budget",
        best_epoch=log["best_epoch"],
        validation_wer_before_training=round(log["validation_wer_before_training"], 4),
        extra_clips_per_epoch=int(settings.get("extra_per_epoch", 0)),
        learning_rate=float(settings["learning_rate"]),
        effective_batch=int(settings["batch_size"]) * int(settings["accumulate"]),
        validation_wer_by_epoch=[round(e["validation_wer"], 4) for e in log["epochs"]],
    )


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
            entry["training"] = training_summary(log_path)
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

    # Runs that were never scored on the test clips: stopped because they never
    # beat their starting point, beaten on validation, or not run at all.
    not_scored = {}
    for name, description in VALIDATION_ONLY.items():
        log_path = OUT / "training" / name / "training_log.json"
        folder = OUT / "evaluation" / f"{name}_validation" / "asr_predictions.jsonl"
        if log_path.exists() and folder.exists():
            val = predictions(f"{name}_validation")
            entry = dict(description=description, training=training_summary(log_path), validation={})
            for mode in MODES:
                edits, words = scores(val, mode)
                entry["validation"][mode] = dict(wer=round(float(edits.sum() / words.sum()), 4),
                                                 word_errors=int(edits.sum()), reference_words=int(words.sum()))
            entry["outcome"] = "worse on validation than its soup with the ATCO2-only model, so not scored on test"
            not_scored[name] = entry
    for name in ["continued", "medium_longer"]:
        log_path = OUT / "training" / name / "training_log.json"
        if log_path.exists():
            not_scored[name] = dict(description=DESCRIPTIONS[name], training=training_summary(log_path),
                                    outcome="never beat its starting point on validation; nothing saved")
    soup3 = OUT / "evaluation" / "soup_all3_validation" / "asr_predictions.jsonl"
    if soup3.exists():
        val = predictions("soup_all3_validation")
        entry = dict(description="Weight average of fixed, synthetic and real_data", validation={})
        for mode in MODES:
            edits, words = scores(val, mode)
            entry["validation"][mode] = dict(wer=round(float(edits.sum() / words.sum()), 4),
                                             word_errors=int(edits.sum()), reference_words=int(words.sum()))
        entry["decoding_chosen_on_validation"] = min(MODES, key=lambda m: entry["validation"][m]["wer"])
        entry["outcome"] = "worse than soup_fixed_real on validation; never scored on test"
        not_scored["soup_all3"] = entry
    not_scored["augmented"] = dict(description=DESCRIPTIONS["augmented"], outcome="not run: dropped for time")
    report["not_scored_on_test"] = not_scored

    def sweep(folder, label):
        path = OUT / "evaluation" / folder / f"noise_sweep_{label}.json"
        if not path.exists():
            return None
        data = json.loads(path.read_text())
        return dict(model=data["model"], band_pass_300_3400_hz=data["band_pass_300_3400_hz"],
                    decoding=data["decoding"],
                    results=[dict(added_noise_snr_db=r["added_noise_snr_db"], wer=round(r["wer"], 4))
                             for r in data["results"]])

    sweeps = {}
    for label in ["zero_shot", "checkpoint_2025", "retrained", "zero_shot_bandpass", "retrained_bandpass"]:
        found = sweep("noise", label)
        if found:
            sweeps[label] = found["results"]
    if sweeps:
        report["noise_sweep"] = dict(
            clips="the 74 unseen test clips",
            noise="white Gaussian noise added at the stated signal-to-noise ratio, measured over the whole clip "
                  "(pauses included) after the optional band-pass filter",
            decoding=json.loads((OUT / "evaluation" / "noise" / "noise_sweep_retrained.json").read_text())["decoding"],
            retrained_model=json.loads((OUT / "evaluation" / "noise" / "noise_sweep_retrained.json").read_text())["model"],
            results=sweeps,
        )
        # The same sweep for two more models, and the first sweep, which had no
        # repetition guards and was redone because greedy decoding looped.
        others = {}
        for name, folder in [("soup_fixed_real", "noise_soup"), ("medium_real", "noise_medium_real"),
                             ("soup_medium", "noise_soup_medium")]:
            for label, key in [("retrained", name), ("retrained_bandpass", f"{name}_bandpass")]:
                found = sweep(folder, label)
                if found:
                    others[key] = found["results"]
        report["noise_sweep"]["other_models"] = others
        first = {label: sweep("noise_greedy", label) for label in ["zero_shot", "checkpoint_2025", "retrained"]}
        report["noise_sweep"]["first_sweep_without_guards"] = {k: v for k, v in first.items() if v}

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
