"""Draw the paper's figures from the saved results.

    python paper/build_figures.py

Every value is read from results/; nothing is estimated here. Needs matplotlib.
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUT = Path(__file__).resolve().parent / "figures"

INK, INK3, LINE = "#15222D", "#5C6A75", "#D2D9DE"
ACCENT, RUST, PALE, MID = "#285B73", "#B45F3C", "#B3C7D3", "#8AA8B9"
LABELS = {
    "zero_shot": "Whisper-small, not fine-tuned",
    "checkpoint_2025": "2025 model",
    "fixed": "Retrained with the fixes",
    "synthetic": "+ synthetic radio speech",
    "real_data": "+ 10.5 h of UWB-ATCC speech",
    "soup_fixed_real": "Soup: fixes-only + UWB-ATCC models",
    "medium_real": "Whisper-medium (LoRA) + UWB-ATCC",
    "soup_medium": "Soup of two Whisper-medium models",
    "medium_longer": "Whisper-medium, trained longer",
}

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 10, "axes.edgecolor": INK3, "axes.labelcolor": INK, "xtick.color": INK3,
    "ytick.color": INK3, "axes.spines.top": False, "axes.spines.right": False,
    "savefig.dpi": 220, "savefig.bbox": "tight",
})


def load(name):
    return json.loads((RESULTS / name).read_text())


def split():
    s = load("rescoring_2025/data_splits.json")
    train, shared = s["asr_train_rows"], s["asr_test_clips_in_shared_sessions"]
    unseen = s["asr_test_rows"] - shared
    fig, ax = plt.subplots(figsize=(7.2, 1.9))
    ax.barh(0, train, color=MID, height=0.5)
    ax.barh(0, shared, left=train, color=PALE, height=0.5)
    ax.barh(0, unseen, left=train + shared, color=ACCENT, height=0.5)
    ax.text(train / 2, 0, f"Training: {train} clips", ha="center", va="center", fontsize=9, color=INK)
    ax.annotate(f"Validation: {shared} clips from\nrecordings that also gave training clips",
                xy=(train + shared / 2, 0.25), xytext=(train - 260, 0.95), fontsize=8.5, color=INK,
                arrowprops=dict(arrowstyle="-", color=INK3, lw=0.8))
    ax.annotate(f"Test: {unseen} clips from\nrecordings never trained on",
                xy=(train + shared + unseen / 2, 0.25), xytext=(train + shared - 20, 0.95), fontsize=8.5,
                color=INK, arrowprops=dict(arrowstyle="-", color=INK3, lw=0.8))
    ax.set_xlim(0, train + shared + unseen)
    ax.set_ylim(-0.45, 1.5)
    ax.set_yticks([])
    ax.set_xlabel(f"ATCO2 clips ({train + shared + unseen} in all). The 2025 split used validation and test "
                  "together as one test set.")
    ax.spines["left"].set_visible(False)
    fig.savefig(OUT / "fig1_split.png")
    plt.close(fig)


def wer_bars():
    r = load("retraining.json")
    runs = [(n, v["headline_test"]) for n, v in r["runs"].items() if n in LABELS]
    fig, ax = plt.subplots(figsize=(7.2, 0.42 * len(runs) + 0.9))
    for i, (name, h) in enumerate(reversed(runs)):
        colour = ACCENT if name == r["chosen_on_validation"] else RUST if name == "checkpoint_2025" else \
            PALE if name == "zero_shot" else MID
        wer, (lo, hi) = 100 * h["wer"], [100 * x for x in h["interval_95"]]
        ax.barh(i, wer, color=colour, height=0.62)
        ax.plot([lo, hi], [i, i], color=INK, lw=1.2)
        ax.plot([lo, lo], [i - 0.13, i + 0.13], color=INK, lw=1.2)
        ax.plot([hi, hi], [i - 0.13, i + 0.13], color=INK, lw=1.2)
        ax.text(hi + 1, i, f"{wer:.2f}%", va="center", fontsize=8.5, color=INK)
    ax.set_yticks(range(len(runs)), [LABELS[n] for n, _ in reversed(runs)])
    ax.set_xlim(0, 75)
    ax.set_xlabel("Word error rate on the 74 test clips (%), with 95% bootstrap intervals")
    ax.grid(axis="x", color=LINE, lw=0.6)
    ax.set_axisbelow(True)
    fig.savefig(OUT / "fig2_word_error_rates.png")
    plt.close(fig)


def noise():
    sweep = load("retraining.json")["noise_sweep"]["results"]
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    series = [("zero_shot", "Whisper-small, not fine-tuned", INK3, "--"),
              ("checkpoint_2025", "2025 model", RUST, "-"),
              ("retrained", "Best 2026 model", ACCENT, "-"),
              ("retrained_bandpass", "Best 2026 model, 300 to 3,400 Hz first", ACCENT, ":")]
    for key, label, colour, style in series:
        pts = sweep[key]
        xs = range(len(pts))
        ax.plot(xs, [100 * p["wer"] for p in pts], style, color=colour, marker="o", ms=3.5, lw=1.8, label=label)
    ticks = ["none" if p["added_noise_snr_db"] is None else f"{p['added_noise_snr_db']:g} dB"
             for p in sweep["checkpoint_2025"]]
    ax.set_xticks(range(len(ticks)), ticks)
    ax.set_xlabel("Signal-to-noise ratio after adding white noise")
    ax.set_ylabel("Word error rate (%)")
    ax.set_ylim(0, 100)
    ax.grid(axis="y", color=LINE, lw=0.6)
    ax.legend(frameon=False, fontsize=8.5)
    fig.savefig(OUT / "fig4_noise.png")
    plt.close(fig)


def instructions():
    r = load("retraining.json")
    a = load("extraction_agreement.json")["models"]
    models = [("zero_shot", PALE), ("checkpoint_2025", RUST), (r["chosen_on_validation"], ACCENT)]
    fields = [("callsign", "Callsign"), ("command", "Command"), ("value", "Numbers")]
    fig, ax = plt.subplots(figsize=(7.2, 2.9))
    width = 0.26
    for j, (name, colour) in enumerate(models):
        m = a[name]
        shares = [100 * m[f]["clips_where_it_matches"] / m[f]["clips_with_field"] for f, _ in fields]
        shares.append(100 * m["callsign_command_value_identical"] / m["clips"])
        shares.append(100 * m["all_fields_identical"] / m["clips"])
        xs = [i + (j - 1) * width for i in range(len(shares))]
        ax.bar(xs, shares, width=width, color=colour, label=LABELS[name])
    ax.set_xticks(range(5), [label for _, label in fields] + ["All three", "All, with\nwaypoints"])
    ax.set_ylabel("Clips where the field matches (%)")
    ax.set_ylim(0, 100)
    ax.grid(axis="y", color=LINE, lw=0.6)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=8.5, loc="upper right")
    fig.savefig(OUT / "fig5_instructions.png")
    plt.close(fig)


def confusion():
    e = load("rescoring_2025/entity_extraction.json")
    cm = e["confusion_matrix"]
    names = ["Other", "Callsign", "Command", "Value", "Waypoint"]
    fig, ax = plt.subplots(figsize=(4.6, 3.8))
    ax.imshow(cm, cmap="Blues")
    for i, row in enumerate(cm):
        for j, v in enumerate(row):
            ax.text(j, i, v, ha="center", va="center", fontsize=9, color="white" if v > 120 else INK)
    ax.set_xticks(range(5), names, rotation=30, ha="right")
    ax.set_yticks(range(5), names)
    ax.set_xlabel("DistilBERT prediction")
    ax.set_ylabel("Label from the word-list rules")
    for side in ax.spines.values():
        side.set_visible(False)
    fig.savefig(OUT / "fig6_tagger_confusion.png")
    plt.close(fig)


def cross_validation():
    path = RESULTS / "cross_validation.json"
    if not path.exists():
        return
    cv = json.loads(path.read_text())
    fig, ax = plt.subplots(figsize=(7.2, 2.6))
    folds = cv["folds"]
    ax.bar([f"Fold {f['fold'] + 1}" for f in folds], [100 * f["wer"] for f in folds], color=MID, width=0.6)
    for level, colour, label in [(cv["retrained"]["wer"], ACCENT, "Retrained, all 874 clips"),
                                 (cv["zero_shot"]["wer"], INK3, "Not fine-tuned, all 874 clips")]:
        ax.axhline(100 * level, color=colour, lw=1.4, ls="--", label=f"{label}: {100 * level:.2f}%")
    ax.set_ylabel("Word error rate (%)")
    ax.set_ylim(0, 70)
    ax.legend(frameon=False, fontsize=8.5, loc="upper right")
    ax.grid(axis="y", color=LINE, lw=0.6)
    ax.set_axisbelow(True)
    fig.savefig(OUT / "fig3_cross_validation.png")
    plt.close(fig)


def main():
    OUT.mkdir(exist_ok=True)
    for old in OUT.glob("*.png"):
        old.unlink()
    for draw in [split, wer_bars, noise, instructions, confusion, cross_validation]:
        draw()
    print("\n".join(sorted(p.name for p in OUT.glob("*.png"))))


if __name__ == "__main__":
    main()
