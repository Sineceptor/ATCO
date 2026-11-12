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

INK, INK3, LINE = "#111111", "#555555", "#dddddd"
ACCENT, BASELINE, PALE, MID = "#111111", "#777777", "#dddddd", "#b5b5b5"
LABELS = {
    "zero_shot": "Whisper-small, not fine-tuned",
    "zero_shot_medium": "Whisper-medium, not fine-tuned",
    "checkpoint_2025": "2025 model",
    "fixed": "Retrained with the fixes",
    "synthetic": "+ synthetic radio speech",
    "real_data": "+ 10.5 h of UWB-ATCC speech",
    "soup_fixed_real": "Soup: fixes-only + UWB-ATCC models",
    "medium_real": "Whisper-medium (LoRA) + UWB-ATCC",
    "soup_medium": "Soup of two Whisper-medium models",
    "soup_medium_long": "Same soup, 12 epochs of training",
    "medium_longer": "Whisper-medium, trained longer",
}

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "STIXGeneral", "DejaVu Serif"],
    "font.size": 10, "axes.edgecolor": INK3, "axes.labelcolor": INK, "xtick.color": INK3,
    "ytick.color": INK3, "axes.spines.top": False, "axes.spines.right": False,
    "savefig.dpi": 300, "savefig.bbox": "tight", "pdf.fonttype": 42,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
})


def load(name):
    return json.loads((RESULTS / name).read_text())


def save(fig, stem):
    """Vector figures for typesetting; PNG copies for the Markdown preview."""
    for extension in ("pdf", "png"):
        fig.savefig(OUT / f"{stem}.{extension}")
    plt.close(fig)


def split():
    s = load("rescoring_2025/data_splits.json")
    train, shared = s["asr_train_rows"], s["asr_test_clips_in_shared_sessions"]
    unseen = s["asr_test_rows"] - shared
    fig, ax = plt.subplots(figsize=(6.3, 1.55), layout="constrained")
    ax.barh(0, train, color=MID, height=0.5)
    ax.barh(0, shared, left=train, color=PALE, height=0.5)
    ax.barh(0, unseen, left=train + shared, color=ACCENT, height=0.5)
    ax.text(train / 2, 0, f"Training: {train} clips", ha="center", va="center", fontsize=9, color=INK)
    ax.annotate(f"Validation: {shared}\nShared recordings",
                xy=(train + shared / 2, 0.25), xytext=(train - 200, 0.95), fontsize=9, color=INK,
                arrowprops=dict(arrowstyle="-", color=INK3, lw=0.8))
    ax.annotate(f"Test: {unseen}\nUnseen recordings",
                xy=(train + shared + unseen / 2, 0.25), xytext=(train + shared + unseen, 0.95), fontsize=9,
                ha="right", color=INK, arrowprops=dict(arrowstyle="-", color=INK3, lw=0.8))
    ax.set_xlim(0, train + shared + unseen)
    ax.set_ylim(-0.45, 1.5)
    ax.set_yticks([])
    ax.set_xlabel(f"ATCO2 clips (n = {train + shared + unseen})")
    ax.spines["left"].set_visible(False)
    save(fig, "fig1_split")


def wer_bars():
    r = load("retraining.json")
    runs = [(n, v["headline_test"]) for n, v in r["runs"].items() if n in LABELS]
    fig, ax = plt.subplots(figsize=(6.3, 0.32 * len(runs) + 0.65), layout="constrained")
    for i, (name, h) in enumerate(reversed(runs)):
        colour = ACCENT if name == r["chosen_on_validation"] else INK3
        wer, (lo, hi) = 100 * h["wer"], [100 * x for x in h["interval_95"]]
        ax.errorbar(wer, i, xerr=[[wer - lo], [hi - wer]],
                    fmt="s" if name == r["chosen_on_validation"] else "o", color=colour,
                    markersize=4, capsize=2.5, elinewidth=0.9, markeredgewidth=0.8)
        ax.text(74, i, f"{wer:.2f}", ha="right", va="center", fontsize=9, color=INK)
    ax.set_yticks(range(len(runs)), [LABELS[n] for n, _ in reversed(runs)])
    ax.set_xlim(0, 75)
    ax.set_xlabel("Word error rate (%)")
    ax.tick_params(axis="y", length=0, labelsize=9)
    ax.spines["left"].set_visible(False)
    ax.grid(axis="x", color=LINE, lw=0.6)
    ax.set_axisbelow(True)
    save(fig, "fig2_word_error_rates")


def noise():
    sweep = load("retraining.json")["noise_sweep"]["results"]
    fig, ax = plt.subplots(figsize=(6.3, 3.1), layout="constrained")
    series = [("zero_shot", "Whisper-small, not fine-tuned", INK3, "--", "^"),
              ("checkpoint_2025", "2025 model", BASELINE, "-.", "D"),
              ("retrained", "Best 2026 model", ACCENT, "-", "o"),
              ("retrained_bandpass", "Best 2026 model, band-pass filtered", ACCENT, ":", "s")]
    for key, label, colour, style, marker in series:
        pts = sweep[key]
        xs = range(len(pts))
        ax.plot(xs, [100 * p["wer"] for p in pts], style, color=colour, marker=marker, ms=3.5, lw=1.2, label=label)
    ticks = ["none" if p["added_noise_snr_db"] is None else f"{p['added_noise_snr_db']:g} dB"
             for p in sweep["checkpoint_2025"]]
    ax.set_xticks(range(len(ticks)), ticks)
    ax.set_xlabel("Signal-to-noise ratio after adding white noise")
    ax.set_ylabel("Word error rate (%)")
    ax.set_ylim(0, 100)
    ax.grid(axis="y", color=LINE, lw=0.6)
    ax.legend(frameon=False, fontsize=9)
    save(fig, "fig4_noise")


def instructions():
    r = load("retraining.json")
    a = load("extraction_agreement.json")["models"]
    models = [("zero_shot", PALE), ("checkpoint_2025", BASELINE), (r["chosen_on_validation"], ACCENT)]
    fields = [("callsign", "Callsign"), ("command", "Command"), ("value", "Numbers")]
    fig, ax = plt.subplots(figsize=(6.3, 2.9), layout="constrained")
    width = 0.26
    for j, (name, colour) in enumerate(models):
        m = a[name]
        shares = [100 * m[f]["clips_where_it_matches"] / m[f]["clips_with_field"] for f, _ in fields]
        shares.append(100 * m["callsign_command_value_identical"] / m["clips"])
        shares.append(100 * m["all_fields_identical"] / m["clips"])
        xs = [i + (j - 1) * width for i in range(len(shares))]
        label = "Best 2026 model" if name == r["chosen_on_validation"] else LABELS[name]
        ax.bar(xs, shares, width=width, color=colour, edgecolor=INK3, linewidth=0.4, label=label)
    ax.set_xticks(range(5), [label for _, label in fields] + ["All three", "All, with\nwaypoints"])
    ax.set_ylabel("Matched clips (%)")
    ax.set_ylim(0, 100)
    ax.grid(axis="y", color=LINE, lw=0.6)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=9, loc="upper right")
    save(fig, "fig5_instructions")


def confusion():
    e = load("rescoring_2025/entity_extraction.json")
    cm = e["confusion_matrix"]
    names = ["Other", "Callsign", "Command", "Value", "Waypoint"]
    fig, ax = plt.subplots(figsize=(4.4, 3.6), layout="constrained")
    ax.imshow(cm, cmap="Greys")
    for i, row in enumerate(cm):
        for j, v in enumerate(row):
            ax.text(j, i, v, ha="center", va="center", fontsize=9, color="white" if v > 120 else INK)
    ax.set_xticks(range(5), names, rotation=30, ha="right")
    ax.set_yticks(range(5), names)
    ax.set_xlabel("DistilBERT prediction")
    ax.set_ylabel("Label from the word-list rules")
    for side in ax.spines.values():
        side.set_visible(False)
    save(fig, "fig6_tagger_confusion")


def cross_validation():
    path = RESULTS / "cross_validation.json"
    if not path.exists():
        return
    cv = json.loads(path.read_text())
    fig, ax = plt.subplots(figsize=(6.3, 2.5), layout="constrained")
    folds = cv["folds"]
    ax.bar([f"Fold {f['fold'] + 1}" for f in folds], [100 * f["wer"] for f in folds], color=MID, width=0.6)
    for level, colour, label, style in [(cv["retrained"]["wer"], ACCENT, "Retrained, all 874 clips", ":"),
                                      (cv["zero_shot"]["wer"], INK3, "Not fine-tuned, all 874 clips", "--")]:
        ax.axhline(100 * level, color=colour, lw=1.2, ls=style, label=f"{label}: {100 * level:.2f}%")
    ax.set_ylabel("Word error rate (%)")
    ax.set_ylim(0, 70)
    ax.legend(frameon=False, fontsize=9, loc="upper right")
    ax.grid(axis="y", color=LINE, lw=0.6)
    ax.set_axisbelow(True)
    save(fig, "fig3_cross_validation")


def main():
    OUT.mkdir(exist_ok=True)
    for draw in [split, wer_bars, noise, instructions, confusion, cross_validation]:
        draw()
    print("\n".join(sorted(p.name for p in OUT.glob("*.png"))))


if __name__ == "__main__":
    main()
