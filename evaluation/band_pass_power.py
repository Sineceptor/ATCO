"""How much power does the 300-3400 Hz filter remove from the real test clips?

The noise sweep sets the noise level from each clip's own power. With
--band-pass the clip is filtered first, so if the filter removes power, the
filtered clip receives weaker noise at the same nominal signal-to-noise ratio.
This measures that difference, 10 log10(P_unfiltered / P_filtered), per clip.

    python -m evaluation.band_pass_power --output results/band_pass_power.json
"""

import argparse
import json
from pathlib import Path

import numpy as np

from .noise_sweep import voice_band

ROOT = Path(__file__).resolve().parents[1]


def main():
    import soundfile as sf

    cli = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    cli.add_argument("--manifest", type=Path, default=ROOT / "datasets/speech_split/test.jsonl")
    cli.add_argument("--audio-dir", type=Path, default=ROOT / "datasets/speech/audio")
    cli.add_argument("--output", type=Path)
    args = cli.parse_args()
    rows = [json.loads(s) for s in args.manifest.read_text().splitlines() if s.strip()]
    ratios, total_before, total_after = [], 0.0, 0.0
    for row in rows:
        audio, rate = sf.read(args.audio_dir / Path(row["audio_file"].replace("\\", "/")).name, dtype="float64")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        assert rate == 16000, rate
        before = float(np.mean(audio ** 2))
        after = float(np.mean(voice_band(audio.astype(np.float32)).astype(np.float64) ** 2))
        ratios.append(10 * np.log10(before / after))
        total_before += before * len(audio)
        total_after += after * len(audio)
    report = dict(
        clips=len(rows),
        filter="the noise sweep's 300-3400 Hz band-pass (4th-order Butterworth, forwards and backwards)",
        db_removed_per_clip_percentiles_10_50_90=[round(float(v), 2) for v in np.percentile(ratios, [10, 50, 90])],
        db_removed_mean_over_clips=round(float(np.mean(ratios)), 2),
        db_removed_pooled=round(float(10 * np.log10(total_before / total_after)), 2),
        meaning="at the same nominal SNR, a filtered clip receives this many dB less noise than the unfiltered clip",
    )
    print(json.dumps(report, indent=2))
    if args.output:
        args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
