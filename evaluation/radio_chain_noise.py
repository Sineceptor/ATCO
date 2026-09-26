"""How much of each noise in the 2025 radio simulation survives its own filter?

The 2025 script (experiments/asr/04_augmentation_and_synthetic_speech/
simulate_radio_channel.py) adds "engine" brown noise and white static, then
applies a 300-3400 Hz band-pass. Brown noise has most of its power at the lowest
frequencies, so this measures, over many random clips drawn exactly as the
script draws them, how much of each noise is left after the filter.

    python -m evaluation.radio_chain_noise --output results/radio_chain_noise.json

Only the two noises are simulated, without speech; the filter is linear, so each
noise's share can be measured on its own. The mu-law stage between them is left
out: without its quantisation it is the identity.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import scipy.signal

RATE = 16000


def brown(n, rng):
    """The 2025 generator: cumulative sum of white noise, scaled to a peak of 1."""
    walk = np.cumsum(rng.normal(0, 1, n))
    return walk / np.max(np.abs(walk))


def band_pass(x):
    """The 2025 filter: scipy butter(4) band-pass, applied once with lfilter."""
    b, a = scipy.signal.butter(4, [300 / (RATE / 2), 3400 / (RATE / 2)], btype="band")
    return scipy.signal.lfilter(b, a, x)


def power(x):
    return float(np.mean(np.square(x)))


def below(x, hz):
    """Share of a signal's power below a frequency."""
    spectrum = np.abs(np.fft.rfft(x)) ** 2
    freqs = np.fft.rfftfreq(len(x), 1 / RATE)
    return float(spectrum[freqs < hz].sum() / spectrum.sum())


def main():
    cli = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    cli.add_argument("--clips", type=int, default=2000)
    cli.add_argument("--seconds", type=float, default=4.5, help="Typical length of a synthetic clip")
    cli.add_argument("--output", type=Path)
    args = cli.parse_args()
    rng = np.random.default_rng(0)
    n = int(args.seconds * RATE)
    kept, low, low_ac, static_kept, gap_before, gap_after = [], [], [], [], [], []
    for _ in range(args.clips):
        engine = brown(n, rng) * rng.uniform(0.01, 0.05)
        static = rng.normal(0, 1, n) * rng.uniform(0.005, 0.02)
        e_out, s_out = band_pass(engine), band_pass(static)
        kept.append(power(e_out) / power(engine))
        static_kept.append(power(s_out) / power(static))
        low.append(below(engine, 20))
        low_ac.append(below(engine - engine.mean(), 20))
        gap_before.append(10 * np.log10(power(engine) / power(static)))
        gap_after.append(10 * np.log10(power(e_out) / power(s_out)))
    q = lambda xs: [round(float(v), 6) for v in np.percentile(xs, [10, 50, 90])]
    report = dict(
        method=f"{args.clips} random {args.seconds}-second clips at {RATE} Hz, noise levels drawn as in the 2025 "
               "script; each noise filtered on its own by the script's band-pass",
        percentiles="10th, 50th and 90th",
        engine_power_below_20_hz_share=q(low),
        engine_power_below_20_hz_share_without_its_constant_offset=q(low_ac),
        engine_power_kept_by_filter_share=q(kept),
        static_power_kept_by_filter_share=q(static_kept),
        engine_minus_static_db_before_filter=[round(v, 2) for v in q(gap_before)],
        engine_minus_static_db_after_filter=[round(v, 2) for v in q(gap_after)],
    )
    print(json.dumps(report, indent=2))
    if args.output:
        args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
