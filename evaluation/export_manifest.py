"""Write results/experiment_manifest.json: how every 2026 model was made.

    python -m evaluation.export_manifest

For each training run: the exact command (with local paths replaced by what they
point to), the seed, the base model and its revision, SHA-256 fingerprints of the
data manifests, and SHA-256 of the saved checkpoint. For each model soup: its
ingredients. Also the decoding settings used for scoring. Model weights and audio
are not published; the fingerprints let anyone with the same files confirm they
have exactly what was used.
"""

import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "training"
SMALL_REVISION = "973afd24965f72e36ca33b3055d56a652f456b4d"

FLAGS = ["base_model", "train", "validation", "audio_dir", "extra", "extra_audio_dir", "extra_per_epoch",
         "epochs", "patience", "learning_rate", "batch_size", "accumulate", "warmup_steps", "optimizer",
         "memory_fraction", "spec_augment", "lora", "letter_weight", "save_every_epoch", "seed"]
DECODING = {
    "greedy": "greedy, English, transcribe, max_new_tokens=128",
    "beam": "beam search, 5 beams, repetition_penalty=1.2, no_repeat_ngram_size=3, max_new_tokens=128 "
            "(the app's decoding)",
    "beamplain": "beam search, 5 beams, max_new_tokens=128, no repetition guards",
    "chosen": "for each model, the decoding with the lowest validation WER (ties go to beam)",
}


def sha256(path, cache={}):
    path = Path(path)
    if path not in cache:
        h = hashlib.sha256()
        with path.open("rb") as f:
            for block in iter(lambda: f.read(1 << 22), b""):
                h.update(block)
        cache[path] = h.hexdigest()
    return cache[path]


def public(value):
    """Replace a local path with what it points to."""
    v = str(value)
    if "atco-run/" in v and ("medium_long" in v or "soup" in v):
        # the 12-epoch run trained from a copy on the internal disk
        return "outputs/training/" + v.split("atco-run/", 1)[1]
    if SMALL_REVISION in v:
        return f"openai/whisper-small@{SMALL_REVISION}"
    if "whisper-medium-base" in v:
        return "openai/whisper-medium (local copy; see base_models)"
    if "uwb_atcc" in v:
        return "datasets/uwb_atcc/" + ("audio" if v.endswith("audio") else Path(v).name)
    if v.endswith("atco2_audio") or v.endswith("speech/audio"):
        return "datasets/speech/audio"
    m = re.search(r"(datasets/.*|outputs/training/.*)", v)
    if m:
        return m.group(1)
    return v


def manifest_file(value):
    """The local file a recorded manifest path refers to, for fingerprinting."""
    v = public(value)
    for candidate in [ROOT / v, ROOT / str(value)]:
        if candidate.is_file():
            return candidate
    return None


def checkpoint(folder):
    for name in ["model.safetensors", "adapter_model.safetensors"]:
        f = folder / name
        if f.is_file():
            return {"file": f"outputs/{folder.relative_to(ROOT / 'outputs')}/{name}", "sha256": sha256(f)}
    return None


def main():
    runs = {}
    for log_path in sorted(OUT.glob("*/training_log.json")):
        name = log_path.parent.name
        if name.startswith("medium_long_lost") or name == "medium_check":
            continue
        log = json.loads(log_path.read_text())
        s = log["settings"]
        args = []
        for flag in FLAGS:
            value = s.get(flag)
            if value in (None, "None", "False", "0", "1.0") and flag not in ("seed",):
                continue
            if value == "True":
                args.append(f"--{flag.replace('_', '-')}")
            else:
                args.append(f"--{flag.replace('_', '-')} {public(value)}")
        data = {}
        for key in ["train", "validation", "extra"]:
            f = manifest_file(s.get(key)) if s.get(key) not in (None, "None") else None
            if f:
                data[key] = {"file": public(s[key]), "clips": sum(1 for line in f.open() if line.strip()),
                             "sha256": sha256(f)}
        saved = {}
        for sub in ["best", "merged"]:
            c = checkpoint(log_path.parent / sub)
            if c:
                saved[sub] = c
        runs[name] = dict(
            command="python -m training.train_whisper --local-only " + " ".join(args)
                    + f" --output outputs/training/{name}",
            seed=int(s.get("seed", 42)),
            base_model=public(s["base_model"]),
            data=data,
            epochs_run=len(log["epochs"]),
            best_epoch=log["best_epoch"],
            checkpoints=saved or "not kept (deleted after scoring to save space, or nothing beat the start)",
        )
    soups = {}
    for soup in sorted(OUT.glob("*/best/SOUP.txt")):
        ingredients = [public(line.strip()) for line in soup.read_text().splitlines()[1:] if line.strip()]
        soups[soup.parent.parent.name] = dict(
            method="python -m training.average_checkpoints " + " ".join(ingredients) +
                   f" --output outputs/training/{soup.parent.parent.name}/best",
            ingredients=ingredients,
            checkpoint=checkpoint(soup.parent),
        )
    medium = ROOT / "models" / "whisper-medium-base" / "model.safetensors"
    report = dict(
        about=__doc__.split("\n\n")[1].strip(),
        base_models={
            "openai/whisper-small": {"revision": SMALL_REVISION},
            "openai/whisper-medium": {"model.safetensors sha256": sha256(medium) if medium.is_file() else None},
        },
        app_model=dict(folder="models/whisper-medium-2026", same_as="soup_medium_long",
                       sha256=sha256(ROOT / "models" / "whisper-medium-2026" / "model.safetensors")),
        decoding=DECODING,
        scoring="python -m evaluation.evaluate_models asr --asr-model MODEL --manifest "
                "datasets/speech_split/{validation,test}.jsonl --audio-dir datasets/speech/audio; "
                "summaries: python -m evaluation.summarise_retraining",
        runs=runs,
        soups=soups,
    )
    path = ROOT / "results" / "experiment_manifest.json"
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(f"{len(runs)} runs, {len(soups)} soups -> {path}")


if __name__ == "__main__":
    main()
