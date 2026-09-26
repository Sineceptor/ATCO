"""Average the weights of several fine-tuned Whisper checkpoints ("model soup").

    python -m training.average_checkpoints outputs/training/fixed/best \\
        outputs/training/real_data/best --output outputs/training/soup/best

Models fine-tuned from the same starting weights often sit in the same valley
of the loss surface, so the average of their weights can be better than any one
of them (Wortsman et al., "Model soups", 2022). Checkpoints are read one at a
time, so memory use stays at about two copies of the model. Which soup to keep
is decided on the validation clips, like every other choice here.
"""

import argparse
import shutil
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file


def main():
    cli = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    cli.add_argument("checkpoints", nargs="+", type=Path)
    cli.add_argument("--output", type=Path, required=True)
    args = cli.parse_args()

    total = None
    for path in args.checkpoints:
        weights = load_file(path / "model.safetensors")
        if total is None:
            total = {k: v.to(torch.float32) for k, v in weights.items()}
        else:
            if weights.keys() != total.keys():
                raise SystemExit(f"{path} has different weights from {args.checkpoints[0]}")
            for k, v in weights.items():
                total[k] += v.to(torch.float32)
        del weights
    n = len(args.checkpoints)
    first = load_file(args.checkpoints[0] / "model.safetensors")
    average = {k: (v / n).to(first[k].dtype) for k, v in total.items()}
    del first, total

    args.output.mkdir(parents=True, exist_ok=True)
    for f in args.checkpoints[0].iterdir():  # config, tokenizer and processor files
        if f.name != "model.safetensors" and f.is_file():
            shutil.copy(f, args.output / f.name)
    save_file(average, args.output / "model.safetensors", metadata={"format": "pt"})
    (args.output / "SOUP.txt").write_text("Average of:\n" + "".join(f"{p}\n" for p in args.checkpoints))
    print(f"averaged {n} checkpoints into {args.output}")


if __name__ == "__main__":
    main()
