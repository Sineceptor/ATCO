"""Merge LoRA adapters into their base Whisper weights, on the CPU.

    python -m training.merge_lora outputs/training/medium/best --output outputs/training/medium/merged

The result is an ordinary checkpoint that every evaluation script can load.
"""

import argparse
from pathlib import Path


def main():
    cli = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    cli.add_argument("adapters", type=Path)
    cli.add_argument("--output", type=Path, required=True)
    args = cli.parse_args()

    from peft import PeftConfig, PeftModel
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    base = PeftConfig.from_pretrained(args.adapters).base_model_name_or_path
    model = WhisperForConditionalGeneration.from_pretrained(base)
    model = PeftModel.from_pretrained(model, args.adapters).merge_and_unload()
    model.save_pretrained(args.output)
    WhisperProcessor.from_pretrained(args.adapters).save_pretrained(args.output)
    print(f"merged {args.adapters} into {base}; saved to {args.output}")


if __name__ == "__main__":
    main()
