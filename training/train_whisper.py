"""Fine-tune Whisper-small on ATC speech, choosing the checkpoint on held-out clips.

    python -m training.train_whisper --output outputs/training/whisper-small-fixed
    python -m training.train_whisper --extra datasets/speech/synthetic.jsonl \
        --extra-audio-dir datasets/speech/audio_synthetic --extra-per-epoch 700 \
        --output outputs/training/whisper-small-synthetic

What changed from the 2025 version, which trained the app's checkpoint:

- Start and task tokens. The old labels began <|startoftranscript|><|notimestamps|>
  and the old collator added a second start token, while at run time the model is
  prompted with <|startoftranscript|><|en|><|transcribe|><|notimestamps|>. Labels
  now carry the same prefix as the prompt, and the start token appears once.
- Checkpoint choice. The 2025 runs picked checkpoints and decoding settings by
  looking at the test clips. Here a separate validation manifest does that job,
  and the test clips are not read at all.
- Hardware. Runs on Apple silicon (MPS), an NVIDIA GPU or the CPU. On a Mac the
  GPU memory is capped (--memory-fraction) so a run fails cleanly instead of
  pushing the whole machine into swap, and --bf16 halves activation memory.
- Extra data. Optional: a fresh random sample of extra clips (my synthetic radio
  speech, or noise and speed copies of the training clips) is mixed with all the
  real clips in every epoch, so real speech is never outnumbered.
"""

import argparse
import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]


def read_manifest(path, audio_dir):
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        name = item["audio_file"].replace("\\", "/").split("/")[-1]
        # Real references are lower case with digits spelled out; the synthetic
        # manifest is upper case. Train on one spelling so the targets agree.
        text = " ".join(item["ASR_transcript_clean"].lower().split())
        if text:
            rows.append(dict(path=Path(audio_dir) / name, text=text))
    return rows


# Phonetic-alphabet letters and digit words, the building blocks of callsigns.
SPELLED = {
    "alpha", "alfa", "bravo", "charlie", "delta", "echo", "foxtrot", "golf", "hotel", "india", "juliett",
    "juliet", "kilo", "lima", "mike", "november", "oscar", "papa", "quebec", "romeo", "sierra", "tango",
    "uniform", "victor", "whiskey", "whisky", "x", "ray", "xray", "yankee", "zulu",
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "niner",
}


def weighted_labels(tokenizer, text, letter_weight):
    """Token ids for a transcript, and a weight per token: letter_weight for the tokens of
    spelled letters and digits, 1 for everything else."""
    ids = tokenizer(text).input_ids
    if letter_weight == 1:
        return ids, [1.0] * len(ids)
    prefix = tokenizer("").input_ids[:-1]  # start, language and task tokens
    pieces, weights = [], []
    for i, word in enumerate(text.split()):
        piece = tokenizer.encode(word if i == 0 else " " + word, add_special_tokens=False)
        pieces += piece
        weights += [letter_weight if word in SPELLED else 1.0] * len(piece)
    if prefix + pieces + ids[-1:] != ids:  # word-by-word tokens must match the whole text
        return ids, [1.0] * len(ids)
    return ids, [1.0] * len(prefix) + weights + [1.0]


class Clips(Dataset):
    def __init__(self, rows, processor, letter_weight=1.0):
        self.rows, self.processor, self.letter_weight = rows, processor, letter_weight

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        audio, rate = sf.read(row["path"], dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if rate != 16000:
            raise ValueError(f"{row['path'].name}: expected 16 kHz audio, got {rate}")
        features = self.processor.feature_extractor(audio, sampling_rate=16000).input_features[0]
        labels, weights = weighted_labels(self.processor.tokenizer, row["text"], self.letter_weight)
        return dict(input_features=features, labels=labels, weights=weights)


@dataclass
class Collator:
    processor: object
    start_token: int
    label_length: int = 64  # the longest label in the data is 60 tokens

    def __call__(self, batch):
        features = torch.tensor(np.stack([b["input_features"] for b in batch]))
        # Every batch is padded to the same label length. On Apple GPUs each new
        # tensor shape compiles and caches another compute graph, so variable
        # lengths make memory creep up until the run fails.
        labels = torch.full((len(batch), self.label_length), -100, dtype=torch.long)
        weights = torch.zeros((len(batch), self.label_length))
        for row, b in enumerate(batch):
            ids = b["labels"][: self.label_length]
            labels[row, : len(ids)] = torch.tensor(ids)
            weights[row, : len(ids)] = torch.tensor(b.get("weights", [1.0] * len(b["labels"]))[: len(ids)])
        # The fix: the tokenizer already starts every label with the start token,
        # and the model prepends it again when it shifts labels into decoder inputs.
        if (labels[:, 0] == self.start_token).all():
            labels, weights = labels[:, 1:], weights[:, 1:]
        return dict(input_features=features, labels=labels, weights=weights)


def pick_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def word_error_rate(references, predictions):
    import jiwer

    normalize = jiwer.Compose(
        [jiwer.ToUpperCase(), jiwer.RemovePunctuation(), jiwer.RemoveMultipleSpaces(), jiwer.Strip()]
    )
    pairs = [(normalize(r), normalize(p)) for r, p in zip(references, predictions) if normalize(r)]
    return jiwer.wer([r for r, _ in pairs], [p for _, p in pairs])


def free_memory(device):
    if device.type == "mps":
        torch.mps.empty_cache()
    elif device.type == "cuda":
        torch.cuda.empty_cache()


def memory_in_use(device):
    if device.type == "mps":
        return f"{torch.mps.driver_allocated_memory() / 2**30:.1f} GB"
    if device.type == "cuda":
        return f"{torch.cuda.max_memory_allocated() / 2**30:.1f} GB"
    return "n/a"


def transcribe(model, processor, rows, device, batch_size=4):
    model.eval()
    predictions = []
    with torch.inference_mode():
        for start in range(0, len(rows), batch_size):
            chunk = rows[start : start + batch_size]
            audio = [sf.read(r["path"], dtype="float32")[0] for r in chunk]
            inputs = processor.feature_extractor(audio, sampling_rate=16000, return_tensors="pt")
            # The same repetition guards as the 2025 decoder: one clip stuck in a loop
            # ("ne sly ne sly ...") otherwise adds dozens of errors and decides which
            # checkpoint looks best.
            ids = model.generate(
                inputs.input_features.to(device), language="en", task="transcribe", max_new_tokens=128,
                use_cache=True, repetition_penalty=1.2, no_repeat_ngram_size=3,
            )
            predictions += processor.batch_decode(ids, skip_special_tokens=True)
    free_memory(device)
    model.train()
    return predictions


def save(model, processor, folder, lora):
    """Save the checkpoint. For LoRA only the small adapters are saved here; merge them
    into the base weights afterwards with python -m training.merge_lora, on the CPU."""
    model.save_pretrained(folder)
    processor.save_pretrained(folder)


def main():
    cli = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    cli.add_argument("--base-model", default="openai/whisper-small")
    cli.add_argument("--train", type=Path, default=ROOT / "datasets/speech/train.jsonl")
    cli.add_argument("--validation", type=Path, default=ROOT / "datasets/speech_split/validation.jsonl")
    cli.add_argument("--audio-dir", type=Path, default=ROOT / "datasets/speech/audio")
    cli.add_argument("--extra", type=Path, help="Optional manifest of extra clips to mix in")
    cli.add_argument("--extra-audio-dir", type=Path, default=ROOT / "datasets/speech/audio_synthetic")
    cli.add_argument("--extra-per-epoch", type=int, default=0)
    cli.add_argument("--output", type=Path, default=ROOT / "outputs/training/whisper-small-fixed")
    cli.add_argument("--epochs", type=int, default=15)
    cli.add_argument("--patience", type=int, default=4,
                     help="Stop after this many epochs without a better validation score")
    cli.add_argument("--learning-rate", type=float, default=1e-5)
    cli.add_argument("--batch-size", type=int, default=1)
    cli.add_argument("--accumulate", type=int, default=16)
    cli.add_argument("--bf16", action="store_true", help="Run the forward pass in bfloat16")
    cli.add_argument("--optimizer", choices=["adamw", "adafactor"], default="adamw")
    cli.add_argument("--memory-fraction", type=float, default=0.6,
                     help="Cap on Apple GPU memory, as a fraction of the recommended maximum")
    cli.add_argument("--warmup-steps", type=int, default=50)
    cli.add_argument("--spec-augment", action="store_true", help="Mask random time and frequency bands")
    cli.add_argument("--lora", type=int, default=0,
                     help="Train LoRA adapters of this rank instead of every weight (for larger models)")
    cli.add_argument("--letter-weight", type=float, default=1.0,
                     help="Weight the loss on spelled letters and digits this many times more")
    cli.add_argument("--save-every-epoch", action="store_true",
                     help="Also keep each epoch's checkpoint (sensible for small LoRA adapters)")
    cli.add_argument("--seed", type=int, default=42)
    cli.add_argument("--local-only", action="store_true", help="Never download the base model")
    args = cli.parse_args()

    from transformers import WhisperForConditionalGeneration, WhisperProcessor, get_linear_schedule_with_warmup

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = pick_device()
    if device.type == "mps":
        torch.mps.set_per_process_memory_fraction(args.memory_fraction)

    processor = WhisperProcessor.from_pretrained(args.base_model, local_files_only=args.local_only)
    processor.tokenizer.set_prefix_tokens(language="en", task="transcribe", predict_timestamps=False)
    model = WhisperForConditionalGeneration.from_pretrained(
        args.base_model, local_files_only=args.local_only, attn_implementation="sdpa"
    )
    model.config.forced_decoder_ids = None
    model.generation_config.forced_decoder_ids = None
    if args.spec_augment:
        model.config.apply_spec_augment = True
        model.config.mask_time_prob = 0.05
        model.config.mask_feature_prob = 0.05
    # Recompute activations in the backward pass instead of storing them, so a
    # batch fits in the memory of a laptop GPU.
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    if args.lora:
        # Freeze the original weights and learn small low-rank updates to every
        # attention and feed-forward layer; merged back in when a checkpoint is saved.
        from peft import LoraConfig, get_peft_model

        model.enable_input_require_grads()
        model = get_peft_model(model, LoraConfig(
            r=args.lora, lora_alpha=2 * args.lora, lora_dropout=0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"],
        ))
        model.print_trainable_parameters()
    model.to(device).train()

    real = read_manifest(args.train, args.audio_dir)
    extra = read_manifest(args.extra, args.extra_audio_dir) if args.extra else []
    validation = read_manifest(args.validation, args.audio_dir)
    per_epoch = len(real) + min(args.extra_per_epoch, len(extra))
    steps_per_epoch = math.ceil(per_epoch / (args.batch_size * args.accumulate))
    total_steps = steps_per_epoch * args.epochs

    collate = Collator(processor, processor.tokenizer.convert_tokens_to_ids("<|startoftranscript|>"))
    if args.optimizer == "adafactor":
        # Adafactor keeps one row and one column of statistics per weight matrix
        # instead of two full copies of the model, which saves about 2 GB here.
        from transformers.optimization import Adafactor

        optimizer = Adafactor(model.parameters(), lr=args.learning_rate, scale_parameter=False,
                              relative_step=False, warmup_init=False, weight_decay=0.01)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01, foreach=False)
    schedule = get_linear_schedule_with_warmup(optimizer, args.warmup_steps, total_steps)

    args.output.mkdir(parents=True, exist_ok=True)
    log = dict(
        settings={k: str(v) for k, v in vars(args).items()},
        device=str(device),
        real_clips=len(real),
        extra_clips_available=len(extra),
        validation_clips=len(validation),
        epochs=[],
    )
    references = [r["text"] for r in validation]
    best = transcribe(model, processor, validation, device)
    best_wer = word_error_rate(references, best)
    log["validation_wer_before_training"] = best_wer
    print(f"validation WER before training: {best_wer:.4f}", flush=True)
    best_epoch, started = 0, time.monotonic()

    for epoch in range(1, args.epochs + 1):
        rows = real + random.sample(extra, min(args.extra_per_epoch, len(extra)))
        loader = DataLoader(Clips(rows, processor, args.letter_weight), batch_size=args.batch_size,
                            shuffle=True, collate_fn=collate)
        losses = []
        for step, batch in enumerate(loader, 1):
            labels = batch["labels"].to(device)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=args.bf16):
                output = model(input_features=batch["input_features"].to(device), labels=labels)
            loss = output.loss
            if args.letter_weight != 1:
                weights = batch["weights"].to(device)
                token_loss = torch.nn.functional.cross_entropy(
                    output.logits.float().transpose(1, 2), labels, ignore_index=-100, reduction="none"
                )
                loss = (token_loss * weights).sum() / weights.sum()
            (loss / args.accumulate).backward()
            losses.append(loss.item())
            if step % 20 == 0:
                print(f"  epoch {epoch} step {step}/{len(loader)} loss {np.mean(losses[-50:]):.4f} "
                      f"{(time.monotonic() - started) / 60:.1f} min, GPU memory {memory_in_use(device)}",
                      flush=True)
            if step % args.accumulate == 0 or step == len(loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                schedule.step()
                optimizer.zero_grad(set_to_none=True)
                free_memory(device)  # hand cached GPU memory back, so it cannot creep up
        wer = word_error_rate(references, transcribe(model, processor, validation, device))
        log["epochs"].append(
            dict(epoch=epoch, train_loss=float(np.mean(losses)), validation_wer=wer,
                 minutes=round((time.monotonic() - started) / 60, 1))
        )
        print(f"epoch {epoch}: loss {np.mean(losses):.4f}, validation WER {wer:.4f}", flush=True)
        if args.save_every_epoch:
            save(model, processor, args.output / f"epoch{epoch}", args.lora)
        if wer < best_wer:
            best_wer, best_epoch = wer, epoch
            save(model, processor, args.output / "best", args.lora)
        log["best_epoch"], log["best_validation_wer"] = best_epoch, best_wer
        (args.output / "training_log.json").write_text(json.dumps(log, indent=2) + "\n")
        if epoch - best_epoch >= args.patience:
            print(f"no improvement for {args.patience} epochs; stopping", flush=True)
            break

    if best_epoch == 0:
        print("No epoch beat the untrained model on the validation clips; nothing saved.")
    else:
        print(f"best validation WER {best_wer:.4f} at epoch {best_epoch}; saved to {args.output / 'best'}")


if __name__ == "__main__":
    main()
