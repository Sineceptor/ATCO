"""Recheck saved models without changing their data or checkpoints.

Detailed predictions stay in the ignored outputs folder.
The BERT targets and the optional ASR cleanup reproduce the old evaluation rules;
they are not independently annotated ground truth.
"""

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import time

from . import reference_labels as labels
from .historical_normalization import final_processing_pipeline

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def edit_distance(reference, prediction):
    """Levenshtein distance, used as an independent check of JiWER's totals."""
    previous = list(range(len(prediction) + 1))
    for i, left in enumerate(reference, 1):
        current = [i]
        for j, right in enumerate(prediction, 1):
            current.append(
                min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (left != right))
            )
        previous = current
    return previous[-1]


def error_rates(references, predictions):
    import jiwer

    words = sum(len(s.split()) for s in references)
    chars = sum(len(s) for s in references)
    word_errors = sum(
        edit_distance(r.split(), p.split())
        for r, p in zip(references, predictions, strict=True)
    )
    char_errors = sum(
        edit_distance(r, p) for r, p in zip(references, predictions, strict=True)
    )
    wer, cer = word_errors / words, char_errors / chars
    assert abs(wer - jiwer.wer(references, predictions)) < 1e-12
    assert abs(cer - jiwer.cer(references, predictions)) < 1e-12
    return dict(
        wer=wer,
        cer=cer,
        word_errors=word_errors,
        reference_words=words,
        character_errors=char_errors,
        reference_characters=chars,
    )


DIGITS = dict(zip("0123456789", "zero one two three four five six seven eight nine".split()))


def spell_digits(text):
    """Write each digit as a word so "4402" and "four four zero two" compare equal."""
    return re.sub(r"\d", lambda m: f" {DIGITS[m.group()]} ", text)


def write_summary(args, summary):
    summary["packages"] = {}
    for package in [
        "torch",
        "transformers",
        "tensorflow",
        "tf-keras",
        "jiwer",
        "scikit-learn",
    ]:
        try:
            summary["packages"][package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            pass
    filenames = {"asr": "speech_recognition.json", "bert": "bert_reference.json", "distilbert": "entity_extraction.json"}
    name = filenames[args.component]
    if args.component == "asr" and args.label:
        name = f"speech_recognition_{args.label}.json"
    path = args.output / name
    path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


def check_asr(args):
    import jiwer
    import soundfile as sf
    import torch
    from transformers import WhisperProcessor, WhisperForConditionalGeneration

    torch.set_num_threads(args.threads)
    manifest = args.manifest or args.data_dir / "speech/test.jsonl"
    audio_dir = args.audio_dir or args.data_dir / "speech/audio"
    # --asr-model scores another checkpoint on the same clips, for example the
    # unmodified "openai/whisper-small" as a zero-shot baseline.
    model_path = args.asr_model or args.models_dir / "whisper-small"
    local_only = not args.allow_downloads
    data = [
        json.loads(line) for line in manifest.read_text().splitlines() if line.strip()
    ]
    if args.limit:
        data = data[: args.limit]
    raw = jiwer.Compose(
        [
            jiwer.ToUpperCase(),
            jiwer.RemovePunctuation(),
            jiwer.RemoveMultipleSpaces(),
            jiwer.Strip(),
        ]
    )
    clean = final_processing_pipeline
    processor = WhisperProcessor.from_pretrained(model_path, local_files_only=local_only)
    model = WhisperForConditionalGeneration.from_pretrained(
        model_path, local_files_only=local_only
    ).eval().to(args.device)
    records, failures = [], []
    prompt = {}
    if args.prompt:
        # Words Whisper sees as "previous text" before each clip, to bias it towards them.
        prompt = dict(prompt_ids=torch.tensor(processor.get_prompt_ids(args.prompt)).to(args.device))
    started = time.monotonic()
    with (args.output / "asr_predictions.jsonl").open("w") as output:
        for index, item in enumerate(data):
            name = item["audio_file"].replace("\\", "/").split("/")[-1]
            path = audio_dir / name
            try:
                audio, sr = sf.read(path, dtype="float32")
                if sr != 16000 or audio.ndim != 1 or not 0 < len(audio) <= 480000:
                    raise ValueError(
                        "Expected a mono 16 kHz clip, between 0 and 30 seconds"
                    )
                inputs = processor(
                    audio,
                    sampling_rate=16000,
                    return_tensors="pt",
                    return_attention_mask=True,
                ).to(args.device)
                predictions = {}
                for mode in ("greedy", "beam", "beamplain"):
                    options = dict(language="en", task="transcribe", max_new_tokens=128, **prompt)
                    if mode == "beam":
                        # The 2025 decoding settings, kept so old scores reproduce.
                        options.update(
                            num_beams=5,
                            repetition_penalty=1.2,
                            length_penalty=1.0,
                            no_repeat_ngram_size=3,
                        )
                    if mode == "beamplain":
                        # Beam search without the repetition rules, which can block
                        # real repeats such as "zero zero zero".
                        options.update(num_beams=5)
                    with torch.inference_mode():
                        ids = model.generate(**inputs, **options)
                    predictions[mode] = processor.batch_decode(
                        ids, skip_special_tokens=True
                    )[0]
                record = dict(
                    file=name,
                    audio_sha256=digest(path),
                    reference=item["ASR_transcript_clean"],
                    **predictions,
                )
                records.append(record)
                output.write(json.dumps(record) + "\n")
                output.flush()
            except Exception as exc:
                failures.append(dict(file=name, error=str(exc)))
            if (index + 1) % 5 == 0:
                print(
                    f"ASR {index + 1}/{len(data)}; {len(failures)} failures; "
                    f"{time.monotonic() - started:.0f}s",
                    flush=True,
                )
    if not records:
        raise RuntimeError(f"No predictions: {failures}")
    metrics = {}
    def spelled(text):
        # An unmodified Whisper writes "4402"; the references say "four four zero two".
        return raw(spell_digits(text))

    modes = [("greedy", raw), ("beam", raw), ("beamplain", raw), ("beam_cleaned", clean)]
    modes += [("greedy_digits_spelled", spelled), ("beam_digits_spelled", spelled)]
    for mode, normalize in modes:
        prediction_key = mode.split("_")[0]
        kept = [r for r in records if normalize(r["reference"])]
        metrics[mode] = error_rates(
            [normalize(r["reference"]) for r in kept],
            [normalize(r[prediction_key]) for r in kept],
        )
        metrics[mode]["scored"] = len(kept)
    write_summary(
        args,
        dict(
            component="Whisper-small",
            attempted=len(data),
            completed=len(records),
            failures=failures,
            metrics=metrics,
            elapsed_seconds=time.monotonic() - started,
            manifest_sha256=digest(manifest),
            model=str(model_path),
            model_sha256=digest(Path(model_path) / "model.safetensors")
            if (Path(model_path) / "model.safetensors").is_file()
            else None,
            decoding=f"English transcription; max_new_tokens=128; {args.device} float32; prompt: {args.prompt!r}. "
            "Beam: num_beams=5, repetition_penalty=1.2, length_penalty=1, "
            "no_repeat_ngram_size=3. Beamplain: num_beams=5 only. "
            "Explicit task/language replace legacy forced IDs.",
            normalization="greedy/beam: uppercase, remove punctuation, collapse spaces. "
            "beam_cleaned: historical phrase/digit mappings and substring removals on BOTH sides.",
        ),
    )


def check_bert(args):
    import numpy as np
    from sklearn.metrics import classification_report, confusion_matrix

    if args.component == "bert":
        os.environ["TF_USE_LEGACY_KERAS"] = "1"
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
        import tensorflow as tf
        from transformers import BertTokenizerFast, TFBertForTokenClassification

        tf.config.threading.set_intra_op_parallelism_threads(args.threads)
        model_path = args.models_dir / "bert-reference"
        manifest = (
            args.data_dir / "entities/raw/test.json"
        )
        data = labels.load_raw_and_tag(str(manifest))
        tokenizer = BertTokenizerFast.from_pretrained(model_path, local_files_only=True)
        model = TFBertForTokenClassification.from_pretrained(
            model_path, local_files_only=True
        )
        tensor_type, weight_file = "tf", "tf_model.h5"
    else:
        import torch
        from transformers import AutoTokenizer, AutoModelForTokenClassification

        torch.set_num_threads(args.threads)
        model_path = args.models_dir / "distilbert"
        manifest = (
            args.data_dir / "entities/test.jsonl"
        )
        data = [
            json.loads(line)
            for line in manifest.read_text().splitlines()
            if line.strip()
        ]
        tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        model = AutoModelForTokenClassification.from_pretrained(
            model_path, local_files_only=True
        ).eval()
        tensor_type, weight_file = "pt", "model.safetensors"
    if args.limit:
        data = data[: args.limit]
    true, predicted = [], []
    truncated_words = 0
    with (args.output / f"{args.component}_predictions.jsonl").open("w") as output:
        for index, item in enumerate(data):
            inputs = tokenizer(
                item["tokens"],
                is_split_into_words=True,
                return_tensors=tensor_type,
                truncation=True,
                max_length=128,
            )
            if tensor_type == "tf":
                logits = model(inputs, training=False).logits.numpy()[0]
            else:
                with torch.inference_mode():
                    logits = model(**inputs).logits.numpy()[0]
            ids = np.argmax(logits, axis=-1)
            seen, expected, actual = set(), [], []
            for token, word in enumerate(inputs.word_ids()):
                if word is None or word in seen:
                    continue
                seen.add(word)
                expected.append(labels.convert_bio_to_entity(item["ner_tags"][word]))
                actual.append(
                    labels.convert_bio_to_entity(labels.ID2LABEL[int(ids[token])])
                )
            truncated_words += len(item["tokens"]) - len(seen)
            true.extend(expected)
            predicted.extend(actual)
            output.write(
                json.dumps(
                    dict(
                        index=index,
                        tokens=item["tokens"],
                        reference=expected,
                        prediction=actual,
                    )
                )
                + "\n"
            )
    entity_labels = labels.ENTITY_LABELS
    report = classification_report(
        true, predicted, labels=entity_labels, output_dict=True, zero_division=0
    )
    # Recompute each F1 and the support-weighted mean directly from the labels.
    weighted_f1 = 0
    for label in entity_labels:
        tp = sum(t == p == label for t, p in zip(true, predicted, strict=True))
        fp = sum(
            t != label and p == label for t, p in zip(true, predicted, strict=True)
        )
        fn = sum(
            t == label and p != label for t, p in zip(true, predicted, strict=True)
        )
        f1 = 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else 0
        assert abs(f1 - report[label]["f1-score"]) < 1e-12
        weighted_f1 += f1 * true.count(label) / len(true)
    assert abs(weighted_f1 - report["weighted avg"]["f1-score"]) < 1e-12
    write_summary(
        args,
        dict(
            component="TensorFlow BERT-base"
            if tensor_type == "tf"
            else "PyTorch DistilBERT (app model)",
            sequences=len(data),
            scored_words=len(true),
            truncated_words=truncated_words,
            correct_words=sum(t == p for t, p in zip(true, predicted, strict=True)),
            report=report,
            labels=entity_labels,
            confusion_matrix=confusion_matrix(true, predicted, labels=entity_labels).tolist(),
            manifest_sha256=digest(manifest),
            model_sha256=digest(model_path / weight_file),
            method="First subword per word; BIO prefixes collapsed; rule-generated targets. "
            "This is word-label F1, including O, not exact entity-span F1.",
        ),
    )


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("component", choices=["asr", "bert", "distilbert"])
    cli.add_argument("--data-dir", type=Path, default=ROOT / "datasets")
    cli.add_argument("--models-dir", type=Path, default=ROOT / "models")
    cli.add_argument("--output", type=Path, default=ROOT / "outputs/evaluation")
    cli.add_argument("--threads", type=int, default=4)
    cli.add_argument("--device", default="cpu", help='"cpu", or "mps" for an Apple GPU')
    cli.add_argument("--prompt", help="Text given to Whisper as context before every clip")
    cli.add_argument(
        "--asr-model",
        help='Score this Whisper checkpoint or hub name instead, e.g. "openai/whisper-small"',
    )
    cli.add_argument("--label", help="Suffix for the ASR summary file, e.g. zero_shot")
    cli.add_argument("--manifest", type=Path, help="Score these clips instead of speech/test.jsonl")
    cli.add_argument("--audio-dir", type=Path, help="Folder holding the manifest's audio")
    cli.add_argument(
        "--allow-downloads", action="store_true", help="Let --asr-model download from the hub"
    )
    cli.add_argument(
        "--limit", type=int, help="Smoke test only; omit for the full saved split"
    )
    args = cli.parse_args()
    args.data_dir = args.data_dir.resolve()
    args.models_dir = args.models_dir.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    (check_asr if args.component == "asr" else check_bert)(args)


if __name__ == "__main__":
    main()
