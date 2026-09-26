"""Run speech recognition, entity extraction and optional spoken readback."""

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def model_paths(args):
    directory = args.models_dir.resolve()
    # The best installed speech model: the 2026 Whisper-medium, then the 2026
    # Whisper-small, then the 2025 checkpoint.
    speech = next((directory / name for name in ["whisper-medium-2026", "whisper-small-2026"]
                   if (directory / name).is_dir()), directory / "whisper-small")
    return (
        args.asr_model or str(speech),
        args.ner_model or str(directory / "distilbert"),
        args.tts_model or str(directory / "speecht5"),
        args.vocoder_model or str(directory / "hifigan"),
    )


def make_result(transcript, entities, parsed_text, warnings):
    """Keep the intermediate output visible; never invent a missing entity."""
    records = [
        {"type": e["entity_group"], "text": e["word"], "score": float(e["score"])} for e in entities
    ]
    # Keep the original entity order and words. The old guard's numeric display
    # is recorded separately because its number conversion is only a heuristic.
    readback = " ".join(e["text"] for e in records).strip()
    return {
        "transcript": transcript,
        "entities": records,
        "parsed_display": parsed_text,
        "rule_warnings": list(warnings),
        "readback_text": readback,
        "status": "entities_found" if records else "no_entities",
        "audio_file": None,
        "note": "Prototype entity readback; omitted words and model errors can change meaning. Not an ATC clearance.",
    }


def run(args):
    asr_path, ner_path, tts_path, vocoder_path = model_paths(args)
    local_only = not args.allow_downloads
    if args.audio:
        if not args.audio.is_file():
            raise FileNotFoundError(f"Audio file not found: {args.audio}")
        from atco.speech_recognition import transcribe

        transcript = transcribe(asr_path, str(args.audio), local_files_only=local_only)
    else:
        transcript = args.text.strip()
    if not transcript:
        raise ValueError("The input or recognised transcript is empty")

    from atco.entity_extraction import AtcParser, drop_function_word_waypoints, repair_bert_output

    parser = AtcParser(ner_path, local_files_only=local_only)
    tokens = parser.pipe.tokenizer(transcript)["input_ids"]
    max_tokens = parser.pipe.model.config.max_position_embeddings
    if len(tokens) > max_tokens:
        raise ValueError(f"Use one short instruction (maximum {max_tokens} model tokens)")
    entities = drop_function_word_waypoints(repair_bert_output(parser.pipe(transcript)))
    parsed_text, warnings = parser.guard.check(entities)
    result = make_result(transcript, entities, parsed_text, warnings)
    result["models"] = {
        "asr": asr_path if args.audio else None,
        "ner": ner_path,
        "tts": tts_path if args.tts == "speecht5" else args.tts,
        "vocoder": vocoder_path,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result_path = args.output_dir / "result.json"
    # Save the intermediate stages even if synthesis subsequently fails.
    result["tts_status"] = "pending" if args.tts != "none" and entities else "skipped"
    result_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    try:
        if result["readback_text"] and args.tts == "speecht5":
            from atco.speech_synthesis import AtcSpeaker

            speaker = AtcSpeaker(
                tts_path, vocoder_path, args.output_dir, local_files_only=local_only
            )
            result["audio_file"] = str(speaker.speak(result["readback_text"], "readback.wav"))
        result["tts_status"] = "generated" if result["audio_file"] else "skipped"
    except Exception as exc:
        result["tts_status"] = "failed"
        result["tts_error"] = str(exc)
        raise
    finally:
        result_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(f"\nTranscript: {result['transcript']}")
    print("Entities:")
    for entity in result["entities"]:
        print(f"  {entity['type']}: {entity['text']} ({entity['score']:.2f})")
    if not result["entities"]:
        print("  No entities found.")
    for warning in result["rule_warnings"]:
        print(f"Check: {warning}")
    print(f"Readback: {result['readback_text'] or '(none)'}")
    if result["audio_file"]:
        print(f"Audio: {result['audio_file']}")
    print(f"Details: {result_path}")
    return result


def main():
    cli = argparse.ArgumentParser(
        description="Transcribe audio, extract instruction fields and generate a readback."
    )
    source = cli.add_mutually_exclusive_group(required=True)
    source.add_argument("--audio", type=Path, help="One audio clip, at most 30 seconds")
    source.add_argument("--text", help="Start at the NLP stage with typed text")
    cli.add_argument(
        "--models-dir",
        type=Path,
        default=ROOT / "models",
        help="Folder containing whisper-small, distilbert, speecht5 and hifigan models",
    )
    cli.add_argument("--asr-model")
    cli.add_argument("--ner-model")
    cli.add_argument("--tts-model")
    cli.add_argument("--vocoder-model")
    cli.add_argument(
        "--tts",
        choices=["none", "speecht5"],
        default="speecht5",
        help="Use local SpeechT5 or skip the spoken readback",
    )
    cli.add_argument(
        "--allow-downloads",
        action="store_true",
        help="Allow Hugging Face to download missing model files",
    )
    cli.add_argument("--output-dir", type=Path, default=ROOT / "outputs/latest")
    args = cli.parse_args()
    try:
        run(args)
    except (OSError, ValueError, ImportError) as exc:
        cli.exit(
            1, f"Could not process the input: {exc}\nSee README.md for model paths and dependencies.\n"
        )


if __name__ == "__main__":
    main()
