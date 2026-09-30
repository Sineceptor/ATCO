"""How often does a transcription error change what the tagger extracts?

For each clip, the app's DistilBERT tagger is run twice: on the reference
transcript and on a model's transcript. The two sets of extracted fields
(callsigns, commands, values, waypoints) are compared, with and without
waypoints.

    python -m evaluation.extraction_agreement \\
        --model "2025=outputs/evaluation/2025_test/asr_predictions.jsonl:beam"

This is not a measure of the tagger's own accuracy: its labels on the
reference transcript are taken as the target, and those can be wrong too. It
measures how much of a speech model's error survives into the structured
output that a person or program would act on.
"""

import argparse
import json
import re
from collections import Counter
from pathlib import Path

from atco.text_prep import prepare_for_tagger

ROOT = Path(__file__).resolve().parents[1]
FIELDS = ["CALLSIGN", "COMMAND", "VALUE", "WAYPOINT"]


def normalise(text):
    """The form the tagger was trained on; the app uses the same step (atco.text_prep)."""
    return prepare_for_tagger(text)


def extract(pipe, text):
    from atco.entity_extraction import repair_bert_output

    if not text.strip():
        return []
    found = repair_bert_output(pipe(text))
    return [(e["entity_group"], " ".join(e["word"].lower().split())) for e in found]


def compare(pairs):
    """pairs: list of (reference fields, model fields), each an ordered list per clip.

    The main counts compare the fields as unordered collections: the same callsign,
    command and number strings, wherever they appear. The in-order count also needs
    them in the same order, which is closer to "the same instruction for the same
    aircraft" when a clip holds more than one instruction.
    """
    # Waypoint labels come from a rule that is known to be too loose, so the
    # headline count leaves them out; the count with them is kept alongside.
    core = lambda c: Counter({k: v for k, v in c.items() if k[0] != "WAYPOINT"})
    in_order = lambda fields: [f for f in fields if f[0] != "WAYPOINT"]
    out = dict(
        clips=len(pairs),
        all_fields_identical=sum(Counter(r) == Counter(h) for r, h in pairs),
        callsign_command_value_identical=sum(core(Counter(r)) == core(Counter(h)) for r, h in pairs),
        callsign_command_value_identical_in_order=sum(in_order(r) == in_order(h) for r, h in pairs),
    )
    pairs = [(Counter(r), Counter(h)) for r, h in pairs]
    for field in FIELDS:
        tp = fp = fn = 0
        clips_with_field = identical = 0
        for ref, hyp in pairs:
            r = Counter({k: v for k, v in ref.items() if k[0] == field})
            h = Counter({k: v for k, v in hyp.items() if k[0] == field})
            tp += sum((r & h).values())
            fp += sum((h - r).values())
            fn += sum((r - h).values())
            if r:
                clips_with_field += 1
                identical += r == h
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        out[field.lower()] = dict(
            reference_items=tp + fn,
            recall=round(recall, 4),
            precision=round(precision, 4),
            clips_with_field=clips_with_field,
            clips_where_it_matches=identical,
        )
    return out


def main():
    cli = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    cli.add_argument("--model", action="append", required=True,
                     help='"Name=path/to/asr_predictions.jsonl:mode"')
    cli.add_argument("--only", type=Path, help="Keep only the clips in this manifest")
    cli.add_argument("--tagger", type=Path, default=ROOT / "models/distilbert")
    cli.add_argument("--train", type=Path, default=ROOT / "datasets/speech_split/train.jsonl",
                     help="Training transcripts, to check whether missed callsigns start with a known word")
    cli.add_argument("--output", type=Path)
    args = cli.parse_args()

    from atco.entity_extraction import AtcParser

    pipe = AtcParser(args.tagger, local_files_only=True).pipe
    keep = None
    if args.only:
        keep = {json.loads(s)["audio_file"] for s in args.only.read_text().splitlines() if s.strip()}
    summary = dict(
        tagger=str(args.tagger.name),
        method="DistilBERT run on the reference and on each model's transcript; extracted "
        "(field, words) pairs compared per clip. Text lower-cased, digits spelled out, punctuation removed.",
        models={},
    )
    known = {w for s in args.train.read_text().splitlines() if s.strip()
             for w in json.loads(s)["ASR_transcript_clean"].split()}
    for spec in args.model:
        name, rest = spec.split("=", 1)
        path, mode = rest.rsplit(":", 1)
        rows = [json.loads(s) for s in Path(path).read_text().splitlines() if s.strip()]
        rows = [r for r in rows if keep is None or r["file"] in keep]
        pairs = [(extract(pipe, normalise(r["reference"])), extract(pipe, normalise(r[mode]))) for r in rows]
        summary["models"][name] = dict(decoding=mode, **compare(pairs))
        missed = [w for ref, hyp in pairs for (field, w) in (Counter(ref) - Counter(hyp)).elements() if field == "CALLSIGN"]
        summary["models"][name]["missed_callsigns"] = dict(
            count=len(missed),
            starting_with_a_word_never_in_training=sum(w.split()[0] not in known for w in missed),
        )
        m = summary["models"][name]
        print(f"{name:18} clips {m['clips']:3}  all fields identical {m['all_fields_identical']:3}  "
              + "  ".join(f"{f.lower()} recall {m[f.lower()]['recall']:.2f}" for f in FIELDS), flush=True)
    if args.output:
        args.output.write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
