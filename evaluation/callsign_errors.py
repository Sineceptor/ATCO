"""Why does the best model miss callsigns? Sort each missed callsign by cause.

The tagger is run on the correct transcript and on the model's transcript, as in
evaluation.extraction_agreement. Each callsign found in the correct transcript
but not in the model's is aligned with the model's transcript and sorted into:

- spelling only: the tagger found the same callsign in the model's transcript,
  spelled differently (alfa and alpha, oskar and oscar)
- tagger: the model's transcript contains it word for word, but the tagger
  labelled it differently there
- letters or digits: only phonetic-alphabet letters or number words were wrong,
  missing or added
- airline word: the airline's name (the first word) was misheard
- other words: some other word of the callsign was misheard
- mostly lost: more than half of its words are missing or wrong

    python -m evaluation.callsign_errors \\
        --model "soup_medium=outputs/evaluation/soup_medium_test/asr_predictions.jsonl:beam"

Only counts are saved, not transcripts, because the ATCO2 transcripts are not
redistributed.
"""

import argparse
import json
from collections import Counter
from pathlib import Path

from .extraction_agreement import ROOT, extract, normalise
from .letters_digits import SAME, SPELLED


NUMBERS = {"ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen",
           "nineteen", "twenty", "thirty", "forty", "fourty", "fifty", "sixty", "seventy", "eighty", "ninety",
           "hundred", "thousand"}


# Spellings in the ATCO2 transcripts that are the same word as the model's.
VARIANTS = {**SAME, "oskar": "oscar", "fourty": "forty"}


def canonical(words):
    return [VARIANTS.get(w, w) for w in words]


def align(ref, hyp):
    """Word-level edit operations turning ref into hyp: (op, ref word, hyp word)."""
    rows, cols = len(ref) + 1, len(hyp) + 1
    cost = [[0] * cols for _ in range(rows)]
    for i in range(rows):
        cost[i][0] = i
    for j in range(cols):
        cost[0][j] = j
    for i in range(1, rows):
        for j in range(1, cols):
            cost[i][j] = min(cost[i - 1][j] + 1, cost[i][j - 1] + 1,
                             cost[i - 1][j - 1] + (ref[i - 1] != hyp[j - 1]))
    ops, i, j = [], len(ref), len(hyp)
    while i or j:
        if i and j and cost[i][j] == cost[i - 1][j - 1] + (ref[i - 1] != hyp[j - 1]):
            ops.append(("equal" if ref[i - 1] == hyp[j - 1] else "sub", ref[i - 1], hyp[j - 1]))
            i, j = i - 1, j - 1
        elif i and cost[i][j] == cost[i - 1][j] + 1:
            ops.append(("del", ref[i - 1], None))
            i -= 1
        else:
            ops.append(("ins", None, hyp[j - 1]))
            j -= 1
    return cost[-1][-1], ops[::-1]


def cause(callsign, hypothesis, found):
    """Compare the callsign with the stretch of the transcript that matches it best.

    found: the callsigns the tagger extracted from the model's transcript.
    """
    ref = canonical(callsign.replace("x ray", "xray").split())
    if any(canonical(c.replace("x ray", "xray").split()) == ref for c in found):
        return "spelling only"
    hyp = canonical(hypothesis.replace("x ray", "xray").split())
    best = None
    for length in range(max(1, len(ref) - 2), len(ref) + 3):
        for start in range(0, max(1, len(hyp) - length + 1)):
            distance, ops = align(ref, hyp[start:start + length])
            if best is None or (distance, abs(length - len(ref))) < best[:2]:
                best = (distance, abs(length - len(ref)), ops)
    distance, _, ops = best
    if distance == 0:
        return "tagger"
    if distance > len(ref) / 2:
        return "mostly lost"
    # What counts is which of the callsign's own words were lost or heard wrong,
    # plus any words wrongly added inside it.
    wrong = [r for op, r, _ in ops if op in ("sub", "del")] + [h for op, _, h in ops if op == "ins"]
    if all(w in SPELLED or w in NUMBERS for w in wrong):
        return "letters or digits"
    if ops[0][0] in ("sub", "del"):
        return "airline word"
    return "other words"


def main():
    cli = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    cli.add_argument("--model", action="append", required=True, help='"Name=path/to/asr_predictions.jsonl:mode"')
    cli.add_argument("--tagger", type=Path, default=ROOT / "models/distilbert")
    cli.add_argument("--output", type=Path)
    cli.add_argument("--show", action="store_true", help="Print each missed callsign (not saved)")
    args = cli.parse_args()

    from atco.entity_extraction import AtcParser

    pipe = AtcParser(args.tagger, local_files_only=True).pipe
    summary = dict(method=__doc__.split("\n\n")[1].strip(), models={})
    for spec in args.model:
        name, rest = spec.split("=", 1)
        path, mode = rest.rsplit(":", 1)
        rows = [json.loads(s) for s in Path(path).read_text().splitlines() if s.strip()]
        kinds, clips = Counter(), set()
        for r in rows:
            ref_text, hyp_text = normalise(r["reference"]), normalise(r[mode])
            hyp_fields = extract(pipe, hyp_text)
            found = [w for f, w in hyp_fields if f == "CALLSIGN"]
            missed = (extract(pipe, ref_text) - hyp_fields).elements()
            for field, words in missed:
                if field != "CALLSIGN":
                    continue
                kind = cause(words, hyp_text, found)
                kinds[kind] += 1
                clips.add(r["file"])
                if args.show:
                    print(f"{kind:18} | {words} | {hyp_text}")
        summary["models"][name] = dict(decoding=mode, missed_callsigns=sum(kinds.values()),
                                       clips_with_a_missed_callsign=len(clips), by_cause=dict(kinds.most_common()))
        print(name, json.dumps(summary["models"][name]))
    if args.output:
        args.output.write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
