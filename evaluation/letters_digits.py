"""How well does a model hear spelled-out letters and digits?

Callsigns and registrations are spelled with the phonetic alphabet ("hotel delta
lima") and digits, and most callsign errors are one of those words heard wrong.
This scores only those words: each reference letter or digit word is aligned
with the model's transcript, and counts as right if it is transcribed exactly.

    python -m evaluation.letters_digits \\
        --model "2025=outputs/evaluation/2025_test/asr_predictions.jsonl:beam"

Spelling variants of the same letter or digit (alfa and alpha, niner and nine)
count as the same word. It also lists the commonest confusions.
"""

import argparse
import json
from collections import Counter
from pathlib import Path

LETTERS = {
    "alpha", "alfa", "bravo", "charlie", "delta", "echo", "foxtrot", "golf", "hotel", "india", "juliett",
    "juliet", "kilo", "lima", "mike", "november", "oscar", "papa", "quebec", "romeo", "sierra", "tango",
    "uniform", "victor", "whiskey", "whisky", "xray", "yankee", "zulu",
}
DIGITS = {"zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "niner"}
SPELLED = LETTERS | DIGITS
# Spelling variants of the same letter or digit. The ATCO2 transcripts write both
# "alfa" and "alpha", and "niner" where the speaker said niner; UWB-ATCC always
# writes "nine". A callsign is equally right either way.
SAME = {"alfa": "alpha", "niner": "nine", "juliet": "juliett", "whisky": "whiskey"}


def normalise(text):
    import jiwer

    from .evaluate_models import spell_digits

    plain = jiwer.Compose(
        [jiwer.ToLowerCase(), jiwer.RemovePunctuation(), jiwer.RemoveMultipleSpaces(), jiwer.Strip()]
    )
    # "x ray" is one letter; keep it as one word so it can be scored like the others.
    words = plain(spell_digits(text)).replace("x ray", "xray").split()
    return " ".join(SAME.get(w, w) for w in words)


def score(records, mode):
    """Count reference letter and digit words transcribed exactly, and the confusions."""
    import jiwer

    right = total = 0
    letters_right = letters_total = 0
    confusions = Counter()
    for r in records:
        ref, hyp = normalise(r["reference"]), normalise(r[mode])
        if not ref:
            continue
        ref_words, hyp_words = ref.split(), hyp.split() if hyp else []
        if not hyp_words:
            hyp_words = [""]
        out = jiwer.process_words(ref, hyp if hyp else "<empty>")
        for chunk in out.alignments[0]:
            for k in range(chunk.ref_end_idx - chunk.ref_start_idx):
                word = ref_words[chunk.ref_start_idx + k]
                if word not in SPELLED:
                    continue
                ok = chunk.type == "equal"
                total += 1
                right += ok
                if word in LETTERS:
                    letters_total += 1
                    letters_right += ok
                if chunk.type == "substitute":
                    heard = out.hypotheses[0][chunk.hyp_start_idx + k]
                    confusions[(word, heard)] += 1
                elif chunk.type == "delete":
                    confusions[(word, "(missed)")] += 1
    return dict(
        spelled_words=total,
        right=right,
        accuracy=round(right / total, 4) if total else None,
        letters=letters_total,
        letters_right=letters_right,
        letter_accuracy=round(letters_right / letters_total, 4) if letters_total else None,
        commonest_confusions=[f"{a} -> {b} ({n})" for (a, b), n in confusions.most_common(12)],
    )


def main():
    cli = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    cli.add_argument("--model", action="append", required=True, help='"Name=path/to/asr_predictions.jsonl:mode"')
    cli.add_argument("--output", type=Path)
    args = cli.parse_args()
    summary = {}
    for spec in args.model:
        name, rest = spec.split("=", 1)
        path, mode = rest.rsplit(":", 1)
        rows = [json.loads(s) for s in Path(path).read_text().splitlines() if s.strip()]
        summary[name] = dict(decoding=mode, **score(rows, mode))
        s = summary[name]
        print(f"{name:18} spelled words {s['right']:3}/{s['spelled_words']:3} = {s['accuracy']:.4f}   "
              f"letters {s['letters_right']:3}/{s['letters']:3} = {s['letter_accuracy']:.4f}")
    if args.output:
        args.output.write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
