"""Turn a transcript into the form the word tagger was trained on.

The app and the evaluation both call prepare_for_tagger, so the tagger sees the
same kind of text in both. The original transcript is kept separately for
display. The tagger was trained on lower-case words with digits spelled out
("four five eight two", not "4582"), so raw digits confuse it.
"""

import re

DIGITS = {"0": "zero", "1": "one", "2": "two", "3": "three", "4": "four", "5": "five",
          "6": "six", "7": "seven", "8": "eight", "9": "nine"}

# Spellings of the same word that should count as one: the ATCO2 transcripts use
# both "alfa" and "alpha", and a callsign is equally right either way.
SPELLING_VARIANTS = {"alfa": "alpha", "niner": "nine", "juliet": "juliett", "whisky": "whiskey",
                     "oskar": "oscar", "fourty": "forty"}

RUNWAY_SIDE = {"l": "left", "r": "right", "c": "centre"}


def spell_digits(text):
    """Write each digit as a word so "4402" and "four four zero two" compare equal."""
    return re.sub(r"\d", lambda m: f" {DIGITS[m.group()]} ", text)


def expand_abbreviations(text):
    """Spell out forms a recogniser may write compactly: FL180 and runway 27L."""
    text = re.sub(r"\bFL\s?(\d{2,3})\b", r"flight level \1", text, flags=re.I)
    text = re.sub(r"\b(\d{1,2})\s?([LRC])\b", lambda m: f"{m.group(1)} {RUNWAY_SIDE[m.group(2).lower()]}",
                  text, flags=re.I)
    return text


def prepare_for_tagger(text):
    """Lower case, digits spelled out, punctuation removed, one spelling per word."""
    text = spell_digits(expand_abbreviations(text)).lower().replace("-", " ")
    return " ".join(SPELLING_VARIANTS.get(w, w) for w in re.sub(r"[^a-z0-9' ]", " ", text).split())
