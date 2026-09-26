"""Rules that reproduce the application model’s stored entity labels."""
import re

AIRLINES = {
    "jetstar",
    "qantas",
    "velocity",
    "virgin",
    "rex",
    "united",
    "singapore",
    "emirates",
    "cathay",
    "air",
    "china",
    "delta",
    "american",
    "shamrock",
    "speedbird",
    "lufthansa",
    "korean",
    "japan",
    "ana",
    "fedex",
    "ups",
    "british",
    "airways",
    "klm",
    "jump",
    "run",
    "medevac",
    "rescue",
    "polair",
    "chopper",
    "cessna",
    "piper",
    "tiger",
    "fiji",
    "tasman",
    "link",
    "nz",
    "new",
    "zealand",
    "bonza",
    "vista",
    "jet",
}


ICAO_PHONETICS = {
    "alfa",
    "alpha",
    "bravo",
    "charlie",
    "delta",
    "echo",
    "foxtrot",
    "golf",
    "hotel",
    "india",
    "juliett",
    "juliet",
    "kilo",
    "lima",
    "mike",
    "november",
    "oscar",
    "papa",
    "quebec",
    "romeo",
    "sierra",
    "tango",
    "uniform",
    "victor",
    "whiskey",
    "x-ray",
    "xray",
    "yankee",
    "zulu",
}


ATC_COMMANDS = {
    "contact",
    "climb",
    "descend",
    "turn",
    "maintain",
    "cleared",
    "land",
    "takeoff",
    "hold",
    "report",
    "squawk",
    "approach",
    "direct",
    "heading",
    "cross",
    "track",
    "identified",
    "cancel",
    "verify",
    "correction",
    "wind",
    "runway",
    "taxi",
    "pushback",
    "line",
    "up",
    "wait",
    "go",
    "around",
    "continue",
    "expect",
    "speed",
    "reduce",
    "increase",
    "intercept",
    "establish",
    "pass",
    "enter",
    "vacate",
    "monitor",
    "approve",
    "approved",
    "check",
    "confirm",
    "negative",
    "affirm",
    "leaving",
    "reaching",
}


NUMBERS_AND_UNITS = {
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "zero",
    "ten",
    "eleven",
    "twelve",
    "thirteen",
    "fourteen",
    "fifteen",
    "sixteen",
    "thousand",
    "hundred",
    "decimal",
    "point",
    "degrees",
    "knots",
    "feet",
}


STOPWORDS = {
    "hello",
    "hi",
    "good",
    "morning",
    "afternoon",
    "evening",
    "day",
    "bye",
    "goodbye",
    "thank",
    "thanks",
    "you",
    "your",
    "we",
    "us",
    "sorry",
    "sir",
    "madam",
    "mister",
    "roger",
    "wilco",
    "affirm",
    "negative",
    "hes",
    "spk",
    "ne",
    "unk",
    "ukn",
    "break",
    "correction",
    "standby",
    "german",
    "french",
    "czech",
    "english",
    "bonjour",
    "ahoj",
    "servus",
    "affirmation",
    "nav",
    "checker",
    "beste",
    "schone",
    "yeah",
    "ok",
    "okay",
    "correction",
    "report",
    "identified",
    "final",
    "ready",
    "departure",
    "arrival",
}


VALUE_MODIFIERS = {"flight", "level", "left", "right", "center", "centre"}


def is_number_or_modifier(w):
    return w.lower() in NUMBERS_AND_UNITS or w.isdigit() or w.lower() in VALUE_MODIFIERS


def is_nato(w):
    return w.lower() in ICAO_PHONETICS


def tag_entities(text, local_context):
    """Assign BIO labels using the existing vocabulary and context rules."""
    clean_text = text.replace(",", "").replace(".", "").replace("?", "").replace("!", "")
    words = clean_text.split()
    labels = ["O"] * len(words)


    local_callsigns = set()
    if "callsigns" in local_context:
        for cs in local_context["callsigns"]:
            if isinstance(cs, str):
                local_callsigns.add(cs.split()[0].lower())
            elif isinstance(cs, dict):
                local_callsigns.add(cs["spoken"].split()[0].lower())

    local_waypoints = set(local_context.get("waypoints", []))

    i = 0
    while i < len(words):
        w_lower = words[i].lower()
        w_upper = words[i].upper()


        is_callsign_start = (w_lower in AIRLINES) or (w_lower in local_callsigns)

        if not is_callsign_start and is_nato(w_lower):
            next_is_nato_or_num = (i + 1 < len(words)) and (
                is_nato(words[i + 1]) or is_number_or_modifier(words[i + 1])
            )
            if next_is_nato_or_num:
                is_callsign_start = True

        if is_callsign_start:
            labels[i] = "B-CALLSIGN"
            j = 1
            while (i + j) < len(words):
                next_w = words[i + j].lower()
                if is_number_or_modifier(next_w) or is_nato(next_w) or (next_w in AIRLINES):
                    labels[i + j] = "I-CALLSIGN"
                    j += 1
                else:
                    break
            i += j
            continue


        if w_lower == "line" and (i + 1 < len(words)) and words[i + 1].lower() == "up":
            labels[i] = "B-COMMAND"
            labels[i + 1] = "I-COMMAND"
            i += 2
            continue
        if w_lower == "go" and (i + 1 < len(words)) and words[i + 1].lower() == "around":
            labels[i] = "B-COMMAND"
            labels[i + 1] = "I-COMMAND"
            i += 2
            continue

        if w_lower in ATC_COMMANDS:
            labels[i] = "B-COMMAND"
            i += 1
            continue


        if w_lower == "flight" and (i + 1 < len(words)) and words[i + 1].lower() == "level":
            labels[i] = "B-VALUE"
            labels[i + 1] = "I-VALUE"
            j = 2
            while (i + j) < len(words) and is_number_or_modifier(words[i + j]):
                labels[i + j] = "I-VALUE"
                j += 1
            i += j
            continue

        if is_number_or_modifier(w_lower):
            if labels[i] == "O":
                labels[i] = "B-VALUE"
                j = 1
                while (i + j) < len(words) and is_number_or_modifier(words[i + j]):
                    labels[i + j] = "I-VALUE"
                    j += 1
                i += j
                continue


        if w_upper in local_waypoints or (
            w_upper.isupper()
            and len(w_upper) >= 3
            and w_upper.isalpha()
            and w_lower not in ATC_COMMANDS
            and w_lower not in STOPWORDS
        ):
            labels[i] = "B-WAYPOINT"
            i += 1
            continue

        i += 1

    return words, labels
