"""Reference-label rules for the separate TensorFlow BERT result."""
import os
import json
import re

ID2LABEL = {
    0: "O",
    1: "B-CALLSIGN",
    2: "I-CALLSIGN",
    3: "B-COMMAND",
    4: "I-COMMAND",
    5: "B-VALUE",
    6: "I-VALUE",
    7: "B-WAYPOINT",
    8: "I-WAYPOINT",
}


ENTITY_LABELS = ["O", "CALLSIGN", "COMMAND", "VALUE", "WAYPOINT"]


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
    "startup",
    "push",
    "start",
    "stop",
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


VALUE_MODIFIERS = {"flight", "level", "left", "right", "center", "centre"}


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
}


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
            elif isinstance(cs, dict) and "spoken" in cs:
                local_callsigns.add(cs["spoken"].split()[0].lower())
    local_waypoints = set(local_context.get("waypoints", []))

    i = 0
    while i < len(words):
        w_lower = words[i].lower()
        w_upper = words[i].upper()
        # CALLSIGN
        is_callsign_start = (w_lower in AIRLINES) or (w_lower in local_callsigns)
        if not is_callsign_start and is_nato(w_lower):
            next_is = (i + 1 < len(words)) and (
                is_nato(words[i + 1]) or is_number_or_modifier(words[i + 1])
            )
            if next_is:
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
        # COMMAND
        if w_lower in ATC_COMMANDS:
            labels[i] = "B-COMMAND"
            i += 1
            continue
        if w_lower == "line" and (i + 1 < len(words)) and words[i + 1].lower() == "up":
            labels[i], labels[i + 1] = "B-COMMAND", "I-COMMAND"
            i += 2
            continue
        # VALUE
        if w_lower == "flight" and (i + 1 < len(words)) and words[i + 1].lower() == "level":
            labels[i], labels[i + 1] = "B-VALUE", "I-VALUE"
            i += 2
            j = 0
            while (i + j) < len(words) and is_number_or_modifier(words[i + j]):
                labels[i + j] = "I-VALUE"
                j += 1
            i += j
            continue
        if is_number_or_modifier(w_lower):
            labels[i] = "B-VALUE" if labels[i] == "O" else labels[i]
            j = 1
            while (i + j) < len(words) and is_number_or_modifier(words[i + j]):
                labels[i + j] = "I-VALUE"
                j += 1
            i += j
            continue
        # WAYPOINT
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


def load_raw_and_tag(filepath):
    """Load transcripts and generate reference labels with the labelling rules."""
    if not os.path.exists(filepath):
        print(f"找不到文件: {filepath}")
        return []
    samples = []
    with open(filepath, "r", encoding="utf-8") as f:
        try:
            raw_data = json.load(f)
            if not isinstance(raw_data, list):
                raw_data = [raw_data]
            for session in raw_data:
                context = session.get("context", {})
                for turn in session.get("dialogue", []):
                    text = turn.get("text", "")
                    if not text:
                        continue
                    tokens, tags = tag_entities(text, context)

                    if any(t != "O" for t in tags):
                        samples.append({"tokens": tokens, "ner_tags": tags})
        except Exception as e:
            print(f"数据读取错误: {e}")
    print(f'成功加载: {len(samples)} 条样本')
    return samples


def convert_bio_to_entity(label):
    """Remove the B- and I- prefixes from token labels."""
    if label == "O":
        return "O"
    return label.split("-")[1]
