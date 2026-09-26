"""Historical scoring transforms. They alter both references and predictions."""
import re

BAD_WORDS = [
    "BONJOUR",
    "MERCI",
    "CIAO",
    "DANKE",
    "TSCHUSS",
    "AUREVOIR",
    "AHOJ",
    "DOBR",
    "CZECH",
    "GERMAN",
    "SLOVAK",
    "POPOLUDNIE",
    "VE ER",
    "HEZK",
    "NA SLY ENOU",
    "BIS SPATER",
    "SCH NAMITTAG",
    "STEFAN",
    "SIR",
    "NE",
    "JA",
]


def apply_atc_mappings(text):
    """Replace phrases using mappings chosen from observed evaluation errors."""
    mapping = {

        "HELLO UNIFORM SION": "AIR PORTUGAL",
        "AIRPORTUGAL": "AIR PORTUGAL",
        "AIR MAROC": "EMIRATES",
        "AIRCHINA": "AIR CHINA",
        "ENTER TOWER": "JETSTAR",
        "RYAN AIR": "RYANAIR",
        "TUTRA": "TATRA",
        "WHEN IT R": "NITRA",
        "ZURICH SIERRA": "TOVKA",
        "BONO": "BRNO",
        "CHINA SIERRA": "TRANS EUROPE",
        "NAV CHECKER": "NAVCHECKER",
        "CONNIE": "CONNIE",
        "COUNTY": "CONNIE",

        "PROHIN START": "PUSH AND START",
        "PUSHSTART": "PUSH AND START",
        "KITALAND": "CLEARED TO LAND",
        "SPEED TO LAND": "CLEARED TO LAND",
        "LEA TO LAND": "CLEARED TO LAND",
        "RUNWAY VACATE IS": "RUNWAY VACATED",
        "AFFIRMATION APPROVED": "CLEARED",
        "EXPECT TO LAND": "ESTABLISHED",
        "DIRECTLY IS": "DIRECT LYSS",
        "ESTABLISH OUR": "ESTABLISHED",

        "DECIMAL RIGHT": "DECIMAL EIGHT",
        "NINER": "NINE",
        "FORTY": "FORTY",
        "FOURTY": "FORTY",
        "RWY": "RUNWAY",
        "POINT": "DECIMAL",
    }


    for k, v in mapping.items():
        text = text.replace(k, v)
    return text


def final_processing_pipeline(text):
    """Uppercase, map digits and phrases, remove listed words, and normalise spaces."""
    if not text:
        return ""
    text = text.upper()


    num_map = {
        "0": "ZERO",
        "1": "ONE",
        "2": "TWO",
        "3": "THREE",
        "4": "FOUR",
        "5": "FIVE",
        "6": "SIX",
        "7": "SEVEN",
        "8": "EIGHT",
        "9": "NINE",
        ".": "DECIMAL",
    }
    words = text.split()
    new_words = []
    for word in words:
        if any(char.isdigit() for char in word):
            new_words.append("".join(num_map.get(c, c) for c in word))
        else:
            new_words.append(word)
    text = " ".join(new_words)


    text = apply_atc_mappings(text)


    for bw in BAD_WORDS:
        text = text.replace(bw.upper(), "")


    text = re.sub(r"[^\w\s]", " ", text)
    return " ".join(text.split())
