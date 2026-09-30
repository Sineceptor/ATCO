"""DistilBERT entity extraction and the saved command heuristics."""
from pathlib import Path
import re

MODEL_DIR = Path(__file__).resolve().parents[1] / "models/distilbert"


def repair_bert_output(predictions):
    """Join WordPiece fragments, keeping the preceding entity label."""
    if not predictions:
        return []

    merged_entities = []

    for pred in predictions:
        word = pred["word"]
        score = pred["score"]

        # WordPiece continuation tokens start with ##.
        if word.startswith("##"):
            if merged_entities:
                prev = merged_entities[-1]
                prev["word"] += word.replace("##", "")
                # Keep the larger score when joining subwords.
                prev["score"] = max(prev["score"], score)
            else:
                # Handle a fragment without a preceding word.
                pred["word"] = word.replace("##", "")
                merged_entities.append(pred)
        else:
            merged_entities.append(pred)

    return merged_entities


# English function words are never waypoints. The word-list rules the tagger was
# trained on label any unfamiliar word of three or more letters as a waypoint, so
# the model learned to tag words like "and" as waypoints; the app drops those.
FUNCTION_WORDS = {
    "a", "an", "and", "the", "on", "in", "at", "to", "for", "of", "with", "from", "via",
    "then", "or", "is", "are", "be", "it", "this", "that", "as", "by", "now",
}


def drop_function_word_waypoints(entities):
    """Remove waypoint labels from function words; keep every other entity as it is."""
    return [e for e in entities
            if not (e["entity_group"] == "WAYPOINT" and e["word"].strip().lower() in FUNCTION_WORDS)]


DIRECTIONS = {"left", "right", "centre", "center", "l", "r", "c"}

# Heuristic checks from the original experiment.


class CommandChecks:
    def __init__(self):
        self.num_map = {
            "one": "1",
            "two": "2",
            "three": "3",
            "four": "4",
            "five": "5",
            "six": "6",
            "seven": "7",
            "eight": "8",
            "nine": "9",
            "zero": "0",
            "niner": "9",
            "fife": "5",
            "thousand": "000",
            "hundred": "00",
            "decimal": ".",
            "point": ".",
        }
        # Tens words, so "seventy seven" and "one sixty" read as numbers.
        self.TENS = {"ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15,
                     "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19, "twenty": 20,
                     "thirty": 30, "forty": 40, "fourty": 40, "fifty": 50, "sixty": 60, "seventy": 70,
                     "eighty": 80, "ninety": 90}
        # Check these words even if the model gives them a different label.
        self.force_commands = [
            "squawk",
            "heading",
            "speed",
            "runway",
            "rwy",
            "contact",
            "climb",
            "descend",
        ]

    def _text_to_digit(self, text):
        """Read a spoken number; returns "" if the text holds no number.

        Digit by digit ("one two one decimal eight") joins the digits: "121.8". With
        "hundred" or "thousand" it is read as a quantity, so "one thousand five hundred"
        is 1500, not 1000500, and "five thousand four eight zero" is 5480. A tens word
        joins the digit after it: "seventy seven" is 77, "one sixty" is 160.
        """
        words = [w for w in text.lower().replace("-", " ").split()
                 if w in self.num_map or w in self.TENS or w.replace(".", "", 1).isdigit()]
        if not words:
            return ""
        if any(w in ("hundred", "thousand") for w in words):
            total = base = current = 0
            after_tens = False
            for w in words:
                if w == "thousand":
                    total += ((base + current) or 1) * 1000
                    base = current = 0
                    after_tens = False
                elif w == "hundred":
                    base += (current or 1) * 100
                    current = 0
                    after_tens = False
                elif w in self.TENS:
                    value = self.TENS[w]
                    current = value if current == 0 else current * 100 + value
                    after_tens = value >= 20
                elif w.isdigit() or self.num_map.get(w, "").isdigit():
                    digits = w if w.isdigit() else self.num_map[w]
                    current = current + int(digits) if after_tens else current * 10 ** len(digits) + int(digits)
                    after_tens = False
            return str(total + base + current)
        out = ""
        after_tens = False
        for w in words:
            if w in self.TENS:
                out += str(self.TENS[w])
                after_tens = self.TENS[w] >= 20
                continue
            piece = self.num_map.get(w, w)
            if after_tens and piece.isdigit() and len(piece) == 1:
                out = out[:-1] + piece
            else:
                out += piece
            after_tens = False
        return out

    def validate_physics(self, command_word, value_str, context_text=""):
        warnings = []
        try:
            val_float = float(value_str)
            val_int = int(float(value_str))
        except ValueError:
            return [f"Could not read a number from '{value_str}' after '{command_word}'; not checked."]

        cmd = command_word.lower()

        # Squawk code checks.

        if "squawk" in cmd:
            # Check the digits as spoken: converting to a number would drop a
            # leading zero and make a valid code such as 0421 look three digits long.
            str_val = value_str.strip()
            if len(str_val) != 4 or not str_val.isdigit():
                warnings.append(f"Squawk code should have four digits; got {str_val}.")

            # Squawk codes use octal digits.
            if "8" in str_val or "9" in str_val:
                warnings.append(f"Squawk code contains 8 or 9; only digits 0 to 7 are allowed.")

            # Special squawk codes.
            if val_int == 7500:
                warnings.append("Special squawk code 7500: unlawful interference.")
            if val_int == 7600:
                warnings.append("Special squawk code 7600: radio failure.")
            if val_int == 7700:
                warnings.append("Special squawk code 7700: emergency.")

        # Heading.
        elif "heading" in cmd or "turn" in cmd:
            # Check the value as read, not a truncated one: 360.5 is not a heading.
            if val_float != val_int:
                warnings.append(f"Heading {value_str} is not a whole number of degrees.")
            if val_float < 0 or val_float > 360:
                warnings.append(f"Heading {value_str} is outside the configured 0 to 360 range.")

        # Frequency.
        elif "contact" in cmd or "centre" in cmd or "approach" in cmd:
            if 118.0 <= val_float <= 137.0:
                pass  # Within the range used by this check.
            elif 108.0 <= val_float < 118.0:
                warnings.append(
                    f"Frequency {val_float} falls in the navigation range used by this check."
                )
            else:
                warnings.append(
                    f"Frequency {val_float} is outside the configured 118 to 137 range."
                )

        # Altitude.
        elif any(x in cmd for x in ["climb", "descend", "level"]):
            # This is a heuristic, not a unit parser.
            is_fl = (
                "flight level" in context_text.lower()
                or "fl" in context_text.lower()
                or val_int < 500
            )

            if is_fl:
                if val_int > 500:
                    warnings.append(f"Flight level {val_int} exceeds the configured limit of 500.")
            else:
                if val_int > 60000:
                    warnings.append(
                        f"Altitude {val_int} ft exceeds the configured limit of 60000 ft."
                    )
                if val_int < 0:
                    warnings.append(f"Altitude is negative.")

        # Speed thresholds used in the original experiment.
        elif "speed" in cmd or "knots" in context_text.lower():
            if val_int < 50:
                warnings.append(f"Speed {val_int} kt is below the configured threshold of 50 kt.")
            if val_int > 1200:
                warnings.append(f"Speed {val_int} kt exceeds the configured threshold of 1200 kt.")

        # Runway number.
        elif "runway" in cmd or "rwy" in cmd:
            # Use the first two digits.
            try:
                rwy = int(str(val_int)[:2])
                if rwy < 1 or rwy > 36:
                    warnings.append(f"Runway number {rwy} is outside 1 to 36.")
            except:
                pass

        return warnings

    def check(self, entities):
        warnings = []
        parsed_text_parts = []

        for i, ent in enumerate(entities):
            word = ent["word"]
            label = ent["entity_group"]

            display_word = word
            if label == "VALUE":
                display_word = self._text_to_digit(word)
                # Include FL when the preceding entity says flight level.
                if (
                    i > 0
                    and "flight level" in entities[i - 1]["word"].lower()
                    and not display_word.startswith("FL")
                ):
                    display_word = "FL" + display_word

            parsed_text_parts.append(display_word.upper())

            # Use the predicted label or a recognised command word.
            is_command_logic = (label == "COMMAND") or (word.lower() in self.force_commands)

            if is_command_logic:
                next_val_str = None
                raw_context = ""

                # Find the next value, stopping at another command.
                for j in range(i + 1, len(entities)):
                    neighbor = entities[j]
                    if neighbor["entity_group"] == "VALUE":
                        next_val_str = self._text_to_digit(neighbor["word"])
                        raw_context = neighbor["word"]  # Keep the text for the unit heuristic.
                        break
                    # Do not match a value across two commands.
                    if (
                        neighbor["entity_group"] == "COMMAND"
                        or neighbor["word"].lower() in self.force_commands
                    ):
                        break

                if next_val_str:
                    w = self.validate_physics(word, next_val_str, raw_context)
                    warnings.extend(w)
                elif raw_context and not set(raw_context.lower().split()) <= DIRECTIONS:
                    # A value was found but no number could be read from it: say so,
                    # rather than silently skipping the check. A direction ("turn right")
                    # is a valid value that simply isn't a number.
                    warnings.append(f"Could not read a number from '{raw_context}' after '{word}'; not checked.")

        return " ".join(parsed_text_parts), warnings


class AtcParser:
    def __init__(self, model_dir=None, local_files_only=False):
        model_dir = model_dir or MODEL_DIR
        print(f"Loading entity model: {model_dir}")
        # Imported here so the rule checks above can be used and tested without them.
        import torch
        from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification

        device = 0 if torch.cuda.is_available() else -1
        tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=local_files_only)
        model = AutoModelForTokenClassification.from_pretrained(
            model_dir, local_files_only=local_files_only
        )
        self.pipe = pipeline(
            "token-classification",
            model=model,
            tokenizer=tokenizer,
            aggregation_strategy="simple",
            device=device,
        )
        self.guard = CommandChecks()
        print("Entity model loaded.")
