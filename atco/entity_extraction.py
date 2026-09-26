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
        words = text.lower().replace("-", " ").split()
        res = []
        for w in words:
            if w in self.num_map:
                res.append(self.num_map[w])
            elif w.replace(".", "", 1).isdigit():
                res.append(w)
        return "".join(res)

    def validate_physics(self, command_word, value_str, context_text=""):
        warnings = []
        try:
            val_float = float(value_str)
            val_int = int(float(value_str))
        except ValueError:
            return warnings

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
            if val_int < 0 or val_int > 360:
                warnings.append(f"Heading {val_int} is outside the configured 0 to 360 range.")

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
