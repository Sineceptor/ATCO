"""SpeechT5 readback with the original ATC pronunciation rules."""
import os
import re
from pathlib import Path


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = Path(__file__).resolve().parents[1] / "outputs/readback"


class AtcTextNormalizer:
    def __init__(self):
        self.digit_map = {
            "0": "zero",
            "1": "one",
            "2": "two",
            "3": "three",
            "4": "four",
            "5": "five",
            "6": "six",
            "7": "seven",
            "8": "eight",
            "9": "niner",
        }

        # Pronunciations used in the original speech experiment.
        self.phonetic_map = {
            "ILS": "Eye. Ell. Ess.",
            "VOR": "Vee. Oh. Are.",
            "KLM": "Kay. Ell. Em.",
            "DME": "Dee. Em. Ee.",
            "NDB": "En. Dee. Bee.",
            "QNH": "Que. En. Aitch.",
            "RWY": "runway",
            "FL": "flight level",
            "KTS": "knots",
            "CTC": "contact",
        }

        self.airline_map = {
            "CSN": "China Southern,",
            "QFA": "Qantas,",
            "JST": "Jetstar,",
            "UAL": "United,",
            "BAW": "Speedbird,",
        }

        self.rwy_suffix_map = {"L": "left", "R": "right", "C": "center"}

    def _expand_digits(self, text):
        result = []
        for char in text:
            if char in self.digit_map:
                result.append(self.digit_map[char])
            else:
                result.append(char)
        return " ".join(result)

    def normalize(self, text):
        # Read decimal points between digits aloud.
        if "." in text:
            text = re.sub(r"(?<=\d)\.(?=\d)", " decimal ", text)

        text = text.upper()

        # Compact forms: FL180 is "flight level one eight zero", not just the digits.
        text = re.sub(r"\bFL\s?(\d{2,3})\b", lambda m: "FLIGHT LEVEL " + self._expand_digits(m.group(1)), text)

        # Expand the listed abbreviations.
        for term, phonetic in self.phonetic_map.items():
            text = re.sub(r"\b" + term + r"\b", phonetic, text)

        # Airline callsigns.
        def replace_callsign(match):
            icao = match.group(1)
            digits = match.group(2)
            airline = self.airline_map.get(icao, icao)
            return f"{airline} {self._expand_digits(digits)}"

        text = re.sub(r"\b([A-Z]{3})(\d{1,4})\b", replace_callsign, text)

        # Runway numbers and left/right/centre suffixes.
        def replace_runway(match):
            num = match.group(1)
            suffix = match.group(2)
            spoken_num = self._expand_digits(num)
            spoken_suffix = (", " + self.rwy_suffix_map.get(suffix, "")) if suffix else ""
            return f"runway {spoken_num}{spoken_suffix}"

        text = re.sub(r"\bRWY(\d{1,2})([LRC]?)\b", replace_runway, text)

        # A runway side written after the number, joined or spaced ("27L", "27 L"),
        # is spoken, not dropped with the other letters.
        def replace_side(match):
            return f"{self._expand_digits(match.group(1))}, {self.rwy_suffix_map[match.group(2)]}"

        text = re.sub(r"\b(\d{1,2})\s?([LRC])\b", replace_side, text)

        # Read remaining digits separately.
        words = text.split()
        final_words = []

        for w in words:
            if w.isdigit():
                final_words.append(self._expand_digits(w))
            elif any(c.isdigit() for c in w) and not w.startswith("decimal"):
                digits_only = "".join([c for c in w if c.isdigit()])
                if digits_only:
                    final_words.append(self._expand_digits(digits_only))
            else:
                if w[0].isupper() and "." in w:
                    final_words.append(w)
                else:
                    final_words.append(w.lower())

        return " ".join(final_words) + " ..."


class AtcSpeaker:
    def __init__(
        self,
        model_path="microsoft/speecht5_tts",
        vocoder_path="microsoft/speecht5_hifigan",
        output_dir=OUTPUT_DIR,
        local_files_only=False,
    ):
        self.output_dir = os.fspath(output_dir)
        # Imported here so the text rules above can be used and tested without them.
        global torch, sf, SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
        import torch
        import soundfile as sf
        from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan

        print("Loading SpeechT5...")
        self.processor = SpeechT5Processor.from_pretrained(
            model_path, local_files_only=local_files_only
        )
        self.model = SpeechT5ForTextToSpeech.from_pretrained(
            model_path, local_files_only=local_files_only
        ).eval()
        self.vocoder = SpeechT5HifiGan.from_pretrained(
            vocoder_path, local_files_only=local_files_only
        ).eval()

        # Keep the original seed-42 random speaker embedding.
        torch.manual_seed(42)
        self.speaker_embeddings = torch.randn(1, 512)

        self.normalizer = AtcTextNormalizer()
        print("Speech model loaded.")

    def speak(self, atc_code_text, filename="output.wav"):
        spoken_text = self.normalizer.normalize(atc_code_text)

        print(f"Readback text: {atc_code_text}")
        print(f"Spoken text: {spoken_text}")

        inputs = self.processor(text=spoken_text, return_tensors="pt")

        with torch.no_grad():
            speech = self.model.generate_speech(
                inputs["input_ids"], self.speaker_embeddings, vocoder=self.vocoder
            )

        os.makedirs(self.output_dir, exist_ok=True)
        save_path = os.path.join(self.output_dir, filename)
        sf.write(save_path, speech.numpy(), samplerate=16000)
        return save_path
