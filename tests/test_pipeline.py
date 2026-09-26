"""Checks for the pipeline, independent of model weights and ML libraries."""

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from atco import pipeline


class ConnectorTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.audio = self.root / "input.wav"
        self.audio.write_bytes(b"test double, not real audio")
        self.calls = []
        self.entities = [{"entity_group": "COMMAND", "word": "climb", "score": 0.8}]
        owner = self

        class Pipe:
            model = SimpleNamespace(config=SimpleNamespace(max_position_embeddings=8))

            def tokenizer(self, text):
                return {"input_ids": list(range(len(text.split()) + 2))}

            def __call__(self, text):
                owner.calls.append(("ner", text))
                return owner.entities

        class Parser:
            def __init__(self, model_path, local_files_only):
                owner.calls.append(("ner_model", model_path, local_files_only))
                self.pipe = Pipe()
                self.guard = SimpleNamespace(
                    check=lambda _: ("CLIMB", ["old rule warning"])
                )

        class Speaker:
            def __init__(self, model, vocoder, output, local_files_only):
                owner.calls.append(("tts_model", model, vocoder, local_files_only))
                self.output = output

            def speak(self, text, filename):
                owner.calls.append(("speak", text))
                return str(self.output / filename)

        def transcribe(model, audio, local_files_only):
            self.calls.append(("asr", model, audio, local_files_only))
            return "climb please"

        self.speaker_class = Speaker
        self.modules = {
            "atco.speech_recognition": SimpleNamespace(transcribe=transcribe),
            "atco.entity_extraction": SimpleNamespace(
                AtcParser=Parser, repair_bert_output=lambda x: x,
                drop_function_word_waypoints=lambda x: x,
            ),
            "atco.speech_synthesis": SimpleNamespace(AtcSpeaker=Speaker),
        }
        self.args = SimpleNamespace(
            models_dir=self.root / "models",
            asr_model="local-asr",
            ner_model="local-ner",
            tts_model="local-tts",
            vocoder_model="local-vocoder",
            tts="speecht5",
            allow_downloads=False,
            audio=self.audio,
            text=None,
            output_dir=self.root / "output",
        )

    def run_pipeline(self):
        with (
            patch.dict("sys.modules", self.modules),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            return pipeline.run(self.args)

    def test_audio_reaches_ner_and_extracted_words_reach_tts(self):
        result = self.run_pipeline()
        self.assertIn(("ner", "climb please"), self.calls)
        self.assertIn(("speak", "climb"), self.calls)
        self.assertIn(("tts_model", "local-tts", "local-vocoder", True), self.calls)
        self.assertEqual(result["rule_warnings"], ["old rule warning"])
        self.assertEqual(result["transcript"], "climb please")
        self.assertEqual(result["tts_status"], "generated")
        self.assertEqual(
            json.loads((self.args.output_dir / "result.json").read_text()), result
        )

    def test_empty_extraction_does_not_invent_readback_or_load_tts(self):
        self.entities = []
        result = self.run_pipeline()
        self.assertEqual(result["status"], "no_entities")
        self.assertIsNone(result["audio_file"])
        self.assertFalse(any(call[0] == "tts_model" for call in self.calls))

    def test_typed_text_skips_asr_and_none_skips_tts(self):
        self.args.audio, self.args.text, self.args.tts = (
            None,
            "  typed instruction  ",
            "none",
        )
        result = self.run_pipeline()
        self.assertEqual(result["transcript"], "typed instruction")
        self.assertFalse(any(call[0] in ("asr", "tts_model") for call in self.calls))

    def test_tts_failure_preserves_intermediate_result(self):
        with patch.object(
            self.speaker_class, "speak", side_effect=OSError("synthesis unavailable")
        ):
            with self.assertRaises(OSError):
                self.run_pipeline()
        result = json.loads((self.args.output_dir / "result.json").read_text())
        self.assertEqual(result["tts_status"], "failed")
        self.assertEqual(result["transcript"], "climb please")
        self.assertEqual(result["tts_error"], "synthesis unavailable")

    def test_missing_audio_fails_before_model_loading(self):
        self.args.audio = self.root / "missing.wav"
        with self.assertRaises(FileNotFoundError):
            self.run_pipeline()
        self.assertEqual(self.calls, [])

    def test_long_text_fails_before_truncated_inference(self):
        self.args.audio, self.args.text = None, "word " * 20
        with self.assertRaisesRegex(ValueError, "short instruction"):
            self.run_pipeline()
        self.assertFalse(any(call[0] == "ner" for call in self.calls))


class FunctionWordTests(unittest.TestCase):
    def test_function_words_lose_their_waypoint_label_and_nothing_else_changes(self):
        try:
            from atco.entity_extraction import drop_function_word_waypoints
        except ImportError:
            self.skipTest("needs torch and transformers")
        found = [
            {"entity_group": "WAYPOINT", "word": "and"},
            {"entity_group": "WAYPOINT", "word": "eurotrans"},
            {"entity_group": "COMMAND", "word": "squawk"},
            {"entity_group": "WAYPOINT", "word": " The "},
        ]
        kept = drop_function_word_waypoints(found)
        self.assertEqual([e["word"] for e in kept], ["eurotrans", "squawk"])



class SquawkCheckTests(unittest.TestCase):
    def setUp(self):
        from atco.entity_extraction import CommandChecks

        self.checks = CommandChecks()

    def warnings(self, spoken):
        return self.checks.validate_physics("squawk", self.checks._text_to_digit(spoken))

    def test_a_leading_zero_is_a_valid_code(self):
        self.assertEqual(self.warnings("zero four two one"), [])

    def test_all_zeros_is_four_digits(self):
        self.assertEqual(self.warnings("zero zero zero zero"), [])

    def test_three_digits_are_flagged(self):
        self.assertEqual(len(self.warnings("four two one")), 1)

    def test_eight_or_nine_is_flagged(self):
        self.assertIn("8 or 9", " ".join(self.warnings("four four eight two")))

    def test_emergency_code_is_named(self):
        self.assertIn("emergency", " ".join(self.warnings("seven seven zero zero")))


if __name__ == "__main__":
    unittest.main()
