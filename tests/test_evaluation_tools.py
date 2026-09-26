"""Checks for the evaluation helpers that need no models or corpus."""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from evaluation import hand_labels, noise_sweep, session_split


class NoiseTests(unittest.TestCase):
    def test_added_noise_has_the_requested_snr(self):
        rng = np.random.default_rng(0)
        audio = np.sin(np.linspace(0, 2000, 160000)).astype(np.float32)
        for snr in [20.0, 5.0, 0.0]:
            noisy = noise_sweep.add_noise(audio, snr, rng)
            noise = noisy.astype(np.float64) - audio
            measured = 10 * np.log10(np.mean(audio.astype(np.float64) ** 2) / np.mean(noise**2))
            self.assertAlmostEqual(measured, snr, delta=0.1)

    def test_silence_is_left_alone_and_seeds_repeat(self):
        silence = np.zeros(100, dtype=np.float32)
        self.assertTrue((noise_sweep.add_noise(silence, 10, np.random.default_rng(0)) == 0).all())
        self.assertEqual(noise_sweep.clip_seed("a.wav", 10), noise_sweep.clip_seed("a.wav", 10))
        self.assertNotEqual(noise_sweep.clip_seed("a.wav", 10), noise_sweep.clip_seed("a.wav", 5))


class SplitTests(unittest.TestCase):
    def test_no_recording_is_shared(self):
        rows = [{"audio_file": f"C:\\data\\REC{r:02d}_{c}.wav"} for r in range(30) for c in range(1 + r % 4)]
        train, test = session_split.split_by_recording(rows, 0.2, seed=1)
        self.assertEqual(len(train) + len(test), len(rows))
        ids = lambda part: {session_split.recording_id(x["audio_file"]) for x in part}
        self.assertFalse(ids(train) & ids(test))
        self.assertAlmostEqual(len(test) / len(rows), 0.2, delta=0.06)
        self.assertEqual(session_split.recording_id("a/LKPR_Tower_20201_12.wav"), "LKPR_Tower_20201")


class HandLabelTests(unittest.TestCase):
    def test_export_then_score(self):
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            test_file = folder / "test.jsonl"
            test_file.write_text(json.dumps({"tokens": ["eurotrans", "one", "and", "climb"],
                                             "ner_tags": ["B-WAYPOINT", "B-VALUE", "B-WAYPOINT", "B-COMMAND"]}) + "\n")
            sheet = folder / "sheet.tsv"
            hand_labels.export(test_file, sheet)
            lines = sheet.read_text().splitlines()
            self.assertNotIn("WAYPOINT", "\n".join(lines))  # rule labels are hidden
            human = ["CALLSIGN", "CALLSIGN", "O", "COMMAND"]
            filled = [lines[0]] + ["\t".join(c[:3] + [h] + c[4:]) for c, h in zip((l.split("\t") for l in lines[1:]), human)]
            sheet.write_text("\n".join(filled) + "\n")
            predictions = folder / "predictions.jsonl"
            predictions.write_text(json.dumps({"index": 0, "prediction": ["CALLSIGN", "VALUE", "O", "COMMAND"]}) + "\n")
            output = folder / "scores.json"
            hand_labels.score(test_file, sheet, predictions, output)
            result = json.loads(output.read_text())
            self.assertEqual(result["rules_vs_human"]["accuracy"], 0.25)
            self.assertEqual(result["distilbert_vs_human"]["accuracy"], 0.75)


if __name__ == "__main__":
    unittest.main()
