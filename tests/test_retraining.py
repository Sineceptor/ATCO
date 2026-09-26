"""Checks for the 2026 retraining code that need no models or audio."""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from evaluation import compare_models, session_split

try:
    import torch

    from training.train_whisper import Collator
except ImportError:  # torch is only needed for the collator test
    torch = None

START = 50258  # <|startoftranscript|> in the Whisper tokenizer


@unittest.skipIf(torch is None, "needs torch")
class CollatorTests(unittest.TestCase):
    def batch(self, *labels):
        return [dict(input_features=np.zeros((80, 3000), np.float32), labels=list(l)) for l in labels]

    def test_start_token_appears_once(self):
        # The tokenizer starts every label with the start token; the model adds another
        # when it builds its decoder inputs, so the collator must remove this one.
        out = Collator(processor=None, start_token=START)(self.batch([START, 1, 2, 3], [START, 4, 5]))
        self.assertTrue((out["labels"][:, 0] != START).all())
        self.assertEqual(out["labels"][0, :3].tolist(), [1, 2, 3])
        self.assertEqual(out["labels"][1, :3].tolist(), [4, 5, -100])

    def test_every_batch_has_the_same_label_length(self):
        # A fixed shape keeps the Apple GPU from compiling a new graph for every length.
        collate = Collator(processor=None, start_token=START, label_length=8)
        a = collate(self.batch([START, 1, 2]))
        b = collate(self.batch([START] + list(range(1, 20))))
        self.assertEqual(a["labels"].shape, b["labels"].shape)
        self.assertEqual(b["labels"].shape[1], 7)


class LetterWeightTests(unittest.TestCase):
    def test_only_letter_and_digit_tokens_get_the_extra_weight(self):
        try:
            from transformers import WhisperTokenizer

            from training.train_whisper import weighted_labels
        except ImportError:
            self.skipTest("needs transformers")
        base = Path("models/whisper-medium-base")
        if not base.is_dir():
            self.skipTest("needs a local Whisper tokenizer")
        tokenizer = WhisperTokenizer.from_pretrained(base)
        tokenizer.set_prefix_tokens(language="en", task="transcribe", predict_timestamps=False)
        text = "hotel delta lima contact tower one two one"
        ids, weights = weighted_labels(tokenizer, text, 3.0)
        self.assertEqual(ids, tokenizer(text).input_ids)
        self.assertEqual(len(ids), len(weights))
        heavy = tokenizer.decode([i for i, w in zip(ids, weights) if w == 3.0])
        self.assertEqual(heavy.split(), ["hotel", "delta", "lima", "one", "two", "one"])
        self.assertEqual(weighted_labels(tokenizer, text, 1.0)[1], [1.0] * len(ids))


class OriginalSplitTests(unittest.TestCase):
    def test_test_clips_come_from_recordings_never_trained_on(self):
        name = lambda rec, clip: {"audio_file": f"{rec}_{clip:04d}.wav"}
        train = [name("A", 0), name("A", 1), name("B", 0)]
        test = [name("A", 2), name("C", 0), name("C", 1), name("B", 3)]
        kept, validation, unseen = session_split.split_original_test(train, test)
        self.assertEqual(kept, train)
        self.assertEqual([r["audio_file"] for r in validation], ["A_0002.wav", "B_0003.wav"])
        self.assertEqual([r["audio_file"] for r in unseen], ["C_0000.wav", "C_0001.wav"])


class FoldTests(unittest.TestCase):
    def test_every_clip_is_tested_once_and_no_recording_crosses_roles(self):
        rows = [{"audio_file": f"REC{r:02d}_{c:04d}.wav"} for r in range(40) for c in range(1 + r % 3)]
        folds = session_split.make_folds(rows, folds=5, validation_fraction=0.1, seed=3)
        tested = [r["audio_file"] for f in folds for r in f["test"]]
        self.assertEqual(sorted(tested), sorted(r["audio_file"] for r in rows))
        rec = lambda part: {session_split.recording_id(r["audio_file"]) for r in part}
        for f in folds:
            self.assertFalse(rec(f["train"]) & rec(f["test"]))
            self.assertFalse(rec(f["validation"]) & rec(f["test"]))
            self.assertFalse(rec(f["train"]) & rec(f["validation"]))
            self.assertEqual(len(f["train"]) + len(f["validation"]) + len(f["test"]), len(rows))


class CompareTests(unittest.TestCase):
    def test_counts_and_digit_spelling(self):
        records = [
            dict(reference="squawk four four zero two", greedy="squawk 4402"),
            dict(reference="hello", greedy="hello there"),
        ]
        edits, words = compare_models.clip_counts(records, "greedy", spelled=True)
        self.assertEqual(edits.tolist(), [0, 1])
        self.assertEqual(words.tolist(), [5, 1])
        edits, _ = compare_models.clip_counts(records, "greedy", spelled=False)
        self.assertEqual(edits.tolist(), [4, 1])

    def test_identical_models_differ_by_zero(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "p.jsonl"
            rows = [dict(file=f"{i}.wav", reference="one two three", greedy="one two") for i in range(5)]
            path.write_text("".join(json.dumps(r) + "\n" for r in rows))
            out = Path(folder) / "out.json"
            import sys
            from unittest.mock import patch

            argv = ["compare", "--model", f"a={path}", "--model", f"b={path}", "--modes", "greedy",
                    "--resamples", "200", "--output", str(out)]
            with patch.object(sys, "argv", argv):
                compare_models.main()
            result = json.loads(out.read_text())
            change = result["models"]["b"]["greedy"]["change_from_a"]
            self.assertEqual(change["points"], 0)
            self.assertEqual(change["interval_95_points"], [0, 0])
            self.assertAlmostEqual(result["models"]["a"]["greedy"]["wer"], 1 / 3)


if __name__ == "__main__":
    unittest.main()
