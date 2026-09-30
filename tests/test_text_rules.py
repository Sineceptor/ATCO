"""Regression tests for the app's text rules: no models or ML libraries needed.

Each case is one the September 2026 review found going wrong.
"""

import unittest

from atco.entity_extraction import CommandChecks
from atco.speech_synthesis import AtcTextNormalizer
from atco.text_prep import prepare_for_tagger


class TaggerInputTests(unittest.TestCase):
    def test_digits_are_spelled_as_the_tagger_was_trained(self):
        # Raw "4582" was tagged as a waypoint, so the squawk check never ran.
        self.assertEqual(prepare_for_tagger("jetstar 1 squawk 4582"), "jetstar one squawk four five eight two")

    def test_compact_flight_level_and_runway_side_are_spelled_out(self):
        self.assertEqual(prepare_for_tagger("climb FL180"), "climb flight level one eight zero")
        self.assertEqual(prepare_for_tagger("runway 27L"), "runway two seven left")

    def test_the_evaluation_uses_the_same_step(self):
        from evaluation.extraction_agreement import normalise

        for text in ["Jetstar 1, squawk 4582", "alfa oskar niner", "cleared runway 16R"]:
            self.assertEqual(normalise(text), prepare_for_tagger(text))

    def test_spelling_matches_the_scorer(self):
        from atco.text_prep import spell_digits as app
        from evaluation.evaluate_models import spell_digits as scorer

        for text in ["4402", "FL 350", "121.8"]:
            self.assertEqual(app(text), scorer(text))


class NumberReadingTests(unittest.TestCase):
    def setUp(self):
        self.checks = CommandChecks()

    def test_quantities_add_up(self):
        read = self.checks._text_to_digit
        self.assertEqual(read("one thousand five hundred"), "1500")  # was 1000500
        self.assertEqual(read("five thousand four eight zero"), "5480")
        self.assertEqual(read("one three thousand"), "13000")

    def test_digit_by_digit_is_unchanged(self):
        read = self.checks._text_to_digit
        self.assertEqual(read("one two one decimal eight"), "121.8")
        self.assertEqual(read("zero four two one"), "0421")
        self.assertEqual(read("seventy seven"), "77")

    def test_altitude_in_hundreds_gets_no_false_warning(self):
        value = self.checks._text_to_digit("one thousand five hundred feet")
        self.assertEqual(self.checks.validate_physics("descend", value, "one thousand five hundred feet"), [])

    def test_fractional_heading_is_flagged(self):
        # Truncating 360.5 to 360 let it pass.
        self.assertTrue(self.checks.validate_physics("heading", "360.5"))
        self.assertEqual(self.checks.validate_physics("heading", "270"), [])

    def test_a_direction_is_not_an_unreadable_number(self):
        entities = [{"entity_group": "COMMAND", "word": "turn"}, {"entity_group": "VALUE", "word": "right"},
                    {"entity_group": "COMMAND", "word": "heading"}, {"entity_group": "VALUE", "word": "one seven two"}]
        self.assertEqual(self.checks.check(entities)[1], [])

    def test_unreadable_value_is_reported_not_skipped(self):
        entities = [{"entity_group": "COMMAND", "word": "squawk"}, {"entity_group": "VALUE", "word": "ident"}]
        _, warnings = self.checks.check(entities)
        self.assertIn("Could not read a number", " ".join(warnings))


class ReadbackTests(unittest.TestCase):
    def setUp(self):
        self.normalise = AtcTextNormalizer().normalize

    def test_flight_level_is_kept(self):
        self.assertIn("flight level one eight zero", self.normalise("climb FL180"))

    def test_runway_side_is_kept_joined_or_spaced(self):
        self.assertIn("two seven, left", self.normalise("runway 27L"))
        self.assertIn("two seven, left", self.normalise("runway 27 L"))


if __name__ == "__main__":
    unittest.main()
