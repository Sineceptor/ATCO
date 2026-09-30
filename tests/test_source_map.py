"""The archived 2025 scripts are a record: their current hashes must match the source map."""

import hashlib
import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class SourceMapTests(unittest.TestCase):
    def test_archived_scripts_match_their_recorded_current_hash(self):
        entries = json.loads((ROOT / "docs/source_map.json").read_text())["archived_experiments"]
        self.assertEqual(len(entries), 52)
        for entry in entries:
            digest = hashlib.sha256((ROOT / entry["file"]).read_bytes()).hexdigest()
            self.assertEqual(digest, entry["current_sha256"], entry["file"])


if __name__ == "__main__":
    unittest.main()
