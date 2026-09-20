from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from data_utils import classify_stance, iter_json_object, participant_sides  # noqa: E402


class DataUtilityTests(unittest.TestCase):
    def setUp(self) -> None:
        self.debate = {
            "participant_1_name": "Ada",
            "participant_1_position": "Pro",
            "participant_2_name": "Grace",
            "participant_2_position": "Con",
        }
        self.sides = participant_sides(self.debate)
        self.criterion = "Agreed with before the debate"

    def vote_map(self, pro=False, con=False, tied=False):
        return {
            "Ada": {self.criterion: pro},
            "Grace": {self.criterion: con},
            "Tied": {self.criterion: tied},
        }

    def test_strict_stances(self):
        self.assertEqual(classify_stance(self.vote_map(pro=True), self.sides, self.criterion), "PRO")
        self.assertEqual(classify_stance(self.vote_map(con=True), self.sides, self.criterion), "CON")

    def test_non_strict_stances_are_preserved(self):
        self.assertEqual(classify_stance(self.vote_map(tied=True), self.sides, self.criterion), "TIE")
        self.assertEqual(classify_stance(self.vote_map(), self.sides, self.criterion), "UNKNOWN")
        self.assertEqual(classify_stance(self.vote_map(pro=True, tied=True), self.sides, self.criterion), "AMBIGUOUS")

    def test_streaming_object_reader(self):
        payload = {"first": {"text": "small"}, "second": [1, 2, 3]}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            self.assertEqual(dict(iter_json_object(path, chunk_size=5)), payload)


if __name__ == "__main__":
    unittest.main()
