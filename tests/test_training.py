from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch


PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from backend.model_service import ModelService  # noqa: E402
from training.features import (  # noqa: E402
    FeatureEncoder,
    TARGET_CLASSES,
    TRANSITIONS,
    load_background_rows,
    load_rows,
    split_background_by_debate,
)
from training.model import TransitionMLP  # noqa: E402


DATA_DIR = PROJECT / "frontend" / "public" / "data" / "debates"
BACKGROUND_DATA = PROJECT / "training" / "data" / "background_debates.jsonl.gz"
BACKGROUND_PROFILES = PROJECT / "training" / "data" / "background_profiles.json.gz"


class TrainingPipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.background_rows = load_background_rows(BACKGROUND_DATA, BACKGROUND_PROFILES)
        cls.train_rows, cls.validation_rows = split_background_by_debate(cls.background_rows)
        cls.test_rows = load_rows(DATA_DIR)
        cls.encoder = FeatureEncoder.fit(cls.train_rows)

    def test_dataset_contains_all_transition_classes(self):
        self.assertEqual({row["transition"] for row in self.background_rows}, set(TRANSITIONS))
        self.assertEqual({row["label"] for row in self.background_rows}, {0, 1})
        self.assertEqual(len(self.test_rows), 1863)
        self.assertGreater(len(self.background_rows), 60000)

    def test_debate_groups_do_not_cross_splits(self):
        groups = [
            {row["debate_id"] for row in split}
            for split in (self.train_rows, self.validation_rows, self.test_rows)
        ]
        self.assertTrue(groups[0].isdisjoint(groups[1]))
        self.assertTrue(groups[0].isdisjoint(groups[2]))
        self.assertTrue(groups[1].isdisjoint(groups[2]))
        self.assertEqual(len(groups[2]), 51)

    def test_feature_dimension_matches_model_input(self):
        vector = self.encoder.transform(self.train_rows[0])
        self.assertEqual(len(vector), self.encoder.dimension)
        model = TransitionMLP(self.encoder.dimension, hidden_units=16, hidden_layers=1)
        output = model(torch.tensor([vector], dtype=torch.float32))
        self.assertEqual(tuple(output.shape), (1, 2))

    def test_saved_model_serves_live_prediction(self):
        service = ModelService()
        self.assertTrue(service.ready)
        example = self.test_rows[0]
        prediction = service.predict(example["debate_id"], example["username"])
        self.assertIn(prediction["predicted_class"], TARGET_CLASSES)
        self.assertIn(prediction["predicted_transition"], TRANSITIONS)
        self.assertAlmostEqual(sum(prediction["probabilities"].values()), 1.0, places=5)
        self.assertGreaterEqual(prediction["switch_probability"], 0.0)
        self.assertLessEqual(prediction["switch_probability"], 1.0)


if __name__ == "__main__":
    unittest.main()
