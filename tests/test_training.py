from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch


PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from backend.model_service import ModelService  # noqa: E402
from training.features import (  # noqa: E402
    BIG_ISSUES,
    FeatureEncoder,
    TARGET_CLASSES,
    TRANSITIONS,
    belief_summary,
    issue_vector,
    load_background_rows,
    load_rows,
    split_background_by_debate,
)
from training.model import TransitionMLP  # noqa: E402
from training.train import showcase_like_indices  # noqa: E402


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

    def test_issue_vector_encodes_answered_stances(self):
        vector = issue_vector({"big_issues": {"Abortion": "Pro", "Torture": "Con", "Welfare": "Und"}})
        self.assertEqual(len(vector), len(BIG_ISSUES))
        self.assertEqual(vector[BIG_ISSUES.index("Abortion")], 1.0)
        self.assertEqual(vector[BIG_ISSUES.index("Torture")], -1.0)
        self.assertEqual(vector[BIG_ISSUES.index("Welfare")], 0.0)
        self.assertEqual(issue_vector({}), [0.0] * len(BIG_ISSUES))

    def test_relative_dissonance_follows_initial_stance(self):
        voter = {"big_issues": {"Abortion": "Pro", "Torture": "Con"}}
        agrees = {"big_issues": {"Abortion": "Pro", "Torture": "Con"}}
        disagrees = {"big_issues": {"Abortion": "Con", "Torture": "Pro"}}
        row = {"before": "PRO", "profile": voter, "pro_debater_profile": agrees, "con_debater_profile": disagrees}
        coverage, same, same_overlap, other, other_overlap, dissonance = belief_summary(row)
        self.assertAlmostEqual(coverage, 2 / len(BIG_ISSUES))
        self.assertEqual((same, other, dissonance), (1.0, -1.0, -2.0))
        self.assertAlmostEqual(same_overlap, 2 / len(BIG_ISSUES))
        # The same voter backing CON first now sits closer to the opposing debater.
        self.assertEqual(belief_summary({**row, "before": "CON"})[5], 2.0)

    def test_showcase_like_calibration_subset_matches_selection_rule(self):
        indices = showcase_like_indices(self.validation_rows)
        self.assertTrue(indices)
        for index in indices[:200]:
            debate = self.validation_rows[index]["debate_id"]
            debate_rows = [row for row in self.validation_rows if row["debate_id"] == debate]
            self.assertGreaterEqual(len(debate_rows), 5)
            self.assertTrue(any(row["label"] == 1 for row in debate_rows))

    def test_saved_model_serves_live_prediction(self):
        service = ModelService()
        self.assertTrue(service.ready)
        example = self.test_rows[0]
        prediction = service.predict(example["debate_id"], example["username"])
        self.assertIn(prediction["predicted_class"], TARGET_CLASSES)
        self.assertIn(prediction["predicted_transition"], TRANSITIONS)
        self.assertAlmostEqual(sum(prediction["probabilities"].values()), 1.0, places=5)
        self.assertIn("beliefs", prediction["input_summary"])
        self.assertGreaterEqual(prediction["switch_probability"], 0.0)
        self.assertLessEqual(prediction["switch_probability"], 1.0)


if __name__ == "__main__":
    unittest.main()
