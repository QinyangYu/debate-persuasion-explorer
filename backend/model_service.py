from __future__ import annotations

import json
from pathlib import Path
from threading import RLock
from typing import Any

import torch

from training.features import (
    FeatureEncoder,
    belief_summary,
    build_inference_row,
    load_test_embeddings,
)
from training.model import TransitionMLP


PROJECT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = PROJECT / "backend" / "model" / "transition_mlp.pt"
DEBATE_DATA_DIR = PROJECT / "frontend" / "public" / "data" / "debates"
TEST_EMBEDDINGS = PROJECT / "backend" / "model" / "test_embeddings.json"


class ModelService:
    def __init__(self, model_path: Path = DEFAULT_MODEL_PATH) -> None:
        self.model_path = model_path
        self._lock = RLock()
        self._model: TransitionMLP | None = None
        self._encoder: FeatureEncoder | None = None
        self._classes: list[str] = []
        self._decision_threshold = 0.5
        self._metrics: dict[str, Any] = {}
        self._embeddings: dict[str, dict[str, list[float]]] = {}
        self.reload()

    @property
    def ready(self) -> bool:
        return self._model is not None and self._encoder is not None

    def reload(self) -> None:
        with self._lock:
            if not self.model_path.exists():
                self._model = None
                self._encoder = None
                self._classes = []
                self._metrics = {}
                return
            artifact = torch.load(self.model_path, map_location="cpu", weights_only=True)
            model = TransitionMLP(**artifact["model_config"])
            model.load_state_dict(artifact["state_dict"])
            model.eval()
            self._model = model
            self._encoder = FeatureEncoder.from_dict(artifact["encoder"])
            self._classes = list(artifact["classes"])
            self._decision_threshold = float(artifact.get("decision_threshold", 0.5))
            self._metrics = artifact.get("metrics") or {}
            self._embeddings = (
                load_test_embeddings(TEST_EMBEDDINGS) if self._encoder.embedding_dimensions else {}
            )

    def info(self) -> dict[str, Any]:
        with self._lock:
            if not self.ready:
                return {"ready": False, "message": "Train the model before requesting predictions."}
            return {
                "ready": True,
                "classes": self._classes,
                "metrics": self._metrics,
                "decision_threshold": self._decision_threshold,
                "model_path": str(self.model_path),
            }

    @torch.no_grad()
    def predict(self, debate_id: str, username: str) -> dict[str, Any]:
        with self._lock:
            if not self.ready or self._model is None or self._encoder is None:
                raise RuntimeError("The trained model is not available.")
            debate_path = DEBATE_DATA_DIR / f"{debate_id}.json"
            if not debate_path.exists():
                raise KeyError("Unknown debate ID")
            debate = json.loads(debate_path.read_text(encoding="utf-8"))
            voter = next(
                (item for item in debate.get("voters") or [] if item.get("username") == username),
                None,
            )
            if voter is None:
                raise KeyError("Voter not found in the selected debate")
            if not voter.get("is_valid_transition"):
                raise ValueError(
                    "The model covers only voters with a PRO or CON stance both before and after the debate."
                )
            profile = (debate.get("profiles") or {}).get(username) or {}
            row = build_inference_row(debate, voter, profile, self._embeddings.get(debate_id))
            features = torch.tensor([self._encoder.transform(row)], dtype=torch.float32)
            probabilities = torch.softmax(self._model(features), dim=1)[0].tolist()
            switch_probability = probabilities[1]
            predicted_index = int(switch_probability >= self._decision_threshold)
            predicted_after = (
                ("CON" if voter["before"] == "PRO" else "PRO")
                if predicted_index == 1
                else voter["before"]
            )
            return {
                "debate_id": debate_id,
                "username": username,
                "before": voter["before"],
                "observed_transition": voter.get("transition"),
                "observed_switched": voter.get("switched"),
                "predicted_class": self._classes[predicted_index],
                "predicted_transition": f'{voter["before"]} -> {predicted_after}',
                "predicted_switched": predicted_index == 1,
                "switch_probability": switch_probability,
                "decision_threshold": self._decision_threshold,
                "probabilities": {
                    label: probability for label, probability in zip(self._classes, probabilities, strict=True)
                },
                "input_summary": {
                    "category": debate.get("category"),
                    "rounds": debate.get("round_count"),
                    "pro_argument_words": len(row.get("pro_tokens") or []),
                    "con_argument_words": len(row.get("con_tokens") or []),
                    "political_ideology": profile.get("political_ideology"),
                    "religious_ideology": profile.get("religious_ideology"),
                    "debates_as_participant": profile.get("number_of_all_debates"),
                    "votes_cast": profile.get("number_of_voted_debates"),
                    "elo_ranking": profile.get("elo_ranking"),
                    "profile_available": bool(profile),
                    "beliefs": self._belief_inputs(debate, voter, row),
                    "text_embeddings": bool(row.get("pro_embedding")) and self._encoder.embedding_dimensions > 0,
                },
            }

    @staticmethod
    def _belief_inputs(debate: dict[str, Any], voter: dict[str, Any], row: dict[str, Any]) -> dict[str, Any]:
        """Human-readable view of the belief features for the voter-detail panel."""

        coverage, same_agreement, same_overlap, other_agreement, other_overlap, dissonance = belief_summary(row)
        names = {
            str(participant.get("position") or "").upper()[:3]: participant.get("username")
            for participant in debate.get("participants") or []
        }
        before = voter["before"]
        opposite = "CON" if before == "PRO" else "PRO"
        same_profile = row.get(f"{before.lower()}_debater_profile") or {}
        other_profile = row.get(f"{opposite.lower()}_debater_profile") or {}
        return {
            "voter_issues_answered": round(coverage * 48),
            "same_side_debater": names.get(before),
            "same_side_debater_elo": same_profile.get("elo_ranking"),
            "same_side_agreement": same_agreement if same_overlap else None,
            "same_side_shared_issues": round(same_overlap * 48),
            "other_side_debater": names.get(opposite),
            "other_side_debater_elo": other_profile.get("elo_ranking"),
            "other_side_agreement": other_agreement if other_overlap else None,
            "other_side_shared_issues": round(other_overlap * 48),
            "relative_dissonance": dissonance if same_overlap and other_overlap else None,
        }

