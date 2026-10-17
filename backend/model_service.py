from __future__ import annotations

import json
from pathlib import Path
from threading import RLock
from typing import Any

import torch

from training.features import FeatureEncoder, build_inference_row
from training.model import TransitionMLP


PROJECT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = PROJECT / "backend" / "model" / "transition_mlp.pt"
DEBATE_DATA_DIR = PROJECT / "frontend" / "public" / "data" / "debates"


class ModelService:
    def __init__(self, model_path: Path = DEFAULT_MODEL_PATH) -> None:
        self.model_path = model_path
        self._lock = RLock()
        self._model: TransitionMLP | None = None
        self._encoder: FeatureEncoder | None = None
        self._classes: list[str] = []
        self._decision_threshold = 0.5
        self._metrics: dict[str, Any] = {}
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
            if voter.get("before") not in {"PRO", "CON"}:
                raise ValueError("A strict before stance is required for prediction.")
            profile = (debate.get("profiles") or {}).get(username) or {}
            row = build_inference_row(debate, voter, profile)
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
                },
            }

