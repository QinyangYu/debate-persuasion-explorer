from __future__ import annotations

import copy
import shutil
import threading
import uuid
from dataclasses import asdict
from typing import Any

from backend.model_service import ModelService
from training.train import DEFAULT_METRICS, DEFAULT_MODEL, PROJECT, TrainConfig, train_model


class TrainingService:
    """Run one interactive training job and expose epoch-level progress for polling."""

    def __init__(self, model_service: ModelService) -> None:
        self.model_service = model_service
        self._lock = threading.RLock()
        self._status: dict[str, Any] = {
            "state": "idle",
            "run_id": None,
            "epoch": 0,
            "history": [],
            "metrics": model_service.info().get("metrics") if model_service.ready else None,
            "promoted": None,
            "error": None,
        }

    def status(self) -> dict[str, Any]:
        with self._lock:
            return copy.deepcopy(self._status)

    def start(self, config: TrainConfig) -> dict[str, Any]:
        with self._lock:
            if self._status["state"] == "running":
                raise RuntimeError("A training run is already in progress.")
            run_id = uuid.uuid4().hex[:10]
            self._status = {
                "state": "running",
                "run_id": run_id,
                "epoch": 0,
                "epochs": config.epochs,
                "config": asdict(config),
                "history": [],
                "metrics": None,
                "promoted": None,
                "error": None,
            }
            thread = threading.Thread(target=self._run, args=(run_id, config), daemon=True)
            thread.start()
            return copy.deepcopy(self._status)

    def _run(self, run_id: str, config: TrainConfig) -> None:
        def progress(record: dict[str, Any]) -> None:
            with self._lock:
                if self._status.get("run_id") != run_id:
                    return
                self._status["epoch"] = record["epoch"]
                self._status["history"].append(record)

        try:
            temp_dir = PROJECT / "tmp" / "interactive-training"
            temp_dir.mkdir(parents=True, exist_ok=True)
            candidate_model = temp_dir / f"{run_id}.pt"
            candidate_metrics = temp_dir / f"{run_id}.json"
            metrics = train_model(
                config,
                model_path=candidate_model,
                metrics_path=candidate_metrics,
                progress=progress,
            )
            current_metrics = self.model_service.info().get("metrics") or {}
            current_validation = current_metrics.get("validation") or {}
            candidate_validation = metrics["validation"]
            quality_metrics = ("balanced_accuracy", "switch_f1", "switch_pr_auc")
            promoted = not self.model_service.ready or all(
                float(candidate_validation[name]) >= float(current_validation.get(name, -1.0))
                for name in quality_metrics
            )
            if promoted:
                DEFAULT_MODEL.parent.mkdir(parents=True, exist_ok=True)
                DEFAULT_METRICS.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(candidate_model, DEFAULT_MODEL)
                shutil.copy2(candidate_metrics, DEFAULT_METRICS)
                self.model_service.reload()
            candidate_model.unlink(missing_ok=True)
            candidate_metrics.unlink(missing_ok=True)
            with self._lock:
                self._status["state"] = "completed"
                self._status["metrics"] = metrics
                self._status["promoted"] = promoted
                self._status["message"] = (
                    "Candidate promoted to the active model."
                    if promoted
                    else "Candidate kept as an experiment; the stronger active model was preserved."
                )
        except Exception as error:  # surfaced to the training monitor
            with self._lock:
                self._status["state"] = "failed"
                self._status["error"] = f"{type(error).__name__}: {error}"

