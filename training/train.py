from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from training.features import (
    FeatureEncoder,
    TARGET_CLASSES,
    load_background_rows,
    load_rows,
    load_test_embeddings,
    split_background_by_debate,
)
from training.model import TransitionMLP


PROJECT = Path(__file__).resolve().parents[1]
DEFAULT_BACKGROUND_DATA = PROJECT / "training" / "data" / "background_debates.jsonl.gz"
DEFAULT_BACKGROUND_PROFILES = PROJECT / "training" / "data" / "background_profiles.json.gz"
DEFAULT_TEST_DATA = PROJECT / "frontend" / "public" / "data" / "debates"
DEFAULT_MODEL = PROJECT / "backend" / "model" / "transition_mlp.pt"
DEFAULT_METRICS = PROJECT / "outputs" / "training_metrics.json"
DEFAULT_BACKGROUND_EMBEDDINGS = PROJECT / "training" / "data" / "background_embeddings.json.gz"
DEFAULT_TEST_EMBEDDINGS = PROJECT / "backend" / "model" / "test_embeddings.json"


@dataclass
class TrainConfig:
    learning_rate: float = 0.001
    epochs: int = 60
    hidden_units: int = 128
    hidden_layers: int = 1
    activation: str = "tanh"
    batch_size: int = 128
    seed: int = 7400
    dropout: float = 0.30
    weight_decay: float = 0.01
    # Optional MiniLM text features (training/embeddings.py); off by default because
    # they lowered validation PR-AUC in our comparison.
    text_embeddings: bool = False


def _tensorize(rows: list[dict[str, Any]], encoder: FeatureEncoder) -> TensorDataset:
    features = torch.tensor([encoder.transform(row) for row in rows], dtype=torch.float32)
    labels = torch.tensor([row["label"] for row in rows], dtype=torch.long)
    return TensorDataset(features, labels)


def _average_precision(labels: list[int], scores: list[float]) -> float:
    positives = sum(labels)
    if positives == 0:
        return 0.0
    ranked = sorted(zip(scores, labels, strict=True), reverse=True)
    hits = 0
    total = 0.0
    for rank, (_, label) in enumerate(ranked, start=1):
        if label:
            hits += 1
            total += hits / rank
    return total / positives


def classification_metrics(
    labels: list[int], probabilities: list[list[float]], loss: float, threshold: float = 0.5
) -> dict[str, Any]:
    predictions = [int(row[1] >= threshold) for row in probabilities]
    confusion = [[0, 0], [0, 0]]
    for actual, predicted in zip(labels, predictions, strict=True):
        confusion[actual][predicted] += 1
    per_class = {}
    recalls = []
    f1_values = []
    for index, name in enumerate(TARGET_CLASSES):
        tp = confusion[index][index]
        fp = sum(confusion[row][index] for row in range(2) if row != index)
        fn = sum(confusion[index][column] for column in range(2) if column != index)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[name] = {"precision": precision, "recall": recall, "f1": f1, "support": sum(confusion[index])}
        recalls.append(recall)
        f1_values.append(f1)

    switch_scores = [row[1] for row in probabilities]
    tp = confusion[1][1]
    fp = confusion[0][1]
    fn = confusion[1][0]
    switch_precision = tp / (tp + fp) if tp + fp else 0.0
    switch_recall = tp / (tp + fn) if tp + fn else 0.0
    switch_f1 = (
        2 * switch_precision * switch_recall / (switch_precision + switch_recall)
        if switch_precision + switch_recall
        else 0.0
    )
    return {
        "loss": loss,
        "accuracy": sum(a == p for a, p in zip(labels, predictions, strict=True)) / len(labels),
        "balanced_accuracy": sum(recalls) / len(recalls),
        "macro_f1": sum(f1_values) / len(f1_values),
        "switch_precision": switch_precision,
        "switch_recall": switch_recall,
        "switch_f1": switch_f1,
        "switch_pr_auc": _average_precision(labels, switch_scores),
        "decision_threshold": threshold,
        "confusion_matrix": confusion,
        "per_class": per_class,
        "examples": len(labels),
    }


@torch.no_grad()
def evaluate(
    model: nn.Module, dataset: TensorDataset, loss_fn: nn.Module, threshold: float = 0.5
) -> dict[str, Any]:
    model.eval()
    features, labels = dataset.tensors
    logits = model(features)
    loss = float(loss_fn(logits, labels).item())
    probabilities = torch.softmax(logits, dim=1)
    return classification_metrics(labels.tolist(), probabilities.tolist(), loss, threshold)


def choose_threshold(model: nn.Module, dataset: TensorDataset, loss_fn: nn.Module) -> float:
    """Choose a validation-only threshold for balanced Switch/Stay performance."""

    candidates = [value / 200 for value in range(10, 191)]
    scored = []
    for threshold in candidates:
        metrics = evaluate(model, dataset, loss_fn, threshold)
        scored.append(
            (
                metrics["balanced_accuracy"],
                metrics["switch_f1"],
                metrics["accuracy"],
                -abs(threshold - 0.5),
                threshold,
            )
        )
    return max(scored)[-1]


def showcase_like_indices(rows: list[dict[str, Any]]) -> list[int]:
    """Rows from debates that pass the showcase selection rule.

    The 51 showcase debates were chosen among debates with at least five strict
    voters and at least one switcher, which doubles their switch rate.  Tuning
    the decision threshold on validation debates chosen by the same rule lets
    the threshold transfer to that test set without touching test labels.
    """

    voters = Counter(row["debate_id"] for row in rows)
    switchers = Counter(row["debate_id"] for row in rows if row["label"] == 1)
    keep = {debate for debate, count in voters.items() if count >= 5 and switchers[debate] > 0}
    return [index for index, row in enumerate(rows) if row["debate_id"] in keep]


def _subset(dataset: TensorDataset, indices: list[int]) -> TensorDataset:
    selected = torch.tensor(indices, dtype=torch.long)
    return TensorDataset(*(tensor[selected] for tensor in dataset.tensors))


def _train_linear_baseline(
    train_data: TensorDataset,
    validation_data: TensorDataset,
    calibration_data: TensorDataset,
    test_data: TensorDataset,
    class_weights: torch.Tensor,
    seed: int,
) -> dict[str, Any]:
    """Fit the roadmap's shallow logistic-regression benchmark."""

    torch.manual_seed(seed)
    input_dim = train_data.tensors[0].shape[1]
    model = nn.Linear(input_dim, 2)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.03, weight_decay=0.001)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights)
    features, labels = train_data.tensors
    best_state = None
    best_loss = float("inf")
    for _ in range(250):
        model.train()
        optimizer.zero_grad()
        loss = loss_fn(model(features), labels)
        loss.backward()
        optimizer.step()
        validation_loss = evaluate(model, validation_data, loss_fn)["loss"]
        if validation_loss < best_loss:
            best_loss = validation_loss
            best_state = {name: tensor.detach().clone() for name, tensor in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    threshold = choose_threshold(model, calibration_data, loss_fn)
    return {
        "description": "Linear softmax/logistic regression on structured baseline inputs",
        "decision_threshold": threshold,
        "validation": evaluate(model, validation_data, loss_fn, threshold),
        "test": evaluate(model, test_data, loss_fn, threshold),
    }


def train_model(
    config: TrainConfig,
    background_data: Path = DEFAULT_BACKGROUND_DATA,
    background_profiles: Path = DEFAULT_BACKGROUND_PROFILES,
    test_data_dir: Path = DEFAULT_TEST_DATA,
    model_path: Path = DEFAULT_MODEL,
    metrics_path: Path = DEFAULT_METRICS,
    progress: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.set_num_threads(max(1, min(torch.get_num_threads(), 4)))

    use_embeddings = config.text_embeddings and DEFAULT_BACKGROUND_EMBEDDINGS.exists()
    background_rows = load_background_rows(
        background_data,
        background_profiles,
        DEFAULT_BACKGROUND_EMBEDDINGS if use_embeddings else None,
    )
    train_rows, validation_rows = split_background_by_debate(background_rows, config.seed)
    test_embeddings = load_test_embeddings(DEFAULT_TEST_EMBEDDINGS if use_embeddings else None)
    test_rows = load_rows(test_data_dir, test_embeddings)
    train_ids = {row["debate_id"] for row in train_rows}
    validation_ids = {row["debate_id"] for row in validation_rows}
    test_ids = {row["debate_id"] for row in test_rows}
    if train_ids & validation_ids or train_ids & test_ids or validation_ids & test_ids:
        raise ValueError("A debate crossed the train/validation/test boundary.")
    if len(test_ids) != 51:
        raise ValueError(f"Expected all 51 showcased debates in test, found {len(test_ids)}")
    embedding_dimensions = next(
        (len(row["pro_embedding"]) for row in train_rows if use_embeddings and row.get("pro_embedding")),
        0,
    )
    encoder = FeatureEncoder.fit(train_rows, embedding_dimensions=embedding_dimensions)
    train_data = _tensorize(train_rows, encoder)
    validation_data = _tensorize(validation_rows, encoder)
    test_data = _tensorize(test_rows, encoder)
    structured_dimension = encoder.dimension - encoder.text_block_dimensions
    structured_train_data = TensorDataset(
        train_data.tensors[0][:, :structured_dimension], train_data.tensors[1]
    )
    structured_validation_data = TensorDataset(
        validation_data.tensors[0][:, :structured_dimension], validation_data.tensors[1]
    )
    structured_test_data = TensorDataset(
        test_data.tensors[0][:, :structured_dimension], test_data.tensors[1]
    )
    calibration_indices = showcase_like_indices(validation_rows)
    calibration_data = _subset(validation_data, calibration_indices)
    structured_calibration_data = _subset(structured_validation_data, calibration_indices)

    model = TransitionMLP(
        encoder.dimension,
        config.hidden_units,
        config.hidden_layers,
        config.activation,  # type: ignore[arg-type]
        config.dropout,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    train_counts = Counter(row["label"] for row in train_rows)
    class_weights = torch.tensor(
        [len(train_rows) / (2 * max(train_counts[index], 1)) for index in range(2)],
        dtype=torch.float32,
    )
    loss_fn = nn.CrossEntropyLoss(weight=class_weights)
    generator = torch.Generator().manual_seed(config.seed)
    loader = DataLoader(
        train_data,
        batch_size=config.batch_size,
        shuffle=True,
        generator=generator,
    )

    history: list[dict[str, Any]] = []
    best_state: dict[str, torch.Tensor] | None = None
    best_validation_score = -1.0
    best_validation_loss = float("inf")
    for epoch in range(1, config.epochs + 1):
        model.train()
        total_loss = 0.0
        seen = 0
        for features, labels in loader:
            optimizer.zero_grad()
            logits = model(features)
            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * len(labels)
            seen += len(labels)

        train_metrics = evaluate(model, train_data, loss_fn)
        validation_metrics = evaluate(model, validation_data, loss_fn)
        record = {
            "epoch": epoch,
            "batch_train_loss": total_loss / max(seen, 1),
            "train": _compact_metrics(train_metrics),
            "validation": _compact_metrics(validation_metrics),
        }
        history.append(record)
        score = validation_metrics["switch_pr_auc"]
        if score > best_validation_score or (
            score == best_validation_score and validation_metrics["loss"] < best_validation_loss
        ):
            best_validation_score = score
            best_validation_loss = validation_metrics["loss"]
            best_state = {name: tensor.detach().clone() for name, tensor in model.state_dict().items()}
        if progress:
            progress(record)

    if best_state is not None:
        model.load_state_dict(best_state)

    decision_threshold = choose_threshold(model, calibration_data, loss_fn)
    train_metrics = evaluate(model, train_data, loss_fn, decision_threshold)
    validation_metrics = evaluate(model, validation_data, loss_fn, decision_threshold)
    calibration_metrics = evaluate(model, calibration_data, loss_fn, decision_threshold)
    test_metrics = evaluate(model, test_data, loss_fn, decision_threshold)
    test_labels = [row["label"] for row in test_rows]
    majority_class, majority_count = Counter(row["label"] for row in train_rows).most_common(1)[0]
    majority_accuracy = sum(label == majority_class for label in test_labels) / len(test_labels)
    linear_baseline = _train_linear_baseline(
        structured_train_data,
        structured_validation_data,
        structured_calibration_data,
        structured_test_data,
        class_weights,
        config.seed + 1,
    )

    metrics = {
        "task": "Binary observed stance-change prediction",
        "classes": list(TARGET_CLASSES),
        "config": asdict(config),
        "architecture": {
            "input_features": encoder.dimension,
            "hidden_layers": config.hidden_layers,
            "hidden_units": config.hidden_units,
            "activation": config.activation,
            "dropout": config.dropout,
            "parameters": sum(parameter.numel() for parameter in model.parameters()),
        },
        "decision_threshold": decision_threshold,
        "class_weights": {
            name: float(class_weights[index]) for index, name in enumerate(TARGET_CLASSES)
        },
        "split": {
            "strategy": "Non-showcase DDO debates: 85% train / 15% validation; all 51 showcased debates: test",
            "background_cache": str(background_data),
            "test_source": str(test_data_dir),
            "train_examples": len(train_rows),
            "validation_examples": len(validation_rows),
            "test_examples": len(test_rows),
            "train_debates": len({row["debate_id"] for row in train_rows}),
            "validation_debates": len({row["debate_id"] for row in validation_rows}),
            "test_debates": len({row["debate_id"] for row in test_rows}),
            "threshold_calibration_examples": len(calibration_indices),
            "threshold_calibration_debates": len(
                {validation_rows[index]["debate_id"] for index in calibration_indices}
            ),
        },
        "baselines": {
            "majority_class": {
                "class": TARGET_CLASSES[majority_class],
                "train_count": majority_count,
                "test_accuracy": majority_accuracy,
                "test_error_rate": 1.0 - majority_accuracy,
            },
            "logistic_regression": linear_baseline,
        },
        "train": train_metrics,
        "validation": validation_metrics,
        "threshold_calibration": calibration_metrics,
        "test": test_metrics,
        "history": history,
        "feature_names": encoder.feature_names,
        "feature_groups": {
            "structured_baseline_features": encoder.feature_names[:structured_dimension],
            "mlp_additional_text_features": encoder.feature_names[structured_dimension:],
        },
        "leakage_controls": [
            "The current after stance is used only as the target.",
            "Post-debate judging fields are excluded.",
            "All voters from a debate stay in one split.",
            "Every debate shown in the browser is excluded from training and validation.",
            "The decision threshold is selected on validation data only, using validation debates "
            "that pass the same selection rule as the showcase test set.",
        ],
    }

    artifact = {
        "state_dict": model.state_dict(),
        "encoder": encoder.to_dict(),
        "model_config": {
            "input_dim": encoder.dimension,
            "hidden_units": config.hidden_units,
            "hidden_layers": config.hidden_layers,
            "activation": config.activation,
            "dropout": config.dropout,
        },
        "classes": list(TARGET_CLASSES),
        "decision_threshold": decision_threshold,
        "metrics": metrics,
    }
    model_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(artifact, model_path)
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def _compact_metrics(metrics: dict[str, Any]) -> dict[str, float]:
    return {
        key: float(metrics[key])
        for key in ("loss", "accuracy", "balanced_accuracy", "macro_f1", "switch_f1")
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the HW2 PyTorch Switch/Stay MLP.")
    parser.add_argument("--background-data", type=Path, default=DEFAULT_BACKGROUND_DATA)
    parser.add_argument("--background-profiles", type=Path, default=DEFAULT_BACKGROUND_PROFILES)
    parser.add_argument("--test-data-dir", type=Path, default=DEFAULT_TEST_DATA)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--metrics-path", type=Path, default=DEFAULT_METRICS)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--hidden-units", type=int, default=128)
    parser.add_argument("--hidden-layers", type=int, default=1)
    parser.add_argument("--activation", choices=("relu", "tanh", "gelu"), default="tanh")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.30)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--text-embeddings", action="store_true")
    parser.add_argument("--seed", type=int, default=7400)
    args = parser.parse_args()
    config = TrainConfig(
        learning_rate=args.learning_rate,
        epochs=args.epochs,
        hidden_units=args.hidden_units,
        hidden_layers=args.hidden_layers,
        activation=args.activation,
        batch_size=args.batch_size,
        seed=args.seed,
        dropout=args.dropout,
        weight_decay=args.weight_decay,
        text_embeddings=args.text_embeddings,
    )
    metrics = train_model(
        config,
        background_data=args.background_data,
        background_profiles=args.background_profiles,
        test_data_dir=args.test_data_dir,
        model_path=args.model_path,
        metrics_path=args.metrics_path,
    )
    summary = {
        "test_accuracy": metrics["test"]["accuracy"],
        "test_balanced_accuracy": metrics["test"]["balanced_accuracy"],
        "test_switch_f1": metrics["test"]["switch_f1"],
        "test_switch_pr_auc": metrics["test"]["switch_pr_auc"],
        "logistic_balanced_accuracy": metrics["baselines"]["logistic_regression"]["test"]["balanced_accuracy"],
        "model_path": str(args.model_path.resolve()),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
