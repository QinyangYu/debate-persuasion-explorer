"""Consistency checks for the frontend-ready DDO sample."""

from __future__ import annotations

import json
import gzip
from collections import Counter
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]
DATA = PROJECT / "frontend" / "public" / "data"


def main() -> None:
    index = json.loads((DATA / "debates_index.json").read_text(encoding="utf-8"))
    assert index, "Debate index is empty"
    assert len({item["id"] for item in index}) == len(index), "Duplicate debate IDs"

    checked_voters = 0
    checked_switchers = 0
    for item in index:
        path = DATA / "debates" / f"{item['id']}.json"
        assert path.exists(), f"Missing detail file: {path.name}"
        detail = json.loads(path.read_text(encoding="utf-8"))
        assert detail["id"] == item["id"]
        assert detail["rounds"], f"No rounds in {item['id']}"

        valid = [voter for voter in detail["voters"] if voter["is_valid_transition"]]
        transition_counts = Counter(voter["transition"] for voter in valid)
        switchers = [voter for voter in valid if voter["switched"]]
        assert len(valid) == detail["summary"]["valid_transitions"]
        assert len(switchers) == detail["summary"]["switchers"]
        assert len(switchers) > 0, f"Sample debate has no visible switch: {item['id']}"
        assert detail["summary"]["transitions"] == {
            name: transition_counts.get(name, 0)
            for name in ("PRO -> PRO", "PRO -> CON", "CON -> PRO", "CON -> CON")
        }
        expected_rate = round(len(switchers) / len(valid), 4)
        assert detail["summary"]["switch_rate"] == expected_rate
        checked_voters += len(valid)
        checked_switchers += len(switchers)

    audit = json.loads((PROJECT / "outputs" / "audit_summary.json").read_text(encoding="utf-8"))
    strict_total = sum(audit["strict_transitions"].values())
    assert strict_total == audit["counts"]["votes_valid_before_and_after"]
    assert (
        audit["strict_transitions"]["PRO -> CON"]
        + audit["strict_transitions"]["CON -> PRO"]
        == audit["counts"]["switchers"]
    )
    metrics_path = PROJECT / "outputs" / "training_metrics.json"
    model_path = PROJECT / "backend" / "model" / "transition_mlp.pt"
    assert metrics_path.exists(), "Missing HW2 training metrics"
    assert model_path.exists(), "Missing trained HW2 weights"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert metrics["architecture"]["hidden_layers"] >= 1
    assert metrics["task"] == "Binary observed stance-change prediction"
    assert metrics["test"]["switch_recall"] > 0, "Model collapsed to all-Stay predictions"
    assert (
        metrics["test"]["balanced_accuracy"]
        > metrics["baselines"]["logistic_regression"]["test"]["balanced_accuracy"]
    ), "MLP did not beat the shallow logistic baseline on balanced accuracy"
    assert metrics["test"]["examples"] == metrics["split"]["test_examples"]
    assert metrics["split"]["test_debates"] == len(index) == 51
    assert metrics["split"]["test_examples"] == checked_voters
    manifest = json.loads(
        (PROJECT / "training" / "data" / "background_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["showcase_test_debates"] == len(index)
    assert manifest["background_examples"] + checked_voters == strict_total
    assert manifest["background_switchers"] + checked_switchers == audit["counts"]["switchers"]
    with gzip.open(PROJECT / "training" / "data" / "background_profiles.json.gz", "rt", encoding="utf-8") as handle:
        assert json.load(handle), "Background profile cache is empty"
    print(
        f"Validated {len(index)} debate files, {checked_voters:,} strict sample transitions, "
        f"{checked_switchers:,} sample switchers, and a binary MLP that beats the logistic "
        "baseline on balanced accuracy."
    )


if __name__ == "__main__":
    main()
