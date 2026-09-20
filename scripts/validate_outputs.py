"""Consistency checks for the frontend-ready DDO sample."""

from __future__ import annotations

import json
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
    print(
        f"Validated {len(index)} debate files, {checked_voters:,} strict sample transitions, "
        f"and {checked_switchers:,} sample switchers."
    )


if __name__ == "__main__":
    main()
