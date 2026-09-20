"""Audit DDO and produce a compact, frontend-ready HW1 sample.

Usage (from the project directory):
    python scripts/preprocess_hw1.py --raw-dir ../DDO_dataset/01_rawdata
"""

from __future__ import annotations

import argparse
import hashlib
import heapq
import json
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from data_utils import STANCE_FIELDS, classify_stance, iter_json_object, parse_vote, participant_sides


def safe_int(value: Any) -> int:
    text = str(value or "0").replace(",", "")
    digits = "".join(character for character in text if character.isdigit())
    return int(digits or 0)


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = round((len(ordered) - 1) * fraction)
    return round(float(ordered[index]), 4)


def detail_id(key: str) -> str:
    return hashlib.sha1(key.encode("utf-8")).hexdigest()[:14]


def participant_records(debate: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "username": debate.get(f"participant_{number}_name"),
            "position": str(debate.get(f"participant_{number}_position") or "").upper(),
            "points": debate.get(f"participant_{number}_points"),
            "status": debate.get(f"participant_{number}_status"),
        }
        for number in (1, 2)
    ]


def prepare_detail(key: str, debate: dict[str, Any], voters: list[dict[str, Any]]) -> dict[str, Any]:
    strict_voters = [voter for voter in voters if voter["is_valid_transition"]]
    transitions = Counter(voter["transition"] for voter in strict_voters)
    switchers = sum(voter["switched"] for voter in strict_voters)
    valid_count = len(strict_voters)
    rounds = []
    for round_number, utterances in enumerate(debate.get("rounds") or [], start=1):
        rounds.append(
            {
                "round": round_number,
                "arguments": [
                    {
                        "side": str(argument.get("side") or "Unknown").upper(),
                        "text": str(argument.get("text") or "").strip(),
                    }
                    for argument in (utterances or [])
                    if isinstance(argument, dict)
                ],
            }
        )

    return {
        "id": detail_id(key),
        "debate_key": key,
        "url": debate.get("url"),
        "title": debate.get("title") or key,
        "category": debate.get("category") or "Uncategorized",
        "start_date": debate.get("start_date"),
        "status": debate.get("debate_status"),
        "voting_style": debate.get("voting_style"),
        "views": safe_int(debate.get("number_of_views")),
        "comments": len(debate.get("comments") or []),
        "round_count": len(rounds),
        "participants": participant_records(debate),
        "voters": voters,
        "summary": {
            "total_votes": len(voters),
            "valid_transitions": valid_count,
            "switchers": switchers,
            "switch_rate": round(switchers / valid_count, 4) if valid_count else 0.0,
            "transitions": {
                transition: transitions.get(transition, 0)
                for transition in ("PRO -> PRO", "PRO -> CON", "CON -> PRO", "CON -> CON")
            },
        },
        "rounds": rounds,
        "profiles": {},
    }


def push_candidate(heap: list, size: int, score: tuple, serial: int, detail: dict[str, Any]) -> None:
    entry = (score, serial, detail)
    if len(heap) < size:
        heapq.heappush(heap, entry)
    elif score > heap[0][0]:
        heapq.heapreplace(heap, entry)


def audit_and_select(debates_path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    counts = Counter()
    before_labels = Counter()
    after_labels = Counter()
    transitions = Counter()
    categories: dict[str, Counter] = defaultdict(Counter)
    years: dict[str, Counter] = defaultdict(Counter)
    valid_counts: list[float] = []
    switch_rates: list[float] = []
    usable_thresholds = Counter()

    top_valid: list = []
    top_switchers: list = []
    top_rate: list = []

    for serial, (key, debate) in enumerate(iter_json_object(debates_path), start=1):
        counts["debates"] += 1
        votes = debate.get("votes") or []
        counts["votes"] += len(votes)
        sides = participant_sides(debate)
        if not sides:
            counts["debates_invalid_participant_mapping"] += 1

        parsed_voters = []
        debate_transitions = Counter()
        debate_switchers = 0
        valid_before = 0
        valid_after = 0
        valid_both = 0

        for vote in votes:
            if not isinstance(vote, dict):
                counts["malformed_votes"] += 1
                continue
            before = classify_stance(vote.get("votes_map"), sides, STANCE_FIELDS["before"])
            after = classify_stance(vote.get("votes_map"), sides, STANCE_FIELDS["after"])
            before_labels[before] += 1
            after_labels[after] += 1
            before_valid = before in {"PRO", "CON"}
            after_valid = after in {"PRO", "CON"}
            valid_before += before_valid
            valid_after += after_valid
            if before_valid and after_valid:
                valid_both += 1
                transition = f"{before} -> {after}"
                transitions[transition] += 1
                debate_transitions[transition] += 1
                debate_switchers += before != after
            parsed_voters.append(parse_vote(vote, sides))

        counts["votes_valid_before"] += valid_before
        counts["votes_valid_after"] += valid_after
        counts["votes_valid_before_and_after"] += valid_both
        counts["switchers"] += debate_switchers
        counts["debates_with_votes"] += bool(votes)
        counts["debates_with_valid_transitions"] += bool(valid_both)
        category = str(debate.get("category") or "Uncategorized")
        category_counter = categories[category]
        category_counter["debates"] += 1
        category_counter["valid_transitions"] += valid_both
        category_counter["switchers"] += debate_switchers
        date = str(debate.get("start_date") or "")
        year = date[-4:] if len(date) >= 4 and date[-4:].isdigit() else "Unknown"
        year_counter = years[year]
        year_counter["debates"] += 1
        year_counter["valid_transitions"] += valid_both
        year_counter["switchers"] += debate_switchers

        valid_counts.append(float(valid_both))
        rate = debate_switchers / valid_both if valid_both else 0.0
        if valid_both:
            switch_rates.append(rate)
        for threshold in (1, 5, 10, 20, 50):
            if valid_both >= threshold:
                usable_thresholds[str(threshold)] += 1

        if valid_both >= 5 and debate_switchers > 0 and debate.get("rounds"):
            detail = prepare_detail(key, debate, parsed_voters)
            push_candidate(top_valid, 30, (valid_both, debate_switchers), serial, detail)
            push_candidate(top_switchers, 20, (debate_switchers, valid_both), serial, detail)
            if valid_both >= 10:
                push_candidate(top_rate, 15, (round(rate, 8), valid_both), serial, detail)

        if serial % 5000 == 0:
            print(
                f"Processed {serial:,} debates | {counts['votes']:,} votes | "
                f"{counts['votes_valid_before_and_after']:,} valid transitions",
                flush=True,
            )

    selected: dict[str, dict[str, Any]] = {}
    for heap in (top_valid, top_switchers, top_rate):
        for _, _, detail in sorted(heap, reverse=True):
            selected[detail["id"]] = detail

    overall_valid = counts["votes_valid_before_and_after"]
    audit = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "source": str(debates_path),
        "definitions": {
            "valid_transition": "Exactly one participant (PRO or CON) selected both before and after; ties, missing selections, and contradictory selections are excluded.",
            "switcher": "A valid transition where before stance differs from after stance.",
        },
        "counts": dict(counts),
        "before_labels": dict(before_labels),
        "after_labels": dict(after_labels),
        "strict_transitions": {
            transition: transitions.get(transition, 0)
            for transition in ("PRO -> PRO", "PRO -> CON", "CON -> PRO", "CON -> CON")
        },
        "overall_switch_rate": round(counts["switchers"] / overall_valid, 6) if overall_valid else 0.0,
        "usable_debates_by_minimum_valid_voters": dict(usable_thresholds),
        "valid_voters_per_debate": {
            "median": percentile(valid_counts, 0.5),
            "p75": percentile(valid_counts, 0.75),
            "p90": percentile(valid_counts, 0.9),
            "p95": percentile(valid_counts, 0.95),
            "p99": percentile(valid_counts, 0.99),
            "max": max(valid_counts, default=0),
        },
        "switch_rate_among_debates_with_valid_transitions": {
            "median": percentile(switch_rates, 0.5),
            "p75": percentile(switch_rates, 0.75),
            "p90": percentile(switch_rates, 0.9),
            "max": round(max(switch_rates, default=0), 4),
        },
        "categories": {name: dict(value) for name, value in sorted(categories.items())},
        "years": {name: dict(value) for name, value in sorted(years.items())},
        "selected_sample_debates": len(selected),
    }
    return audit, list(selected.values())


def attach_profiles(users_path: Path, details: list[dict[str, Any]]) -> None:
    wanted = {
        voter["username"].casefold()
        for detail in details
        for voter in detail["voters"]
        if voter["username"]
    }
    wanted.update(
        str(participant["username"]).casefold()
        for detail in details
        for participant in detail["participants"]
        if participant["username"]
    )
    profiles: dict[str, dict[str, Any]] = {}
    for username, profile in iter_json_object(users_path):
        if username.casefold() not in wanted:
            continue
        profiles[username.casefold()] = {
            "political_ideology": profile.get("political_ideology"),
            "religious_ideology": profile.get("religious_ideology"),
            "elo_ranking": profile.get("elo_ranking"),
            "number_of_all_debates": safe_int(profile.get("number_of_all_debates")),
            "number_of_voted_debates": safe_int(profile.get("number_of_voted_debates")),
        }
    for detail in details:
        names = {voter["username"] for voter in detail["voters"]}
        names.update(participant["username"] for participant in detail["participants"])
        detail["profiles"] = {
            name: profiles[name.casefold()]
            for name in names
            if name and name.casefold() in profiles
        }


def write_outputs(project_dir: Path, audit: dict[str, Any], details: list[dict[str, Any]]) -> None:
    output_dir = project_dir / "outputs"
    public_dir = project_dir / "frontend" / "public" / "data"
    detail_dir = public_dir / "debates"
    output_dir.mkdir(parents=True, exist_ok=True)
    detail_dir.mkdir(parents=True, exist_ok=True)

    details.sort(
        key=lambda detail: (
            detail["summary"]["valid_transitions"],
            detail["summary"]["switchers"],
        ),
        reverse=True,
    )
    index = []
    for detail in details:
        (detail_dir / f"{detail['id']}.json").write_text(
            json.dumps(detail, ensure_ascii=False, separators=(",", ":")),
            encoding="utf-8",
        )
        index.append(
            {
                "id": detail["id"],
                "title": detail["title"],
                "category": detail["category"],
                "start_date": detail["start_date"],
                "participants": detail["participants"],
                **detail["summary"],
            }
        )

    (output_dir / "audit_summary.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (public_dir / "debates_index.json").write_text(
        json.dumps(index, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Wrote audit to {output_dir / 'audit_summary.json'}")
    print(f"Wrote {len(index)} sample debates to {detail_dir}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, required=True, help="Directory containing debates.json and users.json")
    parser.add_argument("--skip-users", action="store_true", help="Do not enrich selected users with profile metadata")
    args = parser.parse_args()

    raw_dir = args.raw_dir.resolve()
    project_dir = Path(__file__).resolve().parents[1]
    debates_path = raw_dir / "debates.json"
    users_path = raw_dir / "users.json"
    if not debates_path.exists():
        raise FileNotFoundError(debates_path)

    audit, details = audit_and_select(debates_path)
    if not args.skip_users and users_path.exists():
        print("Attaching non-sensitive user profile fields...", flush=True)
        attach_profiles(users_path, details)
    write_outputs(project_dir, audit, details)


if __name__ == "__main__":
    main()

