"""Build a compact training cache from every non-showcase DDO debate.

The 51 debates rendered by the web app are excluded by their original DDO
keys and are reserved as the final test set. The remaining raw file is read
once with the streaming parser, then saved as a compact gzip cache so normal
training runs never need to parse the 1.2 GB source file again.
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from data_utils import iter_json_object, parse_vote, participant_sides, profile_record


PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))


def side_text(debate: dict[str, Any], side: str, limit: int) -> str:
    """Concatenate one side's arguments, truncated for the sentence encoder."""

    parts = []
    for round_record in debate.get("rounds") or []:
        for argument in round_record if isinstance(round_record, list) else []:
            if isinstance(argument, dict) and str(argument.get("side") or "").upper().startswith(side):
                parts.append(" ".join(str(argument.get("text") or "").split()))
    return " ".join(parts)[:limit]


def showcase_keys(test_data_dir: Path) -> set[str]:
    keys = set()
    for path in test_data_dir.glob("*.json"):
        debate = json.loads(path.read_text(encoding="utf-8"))
        key = str(debate.get("debate_key") or "")
        if key:
            keys.add(key)
    if len(keys) != 51:
        raise ValueError(f"Expected 51 unique showcase debate keys, found {len(keys)}")
    return keys


def build_cache(
    debates_path: Path,
    users_path: Path,
    test_data_dir: Path,
    output_dir: Path,
    text_output: Path | None = None,
) -> dict[str, Any]:
    from training.features import argument_feature_record

    excluded_keys = showcase_keys(test_data_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    debates_output = output_dir / "background_debates.jsonl.gz"
    profiles_output = output_dir / "background_profiles.json.gz"
    manifest_output = output_dir / "background_manifest.json"

    counts = Counter()
    transitions = Counter()
    wanted_users: set[str] = set()
    seen_showcase: set[str] = set()

    text_handle = gzip.open(text_output, "wt", encoding="utf-8") if text_output else None
    with gzip.open(debates_output, "wt", encoding="utf-8", compresslevel=6) as output:
        for serial, (key, debate) in enumerate(iter_json_object(debates_path), start=1):
            if key in excluded_keys:
                seen_showcase.add(key)
                continue
            sides = participant_sides(debate)
            voters = []
            for raw_vote in debate.get("votes") or []:
                if not isinstance(raw_vote, dict):
                    continue
                voter = parse_vote(raw_vote, sides)
                if not voter["is_valid_transition"]:
                    continue
                label = int(voter["switched"])
                username = str(voter["username"])
                voters.append({"username": username, "before": voter["before"], "label": label})
                wanted_users.add(username.casefold())
                counts["examples"] += 1
                counts["switchers"] += label
                transitions[voter["transition"]] += 1
            if not voters:
                continue

            features = argument_feature_record(debate)
            debaters = {side: name for name, side in (sides or {}).items()}
            wanted_users.update(name.casefold() for name in debaters.values())
            record = {
                "id": f"background:{serial}",
                "category": debate.get("category") or "Uncategorized",
                "round_count": len(debate.get("rounds") or []),
                "pro_debater": debaters.get("PRO"),
                "con_debater": debaters.get("CON"),
                **features,
                "voters": voters,
            }
            output.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
            if text_handle:
                text_record = {
                    "id": record["id"],
                    "pro_text": side_text(debate, "PRO", 4000),
                    "con_text": side_text(debate, "CON", 4000),
                }
                text_handle.write(json.dumps(text_record, ensure_ascii=False) + "\n")
            counts["debates"] += 1
            if serial % 5000 == 0:
                print(
                    f"Scanned {serial:,} debates | cached {counts['debates']:,} debates | "
                    f"{counts['examples']:,} transitions",
                    flush=True,
                )

    if text_handle:
        text_handle.close()
    if seen_showcase != excluded_keys:
        missing = sorted(excluded_keys - seen_showcase)
        raise ValueError(f"The raw corpus did not contain {len(missing)} showcase debates")

    profiles: dict[str, dict[str, Any]] = {}
    if users_path.exists():
        for username, profile in iter_json_object(users_path):
            normalized = username.casefold()
            if normalized not in wanted_users:
                continue
            profiles[normalized] = profile_record(profile)
    with gzip.open(profiles_output, "wt", encoding="utf-8", compresslevel=6) as output:
        json.dump(profiles, output, ensure_ascii=False, separators=(",", ":"))

    manifest = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "source_debates": str(debates_path.resolve()),
        "source_users": str(users_path.resolve()),
        "policy": "All 51 debates rendered by the application are excluded and reserved for test.",
        "showcase_test_debates": len(excluded_keys),
        "background_debates": counts["debates"],
        "background_examples": counts["examples"],
        "background_switchers": counts["switchers"],
        "background_switch_rate": counts["switchers"] / max(counts["examples"], 1),
        "profiles_matched": len(profiles),
        "transition_counts": dict(transitions),
    }
    manifest_output.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2), flush=True)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Cache non-showcase DDO debates for HW2 training.")
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument(
        "--test-data-dir",
        type=Path,
        default=PROJECT / "frontend" / "public" / "data" / "debates",
    )
    parser.add_argument("--output-dir", type=Path, default=PROJECT / "training" / "data")
    parser.add_argument(
        "--text-output",
        type=Path,
        default=None,
        help="Optional gzip JSONL of truncated argument text for sentence embeddings",
    )
    args = parser.parse_args()
    build_cache(
        args.raw_dir / "debates.json",
        args.raw_dir / "users.json",
        args.test_data_dir,
        args.output_dir,
        args.text_output,
    )


if __name__ == "__main__":
    main()
