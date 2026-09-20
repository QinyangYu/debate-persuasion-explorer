"""Streaming helpers and stance parsing for the DDO dataset.

The raw debates file is larger than 1 GB.  These utilities iterate over the
top-level JSON object one debate at a time so preprocessing does not require
loading the entire file into memory.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator


STANCE_FIELDS = {
    "before": "Agreed with before the debate",
    "after": "Agreed with after the debate",
}


def iter_json_object(path: Path, chunk_size: int = 1024 * 1024) -> Iterator[tuple[str, Any]]:
    """Yield key/value pairs from a JSON object without loading it all at once."""

    decoder = json.JSONDecoder()
    with path.open("r", encoding="utf-8") as handle:
        buffer = ""
        position = 0
        eof = False

        def fill() -> bool:
            nonlocal buffer, eof
            chunk = handle.read(chunk_size)
            if not chunk:
                eof = True
                return False
            buffer += chunk
            return True

        def compact() -> None:
            nonlocal buffer, position
            if position > chunk_size * 2:
                buffer = buffer[position:]
                position = 0

        def skip_space() -> None:
            nonlocal position
            while True:
                while position < len(buffer) and buffer[position].isspace():
                    position += 1
                if position < len(buffer) or eof:
                    return
                fill()

        def decode_next() -> Any:
            nonlocal position
            while True:
                try:
                    value, end = decoder.raw_decode(buffer, position)
                    position = end
                    return value
                except json.JSONDecodeError:
                    if not fill():
                        raise

        fill()
        skip_space()
        if position >= len(buffer) or buffer[position] != "{":
            raise ValueError(f"Expected a top-level JSON object in {path}")
        position += 1

        while True:
            compact()
            skip_space()
            if position < len(buffer) and buffer[position] == "}":
                return

            key = decode_next()
            if not isinstance(key, str):
                raise ValueError(f"Expected an object key in {path}")

            skip_space()
            if position >= len(buffer):
                fill()
                skip_space()
            if buffer[position] != ":":
                raise ValueError(f"Expected ':' after key {key!r}")
            position += 1

            skip_space()
            value = decode_next()
            yield key, value

            skip_space()
            if position >= len(buffer) and not eof:
                fill()
                skip_space()
            if position < len(buffer) and buffer[position] == ",":
                position += 1
                continue
            if position < len(buffer) and buffer[position] == "}":
                return
            raise ValueError(f"Expected ',' or '}}' after key {key!r}")


def _casefold_lookup(mapping: dict[str, Any], key: str) -> Any:
    if key in mapping:
        return mapping[key]
    wanted = key.casefold()
    for candidate, value in mapping.items():
        if str(candidate).casefold() == wanted:
            return value
    return None


def participant_sides(debate: dict[str, Any]) -> dict[str, str] | None:
    """Return participant-name -> PRO/CON when metadata is unambiguous."""

    sides: dict[str, str] = {}
    for number in (1, 2):
        name = str(debate.get(f"participant_{number}_name") or "").strip()
        raw_position = str(debate.get(f"participant_{number}_position") or "").strip().upper()
        if raw_position.startswith("PRO"):
            side = "PRO"
        elif raw_position.startswith("CON"):
            side = "CON"
        else:
            return None
        if not name or name.casefold() in {existing.casefold() for existing in sides}:
            return None
        sides[name] = side
    if set(sides.values()) != {"PRO", "CON"}:
        return None
    return sides


def classify_stance(
    votes_map: Any,
    sides: dict[str, str] | None,
    criterion: str,
) -> str:
    """Map one agreement criterion to PRO, CON, TIE, UNKNOWN, or AMBIGUOUS."""

    if not isinstance(votes_map, dict) or not sides:
        return "UNKNOWN"

    selected: list[str] = []
    for participant, side in sides.items():
        record = _casefold_lookup(votes_map, participant)
        if isinstance(record, dict) and record.get(criterion) is True:
            selected.append(side)

    tied = _casefold_lookup(votes_map, "Tied")
    if isinstance(tied, dict) and tied.get(criterion) is True:
        selected.append("TIE")

    if not selected:
        return "UNKNOWN"
    if len(selected) > 1:
        return "AMBIGUOUS"
    return selected[0]


def parse_vote(vote: dict[str, Any], sides: dict[str, str] | None) -> dict[str, Any]:
    before = classify_stance(vote.get("votes_map"), sides, STANCE_FIELDS["before"])
    after = classify_stance(vote.get("votes_map"), sides, STANCE_FIELDS["after"])
    strict = before in {"PRO", "CON"} and after in {"PRO", "CON"}
    return {
        "username": str(vote.get("user_name") or "Unknown voter"),
        "vote_time": vote.get("time"),
        "before": before,
        "after": after,
        "transition": f"{before} -> {after}",
        "is_valid_transition": strict,
        "switched": bool(strict and before != after),
    }

