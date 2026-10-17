from __future__ import annotations

import hashlib
import gzip
import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


TRANSITIONS = ("PRO -> PRO", "PRO -> CON", "CON -> PRO", "CON -> CON")
TARGET_CLASSES = ("STAY", "SWITCH")
TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z']+")
NUMERIC_NAMES = (
    "before_is_pro",
    "round_count",
    "log_pro_words",
    "log_con_words",
    "argument_length_balance",
    "log_user_debates",
    "log_user_votes",
    "log_user_elo",
    "has_profile",
)


def _safe_float(value: Any) -> float:
    try:
        return float(str(value or "0").replace(",", ""))
    except (TypeError, ValueError):
        return 0.0


def _tokens(text: str) -> list[str]:
    return [match.group(0).casefold() for match in TOKEN_RE.finditer(text)]


def _argument_tokens(debate: dict[str, Any]) -> tuple[list[str], list[str]]:
    pro: list[str] = []
    con: list[str] = []
    for round_record in debate.get("rounds") or []:
        arguments = (
            round_record.get("arguments") or []
            if isinstance(round_record, dict)
            else round_record
            if isinstance(round_record, list)
            else []
        )
        for argument in arguments:
            if not isinstance(argument, dict):
                continue
            side = str(argument.get("side") or "").upper()
            target = pro if side.startswith("PRO") else con if side.startswith("CON") else None
            if target is not None:
                target.extend(_tokens(str(argument.get("text") or "")))
    return pro, con


def argument_feature_record(
    debate: dict[str, Any], text_dimensions: int = 32
) -> dict[str, Any]:
    """Create a compact, reusable representation of one debate's arguments."""

    pro_tokens, con_tokens = _argument_tokens(debate)
    return {
        "pro_word_count": len(pro_tokens),
        "con_word_count": len(con_tokens),
        "pro_text_features": _hashed_text(pro_tokens, text_dimensions),
        "con_text_features": _hashed_text(con_tokens, text_dimensions),
    }


def _hashed_text(tokens: Iterable[str], dimensions: int) -> list[float]:
    values = [0.0] * dimensions
    count = 0
    for token in tokens:
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        number = int.from_bytes(digest, "little")
        index = number % dimensions
        sign = 1.0 if (number >> 8) & 1 else -1.0
        values[index] += sign
        count += 1
    scale = math.sqrt(max(count, 1))
    return [value / scale for value in values]


def load_rows(data_dir: Path) -> list[dict[str, Any]]:
    """Build one leakage-safe learning row for every strict voter transition."""

    rows: list[dict[str, Any]] = []
    for path in sorted(data_dir.glob("*.json")):
        debate = json.loads(path.read_text(encoding="utf-8"))
        pro_tokens, con_tokens = _argument_tokens(debate)
        shared = {
            "debate_id": debate["id"],
            "category": str(debate.get("category") or "Unknown"),
            "round_count": float(debate.get("round_count") or len(debate.get("rounds") or [])),
            "pro_tokens": pro_tokens,
            "con_tokens": con_tokens,
            "pro_word_count": len(pro_tokens),
            "con_word_count": len(con_tokens),
        }
        profiles = debate.get("profiles") or {}
        for voter in debate.get("voters") or []:
            transition = str(voter.get("transition") or "")
            if not voter.get("is_valid_transition") or transition not in TRANSITIONS:
                continue
            username = str(voter.get("username") or "Unknown voter")
            profile = profiles.get(username) or {}
            rows.append(build_row(shared, voter, profile))
    return rows


def load_background_rows(data_path: Path, profiles_path: Path) -> list[dict[str, Any]]:
    """Load the compact non-showcase corpus produced by preprocess_training_data.py."""

    if not data_path.exists():
        raise FileNotFoundError(
            f"Missing background training cache: {data_path}. "
            "Run scripts/preprocess_training_data.py first."
        )
    profiles: dict[str, dict[str, Any]] = {}
    if profiles_path.exists():
        with gzip.open(profiles_path, "rt", encoding="utf-8") as handle:
            profiles = json.load(handle)

    rows: list[dict[str, Any]] = []
    with gzip.open(data_path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            debate = json.loads(line)
            shared = {
                "debate_id": debate["id"],
                "category": str(debate.get("category") or "Unknown"),
                "round_count": float(debate.get("round_count") or 0),
                "pro_word_count": int(debate.get("pro_word_count") or 0),
                "con_word_count": int(debate.get("con_word_count") or 0),
                "pro_text_features": debate.get("pro_text_features") or [],
                "con_text_features": debate.get("con_text_features") or [],
            }
            for voter in debate.get("voters") or []:
                before = str(voter.get("before") or "")
                label = int(voter.get("label") or 0)
                after = before if label == 0 else "CON" if before == "PRO" else "PRO"
                username = str(voter.get("username") or "Unknown voter")
                rows.append(
                    {
                        **shared,
                        "username": username,
                        "before": before,
                        "transition": f"{before} -> {after}",
                        "label": label,
                        "profile": profiles.get(username.casefold()) or {},
                    }
                )
    return rows


def build_inference_row(
    debate: dict[str, Any], voter: dict[str, Any], profile: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Create the exact feature row used for a live app prediction."""

    pro_tokens, con_tokens = _argument_tokens(debate)
    shared = {
        "debate_id": debate["id"],
        "category": str(debate.get("category") or "Unknown"),
        "round_count": float(debate.get("round_count") or len(debate.get("rounds") or [])),
        "pro_tokens": pro_tokens,
        "con_tokens": con_tokens,
        "pro_word_count": len(pro_tokens),
        "con_word_count": len(con_tokens),
    }
    return build_row(shared, voter, profile or {})


def build_row(
    shared: dict[str, Any], voter: dict[str, Any], profile: dict[str, Any]
) -> dict[str, Any]:
    transition = str(voter.get("transition") or "")
    return {
        **shared,
        "username": str(voter.get("username") or "Unknown voter"),
        "before": voter.get("before"),
        "transition": transition,
        "label": int(bool(voter.get("switched"))) if transition in TRANSITIONS else -1,
        "profile": profile,
    }


@dataclass
class FeatureEncoder:
    categories: list[str]
    political_values: list[str]
    religious_values: list[str]
    numeric_means: list[float]
    numeric_stds: list[float]
    text_dimensions: int = 32

    @classmethod
    def fit(cls, rows: list[dict[str, Any]], text_dimensions: int = 32) -> "FeatureEncoder":
        categories = _vocabulary(rows, lambda row: row["category"], minimum=1, limit=24)
        political = _vocabulary(
            rows,
            lambda row: (row.get("profile") or {}).get("political_ideology") or "Unknown",
            minimum=2,
            limit=16,
        )
        religious = _vocabulary(
            rows,
            lambda row: (row.get("profile") or {}).get("religious_ideology") or "Unknown",
            minimum=2,
            limit=16,
        )
        raw = [_numeric_values(row) for row in rows]
        means = [sum(column) / len(column) for column in zip(*raw, strict=True)]
        stds = []
        for index, mean in enumerate(means):
            variance = sum((values[index] - mean) ** 2 for values in raw) / max(len(raw) - 1, 1)
            stds.append(max(math.sqrt(variance), 1e-6))
        return cls(categories, political, religious, means, stds, text_dimensions)

    @property
    def dimension(self) -> int:
        return (
            len(NUMERIC_NAMES)
            + len(self.categories)
            + 1
            + len(self.political_values)
            + 1
            + len(self.religious_values)
            + 1
            + self.text_dimensions * 2
        )

    @property
    def feature_names(self) -> list[str]:
        return [
            *NUMERIC_NAMES,
            *[f"category={value}" for value in self.categories],
            "category=Other",
            *[f"politics={value}" for value in self.political_values],
            "politics=Other",
            *[f"religion={value}" for value in self.religious_values],
            "religion=Other",
            *[f"pro_text_hash_{index}" for index in range(self.text_dimensions)],
            *[f"con_text_hash_{index}" for index in range(self.text_dimensions)],
        ]

    def transform(self, row: dict[str, Any]) -> list[float]:
        numeric = [
            (value - mean) / std
            for value, mean, std in zip(
                _numeric_values(row), self.numeric_means, self.numeric_stds, strict=True
            )
        ]
        profile = row.get("profile") or {}
        return [
            *numeric,
            *_one_hot(str(row.get("category") or "Unknown"), self.categories),
            *_one_hot(str(profile.get("political_ideology") or "Unknown"), self.political_values),
            *_one_hot(str(profile.get("religious_ideology") or "Unknown"), self.religious_values),
            *_text_features(row, "pro", self.text_dimensions),
            *_text_features(row, "con", self.text_dimensions),
        ]

    def to_dict(self) -> dict[str, Any]:
        return {
            "categories": self.categories,
            "political_values": self.political_values,
            "religious_values": self.religious_values,
            "numeric_means": self.numeric_means,
            "numeric_stds": self.numeric_stds,
            "text_dimensions": self.text_dimensions,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "FeatureEncoder":
        return cls(**payload)


def _vocabulary(rows, getter, minimum: int, limit: int) -> list[str]:
    counts = Counter(str(getter(row)) for row in rows)
    return [value for value, count in counts.most_common(limit) if count >= minimum]


def _one_hot(value: str, vocabulary: list[str]) -> list[float]:
    output = [0.0] * (len(vocabulary) + 1)
    try:
        output[vocabulary.index(value)] = 1.0
    except ValueError:
        output[-1] = 1.0
    return output


def _numeric_values(row: dict[str, Any]) -> list[float]:
    profile = row.get("profile") or {}
    pro_count = int(row.get("pro_word_count", len(row.get("pro_tokens") or [])))
    con_count = int(row.get("con_word_count", len(row.get("con_tokens") or [])))
    total_words = max(pro_count + con_count, 1)
    return [
        1.0 if row.get("before") == "PRO" else 0.0,
        float(row.get("round_count") or 0.0),
        math.log1p(pro_count),
        math.log1p(con_count),
        (pro_count - con_count) / total_words,
        math.log1p(max(_safe_float(profile.get("number_of_all_debates")), 0.0)),
        math.log1p(max(_safe_float(profile.get("number_of_voted_debates")), 0.0)),
        math.log1p(max(_safe_float(profile.get("elo_ranking")), 0.0)),
        1.0 if profile else 0.0,
    ]


def _text_features(row: dict[str, Any], side: str, dimensions: int) -> list[float]:
    cached = row.get(f"{side}_text_features")
    if isinstance(cached, list) and len(cached) == dimensions:
        return [float(value) for value in cached]
    return _hashed_text(row.get(f"{side}_tokens") or [], dimensions)


def split_by_debate(
    rows: list[dict[str, Any]], seed: int = 7400
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    debates = sorted({row["debate_id"] for row in rows})
    ranked = sorted(
        debates,
        key=lambda value: hashlib.sha256(f"{seed}:{value}".encode("utf-8")).hexdigest(),
    )
    train_end = max(1, round(len(ranked) * 0.70))
    val_end = max(train_end + 1, round(len(ranked) * 0.85))
    groups = (set(ranked[:train_end]), set(ranked[train_end:val_end]), set(ranked[val_end:]))
    return tuple([[row for row in rows if row["debate_id"] in group] for group in groups])  # type: ignore[return-value]


def split_background_by_debate(
    rows: list[dict[str, Any]], seed: int = 7400, validation_fraction: float = 0.15
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split only the non-showcase corpus; the 51 showcased debates are a separate test set."""

    debates = sorted({row["debate_id"] for row in rows})
    ranked = sorted(
        debates,
        key=lambda value: hashlib.sha256(f"{seed}:{value}".encode("utf-8")).hexdigest(),
    )
    validation_count = max(1, round(len(ranked) * validation_fraction))
    validation_ids = set(ranked[:validation_count])
    train_ids = set(ranked[validation_count:])
    train_rows = [row for row in rows if row["debate_id"] in train_ids]
    validation_rows = [row for row in rows if row["debate_id"] in validation_ids]
    return train_rows, validation_rows

