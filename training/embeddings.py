"""Sentence-embedding features for PRO and CON argument text.

Step 1 (`encode`) runs a pretrained MiniLM sentence encoder over every debate's
PRO and CON text and caches the full 384-dimensional vectors.  Step 2
(`reduce`) fits a PCA projection on training-split debates only and writes
compact per-side vectors: the background cache used for training and one
small file for the 51 showcase test debates used for live inference.

The encoder is only needed here, during preprocessing; the backend reads the
saved reduced vectors and never loads the language model.

Usage (from the project directory):
    python scripts/preprocess_training_data.py --raw-dir ../DDO_dataset/01_rawdata \
        --text-output tmp/background_text.jsonl.gz
    python -m training.embeddings encode --text tmp/background_text.jsonl.gz
    python -m training.embeddings reduce --dimensions 32
"""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path
from typing import Any

import torch

from training.features import load_background_rows, split_background_by_debate


PROJECT = Path(__file__).resolve().parents[1]
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_CHARACTERS = 1000
MAX_CHUNKS = 4
FULL_CACHE = PROJECT / "tmp" / "embeddings_full.pt"
BACKGROUND_EMBEDDINGS = PROJECT / "training" / "data" / "background_embeddings.json.gz"
TEST_EMBEDDINGS = PROJECT / "backend" / "model" / "test_embeddings.json"
TEST_DATA = PROJECT / "frontend" / "public" / "data" / "debates"
BACKGROUND_DATA = PROJECT / "training" / "data" / "background_debates.jsonl.gz"
BACKGROUND_PROFILES = PROJECT / "training" / "data" / "background_profiles.json.gz"


def _chunks(text: str) -> list[str]:
    text = " ".join(text.split())
    pieces = [text[start : start + CHUNK_CHARACTERS] for start in range(0, len(text), CHUNK_CHARACTERS)]
    return pieces[:MAX_CHUNKS] or [""]


def _test_side_text(debate: dict[str, Any], side: str) -> str:
    parts = []
    for round_record in debate.get("rounds") or []:
        for argument in round_record.get("arguments") or []:
            if str(argument.get("side") or "").upper().startswith(side):
                parts.append(str(argument.get("text") or ""))
    return " ".join(parts)


def encode(text_path: Path, batch_size: int = 64) -> None:
    from sentence_transformers import SentenceTransformer

    encoder = SentenceTransformer(MODEL_NAME, device="cpu")
    documents: list[tuple[str, str, str]] = []
    with gzip.open(text_path, "rt", encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            documents.append((record["id"], record["pro_text"], record["con_text"]))
    for path in sorted(TEST_DATA.glob("*.json")):
        debate = json.loads(path.read_text(encoding="utf-8"))
        documents.append(
            (f"test:{debate['id']}", _test_side_text(debate, "PRO"), _test_side_text(debate, "CON"))
        )

    chunks: list[str] = []
    owners: list[int] = []
    for index, (_, pro_text, con_text) in enumerate(documents):
        for side, text in ((0, pro_text), (1, con_text)):
            for piece in _chunks(text):
                chunks.append(piece)
                owners.append(index * 2 + side)
    print(f"Encoding {len(chunks):,} chunks from {len(documents):,} debates", flush=True)
    vectors = encoder.encode(
        chunks,
        batch_size=batch_size,
        convert_to_tensor=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    sums = torch.zeros(len(documents) * 2, vectors.shape[1])
    counts = torch.zeros(len(documents) * 2, 1)
    owner_index = torch.tensor(owners)
    sums.index_add_(0, owner_index, vectors.float())
    counts.index_add_(0, owner_index, torch.ones(len(owners), 1))
    side_vectors = torch.nn.functional.normalize(sums / counts.clamp(min=1), dim=1)
    FULL_CACHE.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"ids": [document[0] for document in documents], "vectors": side_vectors.view(len(documents), 2, -1)},
        FULL_CACHE,
    )
    print(f"Saved full embeddings to {FULL_CACHE}", flush=True)


def reduce(dimensions: int, seed: int = 7400) -> None:
    cache = torch.load(FULL_CACHE, weights_only=True)
    ids: list[str] = cache["ids"]
    vectors: torch.Tensor = cache["vectors"]
    rows = load_background_rows(BACKGROUND_DATA, BACKGROUND_PROFILES)
    train_rows, _ = split_background_by_debate(rows, seed)
    train_ids = {row["debate_id"] for row in train_rows}
    fit_mask = torch.tensor([identifier in train_ids for identifier in ids])

    # One shared PCA basis for both sides, fitted on training-split debates only.
    fit = vectors[fit_mask].reshape(-1, vectors.shape[-1])
    mean = fit.mean(dim=0)
    _, singular, basis = torch.linalg.svd(fit - mean, full_matrices=False)
    components = basis[:dimensions]
    scale = singular[:dimensions] / (len(fit) - 1) ** 0.5
    explained = float((singular[:dimensions] ** 2).sum() / (singular**2).sum())
    reduced = ((vectors - mean) @ components.T) / scale

    background: dict[str, list[float]] = {}
    test: dict[str, dict[str, list[float]]] = {}
    for identifier, pair in zip(ids, reduced, strict=True):
        pro = [round(float(value), 4) for value in pair[0]]
        con = [round(float(value), 4) for value in pair[1]]
        if identifier.startswith("test:"):
            test[identifier.removeprefix("test:")] = {"pro": pro, "con": con}
        else:
            background[identifier] = pro + con
    with gzip.open(BACKGROUND_EMBEDDINGS, "wt", encoding="utf-8", compresslevel=6) as handle:
        json.dump({"model": MODEL_NAME, "dimensions": dimensions, "vectors": background}, handle)
    TEST_EMBEDDINGS.write_text(
        json.dumps({"model": MODEL_NAME, "dimensions": dimensions, "vectors": test}), encoding="utf-8"
    )
    print(
        f"Wrote {len(background):,} background and {len(test)} test debates with "
        f"{dimensions} PCA dimensions per side ({explained:.1%} of variance)."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Build sentence-embedding text features.")
    commands = parser.add_subparsers(dest="command", required=True)
    encode_parser = commands.add_parser("encode")
    encode_parser.add_argument("--text", type=Path, required=True)
    reduce_parser = commands.add_parser("reduce")
    reduce_parser.add_argument("--dimensions", type=int, default=32)
    args = parser.parse_args()
    if args.command == "encode":
        encode(args.text)
    else:
        reduce(args.dimensions)


if __name__ == "__main__":
    main()
