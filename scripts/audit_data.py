"""Run the DDO data audit without changing the frontend sample."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from preprocess_hw1 import audit_and_select


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--debates", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("outputs/audit_summary.json"))
    args = parser.parse_args()
    audit, _ = audit_and_select(args.debates.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
