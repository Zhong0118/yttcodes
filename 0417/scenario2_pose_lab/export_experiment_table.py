from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.config import get_preset


def main():
    root = Path(__file__).resolve().parent
    configs = get_preset("scenario2_full")
    rows = [cfg.to_dict() for cfg in configs]
    df = pd.DataFrame(rows)
    df.to_csv(root / "experiment_table.csv", index=False, encoding="utf-8-sig")
    df.to_excel(root / "experiment_table.xlsx", index=False)
    with open(root / "experiment_table.json", "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
    print(f"Exported {len(rows)} experiments")


if __name__ == "__main__":
    main()
