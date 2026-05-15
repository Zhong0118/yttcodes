from __future__ import annotations
from pathlib import Path
import uuid
from typing import Dict, Any

from src.config import RunConfig
from src.io_utils import ensure_dir, save_json, save_dataframe
from src.pipeline import analyze_single_video


def run_and_save(cfg: RunConfig, output_root: Path) -> Dict[str, Any]:
    run_id = f"{cfg.exp_group}_r{cfg.repeat_idx}_{uuid.uuid4().hex[:8]}"
    run_dir = output_root / run_id
    ensure_dir(run_dir)

    chunk_df, summary = analyze_single_video(cfg)
    save_dataframe(chunk_df, run_dir / "chunk_results.csv")

    meta = cfg.to_dict()
    meta["run_id"] = run_id

    save_json(meta, run_dir / "run_meta.json")
    save_json(summary, run_dir / "run_summary.json")

    return {**meta, **summary}
