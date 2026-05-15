from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch

from evaluate_x3d_grid import (
    EvalConfig,
    build_configs,
    evaluate_config,
    load_aliases,
    load_kinetics_categories,
)


def main():
    parser = argparse.ArgumentParser(description="Run X3D grid experiments from a JSON config.")
    parser.add_argument("--config-json", required=True)
    args = parser.parse_args()

    config_path = Path(args.config_json)
    with open(config_path, "r", encoding="utf-8") as f:
        cfg_json = json.load(f)

    manifest_csv = cfg_json["manifest_csv"]
    out_dir = Path(cfg_json.get("out_dir", "outputs_grid"))
    mode = cfg_json.get("mode", "one_factor")
    alias_json = cfg_json.get("alias_json")
    kinetics_json = cfg_json.get("kinetics_json")
    device_str = cfg_json.get("device", "cuda" if torch.cuda.is_available() else "cpu")

    base_cfg_dict = cfg_json["base_config"]
    base_cfg = EvalConfig(**base_cfg_dict)

    grid = cfg_json.get("grid", {})

    configs = build_configs(
        base=base_cfg,
        mode=mode,
        chunk_frames_list=grid.get("chunk_frames"),
        input_resize_list=grid.get("input_resize"),
        sampling_rate_list=grid.get("sampling_rate"),
        stride_frames_list=grid.get("stride_frames"),
        bandwidth_list=grid.get("bandwidth_mbps"),
        delay_list=grid.get("network_delay_ms"),
        loss_list=grid.get("packet_loss"),
        jpeg_quality_list=grid.get("jpeg_quality"),
    )

    manifest_df = pd.read_csv(manifest_csv)
    if "video_path" not in manifest_df.columns or "canonical_label" not in manifest_df.columns:
        raise ValueError("manifest csv must contain columns: video_path, canonical_label")

    from datetime import datetime
    batch_dir = out_dir / f"grid_{base_cfg.model_name}_{mode}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    batch_dir.mkdir(parents=True, exist_ok=True)

    with open(batch_dir / "batch_meta.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "config_json": str(config_path.resolve()),
                "manifest_csv": str(Path(manifest_csv).resolve()),
                "resolved_batch_dir": str(batch_dir.resolve()),
                "device": device_str,
                "mode": mode,
                "n_configs": len(configs),
                "alias_json": alias_json,
                "kinetics_json": kinetics_json,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    alias_map = load_aliases(alias_json)
    categories = load_kinetics_categories(kinetics_json)
    device = torch.device(device_str)

    summaries = []
    for i, cfg in enumerate(configs):
        run_dir = batch_dir / f"run_{i:04d}"
        print(f"\n========== [{i+1}/{len(configs)}] {cfg.name} ==========")

        summary = evaluate_config(
            manifest_df=manifest_df,
            cfg=cfg,
            categories=categories,
            device=device,
            alias_map=alias_map,
            run_dir=run_dir,
        )
        summaries.append(summary)

    runs_df = pd.DataFrame(summaries)
    runs_df.to_csv(batch_dir / "runs_summary.csv", index=False, encoding="utf-8-sig")
    runs_df.to_excel(batch_dir / "runs_summary.xlsx", index=False)

    with open(batch_dir / "runs_summary.json", "w", encoding="utf-8") as f:
        json.dump(summaries, f, ensure_ascii=False, indent=2)

    print(f"\n[DONE] batch dir: {batch_dir.resolve()}")


if __name__ == "__main__":
    main()