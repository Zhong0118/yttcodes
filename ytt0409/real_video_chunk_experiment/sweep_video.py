from __future__ import annotations
import argparse
from pathlib import Path
import itertools
import yaml
import pandas as pd

from src.config import RunConfig
from src.runner import run_and_save


def make_cfg(exp_group: str, repeat_idx: int, video_path: str, model_name: str, params: dict) -> RunConfig:
    return RunConfig(
        exp_group=exp_group,
        repeat_idx=repeat_idx,
        video_path=video_path,
        model_name=model_name,
        ai_fps=int(params["ai_fps"]),
        ai_input_size=int(params["ai_input_size"]),
        clip_len=int(params["clip_len"]),
        stride_len=int(params["stride_len"]),
        bandwidth_kbps=int(params["bandwidth_kbps"]),
        network_delay_ms=int(params["network_delay_ms"]),
        jpeg_quality=int(params["jpeg_quality"]),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--output", type=str, default="outputs")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        plan = yaml.safe_load(f)

    video_path = plan["video_path"]
    model_name = plan.get("model_name", "placeholder")
    baseline = dict(plan["baseline"])
    output_root = Path(args.output)
    rows = []

    for r in range(3):
        cfg = make_cfg("baseline", r, video_path, model_name, baseline)
        rows.append(run_and_save(cfg, output_root))

    for item in plan.get("single_factor", []):
        var = item["variable"]
        values = item["values"]
        name = item["name"]
        for v in values:
            for r in range(3):
                params = dict(baseline)
                params[var] = v
                if params["stride_len"] > params["clip_len"]:
                    params["stride_len"] = params["clip_len"]
                cfg = make_cfg(name, r, video_path, model_name, params)
                rows.append(run_and_save(cfg, output_root))

    for item in plan.get("two_factor", []):
        vx = item["variable_x"]
        vy = item["variable_y"]
        values_x = item["values_x"]
        values_y = item["values_y"]
        name = item["name"]

        for x, y in itertools.product(values_x, values_y):
            for r in range(3):
                params = dict(baseline)
                params[vx] = x
                params[vy] = y
                if params["stride_len"] > params["clip_len"]:
                    params["stride_len"] = params["clip_len"]
                cfg = make_cfg(name, r, video_path, model_name, params)
                rows.append(run_and_save(cfg, output_root))

    df = pd.DataFrame(rows)
    df.to_csv(output_root / "runs_summary.csv", index=False, encoding="utf-8-sig")
    print(f"saved: {output_root / 'runs_summary.csv'}")
    print(df.head())


if __name__ == "__main__":
    main()
