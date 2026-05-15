from __future__ import annotations
import argparse
from pathlib import Path
import yaml

from src.config import RunConfig
from src.runner import run_and_save


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--output", type=str, default="outputs")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg_raw = yaml.safe_load(f)

    p = cfg_raw["params"]
    cfg = RunConfig(
        exp_group="single_video",
        repeat_idx=0,
        video_path=cfg_raw["video_path"],
        model_name=cfg_raw.get("model_name", "placeholder"),
        ai_fps=int(p["ai_fps"]),
        ai_input_size=int(p["ai_input_size"]),
        clip_len=int(p["clip_len"]),
        stride_len=int(p["stride_len"]),
        bandwidth_kbps=int(p["bandwidth_kbps"]),
        network_delay_ms=int(p["network_delay_ms"]),
        jpeg_quality=int(p["jpeg_quality"]),
    )

    result = run_and_save(cfg, Path(args.output))
    print(result)


if __name__ == "__main__":
    main()
