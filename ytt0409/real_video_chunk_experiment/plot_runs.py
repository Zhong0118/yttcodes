from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


def line_plot(df: pd.DataFrame, x: str, y: str, exp_group: str, save_path: Path):
    sub = df[df["exp_group"] == exp_group].copy()
    if sub.empty:
        return
    agg = sub.groupby(x, as_index=False)[y].mean()
    plt.figure(figsize=(6, 4))
    plt.plot(agg[x], agg[y], marker="o")
    plt.xlabel(x)
    plt.ylabel(y)
    plt.title(f"{exp_group}: {x} vs {y}")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def heatmap_plot(df: pd.DataFrame, x: str, y: str, value: str, exp_group: str, save_path: Path):
    sub = df[df["exp_group"] == exp_group].copy()
    if sub.empty:
        return
    pivot = sub.pivot_table(index=y, columns=x, values=value, aggfunc="mean")
    plt.figure(figsize=(6, 4))
    plt.imshow(pivot.values, aspect="auto")
    plt.xticks(range(len(pivot.columns)), pivot.columns)
    plt.yticks(range(len(pivot.index)), pivot.index)
    plt.xlabel(x)
    plt.ylabel(y)
    plt.title(f"{exp_group}: {value}")
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            plt.text(j, i, f"{pivot.values[i, j]:.1f}", ha="center", va="center", fontsize=8)
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="outputs")
    args = parser.parse_args()

    root = Path(args.input)
    csv_path = root / "runs_summary.csv"
    if not csv_path.exists():
        raise RuntimeError("Please run aggregate_runs.py first to generate runs_summary.csv")

    df = pd.read_csv(csv_path)
    plot_dir = root / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    line_plot(df, "ai_fps", "infer_avg_ms", "sweep_fps", plot_dir / "sweep_fps_infer_avg_ms.png")
    line_plot(df, "ai_input_size", "infer_avg_ms", "sweep_input_size", plot_dir / "sweep_input_size_infer_avg_ms.png")
    line_plot(df, "clip_len", "first_result_ms", "sweep_clip_len", plot_dir / "sweep_clip_len_first_result_ms.png")
    line_plot(df, "bandwidth_kbps", "end_to_end_avg_ms", "sweep_bandwidth", plot_dir / "sweep_bandwidth_end_to_end_avg_ms.png")
    line_plot(df, "network_delay_ms", "end_to_end_avg_ms", "sweep_delay", plot_dir / "sweep_delay_end_to_end_avg_ms.png")

    heatmap_plot(df, "ai_fps", "ai_input_size", "infer_avg_ms", "grid_fps_input", plot_dir / "grid_fps_input_infer_avg_ms.png")
    heatmap_plot(df, "bandwidth_kbps", "network_delay_ms", "end_to_end_avg_ms", "grid_bw_delay", plot_dir / "grid_bw_delay_end_to_end_avg_ms.png")

    print(f"plots saved to: {plot_dir}")


if __name__ == "__main__":
    main()
