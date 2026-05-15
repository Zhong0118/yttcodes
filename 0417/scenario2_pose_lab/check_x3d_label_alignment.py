from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Dict, List, Set


def normalize_label(label: str) -> str:
    s = str(label).strip().lower()
    s = s.replace("_", " ").replace("-", " ")
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


DEFAULT_ALIASES = {
    "run": "running",
    "jog": "running",
    "drink": "drinking",
    "eat": "eating",
    "walk": "walking",
    "sit": "sitting",
    "jumping_jacks": "jumping jacks",
}


def canonicalize_label(label: str, alias_map: Dict[str, str]) -> str:
    x = normalize_label(label)
    return alias_map.get(x, x)


def load_aliases(alias_json: str | None) -> Dict[str, str]:
    alias_map = {}
    for k, v in DEFAULT_ALIASES.items():
        alias_map[normalize_label(k)] = normalize_label(v)

    if alias_json:
        with open(alias_json, "r", encoding="utf-8") as f:
            user_aliases = json.load(f)
        for k, v in user_aliases.items():
            alias_map[normalize_label(k)] = normalize_label(v)

    return alias_map


def load_dataset_labels(input_csv: Path, alias_map: Dict[str, str]) -> List[Dict]:
    import pandas as pd

    df = pd.read_csv(input_csv)

    if "canonical_label" in df.columns:
        labels = sorted(set(df["canonical_label"].astype(str).tolist()))
        return [
            {
                "dataset_label_raw": lb,
                "dataset_label_canonical": canonicalize_label(lb, alias_map),
            }
            for lb in labels
        ]

    if "label" in df.columns:
        labels = sorted(set(df["label"].astype(str).tolist()))
        return [
            {
                "dataset_label_raw": lb,
                "dataset_label_canonical": canonicalize_label(lb, alias_map),
            }
            for lb in labels
        ]

    if "raw_label" in df.columns:
        labels = sorted(set(df["raw_label"].astype(str).tolist()))
        return [
            {
                "dataset_label_raw": lb,
                "dataset_label_canonical": canonicalize_label(lb, alias_map),
            }
            for lb in labels
        ]

    raise ValueError(
        f"{input_csv} must contain one of these columns: canonical_label / label / raw_label"
    )


def save_csv(rows: List[Dict], out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        with open(out_path, "w", encoding="utf-8-sig", newline="") as f:
            pass
        return
    with open(out_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def load_x3d_categories(model_name: str = "x3d_xs") -> tuple[list[str], str]:
    """
    返回:
      categories, backend_name
    backend_name 可能是:
      - torchvision
      - pytorchvideo_hub
      - builtin_kinetics_400
    """

    # 1) 先试 torchvision
    try:
        if model_name == "x3d_xs":
            from torchvision.models.video import x3d_xs, X3D_XS_Weights
            weights = X3D_XS_Weights.DEFAULT
            _ = x3d_xs(weights=weights)
            categories = list(weights.meta["categories"])
            return categories, "torchvision"
        elif model_name == "x3d_s":
            from torchvision.models.video import x3d_s, X3D_S_Weights
            weights = X3D_S_Weights.DEFAULT
            _ = x3d_s(weights=weights)
            categories = list(weights.meta["categories"])
            return categories, "torchvision"
        else:
            raise ValueError(f"Unsupported model_name for torchvision path: {model_name}")
    except Exception as e:
        print(f"[WARN] torchvision path failed: {e}")

    # 2) 再试 torch.hub / pytorchvideo
    try:
        import urllib.request

        import torch

        _ = torch.hub.load("facebookresearch/pytorchvideo", model_name, pretrained=True)

        json_url = "https://dl.fbaipublicfiles.com/pyslowfast/dataset/class_names/kinetics_classnames.json"
        json_path = Path("kinetics_classnames.json")
        if not json_path.exists():
            urllib.request.urlretrieve(json_url, str(json_path))

        with open(json_path, "r", encoding="utf-8") as f:
            kinetics_classnames = json.load(f)

        # 官方 hub 示例里是 {label_name: class_id}，这里把它转成按 id 排序的列表
        id_to_name = {}
        for k, v in kinetics_classnames.items():
            id_to_name[int(v)] = str(k).replace('"', "")

        categories = [id_to_name[i] for i in sorted(id_to_name.keys())]
        return categories, "pytorchvideo_hub"
    except Exception as e:
        print(f"[WARN] pytorchvideo hub path failed: {e}")

    # 3) 最后兜底：直接使用内置的 kinetics_classnames.json 文件
    # 你也可以自己手工准备一个 400 类 json
    local_json = Path("kinetics_classnames.json")
    if local_json.exists():
        with open(local_json, "r", encoding="utf-8") as f:
            kinetics_classnames = json.load(f)

        id_to_name = {}
        for k, v in kinetics_classnames.items():
            id_to_name[int(v)] = str(k).replace('"', "")

        categories = [id_to_name[i] for i in sorted(id_to_name.keys())]
        return categories, "builtin_kinetics_400"

    raise RuntimeError(
        "Failed to load X3D / Kinetics categories from torchvision and torch.hub. "
        "You can manually place kinetics_classnames.json in the current directory."
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-csv", required=True, help="label_summary.csv or dataset_manifest.csv")
    parser.add_argument("--out-dir", default="outputs_alignment")
    parser.add_argument("--alias-json", default=None)
    parser.add_argument("--model-name", default="x3d_xs", choices=["x3d_xs", "x3d_s"])
    args = parser.parse_args()

    input_csv = Path(args.input_csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    alias_map = load_aliases(args.alias_json)

    dataset_rows = load_dataset_labels(input_csv, alias_map)
    dataset_canonical_labels: Set[str] = set(x["dataset_label_canonical"] for x in dataset_rows)

    model_categories, backend = load_x3d_categories(args.model_name)

    model_rows = []
    for lb in model_categories:
        model_rows.append(
            {
                "model_label_raw": lb,
                "model_label_canonical": canonicalize_label(lb, alias_map),
            }
        )
    model_canonical_labels: Set[str] = set(x["model_label_canonical"] for x in model_rows)

    matched = []
    unmatched_dataset = []
    unmatched_model = []

    model_canonical_to_raws: Dict[str, List[str]] = {}
    for r in model_rows:
        model_canonical_to_raws.setdefault(r["model_label_canonical"], []).append(r["model_label_raw"])

    for r in dataset_rows:
        c = r["dataset_label_canonical"]
        if c in model_canonical_labels:
            matched.append(
                {
                    "dataset_label_raw": r["dataset_label_raw"],
                    "dataset_label_canonical": c,
                    "matched_model_labels_raw": " | ".join(sorted(model_canonical_to_raws.get(c, []))),
                }
            )
        else:
            unmatched_dataset.append(r)

    for r in model_rows:
        c = r["model_label_canonical"]
        if c not in dataset_canonical_labels:
            unmatched_model.append(r)

    save_csv(matched, out_dir / "matched_labels.csv")
    save_csv(unmatched_dataset, out_dir / "unmatched_dataset_labels.csv")
    save_csv(unmatched_model, out_dir / "unmatched_model_labels.csv")
    save_csv(model_rows, out_dir / "all_model_labels.csv")
    save_csv(dataset_rows, out_dir / "all_dataset_labels.csv")

    summary = {
        "model_name": args.model_name,
        "backend_used": backend,
        "n_dataset_labels": len(dataset_rows),
        "n_model_labels": len(model_rows),
        "n_matched_dataset_labels": len(matched),
        "n_unmatched_dataset_labels": len(unmatched_dataset),
        "n_unmatched_model_labels": len(unmatched_model),
        "dataset_coverage_ratio": float(len(matched) / max(1, len(dataset_rows))),
    }
    with open(out_dir / "alignment_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"[OUT] {out_dir}")


if __name__ == "__main__":
    main()