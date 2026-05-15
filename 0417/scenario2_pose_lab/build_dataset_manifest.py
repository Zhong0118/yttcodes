from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List


VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}


# 你可以在这里补一些常见别名
DEFAULT_ALIASES = {
    "run": "running",
    "jog": "running",
    "drink": "drinking",
    "eat": "eating",
    "walk": "walking",
    "sit": "sitting",
    "jump rope": "jumping rope",
    "jumping_jacks": "jumping jacks",
    "pushup": "push up",
    "pullup": "pull up",
}


def normalize_label(label: str) -> str:
    """
    基础清洗：
    1. 去首尾空白
    2. 转小写
    3. _ 和 - 替换为空格
    4. 去掉多余空白
    5. 去掉大部分非字母数字符号（保留空格）
    """
    s = label.strip().lower()
    s = s.replace("_", " ").replace("-", " ")
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def canonicalize_label(label: str, alias_map: Dict[str, str]) -> str:
    """
    先 normalize，再做别名映射。
    """
    s = normalize_label(label)
    return alias_map.get(s, s)


def load_aliases(alias_json: str | None) -> Dict[str, str]:
    alias_map = dict(DEFAULT_ALIASES)
    if alias_json:
        with open(alias_json, "r", encoding="utf-8") as f:
            user_aliases = json.load(f)

        cleaned = {}
        for k, v in user_aliases.items():
            ck = normalize_label(str(k))
            cv = normalize_label(str(v))
            cleaned[ck] = cv

        alias_map.update(cleaned)
    return alias_map


def scan_dataset(root: Path, recursive_class_depth: int = 1) -> List[Dict]:
    """
    默认假设目录结构：
        root/
          class_a/
            xx.mp4
            yy.mp4
          class_b/
            zz.mp4

    这里 class label 默认取“相对于 root 的第一级目录名”。
    如果你后面目录更复杂，再改这块。
    """
    rows: List[Dict] = []

    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() not in VIDEO_EXTS:
            continue

        rel = path.relative_to(root)
        parts = rel.parts

        if len(parts) < 2:
            # 根目录下直接放视频，不知道标签，跳过
            continue

        raw_label = parts[0]

        rows.append(
            {
                "video_path": str(path.resolve()),
                "rel_path": str(rel.as_posix()),
                "raw_label": raw_label,
                "file_name": path.name,
                "ext": path.suffix.lower(),
            }
        )

    return rows


def write_csv(rows: List[Dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        with open(out_path, "w", encoding="utf-8-sig", newline="") as f:
            pass
        return

    fieldnames = list(rows[0].keys())
    with open(out_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, help="数据集根目录，按类别文件夹组织")
    parser.add_argument("--out-dir", default="manifest_outputs", help="输出目录")
    parser.add_argument("--alias-json", default=None, help="可选，自定义别名映射 json")
    args = parser.parse_args()

    root = Path(args.root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not root.exists():
        raise FileNotFoundError(f"Dataset root not found: {root}")

    alias_map = load_aliases(args.alias_json)
    rows = scan_dataset(root)

    cleaned_rows: List[Dict] = []
    raw_counter = Counter()
    canonical_counter = Counter()
    raw_to_canonical = defaultdict(set)

    for row in rows:
        raw_label = row["raw_label"]
        normalized_label = normalize_label(raw_label)
        canonical_label = canonicalize_label(raw_label, alias_map)

        raw_counter[raw_label] += 1
        canonical_counter[canonical_label] += 1
        raw_to_canonical[raw_label].add(canonical_label)

        new_row = dict(row)
        new_row["normalized_label"] = normalized_label
        new_row["canonical_label"] = canonical_label
        cleaned_rows.append(new_row)

    # 1) 完整 manifest
    manifest_csv = out_dir / "dataset_manifest.csv"
    write_csv(cleaned_rows, manifest_csv)

    manifest_json = out_dir / "dataset_manifest.json"
    with open(manifest_json, "w", encoding="utf-8") as f:
        json.dump(cleaned_rows, f, ensure_ascii=False, indent=2)

    # 2) 标签汇总
    summary_rows = []
    for raw_label in sorted(raw_counter.keys()):
        summary_rows.append(
            {
                "raw_label": raw_label,
                "normalized_label": normalize_label(raw_label),
                "canonical_label": ",".join(sorted(raw_to_canonical[raw_label])),
                "n_videos": raw_counter[raw_label],
            }
        )

    label_summary_csv = out_dir / "label_summary.csv"
    write_csv(summary_rows, label_summary_csv)

    # 3) 规范标签频次
    canonical_rows = [
        {"canonical_label": k, "n_videos": v}
        for k, v in sorted(canonical_counter.items(), key=lambda x: (-x[1], x[0]))
    ]
    canonical_summary_csv = out_dir / "canonical_label_counts.csv"
    write_csv(canonical_rows, canonical_summary_csv)

    # 4) 保存 alias 映射
    alias_out = out_dir / "used_aliases.json"
    with open(alias_out, "w", encoding="utf-8") as f:
        json.dump(alias_map, f, ensure_ascii=False, indent=2)

    # 5) 控制台打印
    print(f"[DONE] scanned videos: {len(cleaned_rows)}")
    print(f"[DONE] raw labels: {len(raw_counter)}")
    print(f"[DONE] canonical labels: {len(canonical_counter)}")
    print(f"[OUT] {manifest_csv}")
    print(f"[OUT] {manifest_json}")
    print(f"[OUT] {label_summary_csv}")
    print(f"[OUT] {canonical_summary_csv}")
    print(f"[OUT] {alias_out}")


if __name__ == "__main__":
    main()