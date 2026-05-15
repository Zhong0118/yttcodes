from __future__ import annotations

import csv
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class ActionSegment:
    action_id: str
    start_sec: float
    end_sec: float

    @property
    def duration_sec(self) -> float:
        return max(0.0, self.end_sec - self.start_sec)


def read_class_names(path: str | Path) -> dict[str, str]:
    classes: dict[str, str] = {}
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            action_id, _, name = line.partition(" ")
            classes[action_id] = name
    return classes


def parse_actions(value: str | float | None) -> list[ActionSegment]:
    if value is None:
        return []
    if isinstance(value, float) and math.isnan(value):
        return []
    text = str(value).strip()
    if not text:
        return []

    actions: list[ActionSegment] = []
    for item in text.split(";"):
        parts = item.strip().split()
        if len(parts) != 3:
            continue
        action_id, start, end = parts
        try:
            start_sec = float(start)
            end_sec = float(end)
        except ValueError:
            continue
        if end_sec > start_sec:
            actions.append(ActionSegment(action_id, start_sec, end_sec))
    return actions


def overlap_seconds(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))


def split_labels(value: str | float | None) -> list[str]:
    if value is None:
        return []
    if isinstance(value, float) and math.isnan(value):
        return []
    return [x for x in str(value).split(";") if x]


def parse_top_labels(value: str | float | None) -> list[str]:
    labels: list[str] = []
    for item in split_labels(value):
        label = item.split(":", 1)[0].strip()
        if label:
            labels.append(label)
    return labels


def parse_top_scores(value: str | float | None) -> dict[str, float]:
    scores: dict[str, float] = {}
    for item in split_labels(value):
        if ":" not in item:
            continue
        label, score = item.split(":", 1)
        try:
            scores[label.strip()] = float(score)
        except ValueError:
            continue
    return scores


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    k = (len(ordered) - 1) * p
    lo = math.floor(k)
    hi = math.ceil(k)
    if lo == hi:
        return ordered[int(k)]
    return ordered[lo] * (hi - k) + ordered[hi] * (k - lo)


def write_json(path: str | Path, data: object) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def read_csv_dicts(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv_dicts(path: str | Path, rows: Iterable[dict], fieldnames: list[str]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def seeded_rng(seed: int) -> random.Random:
    rng = random.Random()
    rng.seed(seed)
    return rng
