from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Dict, Any


@dataclass
class RunConfig:
    exp_group: str
    repeat_idx: int
    video_path: str
    model_name: str
    ai_fps: int
    ai_input_size: int
    clip_len: int
    stride_len: int
    bandwidth_kbps: int
    network_delay_ms: int
    jpeg_quality: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
