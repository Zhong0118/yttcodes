from __future__ import annotations

from dataclasses import dataclass, asdict
from itertools import product
from typing import Any, Dict, List


@dataclass
class Scenario2Config:
    name: str = "base"

    # uplink / AI chain
    uplink_resolution: int = 360
    uplink_fps: float = 8.0
    window_seconds: float = 2.0
    stride_seconds: float = 1.0
    transport: str = "tcp"
    extra_uplink_delay_ms: float = 0.0
    tcp_bandwidth_mbps: float = 8.0
    tcp_fixed_rtt_ms: float = 30.0

    # downlink / user chain
    downlink_resolution: int = 720
    downlink_fps: float = 15.0
    extra_downlink_delay_ms: float = 0.0

    # runtime
    max_duration_sec: float = 20.0

    @property
    def chunk_size_frames(self) -> int:
        return max(1, int(round(self.uplink_fps * self.window_seconds)))

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["chunk_size_frames"] = self.chunk_size_frames
        return d


DEFAULT_CONFIG = Scenario2Config()


SMOKE_EXPERIMENTS: List[Scenario2Config] = [
    Scenario2Config(name="base"),
    Scenario2Config(name="win_1", window_seconds=1.0),
    Scenario2Config(name="low_bw", tcp_bandwidth_mbps=1.0),
]


def build_scenario2_full() -> List[Scenario2Config]:
    experiments: List[Scenario2Config] = []

    # Base
    experiments.append(Scenario2Config(name="base"))

    # Uplink resolution sensitivity
    for res in [224, 320, 360, 480]:
        if res == 360:
            continue
        experiments.append(Scenario2Config(name=f"up_res_{res}", uplink_resolution=res))

    # Uplink FPS sensitivity
    for fps in [4.0, 6.0, 8.0, 12.0]:
        if fps == 8.0:
            continue
        experiments.append(Scenario2Config(name=f"up_fps_{str(fps).replace('.','_')}", uplink_fps=fps))

    # Window sensitivity with 1s included
    for win in [1.0, 2.0, 3.0, 4.0]:
        if win == 2.0:
            continue
        experiments.append(Scenario2Config(name=f"win_{str(win).replace('.','_')}", window_seconds=win, stride_seconds=min(1.0, win)))

    # TCP bandwidth sensitivity
    for bw in [0.5, 1.0, 2.0, 4.0, 8.0, 16.0]:
        if bw == 8.0:
            continue
        experiments.append(Scenario2Config(name=f"tcp_bw_{str(bw).replace('.','_')}", tcp_bandwidth_mbps=bw))

    # TCP RTT sensitivity
    for rtt in [0.0, 30.0, 80.0, 150.0]:
        if rtt == 30.0:
            continue
        experiments.append(Scenario2Config(name=f"tcp_rtt_{int(rtt)}", tcp_fixed_rtt_ms=rtt))

    # Downlink sensitivity
    for dres in [360, 480, 720]:
        if dres == 720:
            continue
        experiments.append(Scenario2Config(name=f"down_res_{dres}", downlink_resolution=dres))
    for dfps in [10.0, 15.0, 24.0]:
        if dfps == 15.0:
            continue
        experiments.append(Scenario2Config(name=f"down_fps_{str(dfps).replace('.','_')}", downlink_fps=dfps))

    # Core cross experiments: bandwidth x window
    for bw, win in product([0.5, 2.0, 8.0, 16.0], [1.0, 2.0, 3.0, 4.0]):
        if bw == 8.0 and win == 2.0:
            continue
        experiments.append(
            Scenario2Config(
                name=f"bw{str(bw).replace('.','_')}_win{str(win).replace('.','_')}",
                tcp_bandwidth_mbps=bw,
                window_seconds=win,
                stride_seconds=min(1.0, win),
            )
        )

    # Core cross experiments: uplink fps x window
    for fps, win in product([4.0, 8.0, 12.0], [1.0, 2.0, 3.0]):
        if fps == 8.0 and win == 2.0:
            continue
        experiments.append(
            Scenario2Config(
                name=f"fps{str(fps).replace('.','_')}_win{str(win).replace('.','_')}",
                uplink_fps=fps,
                window_seconds=win,
                stride_seconds=min(1.0, win),
            )
        )

    # Deduplicate by name
    uniq: Dict[str, Scenario2Config] = {}
    for cfg in experiments:
        uniq[cfg.name] = cfg
    return list(uniq.values())


PRESETS: Dict[str, List[Scenario2Config]] = {
    "smoke": SMOKE_EXPERIMENTS,
    "scenario2_full": build_scenario2_full(),
}


def get_preset(name: str) -> List[Scenario2Config]:
    if name not in PRESETS:
        raise KeyError(f"Unknown preset: {name}. Available: {list(PRESETS)}")
    return PRESETS[name]
