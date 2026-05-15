from dataclasses import dataclass

@dataclass
class EvalConfig:
    name: str
    model_name: str = "x3d_xs"
    chunk_frames: int = 4
    sampling_rate: int = 12
    stride_frames: int = 4
    input_resize: int = 182
    input_crop: int = 182
    bandwidth_mbps: float = 8.0
    network_delay_ms: float = 20.0
    packet_loss: float = 0.0
    max_videos: int | None = None