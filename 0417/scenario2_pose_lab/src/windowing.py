from __future__ import annotations

from typing import Any, List, Optional


class SlidingWindowBuffer:
    def __init__(self, fps: float, window_seconds: float, stride_seconds: float):
        self.fps = max(1e-6, float(fps))
        self.window_size = max(1, int(round(self.fps * window_seconds)))
        self.stride_size = max(1, int(round(self.fps * stride_seconds)))
        self.buffer: List[Any] = []
        self.frames_since_emit = 0

    def push(self, item: Any) -> Optional[List[Any]]:
        self.buffer.append(item)
        if len(self.buffer) > self.window_size:
            self.buffer.pop(0)
        self.frames_since_emit += 1

        if len(self.buffer) == self.window_size and self.frames_since_emit >= self.stride_size:
            self.frames_since_emit = 0
            return list(self.buffer)
        return None
