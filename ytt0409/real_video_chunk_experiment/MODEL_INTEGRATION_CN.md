# 如何把你的真实视频行为分析模型接进来

你现在要的是：
- 输入真实原视频
- 对每个 chunk 做分析
- 输出每个 chunk 的结果
- 统计不同参数下的延迟变化

这套代码里你只需要改一个地方：
- src/models.py

## 当前默认接口

predict_chunk(chunk_frames) 输入：
- 一个 chunk 的 frame 列表
- frame 已经按 ai_fps 采样
- 已经 resize 到 ai_input_size x ai_input_size
- chunk 长度就是 clip_len

它必须输出：
- pred_label
- confidence
- infer_ms

## 最小模板

from typing import Dict, List, Union
import time
import numpy as np

class YourRealModel:
    def __init__(self, ckpt_path: str = None):
        pass

    def predict_chunk(self, chunk_frames: List[np.ndarray]) -> Dict[str, Union[float, str]]:
        t0 = time.perf_counter()

        pred_label = "action_x"
        confidence = 0.91

        infer_ms = (time.perf_counter() - t0) * 1000.0
        return {
            "pred_label": pred_label,
            "confidence": confidence,
            "infer_ms": infer_ms,
        }

然后在 build_model() 里注册。

## 你已经不需要管的部分

你不需要再写：
- 视频读取
- fps 重采样
- resize
- clip 切分
- 带宽 / 延迟模拟
- chunk 级 CSV 输出
- run 级 summary
- sweep
- 聚合
- 作图

你只需要替换模型。
