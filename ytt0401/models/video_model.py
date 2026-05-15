from dataclasses import dataclass
from typing import Optional
import tempfile
import os
import cv2
import torch
import numpy as np
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor


@dataclass
class VideoAnalysisResult:
    text: str
    infer_ms: float


class QwenVideoAnalyzer:
    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-VL-3B-Instruct",
        fps_for_model: float = 1.0,
        max_new_tokens: int = 96,
    ):
        self.model_name = model_name
        self.fps_for_model = fps_for_model
        self.max_new_tokens = max_new_tokens

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map="auto",
        )
        self.processor = AutoProcessor.from_pretrained(model_name)

    def _write_temp_video(self, frames_bgr, save_fps: int = 6) -> str:
        if len(frames_bgr) == 0:
            raise ValueError("frames_bgr is empty")

        h, w = frames_bgr[0].shape[:2]
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tmp_path = tmp.name
        tmp.close()

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(tmp_path, fourcc, save_fps, (w, h))
        if not writer.isOpened():
            raise RuntimeError("Failed to open VideoWriter for temp video.")

        for frame in frames_bgr:
            writer.write(frame)
        writer.release()

        return tmp_path

    def analyze_clip(self, frames_bgr) -> VideoAnalysisResult:
        import time

        if len(frames_bgr) == 0:
            return VideoAnalysisResult("系统正在采样视频片段，暂未生成分析结果。", 0.0)

        t0 = time.perf_counter()
        temp_video_path = self._write_temp_video(frames_bgr)

        try:
            conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "video", "path": temp_video_path},
                        {
                            "type": "text",
                            "text": (
                                "请用中文简洁分析这段视频中的主要行为或画面变化。"
                                "要求输出1到2句，尽量描述当前人物或目标的动作状态、运动趋势和显著变化，"
                                "不要输出项目符号，不要解释模型本身。"
                            ),
                        },
                    ],
                }
            ]

            inputs = self.processor.apply_chat_template(
                conversation,
                fps=self.fps_for_model,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            )

            inputs = inputs.to(self.model.device)

            with torch.no_grad():
                output_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                )

            generated_ids = [
                output_ids[len(input_ids):]
                for input_ids, output_ids in zip(inputs.input_ids, output_ids)
            ]

            output_text = self.processor.batch_decode(
                generated_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            )[0].strip()

            infer_ms = (time.perf_counter() - t0) * 1000.0
            if not output_text:
                output_text = "模型已完成分析，但当前未返回明确文本结果。"

            return VideoAnalysisResult(output_text, infer_ms)

        finally:
            try:
                os.remove(temp_video_path)
            except Exception:
                pass