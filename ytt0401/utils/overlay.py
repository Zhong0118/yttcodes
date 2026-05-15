import os
import time
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def current_time_str():
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())


def _find_font_path():
    candidates = [
        r"C:\Windows\Fonts\msyh.ttc",
        r"C:\Windows\Fonts\msyhbd.ttc",
        r"C:\Windows\Fonts\simhei.ttf",
        r"C:\Windows\Fonts\simsun.ttc",
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return None


FONT_PATH = _find_font_path()


def _to_pil(image_bgr: np.ndarray) -> Image.Image:
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(image_rgb)


def _to_cv2(image_pil: Image.Image) -> np.ndarray:
    image_rgb = np.array(image_pil)
    return cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)


def draw_text(
    image: np.ndarray,
    text: str,
    org,
    font_size: int = 24,
    color=(0, 255, 0),
):
    if FONT_PATH is None:
        cv2.putText(
            image,
            text,
            org,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 0),
            4,
            cv2.LINE_AA,
        )
        cv2.putText(
            image,
            text,
            org,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2,
            cv2.LINE_AA,
        )
        return

    pil_img = _to_pil(image)
    draw = ImageDraw.Draw(pil_img)
    font = ImageFont.truetype(FONT_PATH, font_size, encoding="utf-8")

    x, y = org
    rgb_color = (color[2], color[1], color[0])

    for dx, dy in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
        draw.text((x + dx, y + dy), text, font=font, fill=(0, 0, 0))
    draw.text((x, y), text, font=font, fill=rgb_color)

    image[:] = _to_cv2(pil_img)


def draw_multiline_info(
    image: np.ndarray,
    lines,
    start_xy=(10, 30),
    line_gap=34,
    color=(0, 255, 0),
    font_size=24,
):
    x, y = start_xy
    for i, line in enumerate(lines):
        draw_text(
            image=image,
            text=line,
            org=(x, y + i * line_gap),
            font_size=font_size,
            color=color,
        )


def draw_title_bar(image: np.ndarray, title: str):
    h, w = image.shape[:2]
    bar_h = 42
    overlay = image.copy()
    cv2.rectangle(overlay, (0, 0), (w, bar_h), (30, 30, 30), -1)
    image[:] = cv2.addWeighted(overlay, 0.6, image, 0.4, 0)
    draw_text(image, title, (10, 8), font_size=24, color=(255, 255, 255))