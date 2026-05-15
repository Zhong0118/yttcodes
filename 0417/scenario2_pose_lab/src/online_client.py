from __future__ import annotations

import argparse
import socket
import time

import cv2

from .config import Scenario2Config
from .extractor import PoseStickExtractor, resize_keep_aspect
from .protocol import recv_json, send_json
from .windowing import SlidingWindowBuffer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", default="0", help="video path or webcam index")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9009)
    parser.add_argument("--uplink-resolution", type=int, default=360)
    parser.add_argument("--uplink-fps", type=float, default=8.0)
    parser.add_argument("--window-seconds", type=float, default=2.0)
    parser.add_argument("--stride-seconds", type=float, default=1.0)
    parser.add_argument("--max-duration-sec", type=float, default=20.0)
    args = parser.parse_args()

    video_source = int(args.video) if args.video.isdigit() else args.video
    cfg = Scenario2Config(
        uplink_resolution=args.uplink_resolution,
        uplink_fps=args.uplink_fps,
        window_seconds=args.window_seconds,
        stride_seconds=args.stride_seconds,
        max_duration_sec=args.max_duration_sec,
    )

    cap = cv2.VideoCapture(video_source)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 1e-3:
        fps = 25.0

    extractor = PoseStickExtractor()
    buffer = SlidingWindowBuffer(cfg.uplink_fps, cfg.window_seconds, cfg.stride_seconds)

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((args.host, args.port))

    start_time = time.time()
    frame_idx = 0
    last_sent = 0.0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        now = time.time()
        if now - start_time > cfg.max_duration_sec:
            break
        if now - last_sent < 1.0 / max(cfg.uplink_fps, 1e-6):
            continue
        last_sent = now

        frame_idx += 1
        frame_up = resize_keep_aspect(frame, cfg.uplink_resolution)
        feat = extractor.extract(frame_up)
        feat["frame_idx"] = frame_idx
        feat["ts"] = now

        window = buffer.push(feat)
        display = frame.copy()
        if window is not None:
            send_json(sock, {
                "type": "window",
                "window_id": frame_idx,
                "window": window,
            })
            resp = recv_json(sock)
            text = f"{resp['label']} | conf={resp['confidence']:.2f} | infer={resp['infer_ms']:.1f}ms"
            cv2.putText(display, text, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.imshow("scenario2-client", display)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    send_json(sock, {"type": "stop"})
    sock.close()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
