from __future__ import annotations

import argparse
import socket
import time

from .classifier import ShortPoseUnderstandingModel
from .protocol import recv_json, send_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=9009)
    args = parser.parse_args()

    model = ShortPoseUnderstandingModel()

    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind((args.host, args.port))
    server.listen(1)
    print(f"[server] listening on {args.host}:{args.port}")

    conn, addr = server.accept()
    print(f"[server] connected from {addr}")

    while True:
        req = recv_json(conn)
        if req.get("type") == "stop":
            break
        if req.get("type") != "window":
            continue

        t0 = time.time()
        result = model.infer(req["window"])
        infer_ms = (time.time() - t0) * 1000.0
        resp = {
            "type": "result",
            "window_id": req.get("window_id"),
            "infer_ms": infer_ms,
            **result,
        }
        send_json(conn, resp)

    conn.close()
    server.close()


if __name__ == "__main__":
    main()
