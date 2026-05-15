import json
import math
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st

from src.config import Scenario2Config
from src.simulator import Scenario2Experiment, save_single_outputs
from src.report import write_markdown_report


st.set_page_config(page_title="Scenario2 Pose Understanding Demo", layout="wide")


# -----------------------------
# 页面样式
# -----------------------------
st.markdown(
    """
    <style>
    .metric-card {
        background-color: #111827;
        padding: 16px;
        border-radius: 14px;
        border: 1px solid #374151;
        margin-bottom: 10px;
    }
    .metric-title {
        color: #9CA3AF;
        font-size: 14px;
        margin-bottom: 6px;
    }
    .metric-value {
        color: white;
        font-size: 28px;
        font-weight: 700;
    }
    .section-title {
        font-size: 22px;
        font-weight: 700;
        margin-top: 18px;
        margin-bottom: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# -----------------------------
# UI helper
# -----------------------------
def metric_card(title: str, value: str):
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-title">{title}</div>
            <div class="metric-value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_summary_cards(summary: dict):
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric_card("AI Avg Latency (ms)", f"{summary.get('ai_avg_latency_ms', float('nan')):.1f}")
    with c2:
        metric_card("First Result Latency (ms)", f"{summary.get('first_result_latency_avg_ms', float('nan')):.1f}")
    with c3:
        metric_card("User Perceived Latency (ms)", f"{summary.get('user_perceived_latency_avg_ms', float('nan')):.1f}")
    with c4:
        metric_card("Hand Detect Rate", f"{summary.get('hand_detect_rate_avg', float('nan')):.3f}")

    c5, c6, c7, c8 = st.columns(4)
    with c5:
        metric_card("Uplink TX Avg (ms)", f"{summary.get('uplink_tx_avg_ms', float('nan')):.1f}")
    with c6:
        metric_card("Downlink Render Avg (ms)", f"{summary.get('downlink_render_delay_avg_ms', float('nan')):.1f}")
    with c7:
        metric_card("Display FPS Actual", f"{summary.get('display_fps_actual_avg', float('nan')):.2f}")
    with c8:
        metric_card("Stutter Score", f"{summary.get('display_stutter_score_avg', float('nan')):.3f}")


# -----------------------------
# video helpers
# -----------------------------
def resize_keep(frame, target_short):
    h, w = frame.shape[:2]
    short_side = min(h, w)
    scale = target_short / max(short_side, 1)
    nw, nh = int(w * scale), int(h * scale)
    return cv2.resize(frame, (nw, nh))


def resize_to_h(frame, target_h):
    h, w = frame.shape[:2]
    if h == target_h:
        return frame
    scale = target_h / max(h, 1)
    return cv2.resize(frame, (int(w * scale), target_h))


def enrich_df_with_drift(df: pd.DataFrame, source_video_fps: float) -> pd.DataFrame:
    df = df.copy()
    if "drift_ms" not in df.columns:
        if "user_perceived_latency_ms" in df.columns:
            df["drift_ms"] = df["user_perceived_latency_ms"]
        elif "ai_latency_ms" in df.columns:
            df["drift_ms"] = df["ai_latency_ms"]
        else:
            df["drift_ms"] = np.nan

    if "drift_frames" not in df.columns:
        df["drift_frames"] = df["drift_ms"] / 1000.0 * source_video_fps

    return df


def build_window_ranges(df: pd.DataFrame, cfg: Scenario2Config):
    """
    按窗口时间生成 [start_sec, end_sec] -> row 的映射
    """
    ranges = []
    stride = float(cfg.stride_seconds)
    win = float(cfg.window_seconds)

    for _, row in df.iterrows():
        wid = int(row["window_id"])
        start_sec = wid * stride
        end_sec = start_sec + win
        ranges.append((start_sec, end_sec, row.to_dict()))
    return ranges


def find_window_row_for_time(t_sec: float, ranges):
    for start_sec, end_sec, row in ranges:
        if start_sec <= t_sec < end_sec:
            return row
    if ranges:
        return ranges[-1][2]
    return None


def generate_side_by_side_demo_video(
    video_path: str,
    df: pd.DataFrame,
    cfg: Scenario2Config,
    out_path: str | Path,
    source_video_fps: float,
):
    """
    真正生成“有可见漂移”的左右对比视频：
    左：用户链路当前时刻画面
    右：AI链路对应的延迟后结果画面（按 drift 回退/偏移帧）
    """
    out_path = str(out_path)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    src_fps = cap.get(cv2.CAP_PROP_FPS)
    if src_fps <= 1e-3:
        src_fps = source_video_fps if source_video_fps > 1e-3 else 25.0

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        raise RuntimeError("Video has no frames")

    max_frames = total_frames
    if cfg.max_duration_sec is not None:
        max_frames = min(total_frames, int(cfg.max_duration_sec * src_fps))

    # 先把需要的帧读进内存，避免左右两边反复 seek 导致错误
    frames = []
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    for i in range(max_frames):
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()

    if not frames:
        raise RuntimeError("Failed to load frames for side-by-side demo")

    # 确保 drift 存在
    df = df.copy()
    if "drift_ms" not in df.columns:
        if "user_perceived_latency_ms" in df.columns:
            df["drift_ms"] = df["user_perceived_latency_ms"]
        elif "ai_latency_ms" in df.columns:
            df["drift_ms"] = df["ai_latency_ms"]
        else:
            df["drift_ms"] = 0.0

    if "drift_frames" not in df.columns:
        df["drift_frames"] = df["drift_ms"] / 1000.0 * src_fps

    # 构造窗口映射
    stride = float(cfg.stride_seconds)
    win = float(cfg.window_seconds)
    window_ranges = []
    for _, row in df.iterrows():
        wid = int(row["window_id"])
        start_sec = wid * stride
        end_sec = start_sec + win
        window_ranges.append((start_sec, end_sec, row.to_dict()))

    def find_row(t_sec: float):
        for s, e, row in window_ranges:
            if s <= t_sec < e:
                return row
        if window_ranges:
            return window_ranges[-1][2]
        return None

    writer = None

    for left_idx, left_frame in enumerate(frames):
        t_sec = left_idx / src_fps
        row = find_row(t_sec)

        # 左边：用户当前时刻
        left = resize_keep(left_frame, cfg.downlink_resolution)
        cv2.putText(left, "USER VIEW", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        cv2.putText(
            left,
            f"t={t_sec:.2f}s | downlink={cfg.downlink_resolution}px / {cfg.downlink_fps:.0f}fps",
            (20, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
        )

        # 默认右边就先等于左边，后面如果有 row 再按 drift 偏移
        right_src_idx = left_idx

        label = "N/A"
        confidence = float("nan")
        drift_ms = 0.0
        drift_frames = 0.0
        uplink_tx_ms = float("nan")
        downlink_render_ms = float("nan")
        first_result_ms = float("nan")
        window_id = -1

        if row is not None:
            label = str(row.get("label", "N/A"))
            confidence = float(row.get("confidence", float("nan")))
            drift_ms = float(row.get("drift_ms", 0.0))
            drift_frames = float(row.get("drift_frames", drift_ms / 1000.0 * src_fps))
            uplink_tx_ms = float(row.get("uplink_tx_ms", float("nan")))
            downlink_render_ms = float(row.get("downlink_render_delay_ms", float("nan")))
            first_result_ms = float(row.get("first_result_latency_ms", float("nan")))
            window_id = int(row.get("window_id", -1))

            # 关键修复：右边按 drift 回退帧
            # 直观解释：AI 对当前时刻 t 的结果，实际上来自更早时刻的输入
            right_src_idx = int(round(left_idx - drift_frames))
            right_src_idx = max(0, min(right_src_idx, len(frames) - 1))

        right_frame = frames[right_src_idx]
        right = resize_keep(right_frame, cfg.uplink_resolution)

        cv2.putText(right, "AI VIEW", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
        cv2.putText(
            right,
            f"src_t={right_src_idx / src_fps:.2f}s | uplink={cfg.uplink_resolution}px / {cfg.uplink_fps:.0f}fps",
            (20, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.60,
            (255, 255, 255),
            2,
        )

        overlay_lines = [
            f"window_id: {window_id}",
            f"label: {label}",
            f"conf: {confidence:.2f}" if not np.isnan(confidence) else "conf: N/A",
            f"drift_ms: {drift_ms:.1f}",
            f"drift_frames: {drift_frames:.1f}",
            f"first_result_ms: {first_result_ms:.1f}" if not np.isnan(first_result_ms) else "first_result_ms: N/A",
            f"uplink_tx_ms: {uplink_tx_ms:.1f}" if not np.isnan(uplink_tx_ms) else "uplink_tx_ms: N/A",
            f"downlink_render_ms: {downlink_render_ms:.1f}" if not np.isnan(downlink_render_ms) else "downlink_render_ms: N/A",
        ]
        y = 95
        for line in overlay_lines:
            cv2.putText(right, line, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (255, 255, 255), 2)
            y += 28

        # 再额外叠加一个“偏移提示”
        drift_text = f"VISUAL DRIFT: {drift_ms:.1f} ms / {drift_frames:.1f} frames"
        cv2.putText(right, drift_text, (20, right.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.70, (0, 0, 255), 2)

        # 高度对齐
        h = max(left.shape[0], right.shape[0])
        left = resize_to_h(left, h)
        right = resize_to_h(right, h)
        combo = np.hstack([left, right])

        if writer is None:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(out_path, fourcc, src_fps, (combo.shape[1], combo.shape[0]))

        writer.write(combo)

    if writer is not None:
        writer.release()

    return out_path


def run_experiment_and_collect(video_path: str, cfg: Scenario2Config, out_dir: Path):
    exp = Scenario2Experiment()
    df, summary = exp.run_single(video_path, cfg)
    save_single_outputs(df, summary, out_dir)
    write_markdown_report(
        title=f"Scenario2 Online Report - {cfg.name}",
        summary=summary,
        single_df=df,
        out_path=out_dir / "report.md",
    )
    return df, summary


def render_label_timeline(df: pd.DataFrame):
    st.markdown('<div class="section-title">结果标签时间轴</div>', unsafe_allow_html=True)
    display_cols = [
        "window_id",
        "label",
        "confidence",
        "drift_ms",
        "drift_frames",
        "first_result_latency_ms",
        "uplink_tx_ms",
        "downlink_render_delay_ms",
        "pose_detect_rate",
        "hand_detect_rate",
        "stability_score",
    ]
    display_cols = [c for c in display_cols if c in df.columns]
    st.dataframe(df[display_cols], use_container_width=True)


def render_drift_charts(df: pd.DataFrame):
    st.markdown('<div class="section-title">漂移程度与时延</div>', unsafe_allow_html=True)
    cols = [
        "drift_ms",
        "drift_frames",
        "first_result_latency_ms",
        "ai_latency_ms",
        "user_perceived_latency_ms",
    ]
    cols = [c for c in cols if c in df.columns]
    if cols:
        st.line_chart(df[cols], use_container_width=True)


def render_quality_charts(df: pd.DataFrame):
    st.markdown('<div class="section-title">检测与稳定性</div>', unsafe_allow_html=True)
    cols = [
        "pose_detect_rate",
        "hand_detect_rate",
        "stability_score",
        "keypoint_completeness",
    ]
    cols = [c for c in cols if c in df.columns]
    if cols:
        st.line_chart(df[cols], use_container_width=True)


# -----------------------------
# 主页面
# -----------------------------
def main():
    st.title("场景2：姿势理解（短时序滑动窗口）")
    st.caption("真正可播放的左右双窗口对比 + 运行过程标签 + 漂移程度")

    with st.sidebar:
        st.header("实验参数")

        uploaded = st.file_uploader("上传视频", type=["mp4", "avi", "mov", "mkv"])

        st.markdown("### 上线链路（AI）")
        uplink_resolution = st.selectbox("上线分辨率", [224, 320, 360, 480], index=2)
        uplink_fps = st.selectbox("上线 FPS", [4, 6, 8, 12], index=2)
        window_seconds = st.selectbox("窗口大小（秒）", [1.0, 2.0, 3.0, 4.0], index=1)
        stride_seconds = st.selectbox("滑动步长（秒）", [0.5, 1.0, 2.0], index=1)
        tcp_bandwidth_mbps = st.selectbox("TCP 带宽 (Mbps)", [0.5, 1.0, 2.0, 4.0, 8.0, 16.0], index=4)
        tcp_fixed_rtt_ms = st.selectbox("TCP 固定 RTT (ms)", [0.0, 30.0, 80.0, 150.0], index=0)
        extra_uplink_delay_ms = st.slider("额外上线排队延迟（ms）", 0, 500, 0)

        st.markdown("### 下线链路（用户）")
        downlink_resolution = st.selectbox("下线分辨率", [360, 480, 720], index=1)
        downlink_fps = st.selectbox("下线 FPS", [10.0, 15.0, 24.0], index=1)
        extra_downlink_delay_ms = st.slider("额外下线渲染延迟（ms）", 0, 500, 0)

        max_duration_sec = st.slider("最大分析时长（秒）", 5, 60, 20)

        run_btn = st.button("开始运行")

    if uploaded is None:
        st.info("请先上传一个视频。")
        return

    tmp_dir = Path(tempfile.mkdtemp(prefix="scenario2_streamlit_"))
    video_path = tmp_dir / uploaded.name
    video_path.write_bytes(uploaded.read())

    cfg = Scenario2Config(
        name="streamlit_demo",
        uplink_resolution=uplink_resolution,
        uplink_fps=float(uplink_fps),
        window_seconds=float(window_seconds),
        stride_seconds=float(stride_seconds),
        tcp_bandwidth_mbps=float(tcp_bandwidth_mbps),
        tcp_fixed_rtt_ms=float(tcp_fixed_rtt_ms),
        downlink_resolution=int(downlink_resolution),
        downlink_fps=float(downlink_fps),
        max_duration_sec=float(max_duration_sec),
        extra_uplink_delay_ms=float(extra_uplink_delay_ms),
        extra_downlink_delay_ms=float(extra_downlink_delay_ms),
    )

    st.markdown('<div class="section-title">原始视频</div>', unsafe_allow_html=True)
    st.video(str(video_path))

    if run_btn:
        out_dir = tmp_dir / "outputs"
        out_dir.mkdir(parents=True, exist_ok=True)

        progress = st.progress(0)
        status = st.empty()

        with st.spinner("正在运行实验，请稍候..."):
            status.info("正在执行场景2实验...")
            t0 = time.time()
            df, summary = run_experiment_and_collect(str(video_path), cfg, out_dir)
            wall_time = time.time() - t0
            summary["run_wall_time_sec"] = float(wall_time)
            summary["run_wall_time_min"] = float(wall_time / 60.0)

            source_fps = float(summary.get("source_video_fps", 25.0))
            df = enrich_df_with_drift(df, source_fps)

            # 重新保存带 drift 的版本
            df.to_csv(out_dir / "windows.csv", index=False, encoding="utf-8-sig")
            df.to_excel(out_dir / "windows.xlsx", index=False)

            with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
                json.dump(summary, f, ensure_ascii=False, indent=2)

            status.info("正在生成左右双窗口对比视频...")
            demo_video_path = out_dir / "side_by_side_demo.mp4"
            generate_side_by_side_demo_video(
                video_path=str(video_path),
                df=df,
                cfg=cfg,
                out_path=demo_video_path,
                source_video_fps=source_fps,
            )

            progress.progress(100)
            status.success("实验完成")

        render_summary_cards(summary)

        # 漂移指标卡
        c1, c2, c3 = st.columns(3)
        with c1:
            metric_card("Avg Drift (ms)", f"{df['drift_ms'].mean():.1f}")
        with c2:
            metric_card("Max Drift (ms)", f"{df['drift_ms'].max():.1f}")
        with c3:
            metric_card("Avg Drift (frames)", f"{df['drift_frames'].mean():.1f}")

        st.markdown('<div class="section-title">左右双窗口对比视频（真正可播放）</div>', unsafe_allow_html=True)
        st.write("左边是用户看到的链路，右边是 AI 分析链路，并叠加了实时标签与漂移程度。")
        with open(demo_video_path, "rb") as f:
            st.video(f.read())

        st.markdown('<div class="section-title">实验说明</div>', unsafe_allow_html=True)
        st.write(
            f"""
            - 原视频总时长：{summary.get('source_video_duration_sec', float('nan')):.2f} s  
            - 本轮实际分析视频时长：{summary.get('effective_video_duration_sec', float('nan')):.2f} s  
            - 实际运行耗时：{summary.get('run_wall_time_sec', float('nan')):.2f} s  
            - chunk_size_frames：{summary.get('chunk_size_frames', 'N/A')}
            """
        )

        render_label_timeline(df)
        render_drift_charts(df)
        render_quality_charts(df)

        st.markdown('<div class="section-title">窗口级原始结果表</div>', unsafe_allow_html=True)
        st.dataframe(df, use_container_width=True)

        st.markdown('<div class="section-title">结果下载</div>', unsafe_allow_html=True)
        csv_bytes = df.to_csv(index=False).encode("utf-8-sig")
        st.download_button("下载 windows.csv", data=csv_bytes, file_name="windows.csv", mime="text/csv")

        summary_bytes = json.dumps(summary, ensure_ascii=False, indent=2).encode("utf-8")
        st.download_button("下载 summary.json", data=summary_bytes, file_name="summary.json", mime="application/json")

        report_path = out_dir / "report.md"
        if report_path.exists():
            st.download_button(
                "下载 report.md",
                data=report_path.read_bytes(),
                file_name="report.md",
                mime="text/markdown",
            )

        if demo_video_path.exists():
            st.download_button(
                "下载 side_by_side_demo.mp4",
                data=demo_video_path.read_bytes(),
                file_name="side_by_side_demo.mp4",
                mime="video/mp4",
            )


if __name__ == "__main__":
    main()