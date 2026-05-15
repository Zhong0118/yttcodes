import time
import av
import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode

from pipeline.realtime_processor import (
    SharedState,
    RealtimeProcessor,
    RuntimeConfig,
)


st.set_page_config(page_title="实时行为分析演示系统", layout="wide")

if "shared_state" not in st.session_state:
    st.session_state.shared_state = SharedState()

if "processor" not in st.session_state:
    st.session_state.processor = RealtimeProcessor(st.session_state.shared_state)

shared_state: SharedState = st.session_state.shared_state
processor: RealtimeProcessor = st.session_state.processor


def build_sidebar() -> RuntimeConfig:
    with st.sidebar:
        st.header("控制栏")

        bandwidth_kbps = st.slider("带宽限制 bandwidth_kbps", 500, 10000, 4000, 100)
        network_delay_ms = st.slider("网络基础延迟 network_delay_ms", 0, 500, 30, 10)
        ai_extra_delay_ms = st.slider("AI 额外处理延迟 ai_extra_delay_ms", 0, 1000, 0, 10)

        ai_input_size = st.selectbox("AI 输入分辨率", [224, 256, 320], index=1)
        ai_fps = st.slider("AI 输入 FPS", 4, 60, 12, 1)
        clip_len = st.selectbox("clip 长度", [8, 12, 16, 20, 30], index=2)

        model_name = st.selectbox("模型选择", ["x3d_xs", "x3d_s"], index=1)
        jpeg_quality = st.slider("压缩质量 JPEG Quality", 40, 95, 85, 1)

        start_btn = st.button("开始")
        stop_btn = st.button("停止")

    cfg = RuntimeConfig(
        bandwidth_kbps=bandwidth_kbps,
        network_delay_ms=network_delay_ms,
        ai_extra_delay_ms=ai_extra_delay_ms,
        ai_input_size=ai_input_size,
        ai_fps=ai_fps,
        clip_len=clip_len,
        model_name=model_name,
        jpeg_quality=jpeg_quality,
    )

    processor.update_config(cfg)

    if start_btn:
        shared_state.reset_runtime_cache()
        processor.set_running(True)

    if stop_btn:
        processor.set_running(False)

    return cfg


cfg = build_sidebar()

st.title("实时可视化行为分析演示系统")
st.caption("左边显示客户端收到的视频流，右边显示 AI 分析后的结果视频流。")

col_head_1, col_head_2 = st.columns(2)
with col_head_1:
    st.info(f"当前模型：{cfg.model_name}")
with col_head_2:
    device_str = shared_state.recognizer.device if shared_state.recognizer else "not_loaded"
    st.info(f"当前设备：{device_str}")

video_col_1, video_col_2 = st.columns(2)
with video_col_1:
    st.subheader("客户端收到的视频流")
    client_placeholder = st.empty()

with video_col_2:
    st.subheader("AI 返回结果后的视频流")
    ai_placeholder = st.empty()


def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
    image = frame.to_ndarray(format="bgr24")
    out = processor.process_frame(image)
    return av.VideoFrame.from_ndarray(out, format="bgr24")


ctx = webrtc_streamer(
    key="action-demo-v2",
    mode=WebRtcMode.SENDRECV,
    media_stream_constraints={"video": True, "audio": False},
    video_frame_callback=video_frame_callback,
    async_processing=True,
)

st.markdown("---")
metric_row_1 = st.columns(4)
metric_row_2 = st.columns(4)
metric_row_3 = st.columns(2)

m_user_latency = metric_row_1[0].empty()
m_ai_latency = metric_row_1[1].empty()
m_skew = metric_row_1[2].empty()
m_infer = metric_row_1[3].empty()

m_infer_avg = metric_row_2[0].empty()
m_conf = metric_row_2[1].empty()
m_label = metric_row_2[2].empty()
m_runtime = metric_row_2[3].empty()

m_desc = metric_row_3[0].empty()

m_top3 = metric_row_3[0].empty()
m_desc = metric_row_3[1].empty()




def render_dashboard():
    client_frame, ai_frame = processor.get_display_frames()
    metrics = processor.get_metrics_dict()

    if client_frame is not None:
        client_placeholder.image(client_frame, channels="BGR", use_container_width=True)

    if ai_frame is not None:
        ai_placeholder.image(ai_frame, channels="BGR", use_container_width=True)

    m_user_latency.metric("用户链路时延", f"{metrics['user_latency_ms']:.1f} ms")
    m_ai_latency.metric("AI 链路时延", f"{metrics['ai_latency_ms']:.1f} ms")
    m_skew.metric("Overlay Skew", f"{metrics['overlay_skew_ms']:.1f} ms")
    m_infer.metric("单次推理耗时", f"{metrics['infer_ms']:.1f} ms")

    m_infer_avg.metric("平均推理耗时", f"{metrics['infer_avg_ms']:.1f} ms")
    m_conf.metric("当前置信度", f"{metrics['confidence']:.3f}")
    m_label.metric("当前行为标签", metrics["pred_label"])
    m_runtime.metric(
        "当前参数",
        f"{metrics['resolution']} | {metrics['bandwidth_kbps']} kbps"
    )

    m_top3.markdown(f"### Top-3 动作候选\n{metrics['top3_text']}")
    m_desc.markdown(f"### 当前行为分析描述\n{metrics['description']}")



if ctx and ctx.state.playing:
    refresh_anchor = st.empty()
    while ctx.state.playing:
        render_dashboard()
        time.sleep(0.15)
        refresh_anchor.markdown("")
else:
    st.warning("请点击页面中的 START，并允许浏览器访问摄像头。")