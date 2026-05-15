from __future__ import annotations

from pathlib import Path
import tempfile
import pandas as pd
import streamlit as st

from src.config import ExperimentConfig
from src.experiment import LiteSignExperiment
from src.plotting import plot_batch_summary, plot_single_run, compute_pareto_front
from src.report import write_markdown_report
from src.utils import ensure_dir, now_ts
from src.video_io import write_side_by_side_video

st.set_page_config(page_title='Lite Sign Experiment Platform', layout='wide')
st.title('直播/双链路行为分析实验平台（增强 Demo）')
st.caption('双视频对照 ｜ 时间轴窗口分析 ｜ 参数敏感性热力图 ｜ Pareto 取舍图')

with st.sidebar:
    st.header('单次实验参数')
    bandwidth_mbps = st.slider('带宽 (Mbps)', 0.5, 20.0, 4.0, 0.5)
    net_delay_ms = st.slider('网络延迟 (ms)', 0, 500, 80, 10)
    ai_resolution = st.selectbox('AI 输入分辨率（短边）', [224, 360, 480, 640], index=1)
    ai_fps = st.select_slider('AI 输入 FPS', options=[2, 4, 6, 8, 10, 12], value=8)
    window_seconds = st.select_slider('窗口长度（秒）', options=[1, 2, 3, 4, 5, 6], value=4)
    stride_seconds = st.select_slider('窗口步长（秒）', options=[1, 2, 3, 4, 5, 6], value=4)
    extra_ai_delay_ms = st.slider('AI 额外延迟 (ms)', 0, 1000, 0, 50)
    jpeg_quality = st.slider('压缩质量（JPEG 近似编码损伤）', 20, 95, 75, 5)
    h264_batch_size = st.select_slider('编码 batch size（模拟）', options=[1, 2, 4, 8], value=1)
    max_duration_sec = st.slider('最大分析时长（秒）', 5, 120, 20, 5)

uploaded = st.file_uploader('上传视频', type=['mp4', 'avi', 'mov', 'mkv'])

if uploaded is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp:
        tmp.write(uploaded.read())
        video_path = tmp.name

    st.video(video_path)

    if st.button('运行增强实验', type='primary'):
        with st.spinner('正在运行实验并生成展示图表...'):
            cfg = ExperimentConfig(
                bandwidth_mbps=float(bandwidth_mbps),
                net_delay_ms=float(net_delay_ms),
                ai_resolution=int(ai_resolution),
                ai_fps=float(ai_fps),
                window_seconds=float(window_seconds),
                stride_seconds=float(stride_seconds),
                extra_ai_delay_ms=float(extra_ai_delay_ms),
                jpeg_quality=int(jpeg_quality),
                h264_batch_size=int(h264_batch_size),
                max_duration_sec=float(max_duration_sec),
            )
            exp = LiteSignExperiment()
            baseline_embeddings = exp.build_baseline_embeddings(video_path, max_duration_sec=max_duration_sec, ai_fps=max(ai_fps, 8))
            df, summary = exp.run_single(video_path, cfg, baseline_embeddings=baseline_embeddings)
            out_root = ensure_dir(Path('results') / f'streamlit_enhanced_{now_ts()}')
            df.to_csv(out_root / 'windows.csv', index=False)
            df.to_excel(out_root / 'windows.xlsx', index=False)
            chart_paths = plot_single_run(df, out_root)
            report_path = write_markdown_report('Lite Sign Enhanced Streamlit Report', summary, df, out_root / 'report.md', chart_paths)
            side_by_side_path = write_side_by_side_video(
                original_video_path=video_path,
                out_path=out_root / 'side_by_side_demo.mp4',
                ai_resolution=ai_resolution,
                jpeg_quality=jpeg_quality,
                max_duration_sec=max_duration_sec,
                overlay_text=f'BW={bandwidth_mbps}Mbps Delay={net_delay_ms}ms FPS={ai_fps}',
            )

        top = st.columns(4)
        top[0].metric('AI 平均延迟 (ms)', f"{summary.get('ai_avg_latency_ms', 0):.1f}")
        top[1].metric('P95 延迟 (ms)', f"{summary.get('ai_p95_latency_ms', 0):.1f}")
        top[2].metric('平均 Overlay Skew (ms)', f"{summary.get('overlay_skew_avg_ms', 0):.1f}")
        top[3].metric('基线相似度', f"{summary.get('baseline_similarity_avg', 0):.3f}")

        tabs = st.tabs(['双视频对照', '时间轴分析', '窗口结果表', '导出'])
        with tabs[0]:
            st.subheader('双视频对照 Demo')
            st.video(str(side_by_side_path))
            st.caption('左侧是用户看到的视频，右侧是 AI 链路的压缩/降采样后视图。')

        with tabs[1]:
            st.subheader('时间轴窗口分析')
            st.image(str(out_root / 'timeline_dashboard.png'), use_container_width=True)
            c1, c2 = st.columns(2)
            if (out_root / 'latency_vs_similarity.png').exists():
                c1.image(str(out_root / 'latency_vs_similarity.png'), use_container_width=True)
            if (out_root / 'label_distribution.png').exists():
                c2.image(str(out_root / 'label_distribution.png'), use_container_width=True)
            show_cols = [c for c in ['window_id', 'label', 'confidence', 'ai_latency_ms', 'overlay_skew_ms', 'baseline_similarity', 'hand_detect_rate', 'pose_detect_rate'] if c in df.columns]
            st.line_chart(df.set_index('window_id')[['ai_latency_ms', 'overlay_skew_ms']])
            if 'baseline_similarity' in df.columns:
                st.line_chart(df.set_index('window_id')[['baseline_similarity', 'hand_detect_rate', 'pose_detect_rate']])
            st.dataframe(df[show_cols], use_container_width=True)

        with tabs[2]:
            st.subheader('窗口级结果明细')
            st.dataframe(df.drop(columns=['embedding_str'], errors='ignore'), use_container_width=True)

        with tabs[3]:
            st.subheader('导出文件')
            for filename in ['windows.csv', 'windows.xlsx', 'report.md', 'side_by_side_demo.mp4']:
                path = out_root / filename
                if path.exists():
                    with open(path, 'rb') as f:
                        st.download_button(f'下载 {filename}', data=f.read(), file_name=filename)
            st.write(f'结果目录：{out_root}')
            st.write(f'报告路径：{report_path}')

    st.divider()
    st.subheader('参数敏感性与 Pareto 分析')
    st.caption('下面会基于当前上传视频，跑一个小规模参数网格，用于生成热力图和延迟-效果取舍图。')
    if st.button('生成敏感性热力图与 Pareto 图'):
        with st.spinner('正在跑小规模参数网格...'):
            exp = LiteSignExperiment()
            out_root = ensure_dir(Path('results') / f'grid_{now_ts()}')
            baseline_embeddings = exp.build_baseline_embeddings(video_path, max_duration_sec=max_duration_sec, ai_fps=max(ai_fps, 8))
            rows = []
            grid = {
                'bandwidth_mbps': [1.0, bandwidth_mbps, 8.0],
                'net_delay_ms': [0.0, net_delay_ms, 150.0],
                'ai_resolution': sorted(set([224, ai_resolution, 480])),
                'ai_fps': sorted(set([4.0, ai_fps, 10.0])),
                'window_seconds': sorted(set([2.0, window_seconds, 4.0])),
                'extra_ai_delay_ms': sorted(set([0.0, extra_ai_delay_ms, 200.0])),
                'jpeg_quality': sorted(set([50, jpeg_quality, 85])),
                'h264_batch_size': sorted(set([1, h264_batch_size, 4])),
            }
            from src.experiment import batch_configs
            for i, cfg_dict in enumerate(batch_configs(grid, fast=True)):
                cfg = ExperimentConfig(**cfg_dict, stride_seconds=float(cfg_dict['window_seconds']), max_duration_sec=float(max_duration_sec))
                _, summary = exp.run_single(video_path, cfg, baseline_embeddings=baseline_embeddings)
                rows.append(summary)
            grid_df = pd.DataFrame(rows)
            grid_df.to_csv(out_root / 'grid_summary.csv', index=False)
            grid_df.to_excel(out_root / 'grid_summary.xlsx', index=False)
            heatmap_paths = plot_batch_summary(grid_df, out_root)
            pareto_df = compute_pareto_front(grid_df, 'ai_avg_latency_ms', 'baseline_similarity_avg', minimize_x=True, maximize_y=True)
            if not pareto_df.empty:
                pareto_df.to_csv(out_root / 'pareto_front.csv', index=False)

        st.success(f'敏感性结果已生成：{out_root}')
        for name in [
            'heatmap_bandwidth_resolution_similarity.png',
            'heatmap_bandwidth_fps_latency.png',
            'heatmap_delay_batch_latency.png',
            'heatmap_window_fps_similarity.png',
            'pareto_latency_quality.png',
        ]:
            p = out_root / name
            if p.exists():
                st.image(str(p), use_container_width=True)
        if (out_root / 'grid_summary.xlsx').exists():
            with open(out_root / 'grid_summary.xlsx', 'rb') as f:
                st.download_button('下载 grid_summary.xlsx', data=f.read(), file_name='grid_summary.xlsx')
else:
    st.info('请先上传一个视频文件。')
