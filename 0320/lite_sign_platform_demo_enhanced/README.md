# Lite Sign Experiment Platform

一个面向“直播/双链路行为分析实验”的轻量可运行平台。

## 核心能力
- 单次演示页面（Streamlit）
- 离线批量实验
- 双链路模拟：用户链路 vs AI 分析链路
- 可调因素：带宽、网络延迟、分辨率、FPS、窗口长度、H.264 质量参数、AI 额外延迟
- 无真值标签时的代理指标：
  - 关键点有效检测率
  - 关键点完整度
  - 基线相似度
  - 轨迹稳定性
  - AI latency / p95 latency / overlay skew
- 自动导出 CSV、XLSX、PNG 图表、Markdown 报告

## 安装
```bash
pip install -r requirements.txt
```

## 单次演示
```bash
streamlit run app_streamlit.py
```

## 批量实验
```bash
python run_batch.py --video /path/to/video.mp4 --fast
```

## 默认建议
- 先使用 10~30 秒视频
- 先使用 `--fast`
- 先关注：bandwidth_mbps / net_delay_ms / ai_resolution / ai_fps / window_seconds

## 说明
当前版本使用 MediaPipe Pose + Hands 提取关键点，适合快速搭建系统实验平台。
如果后续需要，可以把 `src/analyzer.py` 替换为更复杂的手语分类或连续识别模型。
