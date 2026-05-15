# Scenario 2 Pose Understanding Lab

面向“场景2：姿势理解（短时序滑动窗口）”的完整实验工程。

## 目标

本工程聚焦：
- 上线链路：`resolution / fps / chunk(window) / TCP payload`
- 下线链路：`resolution / fps` 对用户“顺滑感 / 感知延迟”的影响
- 核心输入：**火柴人/关键点**，不是原始视频整段上传
- 核心任务：**短时序姿势理解**，不是长时序手语翻译

## 架构

```text
视频/摄像头
  -> 本地预处理（Pose + Hands + Face landmarks）
  -> 短窗口滑动缓存（window_seconds, stride_seconds）
  -> 一个窗口打成一个 chunk
  -> TCP / 本地仿真发送
  -> 服务端短时序理解
  -> 输出 label / confidence / latency / quality metrics
  -> 导出 CSV / XLSX / Markdown / PNG 图表
```

## 目录

- `src/config.py`：配置与参数表
- `src/extractor.py`：火柴人/关键点提取
- `src/windowing.py`：滑动窗口
- `src/classifier.py`：轻量姿势理解模型
- `src/simulator.py`：核心批量实验逻辑
- `src/protocol.py`：TCP JSON 协议
- `src/online_server.py`：在线 TCP 服务端
- `src/online_client.py`：在线 TCP 客户端
- `batch_experiment.py`：离线批量实验入口
- `make_charts.py`：实验图表生成
- `experiment_table.csv`：推荐参数表
- `experiment_table.json`：推荐参数表（程序可读）

## 安装

```bash
pip install -r requirements.txt
```

建议使用：
- Python 3.10 / 3.11
- `mediapipe==0.10.15`

## 先跑一个最小实验

```bash
python batch_experiment.py --video /path/to/video.mp4 --preset smoke
```

## 跑完整场景2推荐实验

```bash
python batch_experiment.py --video /path/to/video.mp4 --preset scenario2_full
```

## 生成图表

```bash
python make_charts.py --batch-dir results/batch_YYYYMMDD_HHMMSS
```

## 在线 TCP 演示

先开服务端：

```bash
python -m src.online_server --host 0.0.0.0 --port 9009
```

再开客户端：

```bash
python -m src.online_client --video /path/to/video.mp4 --host 127.0.0.1 --port 9009
```

## 参数解释

- `uplink_resolution`：上线给 AI 的分辨率
- `uplink_fps`：上线给 AI 的实际分析 FPS
- `window_seconds`：窗口时长，算法视角
- `chunk_size_frames`：窗口内帧数，传输视角；满足：
  `chunk_size_frames = round(uplink_fps * window_seconds)`
- `stride_seconds`：窗口滑动步长
- `downlink_resolution`：用户观看链路分辨率
- `downlink_fps`：用户观看链路 FPS

## 场景2推荐参数

默认主实验：
- uplink_resolution = 360
- uplink_fps = 8
- window_seconds = 2 或 3
- stride_seconds = 1
- downlink_resolution = 720
- downlink_fps = 15

## 输出指标

### 上线 / AI 侧
- `first_result_latency_ms`
- `ai_avg_latency_ms`
- `ai_p95_latency_ms`
- `payload_avg_kb`
- `tcp_est_send_ms_avg`
- `pose_detect_rate_avg`
- `hand_detect_rate_avg`
- `face_detect_rate_avg`
- `keypoint_completeness_avg`
- `stability_score_avg`
- `motion_energy_avg`

### 下线 / 用户侧
- `downlink_frame_interval_ms`
- `downlink_perceived_latency_ms`
- `user_smoothness_score`
- `user_visual_quality_score`

## 说明

1. 本工程优先可跑、可对比、可画图。
2. 没有真值标签时，使用代理指标，不伪造 accuracy/F1。
3. 场景2 的重点不是“长语义翻译”，而是“短时序姿势理解”。
