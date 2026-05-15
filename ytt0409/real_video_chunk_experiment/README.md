# 真实原视频 Chunk 分析实验平台

这套代码只做一件事：

输入一段真实原视频，按参数切成 chunk，输出每个 chunk 的分析结果，并在不同参数下统计延迟变化。

当前版本只保留：
- 真实视频输入
- chunk 分析
- 参数 sweep
- 延迟统计
- 自动汇总
- 自动画图

不包含：
- 摄像头
- 假数据集
- 假标签评测
- Accuracy / F1 统计

## 一、你现在能做什么

### 1）分析一段视频
读取一个真实视频，按配置：
- ai_fps
- ai_input_size
- clip_len
- stride_len
- bandwidth_kbps
- network_delay_ms
- jpeg_quality

处理后输出：
- 每个 chunk 的起止时间
- 每个 chunk 的预测结果
- 每个 chunk 的置信度
- 每个 chunk 的推理耗时
- 每个 chunk 的传输耗时
- 每个 chunk 的端到端耗时

### 2）批量跑参数 sweep
对同一个真实视频，批量测试不同参数组合，比较：
- infer_avg_ms
- infer_p95_ms
- first_result_ms
- end_to_end_avg_ms

### 3）自动出图
自动画出：
- ai_fps -> infer_avg_ms
- ai_input_size -> infer_avg_ms
- clip_len -> first_result_ms
- bandwidth_kbps -> end_to_end_avg_ms
- network_delay_ms -> end_to_end_avg_ms
- 双因素热力图

## 二、快速开始

### 1. 安装依赖
pip install -r requirements.txt

### 2. 修改配置文件
把 configs/single_video.yaml 里的 video_path 改成你自己的视频路径。

### 3. 跑一次单视频分析
python analyze_video.py --config configs/single_video.yaml

### 4. 跑参数 sweep
python sweep_video.py --config configs/sweep_video.yaml

### 5. 汇总实验结果
python aggregate_runs.py --input outputs

### 6. 自动画图
python plot_runs.py --input outputs

## 三、输出结果

每次运行都会在 outputs/ 下生成一个 run 目录，里面包含：
- chunk_results.csv：每个 chunk 的详细结果
- run_meta.json：本次参数配置
- run_summary.json：本次汇总指标

聚合后得到：
- outputs/runs_summary.csv

自动绘图输出：
- outputs/plots/*.png

## 四、当前默认模型说明

当前默认使用的是一个“可运行的占位分析模型”，用于把整套实验流程跑通。
它会对每个 chunk 返回：
- pred_label
- confidence
- infer_ms

你后面只需要把自己的真实模型接到 src/models.py 里即可，不需要改实验框架。

## 五、你最关心的字段

### chunk_results.csv
- chunk_id
- start_sec
- end_sec
- pred_label
- confidence
- infer_ms
- tx_ms
- network_delay_ms
- first_result_ms
- end_to_end_ms
- payload_kb

### run_summary.json
- infer_avg_ms
- infer_p95_ms
- first_result_ms
- end_to_end_avg_ms
- n_chunks

## 六、推荐执行顺序

先跑：
python analyze_video.py --config configs/single_video.yaml

确认：
- 能正常读视频
- 能输出 chunk_results.csv
- 能输出 run_summary.json

再跑：
python sweep_video.py --config configs/sweep_video.yaml
python aggregate_runs.py --input outputs
python plot_runs.py --input outputs

## 七、后续如何替换成你的真实模型

只改：
- src/models.py

如果你已有推理函数，只要把它包装成：

输入：
- 一个 chunk 的 frame 列表

输出：
- pred_label
- confidence
- infer_ms

就能接入这套平台。
