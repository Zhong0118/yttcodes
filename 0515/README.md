# Charades 动作切片与指标实验骨架

这个目录现在可以先在“没有训练模型”的情况下跑通老师要求的三件事：

1. **action video clip**：按 Charades 的 `actions` 时间戳切动作，也支持固定窗口切片。
2. **统计 action 时间分布和延时**：输出动作时长、`action_end -> clip_end` 的 delay 分布。
3. **clip 的指标**：输出 clip-level 预测占位结果、hit@k、mAP/F1、推理/传输/端到端延迟。

## 先跑一个 50 视频实验

```bash
python dataset.py --annotations Charades/Charades_v1_train.csv --video-dir Charades_v1_480 --mode action --max-videos 50 --require-video --delay-sec 0.5 --out-manifest outputs/manifest.csv
python simulate_results.py --manifest outputs/manifest.csv --mode oracle --network-delay-ms 50 --out outputs/clip_results.csv
python aggregate_metrics.py --clip-results outputs/clip_results.csv --out-dir outputs/metrics
python plot_results.py --clip-results outputs/clip_results.csv --summary outputs/metrics/action_summary.csv --run-summary outputs/metrics/run_summary.json --out-dir outputs/figures
```

关键输出：

- `outputs/manifest.csv`：每个 clip 的起止时间、动作标签、动作原始起止时间、delay。
- `outputs/clip_results.csv`：clip-level 指标，当前用占位预测，后面可替换成真实模型输出。
- `outputs/metrics/action_summary.csv`：按 action 聚合的 hit@1/3/5、置信度和延迟。
- `outputs/metrics/run_summary.json`：整次实验总指标。
- `outputs/figures/*.png`：动作时长分布、delay 分布、延迟柱状图等。

## 切片方式

按动作切片：

```bash
python dataset.py --mode action --delay-sec 0.5 --context-before-sec 0.25 --require-video
```

这里 `delay-sec` 就是老师说的“终止信号/回溯改 delay 的空间”：clip 在 `action_end + delay_sec` 处结束，用来模拟动作结束后系统延迟多久才切断。

固定窗口切片：

```bash
python dataset.py --mode fixed --clip-seconds 2.56 --stride-seconds 1.28 --require-video
```

固定窗口会把与窗口重叠的动作写入 `action_id`，多个标签用分号连接。

## 真切视频片段

需要本机有 `ffmpeg`：

```bash
python cut_videos.py --manifest outputs/manifest.csv --out-dir outputs/clips --limit 20
```

不想实际切，只看会生成哪些路径：

```bash
python cut_videos.py --manifest outputs/manifest.csv --out-dir outputs/clips --dry-run
```

## 没有模型时怎么处理

`simulate_results.py` 有三种模式：

- `oracle`：用 ground truth 生成理想预测，适合先验证指标管线是否正确。
- `random`：随机预测，适合当负基线。
- `empty`：不预测任何标签，适合检查空结果边界。

以后有真实模型后，只要产出同结构的 `clip_results.csv`，再运行 `aggregate_metrics.py` 即可。

## 有 Charades 模型 checkpoint 后

`run_inference.py` 会读取 manifest 中每个 clip 的 `start_sec/end_sec`，从原视频采样帧，跑真实模型，然后输出和 `simulate_results.py` 同结构的结果：

```bash
python run_inference.py --manifest outputs/manifest.csv --checkpoint checkpoints/charades_r3d18.pth --arch r3d_18 --out outputs/real_clip_results.csv
python aggregate_metrics.py --clip-results outputs/real_clip_results.csv --out-dir outputs/real_metrics
```

注意：必须是 **Charades 157 类多标签 checkpoint**。普通 Kinetics 预训练模型的类别不是 `c000-c156`，不能直接算 Charades hit@k/mAP。

## 训练一个 2000 视频的小模型

先用 train.csv 生成训练 action clips：

```bash
python dataset.py --annotations Charades/Charades_v1_train.csv --video-dir Charades_v1_480 --mode action --max-videos 2000 --require-video --delay-sec 0.0 --out-manifest outputs/manifest_train_action_2000.csv
```

再训练 3 个 epoch。训练脚本会显示 tqdm 进度条，并把 checkpoint 写入带 `2000` 的文件夹：

```bash
python train_charades.py --manifest outputs/manifest_train_action_2000.csv --out-dir checkpoints/r3d18_train2000_e3 --epochs 3 --batch-size 4 --num-frames 8 --resize 112 --amp
```

## 小规模参数 sweep

```bash
python sweep.py --max-videos 50 --delay-sec 0,0.5,1.0 --network-delay-ms 20,50,100 --mode oracle
```

输出：`outputs/sweep/runs_summary.csv`，用于比较 delay 和网络延迟对 `hit@k / mAP / end_to_end_avg_ms` 的影响。
