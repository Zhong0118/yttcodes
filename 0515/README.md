# Charades 全量训练与正式评估流程

本 README 记录正式实验流程。当前目标是推翻之前 `train2000` 的快速迭代结果，重新用 **全部 train 视频** 训练模型，再用 **全部 test 视频** 做评估。

你可以手动删除旧结果：

```text
outputs/
checkpoints/
```

不要删除代码文件、`Charades/` 和 `Charades_v1_480/`。

## 0. 进入环境

```powershell
conda activate dl
cd C:\Users\zx\Desktop\0-Inbox\1-Projects\yttcodes\0515
```

如果你已经手动删除旧结果，先重新创建结果目录：

```powershell
New-Item -ItemType Directory -Force -Path outputs | Out-Null
New-Item -ItemType Directory -Force -Path checkpoints | Out-Null
```

## 1. 生成全量训练 manifest

这里使用 `Charades_v1_train.csv` 的全部可用视频。`--max-videos 0` 表示不限制数量。

```powershell
python dataset.py `
  --annotations Charades/Charades_v1_train.csv `
  --video-dir Charades_v1_480 `
  --mode action `
  --max-videos 0 `
  --require-video `
  --delay-sec 0.0 `
  --out-manifest outputs/manifest_train_action_all.csv
```

输出：

```text
outputs/manifest_train_action_all.csv
```

作用：

- 读取全部训练视频的 Charades action 时间戳
- 为每个 action 生成一个训练 clip
- 训练时不需要提前切出实体 mp4，`train_charades.py` 会按 manifest 里的 `start_sec/end_sec` 从原视频采样帧

## 2. 训练全量模型

先训练 3 个 epoch。输出目录不再带 `2000`，而是：

```text
checkpoints/r3d18_trainall_e3/
```

```powershell
python train_charades.py `
  --manifest outputs/manifest_train_action_all.csv `
  --out-dir checkpoints/r3d18_trainall_e3 `
  --epochs 3 `
  --batch-size 4 `
  --num-frames 8 `
  --resize 112 `
  --amp
```

输出：

```text
checkpoints/r3d18_trainall_e3/best.pth
checkpoints/r3d18_trainall_e3/epoch_001.pth
checkpoints/r3d18_trainall_e3/epoch_002.pth
checkpoints/r3d18_trainall_e3/epoch_003.pth
```

说明：

- `best.pth` 是后续推理默认使用的模型
- `tqdm` 会显示每个 epoch 的训练进度
- 如果显存不够，把 `--batch-size 4` 改成 `--batch-size 2`
- 如果训练很快，可以再训练 5 或 10 epoch，但建议先用 3 epoch 跑完整流程

## 3. 生成全量测试 manifest

正式评估使用 `Charades_v1_test.csv`。test 一共有 1863 个视频，`--max-videos 0` 表示全量 test。

### 3.1 Action 切片，delay=0.5

```powershell
$RUN_ACTION = "outputs/experiments/trainall_testall_action_delay0p5"

New-Item -ItemType Directory -Force -Path "$RUN_ACTION/manifest" | Out-Null
New-Item -ItemType Directory -Force -Path "$RUN_ACTION/predictions" | Out-Null
New-Item -ItemType Directory -Force -Path "$RUN_ACTION/metrics" | Out-Null
New-Item -ItemType Directory -Force -Path "$RUN_ACTION/figures" | Out-Null

python dataset.py `
  --annotations Charades/Charades_v1_test.csv `
  --video-dir Charades_v1_480 `
  --mode action `
  --max-videos 0 `
  --require-video `
  --delay-sec 0.5 `
  --out-manifest "$RUN_ACTION/manifest/manifest.csv"
```

作用：

- 按真实 action 边界切片
- clip 结束位置为 `action_end + 0.5s`
- 用来模拟动作结束后有 0.5 秒终止/检测延迟

### 3.2 Fixed 切片，2.56s / 1.28s

```powershell
$RUN_FIXED = "outputs/experiments/trainall_testall_fixed_2p56_1p28"

New-Item -ItemType Directory -Force -Path "$RUN_FIXED/manifest" | Out-Null
New-Item -ItemType Directory -Force -Path "$RUN_FIXED/predictions" | Out-Null
New-Item -ItemType Directory -Force -Path "$RUN_FIXED/metrics" | Out-Null
New-Item -ItemType Directory -Force -Path "$RUN_FIXED/figures" | Out-Null

python dataset.py `
  --annotations Charades/Charades_v1_test.csv `
  --video-dir Charades_v1_480 `
  --mode fixed `
  --clip-seconds 2.56 `
  --stride-seconds 1.28 `
  --max-videos 0 `
  --require-video `
  --out-manifest "$RUN_FIXED/manifest/manifest.csv"
```

作用：

- 不依赖真实动作结束点
- 每 1.28 秒滑动一次窗口
- 每个窗口长度 2.56 秒
- 用来和 action delay=0.5 的切法做对比

## 4. 用全量模型做真实推理

设置 checkpoint：

```powershell
$CKPT = "checkpoints/r3d18_trainall_e3/best.pth"
```

### 4.1 Action delay=0.5 推理

```powershell
python run_inference.py `
  --manifest "$RUN_ACTION/manifest/manifest.csv" `
  --checkpoint $CKPT `
  --arch r3d_18 `
  --num-frames 8 `
  --resize 112 `
  --batch-size 16 `
  --print-topk 5 `
  --print-every 200 `
  --out "$RUN_ACTION/predictions/clip_results.csv"
```

如果显存不够，把 `--batch-size 16` 改成 `--batch-size 8`。

输出：

```text
outputs/experiments/trainall_testall_action_delay0p5/predictions/clip_results.csv
```

### 4.2 Fixed 2.56/1.28 推理

```powershell
python run_inference.py `
  --manifest "$RUN_FIXED/manifest/manifest.csv" `
  --checkpoint $CKPT `
  --arch r3d_18 `
  --num-frames 8 `
  --resize 112 `
  --batch-size 16 `
  --print-topk 5 `
  --print-every 200 `
  --out "$RUN_FIXED/predictions/clip_results.csv"
```

输出：

```text
outputs/experiments/trainall_testall_fixed_2p56_1p28/predictions/clip_results.csv
```

## 5. 聚合指标

### 5.1 Action delay=0.5

```powershell
python aggregate_metrics.py `
  --clip-results "$RUN_ACTION/predictions/clip_results.csv" `
  --out-dir "$RUN_ACTION/metrics"
```

### 5.2 Fixed 2.56/1.28

```powershell
python aggregate_metrics.py `
  --clip-results "$RUN_FIXED/predictions/clip_results.csv" `
  --out-dir "$RUN_FIXED/metrics"
```

输出：

```text
metrics/clip_metrics.csv
metrics/action_summary.csv
metrics/run_summary.json
```

含义：

- `clip_metrics.csv`：每个 clip 的 hit@1、hit@3、hit@5
- `action_summary.csv`：按 action 类别聚合后的指标
- `run_summary.json`：整次实验的总体 hit@k、mAP、F1、延迟

## 6. 画单实验图

### 6.1 Action delay=0.5

```powershell
python plot_results.py `
  --clip-results "$RUN_ACTION/predictions/clip_results.csv" `
  --summary "$RUN_ACTION/metrics/action_summary.csv" `
  --run-summary "$RUN_ACTION/metrics/run_summary.json" `
  --out-dir "$RUN_ACTION/figures"
```

### 6.2 Fixed 2.56/1.28

```powershell
python plot_results.py `
  --clip-results "$RUN_FIXED/predictions/clip_results.csv" `
  --summary "$RUN_FIXED/metrics/action_summary.csv" `
  --run-summary "$RUN_FIXED/metrics/run_summary.json" `
  --out-dir "$RUN_FIXED/figures"
```

每个实验会生成：

```text
overall_performance.png
action_duration_hist.png
delay_hist.png
latency_bar.png
top_action_hit5.png
```

图片含义：

- `overall_performance.png`：总体 hit@1、hit@3、hit@5、mAP、micro_f1、macro_f1
- `action_duration_hist.png`：动作持续时间分布
- `delay_hist.png`：检测/截断延迟分布
- `latency_bar.png`：推理、传输、端到端延迟
- `top_action_hit5.png`：高频动作类别的 hit@5

## 7. 对比 action 和 fixed

```powershell
python compare_experiments.py `
  --run action_delay0p5 "$RUN_ACTION" `
  --run fixed_2p56_1p28 "$RUN_FIXED" `
  --out-dir outputs/experiments/comparison_trainall_testall_action_vs_fixed
```

输出：

```text
outputs/experiments/comparison_trainall_testall_action_vs_fixed/comparison_summary.csv
outputs/experiments/comparison_trainall_testall_action_vs_fixed/performance_comparison.png
outputs/experiments/comparison_trainall_testall_action_vs_fixed/latency_comparison.png
outputs/experiments/comparison_trainall_testall_action_vs_fixed/delay_comparison.png
```

含义：

- `performance_comparison.png`：比较两种切法的 hit@1、hit@3、hit@5、mAP、F1
- `latency_comparison.png`：比较两种切法的推理延迟、传输延迟、端到端延迟
- `delay_comparison.png`：比较两种切法的 delay p50/p90
- `comparison_summary.csv`：所有关键数字的表格版，方便写报告

## 8. 推荐执行顺序

完整顺序如下：

```text
1. 删除旧 outputs/ 和旧 checkpoints/
2. 生成 outputs/manifest_train_action_all.csv
3. 训练 checkpoints/r3d18_trainall_e3/best.pth
4. 生成 action delay=0.5 test manifest
5. 生成 fixed 2.56/1.28 test manifest
6. action delay=0.5 推理
7. fixed 2.56/1.28 推理
8. 分别 aggregate_metrics
9. 分别 plot_results
10. compare_experiments
```

## 9. 最终重点汇报的结果

优先看：

```text
outputs/experiments/comparison_trainall_testall_action_vs_fixed/performance_comparison.png
outputs/experiments/comparison_trainall_testall_action_vs_fixed/latency_comparison.png
outputs/experiments/comparison_trainall_testall_action_vs_fixed/comparison_summary.csv
```

报告结论围绕：

- action delay=0.5 是否比 fixed 2.56/1.28 准确率更高
- fixed 是否更稳定或延迟更低
- 两种切法在 hit@5、mAP、end_to_end latency 上的 tradeoff


---
---

现在推理完毕后，下一步就是：

```text
1. aggregate_metrics.py：聚合指标
2. plot_results.py：每组单独画图
3. compare_experiments.py：多组横向比较
```

一个总脚本：
[scripts/aggregate_plot_compare_all.ps1](c:/Users/zx/Desktop/0-Inbox/1-Projects/yttcodes/0515/scripts/aggregate_plot_compare_all.ps1)

直接跑：

```powershell
conda activate dl
cd C:\Users\zx\Desktop\0-Inbox\1-Projects\yttcodes\0515

Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\scripts\aggregate_plot_compare_all.ps1
```

它会自动处理你这五组：

```text
action_delay0p5
fixed_1p28_0p64
fixed_2p56_1p28
fixed_5p12_2p56
fixed_7p68_3p84
```

**aggregate 得到什么**

每组会生成：

```text
metrics/clip_metrics.csv
metrics/action_summary.csv
metrics/run_summary.json
```

`clip_metrics.csv`：

```text
每个 clip 的 hit@1 / hit@3 / hit@5
每个 clip 的 infer_ms / tx_ms / end_to_end_ms
每个 clip 的 detection_delay_ms
```

`action_summary.csv`：

```text
每个 action 类别的 clip_count
每个 action 类别的 hit@1 / hit@3 / hit@5
每个 action 类别的平均置信度
每个 action 类别的平均延迟
```

`run_summary.json`：

```text
整组实验的总体 hit@1 / hit@3 / hit@5
mAP
micro_f1 / macro_f1
平均推理延迟
平均传输延迟
平均端到端延迟
delay p50 / p90 / p95
action duration p50 / p90
```

这些满足你老师要求里的：

```text
clip 的指标
action-level 聚合指标
action 时间分布
delay 分布
端到端延迟
```

但提醒一句：当前 `mAP` 是基于 top5 分数近似算的，不是严格全 157 类 score 的 mAP。`hit@1/3/5` 是可靠的，延迟指标也是可靠的。

**plot 得到什么**

每组会生成：

```text
figures/overall_performance.png
figures/action_duration_hist.png
figures/delay_hist.png
figures/latency_bar.png
figures/top_action_hit5.png
```

含义：

`overall_performance.png`

```text
总体准确度/性能图
展示 hit@1、hit@3、hit@5、mAP、micro_f1、macro_f1
```

`action_duration_hist.png`

```text
动作时长分布
说明这个切法下动作片段普遍多长
```

`delay_hist.png`

```text
delay 分布
说明终止信号/窗口结束相对 action_end 的延迟情况
```

`latency_bar.png`

```text
延迟构成
展示 infer_ms、tx_ms、end_to_end_ms
```

`top_action_hit5.png`

```text
高频动作类别 hit@5
说明哪些常见动作识别得好或差
```

**compare 能比较几组**

`compare_experiments.py` 可以比较任意多组，只要你一直加：

```powershell
--run 名字 路径
```

你现在五组完全可以一起比。

脚本会输出到：

```text
outputs/experiments/comparison_trainall_testall_all5/
```

里面有：

```text
comparison_summary.csv
performance_comparison.png
latency_comparison.png
delay_comparison.png
```

`performance_comparison.png`

```text
五组切法的 hit@1 / hit@3 / hit@5 / mAP / F1 对比
这是最核心的准确度对比图
```

`latency_comparison.png`

```text
五组切法的 infer_ms / tx_ms / end_to_end_ms 对比
说明哪种切法更快
```

`delay_comparison.png`

```text
五组切法的 delay p50 / p90 对比
说明哪种切法终止延迟更低或更稳定
```

`comparison_summary.csv`

```text
所有关键指标的表格
最适合写报告、复制到 Excel 或做最终结论
```

你现在最该看的顺序是：

```text
1. comparison_summary.csv
2. performance_comparison.png
3. latency_comparison.png
4. delay_comparison.png
```

这四个基本就能支撑你对五种切法的结论。
