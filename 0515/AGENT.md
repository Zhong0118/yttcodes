------

# 📝 实验执行 Agent 文档：基于真实视频的动作分析（Charades_v1 50 视频示例）

## 1️⃣ 项目概览

**目标**：
构建一个可重复运行的视频动作分析实验平台，实现以下功能：

1. 将视频切片成 clip（按动作或固定长度）
2. 对每个 clip 做动作识别（模型推理）
3. 记录 clip-level 指标：预测、置信度、推理时间、延迟
4. 聚合 action-level 指标：hit@1/3/5, mAP, 延迟分布
5. 支持参数 sweep / 网格实验，分析关键参数对准确率和延迟的影响

**数据集**：

- 主用：Charades_v1_480（可用 50 视频子集快速测试）
- CSV / manifest：记录每个视频 clip 的 start_sec / end_sec / action_id / 文件路径
- 相关视频标签的内容在Charades文件夹，包含训练集和测试集
- 如果要以原视频重新训练模型的话会不会特别慢，还是直接以现成的模型来测

------

## 2️⃣ 数据处理需求

### 视频切片

- **按动作切片（优先）**：利用 Charades 原本的动作时间戳 `start_sec` / `end_sec`
- **固定切片（备用）**：若缺动作标签，则用固定滑动窗口
  - clip_length: 2.56 秒
  - stride: 1.28 秒
  - 支持多参数实验

### manifest CSV 字段示例

| video_id | clip_id | start_sec | end_sec | action_id | video_path      |
| -------- | ------- | --------- | ------- | --------- | --------------- |
| 00001    | 00001   | 0.0       | 2.56    | c009      | /path/00001.mp4 |
| 00001    | 00002   | 1.28      | 3.84    | c009      | /path/00001.mp4 |

> 作用：提供模型推理输入，保证每个 clip 可以独立处理。

------

## 3️⃣ 模型选择

### Backbone

- **推荐**：Slow-R50 (PyTorchVideo)
- **输入要求**：
  - num_frames: 8
  - sampling_rate: 8
  - resize: 224 × 224
- **输出**：
  - multi-label top-k (k=5)
  - 预测类别 + 置信度

### 设计理由

- Charades 是动作时间序列密集数据，Slow-R50 对 temporal information 敏感
- Top5 输出可以覆盖动作片段中多个可能标签，提高 hit@3 / hit@5 指标意义

------

## 4️⃣ Clip-Level 指标

每个 clip 推理后记录：

| 字段          | 含义                        |
| ------------- | --------------------------- |
| video_id      | 视频唯一 ID                 |
| clip_id       | clip 编号                   |
| start_sec     | clip 开始时间（秒）         |
| end_sec       | clip 结束时间（秒）         |
| pred_label    | 模型预测动作                |
| confidence    | 置信度                      |
| infer_ms      | 模型推理耗时（毫秒）        |
| tx_ms         | 网络或传输延迟（模拟/测量） |
| end_to_end_ms | 总时延（从抓帧到输出）      |

输出 CSV：`clip_results.csv`

------

## 5️⃣ Action-Level 聚合指标

将 clip 聚合到对应动作单位（action_id）：

| 指标                                         | 说明                       |
| -------------------------------------------- | -------------------------- |
| hit@1 / hit@3 / hit@5                        | Top-k 是否命中动作         |
| mAP / micro_f1 / macro_f1                    | 多标签准确率指标           |
| avg_confidence                               | clip confidence 平均值     |
| infer_avg_ms / tx_avg_ms / end_to_end_avg_ms | 延迟统计                   |
| clip_count                                   | 每个 action 对应 clip 数量 |

输出 CSV / JSON：`action_summary.csv` 或 `run_summary.json`

------

## 6️⃣ 参数与网格实验

### 可调参数

| 参数                    | 取值示例           | 说明               |
| ----------------------- | ------------------ | ------------------ |
| effective_clip_frames   | 4, 8, 12, 16       | 模型输入帧数       |
| effective_sampling_rate | 4, 8, 10, 12, 14   | 时间采样率         |
| input_resize            | 192, 224, 256, 288 | 视频帧 resize 尺寸 |
| stride_frames           | 4, 8, 16           | clip 滑动窗口步长  |
| bandwidth_mbps          | 1, 4, 8            | 模拟带宽           |
| network_delay_ms        | 20, 50, 100        | 模拟网络延迟       |
| packet_loss             | 0.01, 0.05, 0.2    | 模拟丢包率         |
| jpeg_quality            | 60, 80, 100        | 视频压缩质量       |

### 网格实验思路

- **单因素**：固定其他参数，观察某个参数对 mAP / hit@5 / end_to_end_ms 的影响
- **双因素**：观察交互作用，例如 `network_delay × packet_loss` → end_to_end_ms
- **输出**：每组参数生成 `runs_summary.csv` + 图表

------

## 7️⃣ 分析与图表

### 单因素趋势图

- x轴：参数值
- y轴：指标 (mAP / hit@5 / end_to_end_avg_ms)
- 标注 n=clip 数量（统计有效点）

### 双因素热力图

- 颜色表示平均指标
- 行列分别是两个关键参数
- 可以快速看到哪些组合 tradeoff 最优

------

## 8️⃣ 初步实验建议

- **视频数量**：50 个视频即可验证 pipeline
- **Clip 切片**：动作单位或固定长度
- **指标收集**：clip + action
- **图表输出**：单因素趋势 + 双因素热力图
- **网格实验**：小规模参数 sweep → 先观察趋势

------

## 9️⃣ 可直接执行 Python 脚本（示意）

```bash
# 1. 生成 clip manifest
python dataset.py --video-dir /path/to/videos --out-manifest manifest.csv --clip-seconds 2.56 --stride-seconds 1.28

# 2. 推理 clip
python run_inference.py --manifest manifest.csv --model slow_r50 --topk 5 --out-dir clip_results/

# 3. 聚合 action
python aggregate_metrics.py --clip-results clip_results/clip_results.csv --out-dir action_summary/

# 4. 画图
python plot_results.py --summary action_summary/action_summary.csv --out-dir analysis_outputs/
```

------

✅ **总结**

- 先从小样本 50 视频做验证
- Clip 切片按动作或固定长度
- 使用 **Slow-R50 top5** 预测
- 记录 clip/action-level 指标 → 输出 CSV + 图
- 确认 pipeline 后，再扩展到完整网格实验

