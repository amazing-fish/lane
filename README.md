# lane

- 当前版本：`v0.6.2`（见 `VERSION`）。
- 技术路径与修改日志锚点：`docs/ANCHOR.md`。
- 当前主任务：**segment（坡道区间）级属性识别**（不再以整 clip 作为训练样本单位）。

## 标注：keyframe schema 升级

支持工具：
1. Tkinter：`annotate.py`
2. Web：`annotate_web.py`

输出 schema 统一为：
- `image_path`
- `clip_id`
- `frame_idx`
- `frame_scope`（`slope/transition/non_slope/unknown`）
- `is_bidirectional`
- `lane_count`

其中 `lane_count` 标签空间已收敛为：`1 / 2 / 2+ / unknown`。

规则：
- 当 `frame_scope != slope` 时，工具会自动将 `is_bidirectional` 与 `lane_count` 归一化为 `unknown`。
- 兼容旧 CSV（缺失 `frame_scope` 时自动按 `unknown` 处理）。

### Web 标注（推荐）

```bash
python annotate_web.py --keyframe_dir ./data/keyframes --label_file ./data/keyframe_labels.csv
```

打开：`http://127.0.0.1:8765`

## bag 解码（多 bag 并行 + 单 bag 内并发）

```bash
python decode_bag.py --config config.yaml --bag_dir ./data/bags --output_dir ./data/frames
```

### 并行能力说明（避免误解）

- **bag 级并行（多进程）**：`decode.bag_workers`
  - 作用在 `main()`，多个 bag 可并行处理。
  - 默认 `1`，保持历史串行行为兼容。
- **单 bag 内写盘并发（多线程）**：`decode.write_workers`
  - 仅用于异步写图，降低主链路写盘阻塞。
- **单 bag 内 ffmpeg 解码并发**：`decode.ffmpeg_threads`
  - 作用于 H.265 packet 的 ffmpeg 调用线程数。
- **H.265 解码进程模式**：`decode.h265_decoder_mode`
  - `legacy`：每次尝试解码都启动一次 ffmpeg（默认，兼容旧行为）。
  - `persistent`：长生命周期 ffmpeg 进程，持续喂入 packet，减少进程启动开销（推荐）。
- **H.265 解码降开销策略**：`decode.h265_decode_cooldown_packets`
  - 连续失败后进入冷却，跳过若干 packet 的解码尝试，降低高频 ffmpeg 进程启动/回退开销。

### 配置默认生效方式

- 命令行参数优先级高于 `config.yaml`：
  - `--bag`/`--bag_dir`/`--output_dir` 会覆盖配置文件同名项。
- 其余解码参数均来自 `config.yaml` 的 `decode.*`。

### profiling 输出（每个 bag）

解码汇总会打印：
- 单 bag 耗时（`elapsed_sec`）
- 输出帧数与输出 FPS（`frames`, `fps_out`）
- H.265 packet 统计（`packet_total`, `packet_decode_attempts`, `packet_decode_success`, `ffmpeg_calls` 等）

### 推荐参数组合（12C/24T + NVIDIA）

- 保守稳定（推荐起点）：
  - `bag_workers: 2`
  - `write_workers: 8`
  - `ffmpeg_threads: 6`
  - `ffmpeg_hwaccel: "cuda"`
  - `h265_decoder_mode: "persistent"`
- IO 紧张时：
  - 先降 `bag_workers`（如 `2 -> 1`），再降 `write_workers`。
- GPU 忙/不稳定时：
  - 先改 `ffmpeg_hwaccel: "auto"`，必要时 `"none"` 回 CPU。

## segment 主真值与 manifest 生成

新增主真值文件：`data/segment_labels.csv`，建议字段：
- `segment_id,clip_id,clip_dir,start_frame,end_frame,segment_type,is_bidirectional,lane_count,quality,notes`

生成 train/val：

```bash
python build_manifest.py --config config.yaml
```

规则：
- 仅保留 `segment_type=slope`
- 默认丢弃 `quality=bad`
- 默认按 `clip_id` 切分 train/val（防同 clip 泄漏）

输出 `train_manifest.csv / val_manifest.csv` 字段：
- `sample_id,clip,clip_dir,start_frame,end_frame,frame_count,is_bidirectional,lane_count`

## 训练（segment 级）

```bash
python train.py --config config.yaml
```

`dataset.py` 会根据 manifest 中 `start_frame/end_frame` 仅加载对应区间帧，再执行 snippet 切分。

## 推理（segment 级）

单段：

```bash
python infer.py --config config.yaml --clip_dir ./data/frames/bag001 --start_frame 120 --end_frame 220
```

批量（manifest）：

```bash
python infer.py --config config.yaml --manifest ./data/val_manifest.csv
```

输出会包含 segment 区间信息（`start_frame/end_frame`）。
