# lane

- 当前版本：`v0.6.11`（见 `VERSION`）。
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
  - 控制台进度条按**进程槽位**固定行位展示（非按提交顺序），避免长短任务混跑时同一行覆盖。
- **单 bag 内写盘并发（多线程）**：`decode.write_workers`
  - 仅用于异步写图，降低主链路写盘阻塞。
- **单 bag 内 ffmpeg 解码并发**：`decode.ffmpeg_threads`
  - 作用于 H.265 packet 的 ffmpeg 调用线程数。
- **H.265 解码进程模式**：`decode.h265_decoder_mode`
  - `legacy`：每次尝试解码都启动一次 ffmpeg（默认，当前推荐，稳定性更好）。
  - `persistent`：长生命周期 ffmpeg 进程，持续喂入 packet，减少进程启动开销（建议完成稳定性验证后再启用）。
- **H.265 超时保护参数**：
  - `decode.h265_legacy_timeout_sec`：legacy 单次 ffmpeg 调用超时（默认 15s）。
  - `decode.h265_persistent_write_timeout_sec`：persistent 向 stdin 写入超时（默认 1.0s）。
  - `decode.h265_persistent_read_timeout_sec`：persistent 读取 stdout 等待窗口（默认 1.0s）。
- **H.265 解码降开销策略**：`decode.h265_decode_cooldown_packets`
  - 连续失败后进入冷却，跳过若干 packet 的解码尝试，降低高频 ffmpeg 进程启动/回退开销。
- **异常帧过滤（大面积灰色）**：
  - `decode.frame_quality_filter_enabled`: 是否启用异常帧过滤（默认开启）。
  - `decode.frame_phase_probe_enabled`: 是否启用“前几帧探测相位”（默认开启）。
  - `decode.frame_phase_probe_frames`: 启动探测窗口上限（默认 `20`）。
  - `decode.frame_quality_check_interval`: 仅当关闭 `frame_phase_probe_enabled` 时生效；`1` 表示每帧都判，`N` 表示每 `N` 帧判 1 次。
  - `decode.gray_ratio_max`: 低饱和像素占比上限（默认 `0.92`）。
  - `decode.saturation_mean_min`: 全图平均饱和度下限（默认 `16.0`）。
  - `decode.luma_std_min`: 明度标准差下限（默认 `8.0`）。
  - 默认策略：先用前几帧判别“正常帧起始相位”，锁定后按该相位执行 `frame_step` 抽样。

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
  - `h265_decoder_mode: "legacy"`
- IO 紧张时：
  - 先降 `bag_workers`（如 `2 -> 1`），再降 `write_workers`。
- GPU 忙/不稳定时：
  - 先改 `ffmpeg_hwaccel: "auto"`，必要时 `"none"` 回 CPU。

### 当 bag 出现“第 4 帧才正常、每 5 帧正常 1 帧”时

- 保持 `frame_step: 5`（当前默认）用于控制落盘规模。
- 打开 `frame_phase_probe_enabled: true`（默认已开），让系统先用前几帧自动判断“从哪一帧开始正常”。
- 探测不稳定时可增大 `frame_phase_probe_frames`（如 `20 -> 30`）。
- 若你明确不需要“起始相位探测”，可关闭 `frame_phase_probe_enabled`，再用下面三项做持续过滤：
  - 误杀正常帧：放宽 `gray_ratio_max`（如 `0.92 -> 0.96`），降低 `saturation_mean_min`（如 `16 -> 10`）。
  - 异常帧漏检：收紧 `gray_ratio_max`（如 `0.92 -> 0.88`），提高 `luma_std_min`（如 `8 -> 12`）。
- 解码汇总会打印 `phase_probe_frames/phase_probe_hits/sample_phase`，可直观看到探测结果。

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
