# lane

- 当前版本：`v0.7.3`（见 `VERSION`）。
- 技术路径与修改日志锚点：`docs/ANCHOR.md`。
- 当前主任务：**segment（坡道区间）级属性识别**（不再以整 clip 作为训练样本单位）。

## 标注：双真值源（clip + keyframe）

支持工具：
1. Tkinter：`annotate.py`
2. Web：`annotate_web.py`

输出拆分为两份：

### `clip_labels.csv`
- `clip_id`
- `is_bidirectional`
- `lane_count`（`1 / 2 / 2+ / unknown`）
- `quality`（`ok/review/bad/unknown`）
- `notes`

### `keyframe_labels.csv`
- `image_path`
- `clip_id`
- `frame_idx`
- `frame_scope`（`slope/transition/non_slope/unknown`）

设计原则：
- 进入新 clip 后，只需标一次 clip 属性。
- 浏览关键帧时，只需标 `frame_scope`。

### Web 标注（推荐）

```bash
python annotate_web.py \
  --keyframe_dir ./data/keyframes \
  --keyframe_label_file ./data/keyframe_labels.csv \
  --clip_label_file ./data/clip_labels.csv
```

打开：`http://127.0.0.1:8765`

## bag 解码（多 bag 并行 + 单 bag 内并发）

```bash
python decode_bag.py --config config.yaml --bag_dir ./data/bags --output_dir ./data/frames
```

## 自动标签构建（关键帧 -> segment/manifest）

```bash
python build_training_labels_from_keyframes.py --config config.yaml
```

输入：
- `data/clip_labels.csv`
- `data/keyframe_labels.csv`
- `data/frames/<clip>/timestamp_index.csv`

输出：
- `data/auto_segments.csv`（审计/复核）
- `data/train_manifest.csv`
- `data/val_manifest.csv`

规则（v1）：
- 以 `slope/non_slope` 关键帧作为强锚点。
- 使用相邻锚点中点切分负责区间。
- `unknown/transition` 不直接定边界，但会触发 `quality=review`。
- `quality=bad` 永不进入训练；`review` 默认不进入训练（可配 `include_review_segments=true` 放行）。

## 训练（segment 级）

```bash
python train.py --config config.yaml
```

## 推理（segment 级）

单段：

```bash
python infer.py --config config.yaml --clip_dir ./data/frames/bag001 --start_frame 120 --end_frame 220
```

批量（manifest / auto_segments 同 schema 兼容，支持 `sample_id|segment_id` 与 `clip|clip_id`）：

```bash
python infer.py --config config.yaml --manifest ./data/val_manifest.csv
```
