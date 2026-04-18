# lane

- 当前版本：`v0.4.1`（见 `VERSION`）。
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

规则：
- 当 `frame_scope != slope` 时，工具会自动将 `is_bidirectional` 与 `lane_count` 归一化为 `unknown`。
- 兼容旧 CSV（缺失 `frame_scope` 时自动按 `unknown` 处理）。

### Web 标注（推荐）

```bash
python annotate_web.py --keyframe_dir ./data/keyframes --label_file ./data/keyframe_labels.csv
```

打开：`http://127.0.0.1:8765`

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
