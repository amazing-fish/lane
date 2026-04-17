# lane

- 当前版本：`v0.2.3`（见 `VERSION`）。
- 技术路径与修改日志锚点：`docs/ANCHOR.md`。
- 训练增强策略（baseline）：`Resize + ColorJitter + Normalize`，不使用水平翻转以保持道路拓扑语义一致。

## 关键帧标注工具

项目同时提供两种本地标注方式：

1. **Tkinter GUI（原有）**：`annotate.py`
2. **Web 轻量标注页（新增）**：`annotate_web.py`

### Web 标注页（推荐远程/无桌面环境）

```bash
python annotate_web.py --keyframe_dir ./data/keyframes --label_file ./data/keyframe_labels.csv
```

启动后在浏览器打开：`http://127.0.0.1:8765`。

功能：
- 浏览本地关键帧目录
- 展示当前索引/进度
- 标注字段：`is_bidirectional`（yes/no/unknown）、`lane_count`（1/2/3/4/5/6+/unknown）
- 前后导航与键盘快捷键
- 保存为兼容 CSV：`image_path,is_bidirectional,lane_count`

常用快捷键：
- `←/A` 上一张，`→/D` 下一张
- `B/N/U`：双向 yes/no/unknown
- `1-6`：车道数（`6` = `6+`），`0`：unknown
- `Ctrl/Cmd+S` 保存


补充：后端会对标签值做白名单校验，并对图片访问做目录边界校验，避免路径穿越。

## Bag 解码（含 H.265 包消息）

`decode_bag.py` 现已支持除传统 `sensor_msgs/Image` / `CompressedImage` 外的 H.265 包消息（例如 topic `/cam_1`，字段 `video_format="h265"` + `raw_data`）。

- 默认支持 `/cam_1 ~ /cam_14` topic（可在 `config.yaml -> decode.front_camera_topics` 自定义）。
- 当消息是视频包而非图片缓冲时，会自动走 ffmpeg(H.265/HEVC) 解码路径。
- 若单包无法直接出图，会回退到多包上下文拼接解码（参数：`decode.h265_context_packets`）。
- 默认按 `decode.frame_step=5` 做步进解码（每 5 帧取 1 帧），用于降低异常帧比例；可按数据源特性调整。

> 说明：使用 H.265 包解码需要系统可执行 `ffmpeg`。
