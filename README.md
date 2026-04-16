# lane

- 当前版本：`v0.2.1`（见 `VERSION`）。
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
