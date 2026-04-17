# Anchor 文档（技术路径与修改日志）

> 目标：保证实现路径、版本与变更记录不漂移。

## 1) 技术路径锚点（issue1）

### 问题定义
- **issue1**：训练/验证中，`collate_fn` 会把不同长度 snippet 补零到统一长度，并提供 `masks`。
- 但模型的 `AttentionMIL` 未使用 `masks`，导致注意力可能分配到 padding snippet，稀释真实时序证据，影响分类稳定性与可解释性。

### 修复策略（固定路径）
1. 在 `LaneMVPModel.forward` 增加 `masks` 入参并向 MIL 传递。
2. 在 `AttentionMIL.forward` 中对无效 snippet 的注意力分数置 `-inf` 做 masked softmax。
3. 增加全无效兜底，避免 softmax NaN。
4. 在训练与验证循环中将 batch 的 `masks` 送入模型。
5. 在推理流程中显式构建全 `True` 的 `masks` 并传入模型，统一三条调用链（train/val/infer）。

### 验收标准
- 代码可通过 `python -m py_compile *.py`。
- 新接口与训练调用一致，不破坏现有推理路径（推理仍可不传 `masks`）。
- padding snippet 不再参与注意力竞争。

### 预期与要求（再次深度分析）
- **一致性要求**：train / val / infer 三条路径都走同一接口语义（`forward(snippets, masks)`）。
- **鲁棒性要求**：
  - `masks` 形状必须严格为 `(B, N)`；不满足时应快速失败，防止静默错误。
  - `masks` 全无效时不得出现 NaN；注意力应稳定为全 0，聚合特征稳定为 0。
- **兼容性要求**：历史调用若不传 `masks`，模型内部自动补全全 `True` mask，行为等价于“全部有效”。

## 1.1) 技术路径锚点（issue2）

### 问题定义
- **issue2**：`dataset.py` 训练增强中包含 `RandomHorizontalFlip(p=0.5)`。
- 当前任务是前视道路片段的拓扑识别（是否双向、车道总数），左右翻转会改变道路左右结构语义，属于高风险增强。

### 修复策略（固定路径）
1. 从训练变换中移除 `RandomHorizontalFlip`。
2. baseline 仅保留低风险增强：`Resize + ColorJitter + Normalize`。
3. 同步更新 README 的增强策略说明，并在版本与修改日志中固化。

### 验收标准
- 训练变换不再包含 `RandomHorizontalFlip`。
- 验证变换不受影响，不出现 train/val transform mismatch。
- 版本与锚点日志一致，避免“代码改了但文档未跟进”的漂移。

## 1.2) 技术路径锚点（issue3）

### 问题定义
- **issue3**：当前仅有 `annotate.py`（Tkinter）标注工具。
- 在远程 Linux / 无桌面 / VS Code over SSH 环境中，图形桌面不稳定或不可用，导致标注流程受阻。

### 修复策略（固定路径）
1. 新增 `annotate_web.py`，提供本地 Flask + 单页 HTML 的轻量标注能力。
2. 复用当前 CSV schema：`image_path,is_bidirectional,lane_count`，保持与现有训练流程兼容。
3. 支持关键帧目录浏览、进度展示、上一张/下一张、保存、键盘快捷键。
4. 保留 `annotate.py`，不废弃旧工具。
5. README 增加 Web 启动与快捷键说明，降低上手成本。

### 验收标准
- 可在浏览器打开标注页并加载 `data/keyframes`。
- 可编辑并保存 CSV，且字段与 `annotate.py` 兼容。
- 支持基础导航（上一张/下一张）与快捷键标注。
- Tkinter 工具继续可用。


## 1.3) 技术路径锚点（issue3-review）

### 问题定义
- 针对 issue3 的首版实现进行 review 后，发现可维护性与安全性可继续收敛：
  - 前端页面内嵌于 Python 字符串，维护成本偏高。
  - 图片访问路径校验需要更严格的目录边界约束。
  - 标签入参缺乏白名单校验，异常值可能写入 CSV。

### 修复策略（固定路径）
1. 将前端页面拆分至 `templates/annotate_web.html`，后端通过 `render_template` 渲染。
2. 图片路由采用 `Path.resolve + relative_to` 进行目录边界校验，并使用 `send_from_directory` 提供文件。
3. 后端增加标签白名单归一化：
   - `is_bidirectional` 仅允许 `yes/no/unknown`。
   - `lane_count` 仅允许 `1/2/3/4/5/6+/unknown`。
4. 状态接口补充“已标注计数”语义，提升标注进度可见性。

### 验收标准
- `annotate_web.py` 与 `templates/annotate_web.html` 能正常联动。
- 目录穿越输入返回 400，不可读出 keyframe 根目录外文件。
- 非法标签值不会被原样写入 CSV，会被归一化为 `unknown`。

## 1.4) 技术路径锚点（issue7）

### 问题定义
- **issue7**：`decode_bag.py` 对消息体的假设偏向 image buffer（raw/compressed）。
- 对 `video_format="h265"` + `raw_data` 这类“视频包（packet）”消息无法正确解码，导致无法导出帧与 `timestamp_index.csv`。

### 修复策略（固定路径）
1. 扩展 topic 兼容性：默认候选切换为 `/cam_1 ~ /cam_14`，并将回退匹配扩展为 `front/camera/cam/video/image` 关键词。
2. 在 `decode_bag.py` 增加 packet payload 提取（`raw_data` / `payload` / `data`）。
3. 新增 H.265 解码路径：
   - 优先尝试单包解码；
   - 失败后回退到多包上下文拼接解码（ffmpeg backend）。
4. 保持原有 raw/compressed 图像路径优先，不破坏已有 bag 流程。
5. 在 README 与配置中固化使用说明与参数（`decode.h265_context_packets`）。

### 验收标准
- `/cam_1 ~ /cam_14` 这类 topic 可被默认匹配到。
- `video_format="h265"` 且 `raw_data` 为 Annex B packet 的消息可尝试解码出图像。
- 仍保持输出 `data/frames/<clip_name>/` + `timestamp_index.csv` 结构不变。
- 原 raw/compressed 路径行为保持兼容。

## 1.5) 技术路径锚点（issue8）

### 问题定义
- **issue8**：部分 bag 数据中存在大量异常帧，业务侧确认按固定步进抽样解码更稳妥（建议每 5 帧解 1 帧）。
- 现有 `decode_bag.py` 默认逐消息尝试解码，异常帧会大量落盘并进入后续流程，影响效率与样本质量。

### 修复策略（固定路径）
1. 在 `decode_bag.py` 的 ROS1/ROS2 解码循环中增加“消息步进”参数 `frame_step`。
2. 仅对满足 `msg_idx % frame_step == 0` 的消息执行解码；其余直接跳过。
3. 在 `config.yaml` 中新增 `decode.frame_step`，默认值设为 `5`，保持可配置。
4. 在 README 同步默认行为说明，避免“代码逻辑与文档描述”漂移。

### 验收标准
- 默认配置下解码行为为“每 5 帧取 1 帧”。
- `decode.frame_step` 可调，且最小值兜底为 `1`（不抽样）。
- ROS1 与 ROS2 两条解码路径行为一致。

## 1.6) 技术路径锚点（issue9）

### 问题定义
- **issue9**：在 H.265 packet 流上引入 `frame_step=5` 后，原实现会在“非采样消息”直接 `continue`。
- 这会导致 packet decoder 无法累积上下文包，采样点常因缺少前后文而解码失败，出现 `0 帧 [failed]`。

### 修复策略（固定路径）
1. 保持步进抽样默认行为不变（仍按采样点落盘）。
2. 对 H.265 packet 消息在“非采样点”执行 **buffer 预热**：仅缓存 payload，不做实际落盘。
3. 采样点继续走原解码逻辑（单包 + 上下文拼接回退），利用预热上下文提升成功率。
4. ROS1 / ROS2 两条路径统一接入相同行为，防止链路漂移。

### 验收标准
- `decode.frame_step=5` 时，H.265 packet 流不再因上下文缺失导致普遍 0 帧。
- 抽样语义不变：仅采样点写盘与写 `timestamp_index.csv`。
- ROS1 / ROS2 行为一致。

## 1.7) 技术路径锚点（issue10）

### 问题定义
- **issue10**：在 H.265 packet 流上，当前“消息步进”(`msg_idx % frame_step`) 会先跳过大部分消息，再对采样点尝试解码。
- 该策略会打断 packet 连续性，导致 ffmpeg 难以在采样点复原可用图像，出现“每帧都不可用”的回归。

### 修复策略（固定路径）
1. 将 `frame_step` 语义从“消息步进”改为“**成功解码帧步进**”。
2. 对每条消息仍执行原解码链路（raw/compressed 优先，packet 走单包+上下文回退），确保 packet 序列连续输入。
3. 仅在“成功解码计数”满足步进点时落盘，保持抽样输出规模可控。
4. ROS1 / ROS2 两条链路统一采用同一抽样语义，避免行为分叉。

### 验收标准
- H.265 packet 流在 `decode.frame_step=5` 下不再出现“全部不可用帧”。
- 默认仍是每 5 帧保留 1 帧，但定义为“每 5 个成功解码帧保留 1 帧”。
- ROS1 / ROS2 解码行为一致。

## 1.8) 技术路径锚点（issue11）

### 问题定义
- **issue11**：当前 `decode_bag.py` 解码链路为单线程串行写盘，H.265 包流也仅走 CPU ffmpeg 默认路径。
- 在大 bag 或高分辨率场景下，写盘阻塞与纯 CPU 解码会明显限制吞吐。

### 修复策略（固定路径）
1. 在 `decode_bag.py` 新增 `AsyncFrameWriter`，用线程池执行异步写盘，减小主解码循环阻塞。
2. 新增配置 `decode.write_workers`，统一控制 ROS1 / ROS2 两条链路的写盘并发度。
3. 在 `H265PacketDecoder` 增加 `ffmpeg_threads` 与 `ffmpeg_hwaccel` 参数：
   - 支持 ffmpeg 多线程解码；
   - 支持硬件加速（如 `auto/cuda/vaapi/qsv/videotoolbox`）。
4. 硬件解码失败时自动回退到 CPU 路径，保证兼容性与稳定性。
5. 在 `config.yaml` 与 README 固化新增参数说明，避免实现与文档漂移。

### 验收标准
- `decode.write_workers>1` 时可并发写图，功能与输出结构保持一致。
- 配置 `decode.ffmpeg_hwaccel` 后可尝试 GPU 硬件解码；若不可用会自动回退 CPU。
- ROS1 / ROS2 解码行为一致，`timestamp_index.csv` 结构不变。

## 1.9) 技术路径锚点（issue12）

### 问题定义
- **issue12**：在新增 CPU 多线程 + GPU 解码能力后，缺少针对常见高配机器（如 12C/24T + NVIDIA A4000）的参数选型指引。
- 业务同学难以快速判断“CPU/GPU 该怎么选”，容易导致配置偏保守或不稳定。

### 修复策略（固定路径）
1. 在 README 增加“解码硬件选择建议”小节，明确推荐 `CPU+GPU` 混合方案。
2. 在 `config.yaml` 的解码参数注释中给出可直接落地的调参区间（`write_workers`、`ffmpeg_threads`、`ffmpeg_hwaccel`）。
3. 保持代码行为不变，仅补全选型与调参锚点，避免文档与实现漂移。

### 验收标准
- README 对 12C/24T + A4000 给出明确的推荐组合与理由。
- `config.yaml` 同步给出对应参数建议，便于直接复制使用。

## 2) 版本策略（v主.次.修）

- 使用 `v主.次.修`，本次为 **bugfix**：`v0.3.0 -> v0.3.1`。
- 语义约定：
  - `feature`：新增能力，升次版本。
  - `bugfix`：修复问题，升修订版本。
  - `refactor`：重构不改行为，通常升修订版本（如影响较大可升次版本）。

## 3) 修改日志（防漂移）

## [v0.3.1] - bugfix
- 补充高配机器（12C/24T + A4000）解码参数选型说明，明确推荐 CPU+GPU 混合方案。
- `README.md` 新增硬件选择建议小节，给出 `cuda + write_workers + ffmpeg_threads` 的实践参数范围。
- `config.yaml` 注释补全对应调参建议，避免实现能力与使用路径漂移。
- 版本升级到 `v0.3.1`。

## [v0.3.0] - feature
- 新增解码 CPU 多线程能力：
  - `decode_bag.py` 引入 `AsyncFrameWriter`，支持异步并发写盘；
  - `config.yaml` 新增 `decode.write_workers`。
- 新增解码 GPU/多线程加速能力：
  - `H265PacketDecoder` 支持 `decode.ffmpeg_threads` 与 `decode.ffmpeg_hwaccel`；
  - 硬件加速失败自动回退 CPU，避免解码中断。
- README 补充新增参数说明，版本升级到 `v0.3.0`。

## [v0.2.6] - bugfix
- 修复 `decode_bag.py` 的步进抽样语义：由“消息步进”改为“成功解码帧步进”。
- H.265 packet 流改为持续参与解码链路，仅在步进命中时落盘，避免抽样后出现“全部不可用帧”。
- ROS1 / ROS2 解码路径统一为相同步进策略。
- `config.yaml` 与 README 同步更新 `decode.frame_step` 语义说明。
- 版本升级到 `v0.2.6`。

## [v0.2.5] - bugfix
- 修复 H.265 packet 流在 `decode.frame_step>1` 下的解码回归：非采样点不再直接丢弃，而是预热 packet 上下文缓存。
- 采样点继续按“单包 + 上下文拼接回退”解码，避免出现 `0 帧 [failed]`。
- ROS1 / ROS2 两条解码链路统一引入 packet buffer 预热策略。
- README 同步补充步进抽样与 H.265 上下文保留说明。
- 版本升级到 `v0.2.5`。

## [v0.2.4] - bugfix
- 新增 `decode.frame_step` 配置项，默认值为 `5`，解码阶段按“每 5 帧取 1 帧”步进处理，降低异常帧落盘比例。
- `decode_bag.py` 的 ROS1/ROS2 解码流程统一接入步进过滤逻辑，并对 `frame_step` 做最小值兜底。
- README 同步默认解码步进说明，避免实现与文档漂移。
- 版本升级到 `v0.2.4`。

## [v0.2.3] - bugfix
- 按 issue7 跟进意见，`config.yaml` 的 `decode.front_camera_topics` 调整为仅保留 `/cam_1 ~ /cam_14`，去除历史冗余 topic。
- README 与 Anchor 同步更新默认 topic 说明。
- 版本升级到 `v0.2.3`。

## [v0.2.2] - bugfix
- 修复 `decode_bag.py` 对 H.265 包消息（如 `/cam_1` + `raw_data`）的兼容性缺口：
  - 增加 packet payload 提取与 ffmpeg(H.265/HEVC) 解码路径；
  - 支持“单包解码 + 多包上下文拼接回退”；
  - 扩展前视 topic 回退匹配关键词。
- `config.yaml` 增加默认 topic `/cam_1` 与参数 `decode.h265_context_packets`。
- README 同步新增 H.265 包解码说明。
- 版本升级到 `v0.2.2`。

## [v0.2.1] - bugfix
- 根据 issue3 review 调整 Web 标注工具实现：
  - 前端页面拆分为模板文件 `templates/annotate_web.html`，降低 Python 内嵌 HTML 维护成本。
  - 加强图片路由目录边界校验，避免路径穿越。
  - 增加标签白名单归一化，非法值回退 `unknown`。
  - 新增“已标注/总数”进度展示。
- 版本升级到 `v0.2.1`。

## [v0.2.0] - feature
- 新增 `annotate_web.py`：轻量本地 Web 关键帧标注页面（Flask + 单页 HTML）。
- 支持图片浏览、进度展示、双向与车道标注、CSV 保存、键盘快捷键。
- 输出 CSV 与原工具兼容：`image_path,is_bidirectional,lane_count`。
- README 新增 Web 标注页启动说明与快捷键说明。
- 版本升级到 `v0.2.0`。

## [v0.1.4] - bugfix
- 移除训练增强中的 `RandomHorizontalFlip`，避免左右翻转破坏道路拓扑语义。
- 训练 baseline 增强策略固定为：`Resize + ColorJitter + Normalize`。
- README 同步增强策略说明，避免实现与文档漂移。
- 版本升级到 `v0.1.4`。

## [v0.1.3] - bugfix
- 加强 `AttentionMIL` 的 mask 处理：
  - 增加 `(B, N)` 形状校验，避免调用方传错维度后静默训练。
  - 使用“masked softmax + 乘 mask + 重归一化”，保证全无效时无 NaN。
- `LaneMVPModel.forward` 增加默认全 `True` masks 逻辑，统一接口并保持向后兼容。
- 版本升级到 `v0.1.3`。

## [v0.1.2] - bugfix
- 推理路径 `infer.py` 显式传入 `masks`，与训练/验证保持一致，避免调用链分叉。
- 版本升级到 `v0.1.2`。

## [v0.1.1] - bugfix
- 修复 Attention MIL 未屏蔽 padding snippet 的问题。
- 训练/验证显式传入 `masks`，确保与 `collate_fn` 产物一致。
- 新增版本文件 `VERSION` 与本 Anchor 文档，锁定技术路径与日志。
