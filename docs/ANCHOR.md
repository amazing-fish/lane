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

## 1.10) 技术路径锚点（issue13）

### 问题定义
- **issue13**：异步写盘首版实现将所有 future 持续累积到内存，长时解码可能出现内存增长。
- ROS1/ROS2 解码循环中若中途异常，写盘线程资源回收与 bag 关闭需要更强兜底。

### 修复策略（固定路径）
1. `AsyncFrameWriter` 改为有界 in-flight 队列（默认 `max_workers*4`），边提交边回收完成任务。
2. `decode_bag_ros1` 与 `decode_bag_ros2` 增加 `try/finally`，确保异常路径下也会执行 `bag.close()/writer.close()`。
3. 保持输出行为不变（文件命名、抽样语义、索引结构）。

### 验收标准
- 长时解码时异步写盘不会因 future 无界累积导致明显内存失控。
- 解码异常时可稳定回收线程池与 bag 句柄。
- `timestamp_index.csv` 与历史结构保持一致。

## 1.11) 技术路径锚点（issue14）

### 问题定义
- **issue14**：解码主循环抛出业务异常时，`finally` 中的资源关闭异常可能覆盖原始异常，增加排障难度（P2）。

### 修复策略（固定路径）
1. 在 `decode_bag.py` 增加 `_safe_close_resources`，统一收敛 `writer.close` / `bag.close` 的清理异常。
2. ROS1/ROS2 解码循环记录主异常：
   - 若主异常存在，清理异常仅告警输出，不覆盖主异常；
   - 若无主异常但清理失败，抛出清理异常，避免静默失败。
3. 不改变解码输出行为与抽样语义，仅修复异常处理优先级。

### 验收标准
- 主流程异常不被清理阶段异常覆盖。
- 清理阶段异常在“无主异常”情况下可被显式抛出。
- 解码路径输出结构保持兼容。


## 1.12) 技术路径锚点（issue15）

### 问题定义
- 当前流程按整 clip 训练，但标签仅在 clip 内的坡道子区间成立，导致监督噪声。
- keyframe 标注无法表达 `non_slope/transition`，无法形成 segment 级数据闭环。

### 修复策略（固定路径）
1. 升级 `annotate.py` 与 `annotate_web.py` schema，新增 `frame_scope`，并补齐 `clip_id/frame_idx`。
2. 统一标注归一化规则：`frame_scope != slope` 时，`is_bidirectional/lane_count` 强制为 `unknown`。
3. 新增 `build_manifest.py`：从 `segment_labels.csv` 自动生成 `train_manifest.csv/val_manifest.csv`。
4. `dataset.py` 从 manifest 读取 `start_frame/end_frame`，仅加载 segment 区间帧。
5. `train.py` 适配 segment manifest 并输出基础统计日志（样本数、平均长度、标签分布）。
6. `infer.py` 改为 segment 级推理：支持 `--clip_dir + --start_frame + --end_frame` 与 `--manifest` 批量。
7. `config.yaml` 增加 `segment_label_file/split_by_clip/include_review_segments`，并保留 `clip_label_file` 兼容。

### 验收标准
- 标注工具可输出新 schema，且兼容旧 CSV。
- `build_manifest.py` 仅输出 `segment_type=slope` 样本并按 `clip_id` 切分。
- 训练与推理均按 segment 区间读帧，不再默认读取整 clip。


## 1.13) 技术路径锚点（issue16）

### 问题定义
- review 发现 segment 闭环首版在鲁棒性与兼容性上仍有边界风险：
  - `build_manifest.py` 对脏值（空串/非法数字/大小写）容错不足，可能 `int()` 失败中断。
  - `dataset.py` 在 `clip_dir` 缺失时直接 `os.listdir` 会抛异常。
  - `infer.py` 读取 manifest 时对缺失 `end_frame` 的兼容不充分。

### 修复策略（固定路径）
1. `build_manifest.py` 增加 `safe_int`，并对 `segment_type/quality/is_bidirectional/lane_count` 做 `strip+lower` 归一化。
2. `dataset.py` 的 `_load_frame_list` 增加 `clip_dir` 可访问性检查，异常路径降级为告警并返回空帧。
3. `infer.py` 增加 `safe_int` 与 `manifest_row_to_segment`：
   - 若无 `end_frame` 则回退到 `start_frame + frame_count - 1`；
   - 增加 `start/end` 参数校验与空 `clip_dir` 行过滤。

### 验收标准
- 脏数据不会导致 manifest 生成中断。
- 训练/推理在缺失目录或缺失字段时可告警降级，不直接崩溃。
- 版本与修改日志同步到 `v0.4.1`。

## 1.14) 技术路径锚点（issue17）

### 问题定义
- 多 bag 输入时 `decode_bag.py` 仍按 bag 串行处理，CPU/GPU 资源利用率不足。
- H.265 packet 路径对 ffmpeg 调用频率高，在异常流上存在明显的启动与回退开销。
- README 对并发层级（bag 级 vs 单 bag 内）说明不够清晰，容易误配。
- 业务标签空间需要将 `lane_count` 收敛到 `1/2/2+`，以降低标注和建模复杂度。

### 修复策略（固定路径）
1. 在 `decode_bag.py` 新增 `decode.bag_workers`，在 `main()` 使用 `ProcessPoolExecutor` 做 bag 级并行。
2. 增加单 bag profiling 输出：耗时、输出帧数、输出 FPS、packet/ffmpeg 调用统计。
3. 在 `H265PacketDecoder` 增加冷却参数 `decode.h265_decode_cooldown_packets`，连续失败后短暂跳过解码尝试，降低高频 ffmpeg 调用开销。
4. 保留原单 bag 解码链路和 `write_workers` 行为，默认配置向后兼容（`bag_workers=1`）。
5. 将 `lane_count` 标签空间统一到 `1/2/2+`：
   - 标注工具（Tk/Web）白名单与快捷键更新；
   - 数据映射与推理类别名更新；
   - 配置 `num_lane_classes` 调整为 `3`。
6. README 与 config 注释同步：明确并发层级、默认生效方式、推荐参数组合与注意事项。

### 验收标准
- 多 bag 场景可通过 `bag_workers>1` 获得并行处理能力。
- 默认配置不传 `bag_workers` 时，行为等价历史串行路径。
- 汇总日志可见单 bag profiling 与 packet/ffmpeg 统计字段。
- `lane_count` 相关工具链一致使用 `1/2/2+`（含 `unknown` 兜底）。

## 1.15) 技术路径锚点（issue18）

### 问题定义
- issue17 中虽优化了 H.265 冷却与统计，但未真正实现“长生命周期 ffmpeg 进程”。
- 高频 packet 流仍可能触发大量 ffmpeg 子进程创建，进程启动开销与回退成本偏高。

### 修复策略（固定路径）
1. 在 `H265PacketDecoder` 增加 `decode.h265_decoder_mode`：
   - `legacy`：保留旧的单次 `subprocess.run` 解码路径；
   - `persistent`：引入长生命周期 `ffmpeg` 进程（`Popen`），持续从 `stdin` 喂入 HEVC packet。
2. persistent 模式下通过 `image2pipe + mjpeg` 输出，使用 JPEG 边界（SOI/EOI）从 stdout 流中提取完整帧。
3. 在 persistent 模式加入异常恢复：
   - 写入/读取异常时自动重启进程；
   - 若硬件加速异常，自动降级 CPU 后重启。
4. 增加进程生命周期 profiling：
   - `ffmpeg_process_launches`
   - `ffmpeg_process_restarts`
5. 保持默认配置可控，支持随时回退到 `legacy`，避免行为漂移。

### 验收标准
- `h265_decoder_mode=persistent` 时，单 bag 内复用 ffmpeg 进程，不再对每次解码都新起子进程。
- 发生 pipe 异常时可自动重启并继续解码，且不影响主流程稳定性。
- README/config/ANCHOR/版本日志同步更新。

## 1.16) 技术路径锚点（issue19-review）

### 问题定义
- 对 issue18 的深度 review 发现 persistent 实现有两个关键边界风险：
  1. 冷却期间直接 `return` 会导致 packet 未输入 ffmpeg，破坏码流连续性；
  2. persistent 模式仍执行“单包 + merged 回退”会向同一 ffmpeg 进程重复喂入历史数据，可能引入时序污染。
- 此外，persistent 进程将 `stderr` 设为 PIPE 且未消费，长时运行存在阻塞风险。

### 修复策略（固定路径）
1. persistent 模式下将冷却语义调整为“**feed-only**”：冷却期仍向 ffmpeg 持续喂包，但不取帧。
2. persistent 模式移除 merged 回退，仅按实时 packet 序列单次解码，避免重复喂历史数据。
3. persistent 进程 `stderr` 改为 `DEVNULL`，降低管道阻塞风险。
4. 配置默认值回调为 `h265_decoder_mode=legacy`，保持向后兼容；README 继续给出 persistent 推荐。

### 验收标准
- persistent 冷却期不丢 packet，码流连续。
- persistent 模式不再重复写入 merged 数据。
- 默认配置保持旧行为兼容，显式设置 `persistent` 时启用长生命周期路径。

## 1.17) 技术路径锚点（issue20-review）

### 问题定义
- issue19 将冷却期改为 feed-only 后，persistent 模式在冷却段不读取 stdout 帧数据。
- 对高吞吐码流，ffmpeg 仍会持续输出 mjpeg；若长期不消费 stdout，可能触发 pipe 积压并反压到 ffmpeg，造成吞吐下降或阻塞。

### 修复策略（固定路径）
1. 在 `H265PacketDecoder._decode_stream_persistent` 增加 `_drain_stdout_nonblocking`。
2. feed-only 路径在写入 packet 后执行非阻塞 drain，定额回收 stdout，避免积压。
3. feed-only 路径清空并丢弃冷却段输出，避免旧帧污染与缓存增长。

### 验收标准
- persistent + 冷却场景下 stdout 可持续被回收，不因长时不读造成明显阻塞。
- feed-only 语义保持不变：冷却期不返回帧，仅维护码流连续性。

## 2) 版本策略（v主.次.修）

- 使用 `v主.次.修`，本次为 **bugfix**：`v0.6.2 -> v0.6.3`。
- 语义约定：
  - `feature`：新增能力，升次版本。
  - `bugfix`：修复问题，升修订版本。
  - `refactor`：重构不改行为，通常升修订版本（如影响较大可升次版本）。

## 3) 修改日志（防漂移）

## [v0.6.3] - bugfix
- 进一步修复 persistent feed-only 路径：冷却期改为“清空并丢弃 stdout 输出”而非缓存，避免旧帧污染后续解码结果。
- 调整非阻塞 drain 的回收上限，降低高码率场景下 stdout 反压阻塞风险。
- 版本升级到 `v0.6.3`。

## [v0.6.2] - bugfix
- 修复 persistent feed-only 冷却期的 stdout 积压风险：新增非阻塞 drain 机制，避免 ffmpeg 管道反压阻塞。
- 对冷却期内部缓存增加上限裁剪，避免长期解码内存增长。（已在 v0.6.3 演进为冷却期直接丢弃输出）
- 版本升级到 `v0.6.2`。

## [v0.6.1] - bugfix
- 修复 persistent 模式在冷却期直接丢包的问题：改为 feed-only，保持 ffmpeg 输入连续性。
- 修复 persistent 模式重复喂 merged 数据的问题：仅按实时 packet 单次解码，不再对同一进程重复注入历史缓存。
- persistent 进程 `stderr` 改为 `DEVNULL`，降低长时运行的管道阻塞风险。
- 默认配置回调为 `h265_decoder_mode=legacy` 以保持向后兼容，README 保留 persistent 推荐说明。
- 版本升级到 `v0.6.1`。

## [v0.6.0] - feature
- 新增 H.265 长生命周期 ffmpeg 解码模式：`decode.h265_decoder_mode=persistent`。
- persistent 模式下复用 `ffmpeg` 进程，通过 `image2pipe(mjpeg)` 持续输出并按 JPEG 边界提取帧，减少高频进程启动开销。
- 增加 persistent 相关统计：`ffmpeg_process_launches`、`ffmpeg_process_restarts`。
- 异常恢复增强：pipe 读写异常触发进程重启；硬件解码异常时自动降级 CPU 后重启。
- README/config 同步新增 `h265_decoder_mode` 说明与推荐组合。
- 版本升级到 `v0.6.0`。

## [v0.5.0] - feature
- `decode_bag.py` 新增 bag 级多进程并行参数 `decode.bag_workers`，并在 `main()` 层支持多 bag 并发处理。
- 新增解码 profiling 汇总：单 bag `elapsed_sec/fps_out`，以及 H.265 packet 的 `packet_total/packet_decode_attempts/packet_decode_success/ffmpeg_calls` 等指标。
- H.265 packet 路径新增冷却策略 `decode.h265_decode_cooldown_packets`，在连续失败场景下减少高频 ffmpeg 调用与回退开销。
- README 与 `config.yaml` 明确并发分层（bag 级并行 vs 单 bag 内并发）及推荐参数组合。
- 车道数标签空间由 `1,2,3,4,5,6+` 收敛为 `1,2,2+`，并同步到标注、数据映射、推理与配置。
- 版本升级到 `v0.5.0`。

## [v0.4.1] - bugfix
- `build_manifest.py` 增加脏值容错（`safe_int`）与字符串归一化（`strip/lower`），避免 CSV 异常值导致生成中断。
- `dataset.py` 增加 `clip_dir` 存在性检查，缺失目录时告警并返回空帧，避免训练时直接异常退出。
- `infer.py` 增强 manifest 兼容：支持缺失 `end_frame` 时由 `frame_count` 回退计算区间，并增加参数有效性校验与空路径过滤。
- 版本升级到 `v0.4.1`。

## [v0.4.0] - feature
- 新增 segment 级数据闭环：增加 `build_manifest.py`，从 `segment_labels.csv` 自动生成 train/val manifest。
- 标注工具升级到统一新 schema：`image_path,clip_id,frame_idx,frame_scope,is_bidirectional,lane_count`，并对非 `slope` 自动归一化为 `unknown`。
- 训练数据读取从 clip 级切换为 segment 级：`dataset.py` 仅加载 `[start_frame,end_frame]` 区间帧。
- `train.py` 输出 segment 样本统计（样本数、平均长度、标签分布）。
- `infer.py` 支持区间推理与 manifest 批量 segment 推理，并在结果中输出区间字段。
- `config.yaml` 增加 `segment_label_file/split_by_clip/include_review_segments`。
- 版本升级到 `v0.4.0`。

## [v0.3.3] - bugfix
- 修复解码清理阶段异常覆盖主异常的问题：新增 `_safe_close_resources`，统一处理 writer/bag 关闭异常。
- ROS1/ROS2 解码流程在存在主异常时仅告警清理异常，保证根因异常可见；无主异常时清理失败会显式抛出。
- 版本升级到 `v0.3.3`。

## [v0.3.2] - bugfix
- 修复异步写盘的资源管理问题：`AsyncFrameWriter` 引入有界 in-flight future 回收机制，避免长时解码内存增长。
- ROS1/ROS2 解码循环增加 `try/finally` 兜底，异常路径下也能正确关闭 bag 句柄与写盘线程池。
- 版本升级到 `v0.3.2`。

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
