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

## 1.18) 技术路径锚点（issue21-review）

### 问题定义
- review 指出 persistent 解码路径将 `timeout` 视作致命失败并立即重启 ffmpeg，会丢失跨包解码上下文。
- 在 HEVC 多包成帧场景，`timeout` 本身是常态；若每次超时都重启，会导致“永远攒不齐上下文”。
- 另一个 review 点是 lane 标签收敛后未对历史 `3/4/5/6+` 进行映射，旧数据会被静默降为 `unknown`，损失监督信息。

### 修复策略（固定路径）
1. persistent 模式下将 `timeout` 视作“暂未产出帧”，不重启 ffmpeg，保留解码状态。
2. 在标注工具、manifest 生成与 legacy 数据读取路径增加 `3/4/5/6+ -> 2+` 映射，保持历史标签可用。
3. 前端 Web 标注归一化逻辑同步 legacy 映射，避免保存时误降级。

### 验收标准
- persistent 模式在 timeout miss 后可继续积累 packet 上下文，不出现频繁重启导致的近零出帧。
- 历史 lane 标签（3/4/5/6+）在新标签体系下自动映射为 `2+`，不再静默丢失监督。

## 1.19) 技术路径锚点（issue22-bugfix）

### 问题定义
- 线上反馈在 `h265_decoder_mode=persistent` 下出现“进度条停住、python/ffmpeg 进程常驻、CPU/IO 低占用”的卡住现象。
- 根因定位在 pipe 交互模型：`stdin.write` 阻塞无超时保护，stdout 等待窗口偏短；同时 legacy 模式缺少子进程级 timeout，异常流可能拖住单次调用。

### 修复策略（固定路径）
1. 给 legacy 模式 `subprocess.run` 增加超时保护（`decode.h265_legacy_timeout_sec`，默认 15s）。
2. persistent 模式引入写入侧超时：
   - 改为基于 `select + os.write` 的分块写入；
   - 超时返回 `write_timeout`，避免在 `stdin.write/flush` 处无限阻塞。
3. persistent 模式将 stdout 读取窗口改为可配置（`decode.h265_persistent_read_timeout_sec`，默认 1.0s），降低短窗口误判。
4. 新增 timeout 统计项（`ffmpeg_write_timeouts`、`ffmpeg_read_timeouts`），便于后续观测与排障。
5. 文档与配置收敛到“默认推荐 legacy、persistent 需稳定性验证后启用”。

### 验收标准
- legacy 模式在坏流/异常流上不会无限等待 ffmpeg 子进程。
- persistent 模式在 pipe 反压时可超时返回，不再长时间卡在写入调用。
- 配置、README、版本号与锚点日志同步，避免技术路径漂移。

## 1.20) 技术路径锚点（issue23-bugfix）

### 问题定义
- **issue23**：部分 bag 存在“周期性灰屏/大面积灰色异常帧”，仅靠 `frame_step` 抽样无法稳定筛出可用帧。
- 业务反馈存在“前几帧异常、之后按固定节奏才出现正常帧”的情况，需要自动判别并过滤异常帧。

### 修复策略（固定路径）
1. 在 `decode_bag.py` 新增帧质量判别函数 `is_valid_frame`，基于 HSV/亮度统计做轻量过滤：
   - 低饱和像素占比（`gray_ratio`）
   - 平均饱和度（`sat_mean`）
   - 明度标准差（`luma_std`）
2. 在 ROS1/ROS2 两条解码链路中统一接入过滤逻辑，异常帧直接跳过，不参与落盘。
3. 新增配置项并在 `config.yaml` 固化默认值：
   - `decode.frame_quality_filter_enabled`
   - `decode.gray_ratio_max`
   - `decode.saturation_mean_min`
   - `decode.luma_std_min`
4. 在 README 增加“异常帧过滤”与调参建议，保证实现与文档同步。
5. 解码汇总增加 `skipped_bad_frames` 指标，便于观察过滤命中率。
6. 新增“启动相位探测”策略：
   - `decode.frame_phase_probe_enabled`
   - `decode.frame_phase_probe_frames`
   仅用前若干帧判断“从哪一帧开始正常”，锁定 `sample_phase` 后按 `frame_step` 抽样。
7. 保留 `decode.frame_quality_check_interval` 作为探测关闭时的兼容路径。

### 验收标准
- 灰屏/低纹理异常帧可在解码阶段被过滤，不落盘。
- `frame_step` 语义保持不变（仍按“成功解码且通过质量过滤”的帧计数抽样）。
- ROS1/ROS2 行为一致，索引结构不变，仅输出帧质量提升。
- 启用探测时，解码汇总可见 `phase_probe_frames/phase_probe_hits/sample_phase`，用于校验起始相位判断结果。

## 1.21) 技术路径锚点（issue24-bugfix）

### 问题定义
- **issue24**：多 bag 并行解码时，多个 worker 的 `tqdm` 进度条会在同一控制台行轮流覆盖，无法同时观察各 worker 的实时进度。

### 修复策略（固定路径）
1. 在 `decode_bag_ros1/decode_bag_ros2` 增加 `progress_position` 参数，并传递给 `tqdm(position=...)`。
2. 在 `main()` 的 `ProcessPoolExecutor` 分发阶段，为每个 worker 分配固定行位（`idx % worker_count`）。
3. 单 bag 串行模式固定使用 `position=0`，保持历史输出兼容。
4. README 同步补充“bag 级并行时进度条按 worker 固定行位展示”，防止文档与实现漂移。

### 验收标准
- `bag_workers>1` 时，控制台可同时显示多行进度条，不再在一行内反复覆盖。
- 串行模式输出行为保持兼容。

## 1.22) 技术路径锚点（issue25-bugfix）

### 问题定义
- PR #17 review 指出：按 `idx % worker_count` 分配 `tqdm.position` 仍可能冲突。
- 在 `ProcessPoolExecutor` 中任务完成顺序不确定，若运行时长不均衡，会出现不同生命周期任务复用同一行位并发显示，进度条再次覆盖。

### 修复策略（固定路径）
1. 去掉“按提交序号”分配行位的逻辑（`idx % worker_count`）。
2. 在 worker 进程内部使用 `multiprocessing.current_process()._identity` 计算稳定槽位（`identity-1`）。
3. 保持串行路径 `position=0` 不变，兼容历史输出。

### 验收标准
- 同时运行的 worker 进度条行位唯一且稳定，不随任务提交顺序变化。
- 长短任务混跑时不再出现两条 live 进度条争用同一行位。

## 2) 版本策略（v主.次.修）

- 使用 `v主.次.修`，本次为 **bugfix**：`v0.6.10 -> v0.6.11`。
- 语义约定：
  - `feature`：新增能力，升次版本。
  - `bugfix`：修复问题，升修订版本。
  - `refactor`：重构不改行为，通常升修订版本（如影响较大可升次版本）。

## 3) 修改日志（防漂移）

## [v0.6.11] - bugfix
- 根据 PR #17 review 修复并行进度条行位分配缺陷：不再使用 `idx % worker_count`。
- 改为 worker 进程内按 `multiprocessing.current_process()._identity` 分配稳定 `tqdm.position`，避免长短任务混跑时行位冲突。
- 版本升级到 `v0.6.11`。

## [v0.6.10] - refactor
- 按版本段压缩历史日志：将 `v0.1.x ~ v0.5.x` 改为分段摘要，降低文档冗余并保留演进脉络。
- 不变更代码行为，仅优化 Anchor 可读性与维护成本。
- 版本升级到 `v0.6.10`。

## [v0.6.9] - bugfix
- 修复多 bag 并行解码时进度条同一行覆盖的问题：ROS1/ROS2 `tqdm` 新增固定 `position` 行位。
- 进程池分发按 worker 固定行位（`idx % worker_count`），并保持串行模式 `position=0` 兼容行为。
- README 同步补充并行进度条展示语义。
- 版本升级到 `v0.6.9`。

## [v0.6.8] - bugfix
- 调整异常帧方案为“前几帧探测相位 + 固定相位抽样”：通过 `frame_phase_probe_enabled/frame_phase_probe_frames` 判断从哪一帧开始正常。
- ROS1/ROS2 新增 `phase_probe_frames/phase_probe_hits/sample_phase` 统计，便于验证探测效果。
- `frame_quality_check_interval` 保留为探测关闭时的兼容模式，README/config/ANCHOR 同步更新。
- 版本升级到 `v0.6.8`。

## [v0.6.7] - bugfix
- 新增 `decode.frame_quality_check_interval`，支持帧质量“每帧判别/按周期判别”。
- ROS1/ROS2 解码链路统一接入周期判别逻辑，新增 `skipped_quality_checks` 统计字段。
- README 与 config 同步补充“每帧还是按周期”的行为说明和调参建议。
- 版本升级到 `v0.6.7`。

## [v0.6.6] - bugfix
- 新增解码阶段异常帧过滤：基于 `gray_ratio/sat_mean/luma_std` 判别并跳过大面积灰色、低纹理帧。
- ROS1/ROS2 解码链路统一接入质量过滤逻辑，并新增 `skipped_bad_frames` 统计指标。
- `config.yaml` 新增帧质量过滤参数（`frame_quality_filter_enabled`、`gray_ratio_max`、`saturation_mean_min`、`luma_std_min`）。
- README 补充“异常帧过滤”机制与“每 5 帧正常 1 帧”场景调参建议。
- 版本升级到 `v0.6.6`。

## [v0.6.5] - bugfix
- 修复 legacy 模式缺少超时保护的问题：为单次 ffmpeg 调用增加 `timeout`（默认 15s）。
- 修复 persistent 模式可能在 `stdin.write` 阻塞导致卡住的问题：改为 `select + os.write` 分块写入并增加写超时保护。
- 放宽并参数化 persistent 读窗口（默认 1.0s），降低短轮询导致的误判；新增写/读超时统计指标便于排障。
- README 与 `config.yaml` 同步新增超时参数，并将推荐模式收敛为 `legacy`。
- 版本升级到 `v0.6.5`。

## [v0.6.4] - bugfix
- 修复 persistent 解码将 `timeout` 误判为致命失败的问题：timeout miss 不再重启 ffmpeg，保留跨包上下文。
- 新增 legacy lane 映射（`3/4/5/6+ -> 2+`）并同步到 Tk/Web 标注、manifest/prepare 数据路径，避免历史监督被静默降级为 `unknown`。
- 版本升级到 `v0.6.4`。

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

## [v0.5.x] - 历史压缩（feature）
- 建立多 bag 并行与 profiling 框架：引入 `bag_workers`、单 bag 耗时/FPS 与 packet 统计。
- 引入 H.265 冷却策略，降低异常流高频 ffmpeg 回退开销。
- 同步收敛 lane 标签空间到 `1/2/2+`，并更新标注/推理/配置说明。

## [v0.4.x] - 历史压缩（feature+bugfix）
- 从 clip 级切换到 segment 级闭环：标注 schema 升级、manifest 生成、区间训练与区间推理打通。
- 补齐脏值容错与兼容兜底：`safe_int`、缺失目录降级告警、manifest 缺字段回退逻辑。

## [v0.3.x] - 历史压缩（feature+bugfix）
- 解码性能能力建设：异步写盘、多线程/硬件解码参数、机型调参建议。
- 资源稳定性修复：有界 future 回收、异常路径资源关闭、主异常优先级保护。

## [v0.2.x] - 历史压缩（feature+bugfix）
- 新增并完善 Web 标注工具（模板化、路径校验、白名单归一化、进度展示）。
- 打通 H.265 packet 解码链路（payload 提取、单包+上下文回退、topic 回退匹配）。
- 逐步修正抽样语义：从消息步进演进到成功帧步进，并加入 packet 预热避免 0 帧回归。

## [v0.1.x] - 历史压缩（bugfix）
- 修复 Attention MIL 对 padding 未屏蔽的问题，训练/验证/推理统一 `masks` 语义。
- 增强 mask 鲁棒性（形状校验、全无效防 NaN）并保持向后兼容。
- 移除高风险左右翻转增强，固定低风险 baseline 增强策略。

## 1.22) 技术路径锚点（issue25-feature）

### 问题定义
- 现有主链路虽已是 segment 级训练，但真实标注仍依赖“关键帧标注后再手工维护 `segment_labels.csv`”。
- 该流程导致重复劳动、边界与证据不一致、数据规模扩大后维护成本快速上升。

### 修复策略（固定路径）
1. 标注真值拆分为两层：
   - `clip_labels.csv`：clip 级常量属性（`is_bidirectional/lane_count/quality/notes`）；
   - `keyframe_labels.csv`：关键帧定位标签（`frame_scope`）。
2. 重构标注工具（Tk/Web）：
   - 进入 clip 时仅维护一次 clip 属性；
   - 浏览关键帧时仅维护 `frame_scope`；
   - 保存时同时输出两份 CSV。
3. 新增 `build_training_labels_from_keyframes.py`：
   - 输入 `clip_labels.csv + keyframe_labels.csv + timestamp_index.csv`；
   - 输出 `auto_segments.csv + train_manifest.csv + val_manifest.csv`。
4. 第一版区间推断采用“锚点中点切分 + review 降级”：
   - `slope/non_slope` 作为强锚点；
   - `unknown/transition` 不直接定边界，但触发 `quality=review`；
   - 边界证据不足（例如触边）同样降级 `review`。
5. 保持训练 manifest 契约不变（`sample_id/clip/clip_dir/start_frame/end_frame/frame_count/is_bidirectional/lane_count`），降低训练链路改动面。

### 验收标准
- 无需人工维护 `segment_labels.csv`，即可由关键帧标注直接生成可训练 manifest。
- `auto_segments.csv` 可用于审计与复核，`review` 样本默认不进入训练（可配置放行）。
- 推理仍保持“给定 segment 做属性预测”的能力边界，不虚构 full-clip 端到端能力。
- `README/config/VERSION/ANCHOR` 同步更新，版本升级为 `v0.7.0`。

## 1.23) 技术路径锚点（issue26-bugfix）

### 问题定义
- review 反馈 `infer.py` 对 `auto_segments.csv` 的兼容不完整：仅识别 `sample_id/clip`，导致读取 `segment_id/clip_id` 时样本标识退化为 `unknown`。
- `build_training_labels_from_keyframes.py` 首版仅遍历 `clip_labels.csv`，当关键帧存在但 clip 标签暂缺时会被静默忽略，影响可用性与排障。

### 修复策略（固定路径）
1. `infer.py` 的 manifest 解析补齐别名兼容：
   - `sample_id <- sample_id | segment_id`
   - `clip <- clip | clip_id`
2. 自动构建脚本改为遍历 `clip_labels ∪ keyframe_labels` 的并集 clip：
   - 缺失 `clip_label` 时，自动注入 `unknown + review + missing_clip_label` 兜底并产出审计记录。
3. `clip_max_frame` 增加 keyframe 回退：当帧目录/索引缺失时，使用关键帧最大 `frame_idx` 估计上界，避免整 clip 被误跳过。
4. README/VERSION/ANCHOR 同步更新，版本升级为 `v0.7.1`（bugfix）。

### 验收标准
- `infer.py --manifest data/auto_segments.csv` 可正确识别 `segment_id/clip_id`。
- `build_training_labels_from_keyframes.py` 在 clip 标签缺失时仍能输出 `auto_segments.csv`（并以 `review` 标记）。
- 无数据时行为保持可解释，不出现静默丢样本。
