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

## 2) 版本策略（v主.次.修）

- 使用 `v主.次.修`，本次为 **bugfix**：`v0.2.0 -> v0.2.1`。
- 语义约定：
  - `feature`：新增能力，升次版本。
  - `bugfix`：修复问题，升修订版本。
  - `refactor`：重构不改行为，通常升修订版本（如影响较大可升次版本）。

## 3) 修改日志（防漂移）

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
