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

## 2) 版本策略（v主.次.修）

- 使用 `v主.次.修`，本次为 **bugfix**：`v0.1.3 -> v0.1.4`。
- 语义约定：
  - `feature`：新增能力，升次版本。
  - `bugfix`：修复问题，升修订版本。
  - `refactor`：重构不改行为，通常升修订版本（如影响较大可升次版本）。

## 3) 修改日志（防漂移）

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
