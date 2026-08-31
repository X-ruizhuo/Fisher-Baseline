# Bi-C2R + Historical Fisher Consolidation

本文档说明如何将 **Historical Fisher Consolidation** 融合到当前
Bi-C2R lifelong person re-identification baseline 中。目标是在不破坏
BiCT、BiCD、BiAD 和 DFF 原有逻辑的前提下，增加参数空间的历史知识保护。

本文档是代码改造说明，不是最终实验结果报告。所有超参数均为初始建议，
最终取值需要通过验证集或消融实验确定。

## 1. 方法概述

当前 baseline 已经在特征空间和关系空间进行跨阶段兼容学习：

```text
BiCT  : 双向特征转换
BiCD  : 兼容特征和关系蒸馏
BiAD  : 双向抗遗忘蒸馏
DFF   : 动态模型参数与历史 gallery 特征融合
```

新增模块在参数空间工作：

```text
Historical Fisher Consolidation
```

它使用上一阶段数据和上一阶段最终部署模型，估计历史身份知识对应的参数
重要性，并在下一阶段约束重要参数不要发生过大漂移。

对于第 `t` 个阶段：

```text
D_(t-1) + theta_(t-1) -> F_(t-1)
```

新增损失为：

```text
L_fim =
1 / (2 |P|)
* sum_{i in P}
  F_(t-1,i) * (theta_(t,i) - theta_(t-1,i))^2
```

最终训练目标为：

```text
L_total = L_BiC2R + lambda_fim * L_fim
```

其中 `P` 是需要保护的参数集合。

## 2. 设计原则

### 2.1 Fisher 的来源

采用历史 Fisher：

```text
上一阶段数据 D_(t-1)
+ 上一阶段最终部署模型 theta_(t-1)
-> F_(t-1)
```

当前阶段 `D_t` 不用于重新计算旧 Fisher。当前阶段数据仍然按照 baseline
原有流程用于当前模型训练、BiCD 和 BiAD。

旧模型在当前阶段的作用与旧 Fisher 的来源需要分开：

```text
old_model + 当前数据 -> BiCD / BiAD
上一阶段数据 + 上一阶段模型 -> old Fisher
```

### 2.2 Fisher 的计算损失

主方案使用身份分类交叉熵：

```text
L_fisher_source = L_ID
```

理想的经验 Fisher 为逐样本梯度平方的平均：

```text
F_i = 1 / N * sum_n (d L_ID,n / d theta_i)^2
```

第一版代码优先实现逐样本 CE 梯度平方。如果显存或速度开销过大，再提供
mini-batch 平均梯度平方作为工程近似，并在实验中单独标注：

```text
per-sample empirical Fisher       主实现
batch-mean gradient square        工程近似消融
```

不建议第一版改用 Hessian、完整二阶矩阵或完整 Bi-C2R 损失计算 Fisher。
这些方案计算代价更高，而且会混入 Triplet、BiCD、BiAD 等损失的权重和梯度。

### 2.3 Fisher 的模型状态

Fisher 必须对应下一阶段实际加载的模型：

```text
阶段 t 训练
-> DFF
-> 保存最终部署模型 theta_t
-> 使用 theta_t 计算 F_new,t
-> 保存 theta_t 和 F_t
```

如果 DFF 仅用于推理期的临时特征融合，而没有修改模型参数，则 Fisher
应当基于真正用于下一阶段初始化的模型状态计算。代码实现前需要确认
DFF 的具体行为。

### 2.4 参数保护范围

初版保护以下参数：

```text
model.base       ResNet backbone
model.bnneck     可训练 BN neck 参数
旧 classifier 行
```

初版不保护：

```text
model_trans      BiCT 转换模块
model_trans2     BiCT 转换模块
新 classifier 行
```

这样可以同时保持：

```text
旧 backbone 和旧身份分类边界的稳定性
新身份 classifier 的可塑性
BiCT 对新域的适应能力
```

## 3. 建议的代码改造结构

预计修改或新增以下位置：

```text
baseline/Bi-C2R/continual_train.py
baseline/Bi-C2R/reid/trainer.py
baseline/Bi-C2R/reid/utils/fisher.py       新增
模型 checkpoint 保存/加载逻辑
配置文件或命令行参数
```

建议不要把 Fisher 的计算和累计逻辑全部塞入 `continual_train.py`。
训练入口只负责阶段级调度，具体数学操作放在 `reid/utils/fisher.py`。

## 4. 新增 Fisher 工具模块

建议在 `reid/utils/fisher.py` 中提供以下接口：

```python
estimate_fisher(
    model,
    dataloader,
    device,
    protected_names=None,
    class_mapping=None,
    max_batches=None,
    estimator="per_sample",
)

normalize_fisher(
    fisher,
    mode="per_parameter_mean",
    clip_value=None,
)

merge_fisher(
    old_fisher,
    new_fisher,
    gamma=1.0,
)

fisher_consolidation_loss(
    model,
    old_params,
    old_fisher,
    protected_names=None,
    classifier_rows=None,
)

save_fisher_checkpoint(
    path,
    fisher,
    stage,
    class_mapping,
    metadata=None,
)

load_fisher_checkpoint(path)
```

### 4.1 Fisher 统计伪代码

```python
model.eval()

fisher = {
    name: torch.zeros_like(param)
    for name, param in model.named_parameters()
    if param.requires_grad and is_protected(name)
}

for images, labels in dataloader:
    model.zero_grad(set_to_none=True)
    logits = get_identity_logits(model(images))
    loss = cross_entropy(logits, labels)
    loss.backward()

    for name, param in model.named_parameters():
        if param.grad is not None and name in fisher:
            fisher[name] += param.grad.detach().pow(2)

fisher /= number_of_batches_or_samples
```

实际主实现应优先支持逐样本梯度平方。若 baseline 的 forward 输出不是单一
logits，需要在 `get_identity_logits` 中适配其返回格式，不能误把 feature、
transformed feature 或蒸馏输出当作分类 logits。

Fisher 计算时必须保留梯度，不能使用 `torch.no_grad()`。同时使用
`model.eval()`，避免 BatchNorm running statistics 和 dropout 影响 Fisher。

### 4.2 Fisher 归一化和累计

建议对每个参数张量独立归一化：

```python
fisher[name] = fisher[name] / (
    fisher[name].mean() + epsilon
)
```

阶段累计：

```python
F_t = gamma * F_(t-1) + normalize(F_new_t)
```

初始默认：

```text
gamma = 1.0
```

消融实验比较：

```text
gamma in {0.5, 0.8, 1.0}
```

建议保留可选的 Fisher 截断，防止极少数异常参数主导正则项：

```python
fisher[name] = fisher[name].clamp(min=0, max=clip_value)
```

## 5. `continual_train.py` 的阶段级改造

### 5.1 阶段开始

阶段 `t > 1` 开始时：

```python
previous_checkpoint = load_stage_checkpoint(stage - 1)
previous_model = load_model(previous_checkpoint)
previous_fisher = load_fisher(previous_checkpoint)

old_model = deepcopy(previous_model)
old_model.eval()
for parameter in old_model.parameters():
    parameter.requires_grad_(False)

old_params = {
    name: value.detach().clone()
    for name, value in previous_model.named_parameters()
}

model = expand_classifier(previous_model, current_classes)
```

`old_model` 只用于 baseline 的 BiCD/BiAD 分支；`old_params` 只用于
`L_fim` 的参数锚点。二者职责分离，避免旧模型副本被错误更新。

阶段 1 没有旧 Fisher：

```python
loss = loss_bic2r
```

阶段 `t > 1`：

```python
loss_fim = fisher_consolidation_loss(
    model=model,
    old_params=old_params,
    old_fisher=previous_fisher,
    ...
)

loss = loss_bic2r + lambda_fim * loss_fim
```

### 5.2 阶段结束

```python
train_current_stage(...)

model = apply_dff(model, ...)

deployed_state = deepcopy(model.state_dict())

new_fisher = estimate_fisher(
    model=model,
    dataloader=current_stage_loader,
    device=device,
    estimator=fisher_estimator,
)

new_fisher = normalize_fisher(new_fisher)
current_fisher = merge_fisher(
    old_fisher=previous_fisher,
    new_fisher=new_fisher,
    gamma=fisher_gamma,
)

save_stage_checkpoint(
    model_state=deployed_state,
    fisher=current_fisher,
    class_mapping=current_class_mapping,
)
```

保存顺序必须保证：

```text
保存的模型状态
和
计算 Fisher 时使用的模型状态
```

完全一致。

## 6. `trainer.py` 的改造方式

原有 Bi-C2R 损失保持不变，仅在最终总损失处添加：

```python
loss_total = loss_bic2r + lambda_fim * loss_fim
```

建议 `loss_fim` 返回一个按参数元素平均的标量：

```python
loss_fim = 0.5 / protected_parameter_count * weighted_squared_distance
```

这样 `lambda_fim` 不会直接随模型参数量变化。

不要把 `L_fim` 加入 BiCD、BiAD 或 BiCT 内部。Fisher 是独立的参数空间
正则项，保持独立便于解释和消融。

## 7. classifier 扩展和参数对齐

classifier 是实现中的最高风险点。不能仅凭当前 tensor 的行号假设旧类别
始终对应相同位置，除非 baseline 明确保证所有阶段采用固定的全局身份映射。

建议每个阶段保存：

```python
class_mapping = {
    identity_id: classifier_row_index
}
```

固化旧 classifier 时执行：

```text
旧 identity
-> 旧阶段 classifier row
-> 当前阶段 classifier row
-> 只对当前对应 row 施加 Fisher 约束
```

新身份 classifier rows 不参与 `L_fim`。

第一版如果确认旧类别始终位于 classifier 前缀，可以先实现 prefix 版本；
但应保留 `class_mapping` 参数，为后续多数据集身份标签映射做准备。

## 8. checkpoint 目录建议

建议每个阶段保存完整的阶段状态：

```text
checkpoints/
  stage_1/
    model.pth
    fisher.pth
    class_mapping.json
  stage_2/
    model.pth
    fisher.pth
    class_mapping.json
```

`fisher.pth` 建议包含：

```python
{
    "fisher": fisher_dict,
    "stage": stage,
    "class_mapping": class_mapping,
    "param_shapes": param_shapes,
    "gamma": gamma,
    "normalization": normalization_name,
    "estimator": estimator_name,
}
```

加载时检查：

```text
参数名是否存在
参数 shape 是否匹配
Fisher 是否包含 NaN 或 Inf
旧 classifier 部分是否可对齐
```

模型 checkpoint 和 Fisher checkpoint 可以分开保存，但必须使用同一阶段编号。

## 9. 配置项建议

建议加入以下配置或命令行参数：

```text
use_fisher                false
lambda_fim                1e-5
fisher_gamma              1.0
fisher_estimator          per_sample
fisher_norm               per_parameter_mean
fisher_clip_value         null
fisher_max_batches        100
fisher_scope              backbone_bn_old_classifier
```

为了保证原始 baseline 可复现，默认应为：

```text
use_fisher = false
```

只有显式开启时才加载和计算 Fisher。

## 10. 测试计划

在完整实验前，先完成以下测试。

### 10.1 Fisher 数值测试

构造一个小型线性分类器，手工计算单样本 CE 梯度平方，并与
`estimate_fisher` 的结果比较。

验证：

```text
Fisher 非负
Fisher 无 NaN / Inf
逐样本平均结果正确
batch 近似路径可以运行
```

### 10.2 累计测试

输入两个已知 Fisher：

```text
F_old = 2
F_new = 3
gamma = 0.5
```

检查输出是否为：

```text
0.5 * 2 + 3 = 4
```

### 10.3 参数固化测试

验证：

```text
Fisher 越大的参数，梯度约束越强
Fisher 为零的参数不产生固化梯度
旧 classifier rows 产生固化梯度
新 classifier rows 不产生固化梯度
BiCT 参数不产生固化梯度
```

### 10.4 两阶段小规模训练

至少运行两个阶段，检查：

```text
阶段 1 是否生成 model.pth 和 fisher.pth
阶段 2 是否能正常加载旧 Fisher
L_fim 是否非零
old_model 是否仍用于 BiCD/BiAD
新身份 classifier 是否可以更新
```

## 11. 实验矩阵

主数据顺序沿用 baseline：

```text
Order 1:
Market1501 -> CUHK-SYSU -> DukeMTMC -> MSMT17 -> CUHK03

Order 2:
DukeMTMC -> MSMT17 -> Market1501 -> CUHK-SYSU -> CUHK03
```

建议至少比较：

```text
Bi-C2R
Bi-C2R + Uniform L2
Bi-C2R + Recent Fisher
Bi-C2R + Historical Fisher
```

关键消融：

```text
CE Fisher vs batch-gradient-square Fisher
Recent Fisher vs accumulated Fisher
gamma in {0.5, 0.8, 1.0}
backbone vs backbone+BN vs backbone+BN+old classifier
lambda_fim in {1e-6, 1e-5, 1e-4, 1e-3}
```

建议记录：

```text
最终平均 mAP
最终平均 Rank-1
每阶段旧任务和新任务性能
Average Forgetting
参数漂移
Fisher-weighted parameter drift
Fisher 计算时间
checkpoint 存储开销
```

## 12. 实现顺序

按以下顺序进行代码改造：

1. 检查 `model.forward()` 的分类 logits 返回格式。
2. 确认 DFF 是否修改模型参数。
3. 确认 classifier 扩展和身份映射规则。
4. 新增 `reid/utils/fisher.py`。
5. 完成 Fisher 工具单元测试。
6. 在 checkpoint 中增加 Fisher 和 class mapping。
7. 在 `trainer.py` 加入 `L_fim`。
8. 在 `continual_train.py` 加入阶段末端 Fisher 统计。
9. 运行单阶段和两阶段小规模训练。
10. 再运行完整阶段序列和消融实验。

## 13. 论文中的方法定位

本文不应声称 Fisher 本身是新的持续学习思想，也不应将简单加入 EWC
描述为完整创新。更准确的表述是：

> We introduce a historical Fisher parameter memory mechanism for lifelong
> person re-identification. The mechanism estimates parameter importance from
> the previous stage data and the final deployed model, and complements
> Bi-C2R's feature-space compatibility learning with parameter-space
> consolidation.

对应的技术贡献为：

```text
1. 历史阶段 Fisher 的阶段末端统计与在线累计
2. 参数空间固化与 Bi-C2R 特征兼容学习的协同
3. 面向扩展身份分类器的选择性旧参数保护
```

## 14. 当前版本的默认方案

第一版代码和实验统一采用：

```text
Fisher source:
    上一阶段数据 + 上一阶段最终部署模型

Fisher loss:
    identity classification CE

Estimator:
    per-sample gradient square

Accumulation:
    F_t = gamma * F_(t-1) + Normalize(F_new,t)

gamma:
    1.0

Protected parameters:
    backbone + BN neck + old classifier rows

Unprotected parameters:
    BiCT transformation modules + new classifier rows

Distillation temperature:
    沿用 Bi-C2R 原有 T=2

Total loss:
    L_BiC2R + lambda_fim * L_fim
```

在完成单元测试和两阶段验证之前，不建议直接启动完整五阶段实验。
