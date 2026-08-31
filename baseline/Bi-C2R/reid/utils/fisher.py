"""Historical Fisher 信息估计与参数固化工具。

本模块负责参数重要性统计、历史 Fisher 累计、参数快照和固化损失计算。
阶段训练时序由 continual_train.py 管理，Bi-C2R 原有特征损失由 trainer.py 管理。
"""
from __future__ import absolute_import
import os
import torch
from torch.nn import functional as F

DEFAULT_PROTECTED_PREFIXES = ("base.", "module.base.", "bottleneck.", "module.bottleneck.")


def _is_classifier_name(name):
    """判断参数是否属于身份分类器，兼容 DataParallel 参数前缀。"""
    return name in ("classifier.weight", "module.classifier.weight")


def is_protected_parameter(name, protect_classifier=True):
    """判断参数是否进入 Fisher 保护集合。"""
    if name.startswith(DEFAULT_PROTECTED_PREFIXES):
        return True
    return protect_classifier and _is_classifier_name(name)


def snapshot_parameters(model, names=None):
    """复制旧模型参数快照，并切断计算图，防止锚点随优化器更新。"""
    selected = {}
    allowed = None if names is None else set(names)
    for name, parameter in model.named_parameters():
        if allowed is None or name in allowed:
            selected[name] = parameter.detach().clone()
    return selected


def _extract_batch(batch):
    """从 baseline batch 中提取图像和身份标签。"""
    if isinstance(batch, (tuple, list)) and len(batch) >= 3:
        return batch[0], batch[2]
    if isinstance(batch, (tuple, list)) and len(batch) == 2:
        return batch[0], batch[1]
    raise TypeError("Fisher 数据加载器必须返回 (images, labels) 或 baseline 五元组")


def _forward_identity_logits(model, images):
    """调用 baseline forward，并提取身份分类 logits。"""
    outputs = model(images, get_all_feat=True)
    if isinstance(outputs, (tuple, list)) and len(outputs) >= 3:
        return outputs[2]
    raise RuntimeError("Fisher 计算要求模型返回身份分类 logits")


def _parameter_filter(model, protect_classifier=True):
    """返回需要统计 Fisher 的可训练参数名称。"""
    return [
        name for name, parameter in model.named_parameters()
        if parameter.requires_grad
        and is_protected_parameter(name, protect_classifier=protect_classifier)
    ]


def estimate_fisher(model, dataloader, device=None, max_batches=None, label_offset=0,
                    estimator="per_sample", protect_classifier=True):
    """估计对角经验 Fisher 信息矩阵。

    per_sample 是主方案：逐样本计算身份 CE 梯度平方后求均值。
    batch_mean 是速度优先的工程近似：先求 batch 平均梯度再平方。
    """
    if estimator not in ("per_sample", "batch_mean"):
        raise ValueError("estimator 必须是 per_sample 或 batch_mean")
    if device is None:
        device = next(model.parameters()).device
    device = torch.device(device)

    parameter_names = _parameter_filter(model, protect_classifier)
    named_parameters = dict(model.named_parameters())
    fisher = {
        name: torch.zeros_like(named_parameters[name], device=device)
        for name in parameter_names
    }

    was_training = model.training
    model.eval()
    processed = 0
    try:
        for batch_index, batch in enumerate(dataloader):
            if max_batches is not None and batch_index >= max_batches:
                break
            images, labels = _extract_batch(batch)
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True).long() + label_offset

            if estimator == "batch_mean":
                model.zero_grad(set_to_none=True)
                logits = _forward_identity_logits(model, images)
                loss = F.cross_entropy(logits, labels, reduction="mean")
                loss.backward()
                for name in parameter_names:
                    gradient = named_parameters[name].grad
                    if gradient is not None:
                        fisher[name] += gradient.detach().pow(2) * images.size(0)
                processed += images.size(0)
                continue

            # 逐样本路径避免 batch 内正负梯度抵消，更符合经验 Fisher 定义。
            for sample_index in range(images.size(0)):
                model.zero_grad(set_to_none=True)
                logits = _forward_identity_logits(model, images[sample_index:sample_index + 1])
                loss = F.cross_entropy(logits, labels[sample_index:sample_index + 1])
                loss.backward()
                for name in parameter_names:
                    gradient = named_parameters[name].grad
                    if gradient is not None:
                        fisher[name] += gradient.detach().pow(2)
                processed += 1
    finally:
        model.zero_grad(set_to_none=True)
        if was_training:
            model.train()

    if processed == 0:
        raise ValueError("Fisher 数据加载器没有提供有效样本")
    for name in fisher:
        fisher[name].div_(float(processed))
    return fisher


def normalize_fisher(fisher, mode="per_parameter_mean", epsilon=1e-12, clip_value=None):
    """归一化 Fisher，缓解不同参数张量梯度尺度差异。"""
    if mode not in ("none", "global_mean", "per_parameter_mean"):
        raise ValueError("mode 必须是 none、global_mean 或 per_parameter_mean")
    if not fisher:
        return {}
    denominator = None
    if mode == "global_mean":
        denominator = torch.cat([v.detach().float().reshape(-1) for v in fisher.values()])
        denominator = denominator.mean().clamp_min(epsilon)

    normalized = {}
    for name, value in fisher.items():
        current = value.detach().clone()
        if mode == "per_parameter_mean":
            current = current / current.float().mean().clamp_min(epsilon)
        elif mode == "global_mean":
            current = current / denominator
        if clip_value is not None:
            current = current.clamp(min=0.0, max=clip_value)
        normalized[name] = current
    return normalized


def merge_fisher(old_fisher, new_fisher, gamma=1.0):
    """累计历史 Fisher，并兼容 classifier 行数随阶段扩展。"""
    if gamma < 0:
        raise ValueError("gamma 不能小于 0")
    merged = {}
    for name, value in new_fisher.items():
        if name not in old_fisher:
            merged[name] = value.detach().clone()
            continue
        old_value = old_fisher[name].to(value.device)
        if old_value.shape == value.shape:
            merged[name] = gamma * old_value + value
            continue
        # classifier 只沿第 0 维增加新身份行；新行没有历史重要性。
        if _is_classifier_name(name) and old_value.ndim == value.ndim:
            if old_value.shape[1:] != value.shape[1:] or old_value.shape[0] > value.shape[0]:
                raise ValueError("classifier Fisher 形状不兼容: {}".format(name))
            merged_value = value.detach().clone()
            old_rows = old_value.shape[0]
            merged_value[:old_rows] = gamma * old_value + value[:old_rows]
            merged[name] = merged_value
            continue
        raise ValueError("Fisher 参数形状不一致: {}: {} vs {}".format(
            name, tuple(old_value.shape), tuple(value.shape)))
    return merged


def fisher_consolidation_loss(model, old_params, old_fisher, lambda_normalization=True,
                              old_classifier_rows=None):
    """计算 Fisher 加权固化损失，只约束旧 classifier 行。"""
    total = None
    parameter_count = 0
    for name, parameter in model.named_parameters():
        if name not in old_params or name not in old_fisher:
            continue
        current_value = parameter
        old_value = old_params[name].to(parameter.device)
        fisher_value = old_fisher[name].to(parameter.device)
        if _is_classifier_name(name) and old_classifier_rows is not None:
            rows = int(old_classifier_rows)
            current_value = current_value[:rows]
            old_value = old_value[:rows]
            fisher_value = fisher_value[:rows]
        term = (fisher_value * (current_value - old_value).pow(2)).sum()
        total = term if total is None else total + term
        parameter_count += current_value.numel()
    if total is None or parameter_count == 0:
        return next(model.parameters()).sum() * 0.0
    if lambda_normalization:
        total = total / float(parameter_count)
    return 0.5 * total


def save_fisher_checkpoint(path, fisher, stage, class_mapping=None, metadata=None):
    """保存 Fisher、阶段编号、参数形状和统计元数据。"""
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    torch.save({
        "fisher": {name: value.detach().cpu() for name, value in fisher.items()},
        "stage": int(stage),
        "class_mapping": class_mapping or {},
        "param_shapes": {name: tuple(value.shape) for name, value in fisher.items()},
        "metadata": metadata or {},
    }, path)


def load_fisher_checkpoint(path, map_location="cpu"):
    """加载 Fisher checkpoint，并检查核心字段。"""
    payload = torch.load(path, map_location=map_location)
    if "fisher" not in payload:
        raise KeyError("Fisher checkpoint 缺少 fisher 字段: {}".format(path))
    return payload