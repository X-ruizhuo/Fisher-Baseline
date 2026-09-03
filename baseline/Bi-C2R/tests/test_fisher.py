import os
import sys

import torch
from torch.utils.data import DataLoader, TensorDataset


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from reid.utils.fisher import (  # noqa: E402
    estimate_fisher,
    fisher_aware_state_dict_fusion,
    fisher_consolidation_loss,
    merge_fisher,
    normalize_fisher,
    snapshot_parameters,
)


class TinyClassifier(torch.nn.Module):
    """用于验证 Fisher 数值和参数固化行为的最小分类模型。"""

    def __init__(self):
        super(TinyClassifier, self).__init__()
        self.base = torch.nn.Linear(2, 2, bias=False)
        self.classifier = torch.nn.Linear(2, 2, bias=False)

    def forward(self, images, get_all_feat=False):
        feature = self.base(images)
        logits = self.classifier(feature)
        if get_all_feat:
            return feature, feature, logits, feature
        if self.training:
            return feature, feature, logits, feature
        return feature


def test_fisher_is_nonnegative_and_uses_model_logits():
    model = TinyClassifier()
    images = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    labels = torch.tensor([0, 1])
    loader = DataLoader(TensorDataset(images, labels), batch_size=2)

    fisher = estimate_fisher(
        model, loader, device="cpu", estimator="per_sample"
    )

    assert fisher
    assert all(torch.all(value >= 0) for value in fisher.values())
    assert all(torch.isfinite(value).all() for value in fisher.values())


def test_fisher_merge_and_normalization():
    old = {"base.weight": torch.tensor([[2.0]])}
    new = {"base.weight": torch.tensor([[3.0]])}

    merged = merge_fisher(old, new, gamma=0.5)
    assert torch.equal(merged["base.weight"], torch.tensor([[4.0]]))

    normalized = normalize_fisher(merged, mode="per_parameter_mean")
    assert torch.equal(normalized["base.weight"], torch.tensor([[1.0]]))


def test_new_classifier_rows_are_not_consolidated():
    model = TinyClassifier()
    old_params = snapshot_parameters(model)
    old_fisher = {
        name: torch.ones_like(value)
        for name, value in old_params.items()
    }

    with torch.no_grad():
        model.classifier.weight[0].add_(1.0)
        model.classifier.weight[1].add_(2.0)

    loss = fisher_consolidation_loss(
        model,
        old_params,
        old_fisher,
        old_classifier_rows=1,
    )
    loss.backward()

    assert torch.allclose(model.classifier.weight.grad[1], torch.zeros(2))
    assert torch.any(model.classifier.weight.grad[0] != 0)

def test_fisher_aware_dff_beta_zero_matches_original_dff():
    current = {"module.base.weight": torch.tensor([2.0, 4.0])}
    old = {"module.base.weight": torch.tensor([0.0, 0.0])}
    fisher = {"module.base.weight": torch.tensor([1.0, 10.0])}

    fused = fisher_aware_state_dict_fusion(
        current, old, fisher, alpha=0.25, beta=0.0
    )

    assert torch.allclose(
        fused["module.base.weight"], torch.tensor([0.5, 1.0])
    )


def test_fisher_aware_dff_protects_high_importance_parameters_more():
    current = {"module.base.weight": torch.tensor([2.0, 2.0])}
    old = {"module.base.weight": torch.tensor([0.0, 0.0])}
    fisher = {"module.base.weight": torch.tensor([0.0, 9.0])}

    fused = fisher_aware_state_dict_fusion(
        current, old, fisher, alpha=0.5, beta=1.0
    )

    assert fused["module.base.weight"][1] < fused["module.base.weight"][0]
    assert torch.allclose(fused["module.base.weight"][0], torch.tensor(1.0))
    assert torch.allclose(fused["module.base.weight"][1], torch.tensor(0.1))


def test_fisher_aware_dff_keeps_new_classifier_rows():
    current = {
        "module.classifier.weight": torch.tensor(
            [[2.0, 2.0], [4.0, 4.0], [8.0, 8.0]]
        )
    }
    old = {
        "module.classifier.weight": torch.tensor(
            [[0.0, 0.0], [0.0, 0.0]]
        )
    }
    fisher = {
        "module.classifier.weight": torch.ones(2, 2)
    }

    fused = fisher_aware_state_dict_fusion(
        current, old, fisher, alpha=0.5, beta=1.0
    )

    assert torch.allclose(
        fused["module.classifier.weight"][:2],
        current["module.classifier.weight"][:2] * 0.25,
    )
    assert torch.equal(
        fused["module.classifier.weight"][2],
        current["module.classifier.weight"][2],
    )