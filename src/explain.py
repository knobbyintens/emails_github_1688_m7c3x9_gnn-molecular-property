"""Attention-path explanations for tabular transformers."""

import torch


@torch.no_grad()
def feature_attribution(model, features: torch.Tensor, target_class: int = 1) -> torch.Tensor:
    """Integrated-gradient style attribution over input features."""
    baseline = torch.zeros_like(features)
    steps = 32
    attributions = torch.zeros_like(features)
    for alpha in torch.linspace(0, 1, steps):
        interpolated = baseline + alpha * (features - baseline)
        interpolated.requires_grad_(True)
        logits = model(interpolated)
        score = logits[:, target_class].sum()
        gradients = torch.autograd.grad(score, interpolated)[0]
        attributions += gradients / steps
    return attributions * (features - baseline)


def top_k_features(attributions: torch.Tensor, feature_names: list[str], k: int = 5):
    values, indices = attributions.abs().topk(k, dim=-1)
    return [[feature_names[i] for i in row] for row in indices.tolist()]
