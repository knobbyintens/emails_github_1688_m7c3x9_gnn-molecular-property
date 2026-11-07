"""Tabular transformer for clinical risk prediction."""

import torch
import torch.nn as nn


class FeatureTokenizer(nn.Module):
    def __init__(self, num_features: int, d_model: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.randn(num_features, d_model) * 0.02)
        self.bias = nn.Parameter(torch.zeros(num_features, d_model))

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return values.unsqueeze(-1) * self.weight.unsqueeze(0) + self.bias.unsqueeze(0)


class TabularTransformer(nn.Module):
    def __init__(self, num_features: int, d_model: int = 128, num_heads: int = 8, num_layers: int = 4, num_classes: int = 2) -> None:
        super().__init__()
        self.tokenizer = FeatureTokenizer(num_features, d_model)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        layer = nn.TransformerEncoderLayer(d_model, num_heads, d_model * 4, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers)
        self.head = nn.Linear(d_model, num_classes)

    def forward(self, features: torch.Tensor, return_attention: bool = False):
        tokens = self.tokenizer(features)
        batch = tokens.size(0)
        tokens = torch.cat([self.cls_token.expand(batch, -1, -1), tokens], dim=1)
        encoded = self.encoder(tokens)
        logits = self.head(encoded[:, 0])
        if return_attention:
            return logits, encoded
        return logits
