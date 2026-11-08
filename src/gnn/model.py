"""Edge-aware message passing network for molecular property prediction."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class EdgeAwareConv(nn.Module):
    """Message passing layer that conditions messages on bond features."""

    def __init__(self, hidden_dim, edge_dim):
        super().__init__()
        self.message = nn.Sequential(
            nn.Linear(2 * hidden_dim + edge_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.update = nn.GRUCell(hidden_dim, hidden_dim)

    def forward(self, node_features, edge_index, edge_features):
        src, dst = edge_index
        messages = self.message(
            torch.cat([node_features[src], node_features[dst], edge_features], dim=-1)
        )
        aggregated = torch.zeros_like(node_features)
        aggregated.index_add_(0, dst, messages)
        return self.update(aggregated, node_features)


class MolecularGNN(nn.Module):
    """Stack of edge-aware layers with a readout head."""

    def __init__(self, node_dim, edge_dim=6, hidden_dim=128, num_layers=4, num_tasks=12):
        super().__init__()
        self.input_proj = nn.Linear(node_dim, hidden_dim)
        self.layers = nn.ModuleList(
            [EdgeAwareConv(hidden_dim, edge_dim) for _ in range(num_layers)]
        )
        self.readout = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_tasks),
        )

    def forward(self, node_features, edge_index, edge_features, batch_size):
        hidden = self.input_proj(node_features)
        for layer in self.layers:
            hidden = hidden + layer(hidden, edge_index, edge_features)
        pooled = torch.zeros(batch_size, hidden.size(-1), device=hidden.device)
        batch_ids = node_features.new_zeros(hidden.size(0), dtype=torch.long)
        pooled.index_add_(0, batch_ids, hidden)
        return self.readout(pooled)

    def loss(self, logits, targets, mask):
        return F.binary_cross_entropy_with_logits(logits[mask], targets[mask])
