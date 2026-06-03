from __future__ import annotations

import torch
from torch import nn


class CoarseEdgeProposer(nn.Module):
    def __init__(self, input_dim: int = 8, hidden_dim: int = 64, num_relations: int = 5) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.relation_head = nn.Linear(hidden_dim, num_relations)
        self.score_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
        hidden = self.encoder(features)
        relation_logits = self.relation_head(hidden)
        edge_scores = self.score_head(hidden).squeeze(-1)
        return {
            "relation_logits": relation_logits,
            "edge_scores": edge_scores,
        }
