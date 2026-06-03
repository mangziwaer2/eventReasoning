from __future__ import annotations

import torch
from torch import nn


class TemporalRelationalEdgeRefiner(nn.Module):
    def __init__(
        self,
        node_dim: int = 6,
        edge_dim: int = 7,
        query_dim: int = 4,
        hidden_dim: int = 64,
        num_message_passing_steps: int = 2,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_message_passing_steps = num_message_passing_steps

        self.node_encoder = nn.Sequential(
            nn.Linear(node_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.edge_encoder = nn.Sequential(
            nn.Linear(edge_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.query_encoder = nn.Sequential(
            nn.Linear(query_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.message_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.node_update = nn.GRUCell(hidden_dim, hidden_dim)

        self.keep_head = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.type_head = nn.Linear(hidden_dim * 3, 4)
        self.strength_head = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )
        self.frontier_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_features: torch.Tensor,
        query_features: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        node_states = self.node_encoder(node_features)
        edge_states = self.edge_encoder(edge_features)
        query_state = self.query_encoder(query_features.unsqueeze(0)).squeeze(0)

        if edge_index.numel() == 0:
            frontier_scores = self.frontier_head(
                torch.cat([node_states, query_state.expand(node_states.size(0), -1)], dim=-1)
            ).squeeze(-1)
            return {
                "edge_keep_logits": torch.empty(0, device=node_features.device),
                "edge_type_logits": torch.empty((0, 4), device=node_features.device),
                "edge_strengths": torch.empty(0, device=node_features.device),
                "frontier_scores": frontier_scores,
            }

        source_index = edge_index[:, 0]
        target_index = edge_index[:, 1]

        for _ in range(self.num_message_passing_steps):
            source_states = node_states[source_index]
            target_states = node_states[target_index]
            query_states = query_state.expand(edge_states.size(0), -1)

            messages = self.message_mlp(torch.cat([source_states, edge_states, query_states], dim=-1))
            aggregated = torch.zeros_like(node_states)
            aggregated.index_add_(0, target_index, messages)
            node_states = self.node_update(aggregated, node_states)

        source_states = node_states[source_index]
        target_states = node_states[target_index]
        query_states = query_state.expand(edge_states.size(0), -1)
        edge_context = torch.cat([source_states + target_states, edge_states, query_states], dim=-1)

        edge_keep_logits = self.keep_head(edge_context).squeeze(-1)
        edge_type_logits = self.type_head(edge_context)
        edge_strengths = self.strength_head(edge_context).squeeze(-1)
        frontier_scores = self.frontier_head(
            torch.cat([node_states, query_state.expand(node_states.size(0), -1)], dim=-1)
        ).squeeze(-1)

        return {
            "edge_keep_logits": edge_keep_logits,
            "edge_type_logits": edge_type_logits,
            "edge_strengths": edge_strengths,
            "frontier_scores": frontier_scores,
        }
