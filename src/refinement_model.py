from __future__ import annotations

import torch
from torch import nn


class TemporalRelationalEdgeRefiner(nn.Module):
    def __init__(
        self,
        node_dim: int = 6,
        edge_dim: int = 7,
        query_dim: int = 4,
        hidden_dim: int = 96,
        num_message_passing_steps: int = 3,
        num_relations: int = 4,
        num_time_buckets: int = 3,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_message_passing_steps = num_message_passing_steps
        self.num_relations = num_relations
        self.num_time_buckets = num_time_buckets

        # edge_dim includes one scalar slot used for the coarse relation id
        self.edge_scalar_dim = edge_dim - 1
        relation_embed_dim = hidden_dim // 4
        time_embed_dim = hidden_dim // 8

        self.node_encoder = nn.Sequential(
            nn.Linear(node_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.edge_scalar_encoder = nn.Sequential(
            nn.Linear(self.edge_scalar_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.query_encoder = nn.Sequential(
            nn.Linear(query_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.relation_embedding = nn.Embedding(num_relations, relation_embed_dim)
        self.time_bucket_embedding = nn.Embedding(num_time_buckets, time_embed_dim)
        self.edge_context_proj = nn.Sequential(
            nn.Linear(hidden_dim + relation_embed_dim + time_embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        message_input_dim = hidden_dim * 4
        self.message_mlp = nn.Sequential(
            nn.Linear(message_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.message_gate = nn.Sequential(
            nn.Linear(message_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )
        self.node_update = nn.GRUCell(hidden_dim, hidden_dim)
        self.node_norm = nn.LayerNorm(hidden_dim)

        edge_head_input_dim = hidden_dim * 4
        self.keep_head = nn.Sequential(
            nn.Linear(edge_head_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.type_head = nn.Sequential(
            nn.Linear(edge_head_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_relations),
        )
        self.strength_head = nn.Sequential(
            nn.Linear(edge_head_input_dim, hidden_dim),
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

    def _split_edge_features(self, edge_features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        relation_ids = edge_features[:, 1].round().clamp(min=0, max=self.num_relations - 1).long()
        temporal_scores = edge_features[:, 2]
        # Current coarse graphs use temporal scores around 0.9 / 0.55 / 0.3.
        # Bucket them into strong-local, chronological, and weak/unknown time evidence.
        time_buckets = torch.bucketize(
            temporal_scores,
            boundaries=torch.tensor([0.45, 0.75], device=edge_features.device),
        ).long()
        edge_scalar_features = torch.cat([edge_features[:, :1], edge_features[:, 2:]], dim=-1)
        return edge_scalar_features, relation_ids, time_buckets

    def forward(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_features: torch.Tensor,
        query_features: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        node_states = self.node_encoder(node_features)
        query_state = self.query_encoder(query_features.unsqueeze(0)).squeeze(0)

        if edge_index.numel() == 0:
            frontier_scores = self.frontier_head(
                torch.cat([node_states, query_state.expand(node_states.size(0), -1)], dim=-1)
            ).squeeze(-1)
            return {
                "edge_keep_logits": torch.empty(0, device=node_features.device),
                "edge_type_logits": torch.empty((0, self.num_relations), device=node_features.device),
                "edge_strengths": torch.empty(0, device=node_features.device),
                "frontier_scores": frontier_scores,
            }

        edge_scalar_features, relation_ids, time_buckets = self._split_edge_features(edge_features)
        edge_scalar_states = self.edge_scalar_encoder(edge_scalar_features)
        relation_states = self.relation_embedding(relation_ids)
        time_states = self.time_bucket_embedding(time_buckets)
        edge_states = self.edge_context_proj(
            torch.cat([edge_scalar_states, relation_states, time_states], dim=-1)
        )

        source_index = edge_index[:, 0]
        target_index = edge_index[:, 1]

        for _ in range(self.num_message_passing_steps):
            source_states = node_states[source_index]
            target_states = node_states[target_index]
            query_states = query_state.expand(edge_states.size(0), -1)

            message_inputs = torch.cat(
                [source_states, target_states, edge_states, query_states],
                dim=-1,
            )
            messages = self.message_mlp(message_inputs)
            gates = self.message_gate(message_inputs)
            messages = messages * gates

            aggregated = torch.zeros_like(node_states)
            aggregated.index_add_(0, target_index, messages)
            updated = self.node_update(aggregated, node_states)
            node_states = self.node_norm(updated)

        source_states = node_states[source_index]
        target_states = node_states[target_index]
        query_states = query_state.expand(edge_states.size(0), -1)
        edge_context = torch.cat(
            [source_states, target_states, edge_states, query_states],
            dim=-1,
        )

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
