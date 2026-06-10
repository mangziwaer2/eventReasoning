from __future__ import annotations

import torch
from torch import nn


class ContinuousTimeEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(input_dim, hidden_dim)
        self.periodic = nn.Linear(input_dim, hidden_dim)
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, time_features: torch.Tensor) -> torch.Tensor:
        linear_part = self.linear(time_features)
        periodic_part = torch.sin(self.periodic(time_features))
        return self.proj(torch.cat([linear_part, periodic_part], dim=-1))


class TemporalRelationalEdgeRefiner(nn.Module):
    def __init__(
        self,
        node_dim: int = 10,
        edge_dim: int = 13,
        query_dim: int = 6,
        hidden_dim: int = 128,
        num_message_passing_steps: int = 3,
        num_relations: int = 4,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_message_passing_steps = num_message_passing_steps
        self.num_relations = num_relations

        self.edge_scalar_dim = edge_dim - 1
        relation_embed_dim = hidden_dim // 4
        self.time_feature_indices = [2, 7, 8, 9, 10, 11]

        self.node_encoder = nn.Sequential(
            nn.Linear(node_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.edge_scalar_encoder = nn.Sequential(
            nn.Linear(self.edge_scalar_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.query_encoder = nn.Sequential(
            nn.Linear(query_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.relation_embedding = nn.Embedding(num_relations, relation_embed_dim)
        self.time_encoder = ContinuousTimeEncoder(input_dim=len(self.time_feature_indices), hidden_dim=hidden_dim)
        self.edge_context_proj = nn.Sequential(
            nn.Linear(hidden_dim * 2 + relation_embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.forward_relation_linears = nn.ModuleList(
            [nn.Linear(hidden_dim, hidden_dim, bias=False) for _ in range(num_relations)]
        )
        self.inverse_relation_linears = nn.ModuleList(
            [nn.Linear(hidden_dim, hidden_dim, bias=False) for _ in range(num_relations)]
        )
        self.self_loop_linear = nn.Linear(hidden_dim, hidden_dim, bias=False)

        message_input_dim = hidden_dim * 4
        attention_input_dim = hidden_dim * 5
        self.forward_message_mlp = nn.Sequential(
            nn.Linear(message_input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.reverse_message_mlp = nn.Sequential(
            nn.Linear(message_input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.forward_attention = nn.Sequential(
            nn.Linear(attention_input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        self.reverse_attention = nn.Sequential(
            nn.Linear(attention_input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        self.message_residual = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.node_update = nn.GRUCell(hidden_dim, hidden_dim)
        self.node_norm = nn.LayerNorm(hidden_dim)

        edge_head_input_dim = hidden_dim * 7
        self.keep_head = nn.Sequential(
            nn.Linear(edge_head_input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        self.type_head = nn.Sequential(
            nn.Linear(edge_head_input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_relations),
        )
        self.strength_head = nn.Sequential(
            nn.Linear(edge_head_input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )
        self.frontier_head = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def _split_edge_features(
        self,
        edge_features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        relation_ids = edge_features[:, 1].round().clamp(min=0, max=self.num_relations - 1).long()
        edge_scalar_features = torch.cat([edge_features[:, :1], edge_features[:, 2:]], dim=-1)
        time_features = edge_features[:, self.time_feature_indices]
        reverse_time_features = torch.stack(
            [
                time_features[:, 0],
                time_features[:, 2],
                time_features[:, 1],
                -time_features[:, 3],
                time_features[:, 4],
                -time_features[:, 5],
            ],
            dim=-1,
        )
        return edge_scalar_features, relation_ids, time_features, reverse_time_features

    def _apply_relation_transforms(
        self,
        states: torch.Tensor,
        relation_ids: torch.Tensor,
        linears: nn.ModuleList,
    ) -> torch.Tensor:
        transformed = torch.zeros_like(states)
        for relation_id, linear in enumerate(linears):
            mask = relation_ids == relation_id
            if torch.any(mask):
                transformed[mask] = linear(states[mask])
        return transformed

    def _segment_softmax(
        self,
        logits: torch.Tensor,
        target_index: torch.Tensor,
        num_nodes: int,
    ) -> torch.Tensor:
        weights = torch.zeros_like(logits)
        for node_id in range(num_nodes):
            mask = target_index == node_id
            if torch.any(mask):
                weights[mask] = torch.softmax(logits[mask], dim=0)
        return weights

    def _message_pass(
        self,
        node_states: torch.Tensor,
        edge_states: torch.Tensor,
        time_states: torch.Tensor,
        reverse_time_states: torch.Tensor,
        relation_ids: torch.Tensor,
        source_index: torch.Tensor,
        target_index: torch.Tensor,
        query_state: torch.Tensor,
    ) -> torch.Tensor:
        source_states = node_states[source_index]
        target_states = node_states[target_index]
        query_states = query_state.expand(edge_states.size(0), -1)

        transformed_sources = self._apply_relation_transforms(
            source_states,
            relation_ids,
            self.forward_relation_linears,
        )
        forward_messages = self.forward_message_mlp(
            torch.cat([transformed_sources, target_states, edge_states, query_states], dim=-1)
        )
        forward_attention_logits = self.forward_attention(
            torch.cat([transformed_sources, target_states, edge_states, time_states, query_states], dim=-1)
        ).squeeze(-1)
        forward_weights = self._segment_softmax(forward_attention_logits, target_index, node_states.size(0))
        forward_messages = forward_messages * forward_weights.unsqueeze(-1)

        transformed_targets = self._apply_relation_transforms(
            target_states,
            relation_ids,
            self.inverse_relation_linears,
        )
        reverse_messages = self.reverse_message_mlp(
            torch.cat([transformed_targets, source_states, edge_states, query_states], dim=-1)
        )
        reverse_attention_logits = self.reverse_attention(
            torch.cat([transformed_targets, source_states, edge_states, reverse_time_states, query_states], dim=-1)
        ).squeeze(-1)
        reverse_weights = self._segment_softmax(reverse_attention_logits, source_index, node_states.size(0))
        reverse_messages = reverse_messages * reverse_weights.unsqueeze(-1)

        aggregated = self.self_loop_linear(node_states)
        aggregated.index_add_(0, target_index, forward_messages)
        aggregated.index_add_(0, source_index, reverse_messages)
        aggregated = self.message_residual(torch.cat([aggregated, node_states], dim=-1))
        updated = self.node_update(aggregated, node_states)
        return self.node_norm(updated + node_states)

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
                torch.cat(
                    [
                        node_states,
                        query_state.expand(node_states.size(0), -1),
                        node_states * query_state.expand(node_states.size(0), -1),
                    ],
                    dim=-1,
                )
            ).squeeze(-1)
            return {
                "edge_keep_logits": torch.empty(0, device=node_features.device),
                "edge_type_logits": torch.empty((0, self.num_relations), device=node_features.device),
                "edge_strengths": torch.empty(0, device=node_features.device),
                "frontier_scores": frontier_scores,
            }

        edge_scalar_features, relation_ids, time_features, reverse_time_features = self._split_edge_features(edge_features)
        edge_scalar_states = self.edge_scalar_encoder(edge_scalar_features)
        relation_states = self.relation_embedding(relation_ids)
        time_states = self.time_encoder(time_features)
        reverse_time_states = self.time_encoder(reverse_time_features)
        edge_states = self.edge_context_proj(
            torch.cat([edge_scalar_states, time_states, relation_states], dim=-1)
        )

        source_index = edge_index[:, 0]
        target_index = edge_index[:, 1]
        for _ in range(self.num_message_passing_steps):
            node_states = self._message_pass(
                node_states=node_states,
                edge_states=edge_states,
                time_states=time_states,
                reverse_time_states=reverse_time_states,
                relation_ids=relation_ids,
                source_index=source_index,
                target_index=target_index,
                query_state=query_state,
            )

        source_states = node_states[source_index]
        target_states = node_states[target_index]
        query_states = query_state.expand(edge_states.size(0), -1)
        edge_context = torch.cat(
            [
                source_states,
                target_states,
                torch.abs(source_states - target_states),
                source_states * target_states,
                edge_states,
                time_states,
                query_states,
            ],
            dim=-1,
        )

        edge_keep_logits = self.keep_head(edge_context).squeeze(-1)
        edge_type_logits = self.type_head(edge_context)
        edge_strengths = self.strength_head(edge_context).squeeze(-1)
        frontier_scores = self.frontier_head(
            torch.cat(
                [
                    node_states,
                    query_state.expand(node_states.size(0), -1),
                    node_states * query_state.expand(node_states.size(0), -1),
                ],
                dim=-1,
            )
        ).squeeze(-1)

        return {
            "edge_keep_logits": edge_keep_logits,
            "edge_type_logits": edge_type_logits,
            "edge_strengths": edge_strengths,
            "frontier_scores": frontier_scores,
        }
