"""Implementation of the ABMAP regression model."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import torch
from torch import nn

from .modules import InteractionHead, SequenceEncoder, SequenceEncoderConfig


@dataclass
class AbmapModelConfig:
    vocab_size: int
    pad_id: int
    embedding_dim: int
    transformer_layers: int
    transformer_heads: int
    transformer_ff_dim: int
    dropout: float
    max_lengths: Tuple[int, int, int]
    interaction_hidden_dim: int
    use_light_chain: bool = True


class AbmapModel(nn.Module):
    def __init__(self, config: AbmapModelConfig) -> None:
        super().__init__()
        heavy_max, light_max, antigen_max = config.max_lengths

        antibody_encoder_config = SequenceEncoderConfig(
            vocab_size=config.vocab_size,
            embedding_dim=config.embedding_dim,
            nhead=config.transformer_heads,
            num_layers=config.transformer_layers,
            dim_feedforward=config.transformer_ff_dim,
            dropout=config.dropout,
            pad_id=config.pad_id,
            max_length=max(heavy_max, light_max),
        )
        antigen_encoder_config = SequenceEncoderConfig(
            vocab_size=config.vocab_size,
            embedding_dim=config.embedding_dim,
            nhead=config.transformer_heads,
            num_layers=config.transformer_layers,
            dim_feedforward=config.transformer_ff_dim,
            dropout=config.dropout,
            pad_id=config.pad_id,
            max_length=antigen_max,
        )

        self.heavy_encoder = SequenceEncoder(antibody_encoder_config)
        self.light_encoder = SequenceEncoder(antibody_encoder_config) if config.use_light_chain else None
        self.antigen_encoder = SequenceEncoder(antigen_encoder_config)
        interaction_input_dim = config.embedding_dim * (3 if config.use_light_chain else 2)
        interaction_input_dim += config.embedding_dim * (2 if config.use_light_chain else 1)
        self.interaction = InteractionHead(
            input_dim=interaction_input_dim,
            hidden_dim=config.interaction_hidden_dim,
            dropout=config.dropout,
        )
        self.use_light_chain = config.use_light_chain

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        heavy = batch["heavy"].to(torch.long)
        antigen = batch["antigen"].to(torch.long)
        heavy_repr = self.heavy_encoder(heavy)
        antigen_repr = self.antigen_encoder(antigen)

        features = [heavy_repr, antigen_repr, heavy_repr * antigen_repr]

        if self.use_light_chain and "light" in batch:
            light = batch["light"].to(torch.long)
            light_repr = self.light_encoder(light)
            features.extend([light_repr, light_repr * antigen_repr])

        combined = torch.cat(features, dim=-1)
        prediction = self.interaction(combined)
        return prediction
