"""Neural network building blocks used by the ABMAP model."""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class SequenceEncoderConfig:
    vocab_size: int
    embedding_dim: int
    nhead: int
    num_layers: int
    dim_feedforward: int
    dropout: float
    pad_id: int
    max_length: int


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float, max_len: int) -> None:
        super().__init__()
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class SequenceEncoder(nn.Module):
    """Transformer-based encoder for amino acid sequences."""

    def __init__(self, config: SequenceEncoderConfig) -> None:
        super().__init__()
        self.embedding = nn.Embedding(config.vocab_size, config.embedding_dim, padding_idx=config.pad_id)
        self.positional = PositionalEncoding(config.embedding_dim, config.dropout, config.max_length)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.embedding_dim,
            nhead=config.nhead,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            batch_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=config.num_layers)
        self.norm = nn.LayerNorm(config.embedding_dim)
        self.pad_id = config.pad_id

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        mask = tokens == self.pad_id
        embedded = self.embedding(tokens)
        encoded = self.positional(embedded)
        encoded = self.transformer(encoded, src_key_padding_mask=mask)
        encoded = self.norm(encoded)
        mask_float = (~mask).unsqueeze(-1)
        summed = (encoded * mask_float).sum(dim=1)
        lengths = mask_float.sum(dim=1).clamp(min=1.0)
        pooled = summed / lengths
        return pooled


class InteractionHead(nn.Module):
    """Final prediction head that fuses antibody and antigen representations."""

    def __init__(self, input_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features).squeeze(-1)
