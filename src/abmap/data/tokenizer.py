"""Tokenization utilities for amino acid sequences."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

import torch

AMINO_ACID_ALPHABET = "ACDEFGHIKLMNPQRSTVWY"
PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"


@dataclass(frozen=True)
class TokenizerConfig:
    alphabet: str = AMINO_ACID_ALPHABET
    pad_token: str = PAD_TOKEN
    unk_token: str = UNK_TOKEN


class AminoAcidTokenizer:
    """Simple amino acid tokenizer with padding utilities."""

    def __init__(self, config: TokenizerConfig | None = None) -> None:
        self.config = config or TokenizerConfig()
        self._vocab = {token: index for index, token in enumerate(self.config.alphabet, start=2)}
        self._vocab[self.config.pad_token] = 0
        self._vocab[self.config.unk_token] = 1
        self._inverse_vocab = {index: token for token, index in self._vocab.items()}

    @property
    def pad_id(self) -> int:
        return self._vocab[self.config.pad_token]

    @property
    def vocab_size(self) -> int:
        return len(self._vocab)

    def encode(self, sequence: str) -> List[int]:
        sequence = sequence.upper()
        return [self._vocab.get(symbol, self._vocab[self.config.unk_token]) for symbol in sequence]

    def batch_encode(self, sequences: Iterable[str], *, max_length: int | None = None) -> torch.Tensor:
        encoded = [self.encode(sequence) for sequence in sequences]
        if max_length is None:
            max_length = max(len(item) for item in encoded)
        padded = torch.full((len(encoded), max_length), self.pad_id, dtype=torch.long)
        for row, item in enumerate(encoded):
            length = min(len(item), max_length)
            padded[row, :length] = torch.tensor(item[:length], dtype=torch.long)
        return padded

    def decode(self, indices: Iterable[int]) -> str:
        return "".join(self._inverse_vocab.get(index, self.config.unk_token) for index in indices)
