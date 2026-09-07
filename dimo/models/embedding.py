"""NeRF-style sinusoidal positional encoding."""

import torch
from torch import nn


class PositionalEncoding(nn.Module):
    """Maps ``(..., D)`` inputs to ``[sin(f_0 x), cos(f_0 x), sin(f_1 x), ...]`` with ``f_k = 2^k``."""

    def __init__(self, num_freqs: int, input_dims: int, include_input: bool = False):
        super().__init__()
        self.include_input = include_input
        self.input_dims = input_dims
        self.register_buffer("freq_bands", 2.0 ** torch.linspace(0.0, num_freqs - 1, steps=num_freqs), persistent=False)
        self.out_dim = input_dims * (2 * num_freqs + int(include_input))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = [x] if self.include_input else []
        for freq in self.freq_bands:
            outputs.append(torch.sin(x * freq))
            outputs.append(torch.cos(x * freq))
        return torch.cat(outputs, dim=-1)
