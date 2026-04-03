import math

import torch
from torch import nn

class FourierPositionalEncoding(nn.Module):
    """
    Implementation inspried by Kornia
    """

    def __init__(self, M: int, target_dim: int, gamma: float = 1.0) -> None:
        super().__init__()
        self.gamma = gamma
        self.Wr = nn.Linear(M, target_dim // 2, bias=False)
        nn.init.normal_(self.Wr.weight.data, mean=0, std=self.gamma**-2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode position vector."""
        projected = self.Wr(x)
        cosines, sines = torch.cos(projected), torch.sin(projected)
        emb = torch.cat([cosines, sines], dim=-1)
        return emb


class FourierEncoding(nn.Module):
    def __init__(
        self,
        target_dim: int,
        max_freq = 1,
        freq_init="uniform",
        include_raw:bool=True,
        learnable_freqs:bool=True,
        device='cuda'
    ) -> None:
        super().__init__()
        self.target_dim = target_dim
        freq_dim = ((target_dim - int(include_raw)) // 2) + 1  # +1 for padding with odd target_dim
        adj_max_freq = max_freq * 2 * torch.pi
        if freq_init == "uniform":
            self.freqs = nn.Parameter(torch.rand(freq_dim, device=device) * adj_max_freq, requires_grad=learnable_freqs)
        elif freq_init == "log":
            self.freqs = nn.Parameter(torch.exp(torch.rand(freq_dim, device=device) * math.log(adj_max_freq)), requires_grad=learnable_freqs)
        else:
            raise ValueError(f"Unsupported freq_init: {freq_init}")
        self.include_raw = include_raw

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode position vector."""
        x = x.unsqueeze(-1)  # (N, M, 1)
        projected = x * self.freqs
        cosines, sines = torch.cos(projected), torch.sin(projected)
        if self.include_raw:
            emb = torch.cat([x, cosines, sines], dim=-1)
        else:
            emb = torch.cat([cosines, sines], dim=-1)
        trunc_emb = emb[:, :self.target_dim]
        return trunc_emb.contiguous()