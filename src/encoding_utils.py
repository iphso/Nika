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
