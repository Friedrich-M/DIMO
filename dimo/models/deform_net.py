"""Latent-conditioned motion decoder.

Given a canonical position, a time stamp and a per-motion latent code, the network predicts a
translation and a rotation (quaternion) for that position.
"""

from typing import List, Tuple, Union

import torch
import torch.nn.functional as F
from torch import nn

from dimo.models.embedding import PositionalEncoding


def _init_xavier(m: nn.Module):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight, gain=1)


def _init_zero(m: nn.Module):
    if isinstance(m, nn.Linear):
        nn.init.constant_(m.weight, 0)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


def _init_identity_quaternion(m: nn.Module):
    if isinstance(m, nn.Linear):
        nn.init.constant_(m.weight, 0)
        if m.bias is not None:
            m.bias.data = torch.tensor([1.0, 0.0, 0.0, 0.0])


class DeformNet(nn.Module):
    """MLP ``(pos-enc(x), pos-enc(t), z) -> (dx, dq)``.

    Sub-module names (``deformnet``, ``pts_layers``, ``rot_layers``) are part of the checkpoint
    format and must not be renamed.
    """

    def __init__(self, depth: int = 8, width: int = 256, skips: Tuple[int, ...] = (4,),
                 latent_dim: int = 32, pts_freqs: int = 10, time_freqs: int = 6):
        super().__init__()
        self.pts_encoder = PositionalEncoding(pts_freqs, 3)
        self.time_encoder = PositionalEncoding(time_freqs, 1)
        self.input_ch = self.pts_encoder.out_dim + self.time_encoder.out_dim + latent_dim
        self.skips = set(skips)

        layers = [nn.Linear(self.input_ch, width)]
        for i in range(depth - 1):
            layers.append(nn.Linear(width + self.input_ch, width) if i in self.skips else nn.Linear(width, width))
        self.deformnet = nn.ModuleList(layers)
        self.pts_layers = nn.Sequential(nn.Linear(width, width), nn.ReLU(), nn.Linear(width, 3))
        self.rot_layers = nn.Sequential(nn.Linear(width, width), nn.ReLU(), nn.Linear(width, 4))

        self.deformnet.apply(_init_xavier)
        self.pts_layers.apply(_init_xavier)
        self.rot_layers.apply(_init_xavier)
        self.pts_layers[-1].apply(_init_zero)  # start from zero translation
        self.rot_layers[-1].apply(_init_identity_quaternion)  # start from identity rotation

    def forward(self, pts: torch.Tensor, t: Union[float, torch.Tensor], latent: torch.Tensor):
        """
        Args:
            pts: ``(N, 3)`` or ``(B, N, 3)`` canonical positions.
            t: a scalar time in [0, 1), or a ``(B, N, 1)`` tensor of times.
            latent: ``(D,)`` latent code, or ``(B, N, D)`` codes.
        Returns:
            translation ``(N, 3)`` / ``(B, N, 3)`` and quaternion ``(N, 4)`` / ``(B, N, 4)``.
        """
        unbatched = pts.dim() == 2
        if unbatched:
            pts = pts[None]
        B, N = pts.shape[:2]
        device = pts.device

        if torch.is_tensor(t):
            times = t.to(device)
            if times.shape[0] != B:
                pts = pts.expand(times.shape[0], N, 3)
                B = pts.shape[0]
        else:
            times = torch.full((B, N, 1), float(t), device=device)
        if latent.dim() == 1:
            latent = latent[None, None, :].expand(B, N, -1)

        inputs = torch.cat([self.pts_encoder(pts), self.time_encoder(times), latent], dim=-1)
        h = inputs
        for i, layer in enumerate(self.deformnet):
            h = F.relu(layer(h))
            if i in self.skips:
                h = torch.cat([inputs, h], dim=-1)
        translation, rotation = self.pts_layers(h), self.rot_layers(h)
        if unbatched:
            translation, rotation = translation[0], rotation[0]
        return translation, rotation

    def translation_parameters(self) -> List[nn.Parameter]:
        """Trunk + translation head parameters."""
        return [p for n, p in self.named_parameters() if not n.startswith("rot_layers")]

    def rotation_parameters(self) -> List[nn.Parameter]:
        return list(self.rot_layers.parameters())
