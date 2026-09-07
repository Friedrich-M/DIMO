"""Per-motion latent codes: a plain learnable table, or a Gaussian (VAE-style) table."""

import os
from typing import List, Optional

import torch
from torch import nn

from dimo.losses.regularization import kl_divergence
from dimo.utils.io import resolve_step_path


class LatentCodes(nn.Module):
    """One learnable code per training motion."""

    GROUP_NAMES = ("latent_code",)

    def __init__(self, num_codes: int, dim: int, device: str = "cuda"):
        super().__init__()
        self.dim = dim
        self.codes = nn.Parameter(torch.randn((num_codes, dim), device=device))

    def __len__(self) -> int:
        return self.codes.shape[0]

    def forward(self, index: int) -> torch.Tensor:
        return self.codes[index]

    def mean(self, index: int) -> torch.Tensor:
        return self.codes[index]

    def kl_loss(self, index: int) -> Optional[torch.Tensor]:
        return None

    def numpy(self, index: int):
        return self.codes[index].detach().cpu().numpy()

    # -- editing -------------------------------------------------------------------------------

    def set_codes(self, codes: torch.Tensor):
        """Replace the whole table with ``codes (M, D)`` (used by the latent-space applications)."""
        self.codes = nn.Parameter(codes.detach().clone().to(self.codes.device).requires_grad_(True))

    def interpolate(self, i: int, j: int, alpha: float = 0.5) -> torch.Tensor:
        return (1 - alpha) * self.codes[i] + alpha * self.codes[j]

    def reset_single(self):
        """Replace the table by a single freshly initialised code (for fitting a new motion)."""
        self.set_codes(torch.randn((1, self.dim), device=self.codes.device))

    # -- optimisation --------------------------------------------------------------------------

    def param_groups(self, lr: float) -> List[dict]:
        return [{"params": [self.codes], "lr": lr, "name": "latent_code"}]

    # -- checkpoints ---------------------------------------------------------------------------

    def save(self, directory: str, step: Optional[int] = None):
        torch.save(self.codes.detach().cpu(), resolve_step_path(os.path.join(directory, "latent_codes.pth"), step))

    def load(self, directory: str, step: Optional[int] = None):
        codes = torch.load(resolve_step_path(os.path.join(directory, "latent_codes.pth"), step), map_location="cuda")
        self.set_codes(codes)
        print(f"[INFO] loaded {len(self)} x {self.dim} latent codes")


class GaussianLatentCodes(LatentCodes):
    """Codes parameterised as ``N(mu, exp(log_var))``; samples with the reparameterisation trick
    in training mode and returns ``mu`` in eval mode."""

    GROUP_NAMES = ("latent_code_mu", "latent_code_log_var")

    def __init__(self, num_codes: int, dim: int, device: str = "cuda"):
        nn.Module.__init__(self)
        self.dim = dim
        self.mu = nn.Parameter(torch.zeros((num_codes, dim), device=device))
        self.log_var = nn.Parameter(torch.zeros((num_codes, dim), device=device))

    def __len__(self) -> int:
        return self.mu.shape[0]

    @property
    def codes(self) -> torch.Tensor:  # keeps the parent interface (interpolation etc.) working
        return self.mu

    @staticmethod
    def reparameterize(mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * log_var)
        return mu + torch.randn_like(std) * std

    def forward(self, index: int) -> torch.Tensor:
        if self.training:
            return self.reparameterize(self.mu[index], self.log_var[index])
        return self.mu[index]

    def mean(self, index: int) -> torch.Tensor:
        return self.mu[index]

    def kl_loss(self, index: int) -> torch.Tensor:
        return kl_divergence(self.mu[index], self.log_var[index])

    def numpy(self, index: int):
        return self.forward(index).detach().cpu().numpy()

    def set_codes(self, codes: torch.Tensor, log_var: Optional[torch.Tensor] = None):
        device = self.mu.device
        self.mu = nn.Parameter(codes.detach().clone().to(device).requires_grad_(True))
        if log_var is None:
            log_var = torch.zeros_like(self.mu)
        self.log_var = nn.Parameter(log_var.detach().clone().to(device).requires_grad_(True))

    def interpolate(self, i: int, j: int, alpha: float = 0.5) -> torch.Tensor:
        return (1 - alpha) * self.mu[i] + alpha * self.mu[j]

    def reset_single(self):
        self.set_codes(torch.zeros((1, self.dim), device=self.mu.device))

    def param_groups(self, lr: float) -> List[dict]:
        return [
            {"params": [self.mu], "lr": lr, "name": "latent_code_mu"},
            {"params": [self.log_var], "lr": lr, "name": "latent_code_log_var"},
        ]

    def save(self, directory: str, step: Optional[int] = None):
        torch.save(self.mu.detach().cpu(), resolve_step_path(os.path.join(directory, "mu.pth"), step))
        torch.save(self.log_var.detach().cpu(), resolve_step_path(os.path.join(directory, "log_var.pth"), step))

    def load(self, directory: str, step: Optional[int] = None):
        mu = torch.load(resolve_step_path(os.path.join(directory, "mu.pth"), step), map_location="cuda")
        log_var = torch.load(resolve_step_path(os.path.join(directory, "log_var.pth"), step), map_location="cuda")
        self.set_codes(mu, log_var)
        print(f"[INFO] loaded {len(self)} x {self.dim} Gaussian latent codes")


def build_latent_codes(num_codes: int, dim: int, vae: bool, device: str = "cuda") -> LatentCodes:
    cls = GaussianLatentCodes if vae else LatentCodes
    return cls(num_codes, dim, device=device)
