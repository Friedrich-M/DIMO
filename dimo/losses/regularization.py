"""Geometry regularisers: depth / normal smoothness, latent KL and key-point anchoring."""

import torch
from pytorch3d.ops import knn_points


def _image_gradients(rgb: torch.Tensor):
    """Mean absolute colour gradient along x and y for ``(B, H, W, 3)`` images."""
    grad_x = torch.mean(torch.abs(rgb[..., :, :-1, :] - rgb[..., :, 1:, :]), -1, keepdim=True)
    grad_y = torch.mean(torch.abs(rgb[..., :-1, :, :] - rgb[..., 1:, :, :]), -1, keepdim=True)
    return grad_x, grad_y


def edge_aware_depth_smoothness_loss(depth: torch.Tensor, rgb: torch.Tensor) -> torch.Tensor:
    """Penalise depth gradients where the image is smooth. ``depth (B, H, W, 1)``, ``rgb (B, H, W, 3)``."""
    grad_depth_x = torch.abs(depth[..., :, :-1, :] - depth[..., :, 1:, :])
    grad_depth_y = torch.abs(depth[..., :-1, :, :] - depth[..., 1:, :, :])
    grad_img_x, grad_img_y = _image_gradients(rgb)
    grad_depth_x = grad_depth_x * torch.exp(-grad_img_x)
    grad_depth_y = grad_depth_y * torch.exp(-grad_img_y)
    return grad_depth_x.mean() + grad_depth_y.mean()


def bilateral_normal_smoothness_loss(normal: torch.Tensor, rgb: torch.Tensor) -> torch.Tensor:
    """``L = e^(-3 |grad I|) * sqrt(1 + |grad n|^2)``. ``normal (B, H, W, 3)``, ``rgb (B, H, W, 3)``."""
    grad_normal_x = torch.abs(normal[..., :, :-1, :] - normal[..., :, 1:, :])
    grad_normal_y = torch.abs(normal[..., :-1, :, :] - normal[..., 1:, :, :])
    grad_img_x, grad_img_y = _image_gradients(rgb)
    grad_normal_x = grad_normal_x * torch.exp(-3 * grad_img_x)
    grad_normal_y = grad_normal_y * torch.exp(-3 * grad_img_y)
    return torch.sqrt(1 + grad_normal_x**2).mean() + torch.sqrt(1 + grad_normal_y**2).mean()


def kl_divergence(mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
    """KL(N(mu, var) || N(0, I)), summed over dimensions."""
    return -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())


def chamfer_forward(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """One-directional Chamfer distance: sum over ``source (N, 3)`` of the squared distance to ``target (M, 3)``."""
    dists = knn_points(source[None], target[None], K=1).dists  # (1, N, 1)
    return dists.sum()
