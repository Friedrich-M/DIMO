"""Image reconstruction losses."""

from math import exp

import torch
import torch.nn.functional as F


def l1_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return (pred - target).abs().mean()


def _gaussian(window_size: int, sigma: float) -> torch.Tensor:
    gauss = torch.tensor([exp(-((x - window_size // 2) ** 2) / (2 * sigma**2)) for x in range(window_size)])
    return gauss / gauss.sum()


def _create_window(window_size: int, channel: int) -> torch.Tensor:
    window_1d = _gaussian(window_size, 1.5).unsqueeze(1)
    window_2d = window_1d.mm(window_1d.t()).float().unsqueeze(0).unsqueeze(0)
    return window_2d.expand(channel, 1, window_size, window_size).contiguous()


def ssim(img1: torch.Tensor, img2: torch.Tensor, window_size: int = 11, size_average: bool = True) -> torch.Tensor:
    """Structural similarity of ``(B, C, H, W)`` images with an 11x11 Gaussian window."""
    channel = img1.size(-3)
    window = _create_window(window_size, channel).to(img1.device, img1.dtype)

    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)
    mu1_sq, mu2_sq, mu1_mu2 = mu1.pow(2), mu2.pow(2), mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1, C2 = 0.01**2, 0.03**2
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    if size_average:
        return ssim_map.mean()
    return ssim_map.mean(1).mean(1).mean(1)
