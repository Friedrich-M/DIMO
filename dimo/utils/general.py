"""Small numeric helpers shared across the code base."""

import os
import random
from typing import NamedTuple, Optional

import numpy as np
import torch


class BasicPointCloud(NamedTuple):
    points: np.ndarray
    colors: np.ndarray
    normals: np.ndarray


def seed_everything(seed: Optional[int]) -> int:
    """Seed python, numpy and torch. A ``None`` seed draws a random one."""
    if seed is None:
        seed = int(np.random.randint(0, 1_000_000))
    seed = int(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    return seed


def inverse_sigmoid(x: torch.Tensor) -> torch.Tensor:
    return torch.log(x / (1 - x))


def get_expon_lr_func(lr_init, lr_final, lr_delay_steps=0, lr_delay_mult=1.0, max_steps=1_000_000):
    """Log-linear learning-rate decay from ``lr_init`` to ``lr_final`` (adapted from Plenoxels)."""

    def helper(step):
        if lr_init == lr_final:
            return lr_init
        if step < 0 or (lr_init == 0.0 and lr_final == 0.0):
            return 0.0
        if lr_delay_steps > 0:
            delay_rate = lr_delay_mult + (1 - lr_delay_mult) * np.sin(
                0.5 * np.pi * np.clip(step / lr_delay_steps, 0, 1)
            )
        else:
            delay_rate = 1.0
        t = np.clip(step / max_steps, 0, 1)
        log_lerp = np.exp(np.log(lr_init) * (1 - t) + np.log(lr_final) * t)
        return delay_rate * log_lerp

    return helper


def sample_ball(num_points: int, radius: float) -> np.ndarray:
    """Uniformly sample ``num_points`` positions inside a ball of the given radius. Returns (N, 3)."""
    phis = np.random.random((num_points,)) * 2 * np.pi
    costheta = np.random.random((num_points,)) * 2 - 1
    thetas = np.arccos(costheta)
    r = radius * np.cbrt(np.random.random((num_points,)))
    x = r * np.sin(thetas) * np.cos(phis)
    y = r * np.sin(thetas) * np.sin(phis)
    z = r * np.cos(thetas)
    return np.stack((x, y, z), axis=1)


def random_point_cloud(num_points: int, radius: float) -> BasicPointCloud:
    """A point cloud uniformly filling a ball, with near-gray random colours."""
    from dimo.utils.sh import SH2RGB

    xyz = sample_ball(num_points, radius)
    shs = np.random.random((num_points, 3)) / 255.0
    return BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_points, 3)))


# ----------------------------------------------------------------------------- rotations


def build_rotation(q: torch.Tensor) -> torch.Tensor:
    """Quaternion(s) ``(..., 4)`` in (w, x, y, z) order to rotation matrices ``(..., 3, 3)``."""
    q = q / torch.linalg.norm(q, dim=-1, keepdim=True)
    r, x, y, z = q.unbind(-1)
    R = torch.stack(
        [
            1 - 2 * (y * y + z * z), 2 * (x * y - r * z), 2 * (x * z + r * y),
            2 * (x * y + r * z), 1 - 2 * (x * x + z * z), 2 * (y * z - r * x),
            2 * (x * z - r * y), 2 * (y * z + r * x), 1 - 2 * (x * x + y * y),
        ],
        dim=-1,
    )
    return R.reshape(q.shape[:-1] + (3, 3))


def quat_mul(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """Hamilton product of two ``(N, 4)`` quaternion batches in (w, x, y, z) order."""
    r1, x1, y1, z1 = q1.unbind(-1)
    r2, x2, y2, z2 = q2.unbind(-1)
    return torch.stack(
        [
            r1 * r2 - x1 * x2 - y1 * y2 - z1 * z2,
            r1 * x2 + x1 * r2 + y1 * z2 - z1 * y2,
            r1 * y2 - x1 * z2 + y1 * r2 + z1 * x2,
            r1 * z2 + x1 * y2 - y1 * x2 + z1 * r2,
        ],
        dim=-1,
    )


def build_scaling_rotation(s: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """``R @ diag(s)`` for scales ``(N, 3)`` and quaternions ``(N, 4)``."""
    L = torch.diag_embed(s)
    return build_rotation(q) @ L


def strip_symmetric(sym: torch.Tensor) -> torch.Tensor:
    """Upper triangle of symmetric ``(N, 3, 3)`` matrices as ``(N, 6)``."""
    return torch.stack(
        [sym[:, 0, 0], sym[:, 0, 1], sym[:, 0, 2], sym[:, 1, 1], sym[:, 1, 2], sym[:, 2, 2]], dim=1
    )
