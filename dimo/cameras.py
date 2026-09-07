"""Orbit cameras and the camera object consumed by the Gaussian rasterizers."""

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch


def _safe_normalize(x: np.ndarray, eps: float = 1e-20) -> np.ndarray:
    return x / np.sqrt(np.maximum(np.sum(x * x, axis=-1, keepdims=True), eps))


def look_at(campos: np.ndarray, target: np.ndarray, opengl: bool = True) -> np.ndarray:
    """Rotation matrix (3, 3) of a camera at ``campos`` looking at ``target`` with +y up."""
    up = np.array([0, 1, 0], dtype=np.float32)
    if not opengl:  # camera forward aligns with -z
        forward = _safe_normalize(target - campos)
        right = _safe_normalize(np.cross(forward, up))
        up = _safe_normalize(np.cross(right, forward))
    else:  # camera forward aligns with +z
        forward = _safe_normalize(campos - target)
        right = _safe_normalize(np.cross(up, forward))
        up = _safe_normalize(np.cross(forward, right))
    return np.stack([right, up, forward], axis=1)


def orbit_camera(elevation: float, azimuth: float, radius: float = 1.0, is_degree: bool = True,
                 target: Optional[np.ndarray] = None, opengl: bool = True) -> np.ndarray:
    """Camera-to-world (4, 4) pose on an orbit around ``target``.

    ``elevation`` in (-90, 90) goes from +y to -y; ``azimuth`` goes from +z towards +x.
    """
    if is_degree:
        elevation = np.deg2rad(elevation)
        azimuth = np.deg2rad(azimuth)
    x = radius * np.cos(elevation) * np.sin(azimuth)
    y = -radius * np.sin(elevation)
    z = radius * np.cos(elevation) * np.cos(azimuth)
    if target is None:
        target = np.zeros([3], dtype=np.float32)
    campos = np.array([x, y, z]) + target
    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = look_at(campos, target, opengl)
    T[:3, 3] = campos
    return T


def projection_matrix(znear: float, zfar: float, fovx: float, fovy: float) -> torch.Tensor:
    P = torch.zeros(4, 4)
    P[0, 0] = 1 / math.tan(fovx / 2)
    P[1, 1] = 1 / math.tan(fovy / 2)
    P[3, 2] = 1.0
    P[2, 2] = zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P


class Camera:
    """A single view for the rasterizer, built from a NeRF-convention camera-to-world pose."""

    def __init__(self, c2w: np.ndarray, width: int, height: int, fovy: float, fovx: float,
                 znear: float, zfar: float, device: str = "cuda"):
        self.image_width = int(width)
        self.image_height = int(height)
        self.FoVy = fovy
        self.FoVx = fovx
        self.znear = znear
        self.zfar = zfar

        w2c = np.linalg.inv(c2w)
        w2c[1:3, :3] *= -1  # flip y and z axes (OpenGL -> rasterizer convention)
        w2c[:3, 3] *= -1

        self.world_view_transform = torch.tensor(w2c).transpose(0, 1).to(device)
        self.projection_matrix = projection_matrix(znear, zfar, fovx, fovy).transpose(0, 1).to(device)
        self.full_proj_transform = self.world_view_transform @ self.projection_matrix
        self.camera_center = -torch.tensor(c2w[:3, 3]).to(device)

    def project(self, points: torch.Tensor) -> torch.Tensor:
        """World points ``(N, 3)`` to pixel coordinates ``(N, 2)`` on this view."""
        hom = torch.cat([points, torch.ones_like(points[..., :1])], dim=-1)
        clip = hom @ self.full_proj_transform
        ndc = clip[..., :2] / clip[..., -1:]
        size = torch.tensor([self.image_width, self.image_height], device=points.device, dtype=ndc.dtype)
        return (ndc + 1) / 2 * size


@dataclass
class OrbitIntrinsics:
    """Intrinsics shared by all orbit views of a scene."""

    width: int
    height: int
    radius: float
    fovy_deg: float
    near: float = 0.01
    far: float = 100.0

    @property
    def fovy(self) -> float:
        return float(np.deg2rad(self.fovy_deg))

    @property
    def fovx(self) -> float:
        return float(2 * np.arctan(np.tan(self.fovy / 2) * self.width / self.height))

    def view(self, elevation: float, azimuth: float, resolution: Optional[int] = None) -> Camera:
        """Camera on the orbit; ``resolution`` overrides the image size (square) if given."""
        pose = orbit_camera(elevation, azimuth, self.radius)
        if resolution is None:
            width, height = self.width, self.height
        else:
            width = height = resolution
        return Camera(pose, width, height, self.fovy, self.fovx, self.near, self.far)
