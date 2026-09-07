"""File-system and media output helpers."""

import os
from typing import Iterable, Optional

import imageio
import numpy as np
import torch
from PIL import Image

VIDEO_FPS = 8


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def write_video(path: str, frames: Iterable[np.ndarray], fps: int = VIDEO_FPS, verbose: bool = True) -> str:
    """Write a list/array of uint8 HxWx3 frames as an mp4."""
    ensure_dir(os.path.dirname(path) or ".")
    frames = [np.ascontiguousarray(f.astype(np.uint8)) for f in frames]
    imageio.mimwrite(path, frames, fps=fps, quality=8, macro_block_size=1)
    if verbose:
        print(f"[INFO] saved video to {os.path.abspath(path)}")
    return path


def write_image(path: str, image: np.ndarray, verbose: bool = False) -> str:
    ensure_dir(os.path.dirname(path) or ".")
    Image.fromarray(image.astype(np.uint8)).save(path)
    if verbose:
        print(f"[INFO] saved image to {os.path.abspath(path)}")
    return path


def write_frames(directory: str, frames: Iterable[np.ndarray], fmt: str = "{:02d}.png") -> str:
    ensure_dir(directory)
    for i, frame in enumerate(frames):
        write_image(os.path.join(directory, fmt.format(i)), frame)
    return directory


def tensor_to_uint8(image: torch.Tensor) -> np.ndarray:
    """``(3, H, W)`` float image in [0, 1] to an ``(H, W, 3)`` uint8 array."""
    return (image.detach().clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)


def side_by_side(*images: np.ndarray) -> np.ndarray:
    return np.concatenate(images, axis=1)


def resolve_step_path(path: str, step: Optional[int]) -> str:
    """``foo/bar.ply`` + step 500 -> ``foo/bar_500.ply``; unchanged when step is falsy."""
    if not step:
        return path
    root, ext = os.path.splitext(path)
    return f"{root}_{step}{ext}"
