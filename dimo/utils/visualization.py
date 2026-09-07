"""Key-point trajectory drawing (2D overlays and 3D matplotlib plots)."""

from typing import List, Optional

import cv2
import matplotlib
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure


def trajectory_colors(num_points: int, cmap: str = "hsv") -> np.ndarray:
    """One RGB colour in [0, 1] per point, spread over ``cmap``. Returns ``(N, 3)``."""
    color_map = matplotlib.colormaps.get_cmap(cmap)
    return np.array([color_map(i / max(1, float(num_points - 1)))[:3] for i in range(num_points)])


def draw_trajectories(traj_2d: np.ndarray, height: int, width: int, colors: np.ndarray,
                      end_frame: Optional[int] = None, trace_length: Optional[int] = None,
                      thickness: int = 2, end_points: bool = False) -> np.ndarray:
    """Polylines of 2D trajectories ``(N, T, 2)`` as an RGBA uint8 image (alpha = coverage).

    ``end_frame`` limits the drawing to frames ``<= end_frame``; ``trace_length`` keeps only the
    last frames before it. ``end_points`` also draws a dot at the current position.
    """
    rgb = np.zeros([height, width, 3])
    alpha = np.zeros([height, width, 3])
    T = traj_2d.shape[1]
    end = T - 1 if end_frame is None else end_frame
    start = 0 if trace_length is None else max(0, end - trace_length)
    for i in range(traj_2d.shape[0]):
        pts = traj_2d[i, start:end + 1].astype(np.int32)
        color = [float(c) for c in colors[i]]
        rgb = cv2.polylines(rgb, [pts], isClosed=False, color=color, thickness=thickness)
        alpha = cv2.polylines(alpha, [pts], isClosed=False, color=[1, 1, 1], thickness=thickness)
        if end_points:
            center = tuple(int(v) for v in traj_2d[i, end])
            rgb = cv2.circle(rgb, center, 2, color, -1, lineType=cv2.LINE_AA)
            alpha = cv2.circle(alpha, center, 2, [1, 1, 1], -1, lineType=cv2.LINE_AA)
    return (np.concatenate([rgb, alpha[..., :1]], axis=-1) * 255).astype(np.uint8)


def draw_trajectory_frames(traj_2d: np.ndarray, height: int, width: int, colors: np.ndarray,
                           trace_length: int = 5, thickness: int = 2) -> List[np.ndarray]:
    """One RGBA overlay per frame showing the recent trace of every trajectory."""
    T = traj_2d.shape[1]
    return [draw_trajectories(traj_2d, height, width, colors, end_frame=t, trace_length=trace_length,
                              thickness=thickness, end_points=True) for t in range(T)]


def overlay(frame: np.ndarray, rgba: np.ndarray, grayscale: bool = True) -> np.ndarray:
    """Composite an RGBA overlay onto an RGB uint8 frame (optionally converted to gray first)."""
    base = frame[..., :3].astype(np.uint8)
    if grayscale:
        base = cv2.cvtColor(base, cv2.COLOR_RGB2GRAY)[..., None]
    mask = rgba[..., -1:].astype(np.float32) / 255
    out = base * (1 - mask) + rgba[..., :3] * mask
    return out.astype(np.uint8)


def _setup_3d_axes(fig: Figure, points: np.ndarray):
    x_min, x_max = np.min(points[..., 0]), np.max(points[..., 0])
    y_min, y_max = np.min(points[..., 2]), np.max(points[..., 2])
    z_min, z_max = np.min(points[..., 1]), np.max(points[..., 1])
    interval = np.max([x_max - x_min, y_max - y_min, z_max - z_min])
    x_min = (x_min + x_max) / 2 - interval / 2
    y_min = (y_min + y_max) / 2 - interval / 2
    z_min = (z_min + z_max) / 2 - interval / 2

    ax = fig.add_subplot(111, projection="3d", computed_zorder=False)
    ax.set_xlim([x_min, x_min + interval])
    ax.set_ylim([y_min, y_min + interval])
    ax.set_zlim([z_min, z_min + interval])
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])
    ax.invert_yaxis()
    ax.view_init()
    return ax


def _render_figure(fig: Figure, canvas: FigureCanvasAgg) -> np.ndarray:
    fig.subplots_adjust(left=-0.05, right=1.05, top=1.05, bottom=-0.05)
    fig.canvas.draw()
    return np.array(canvas.buffer_rgba())[..., :3]


def plot_3d_tracks(points: np.ndarray, trace_length: int = 16, cmap: str = "hsv") -> np.ndarray:
    """Animate 3D trajectories ``(T, N, 3)``; returns ``(T, H, W, 3)`` uint8 frames."""
    num_frames, num_points = points.shape[:2]
    colors = trajectory_colors(num_points, cmap)
    frames = []
    for t in range(num_frames):
        fig = Figure(figsize=(6.4, 4.8))
        canvas = FigureCanvasAgg(fig)
        ax = _setup_3d_axes(fig, points)
        for i in range(num_points):
            line = points[max(0, t - trace_length): t + 1, i]
            ax.plot(xs=line[:, 0], ys=line[:, 2], zs=line[:, 1], color=colors[i], linewidth=1)
            end = points[t, i]
            ax.scatter(xs=end[0], ys=end[2], zs=end[1], color=colors[i], s=3)
        frames.append(_render_figure(fig, canvas))
    return np.stack(frames)


def plot_3d_tracks_image(points: np.ndarray, cmap: str = "hsv") -> np.ndarray:
    """Full 3D trajectories ``(T, N, 3)`` in one ``(H, W, 3)`` uint8 image."""
    num_points = points.shape[1]
    colors = trajectory_colors(num_points, cmap)
    fig = Figure(figsize=(6.4, 4.8))
    canvas = FigureCanvasAgg(fig)
    ax = _setup_3d_axes(fig, points)
    for i in range(num_points):
        line = points[:, i]
        ax.plot(xs=line[:, 0], ys=line[:, 2], zs=line[:, 1], color=colors[i], linewidth=1)
    return _render_figure(fig, canvas)


def downscale_video(video: np.ndarray, size: int) -> np.ndarray:
    """Resize a ``(T, H, W, C)`` video so its longer side is ``size`` (no-op if already smaller).

    Used before a video is retained for the motion grid: keeping 51 motions at the render resolution
    costs several GB of host memory, and a grid of 800x800 tiles is unviewable anyway.
    """
    import cv2

    height, width = video.shape[1:3]
    if max(height, width) <= size:
        return video
    scale = size / max(height, width)
    target = (max(1, int(round(width * scale))), max(1, int(round(height * scale))))
    return np.stack([cv2.resize(frame, target, interpolation=cv2.INTER_AREA) for frame in video])


def tile_videos(videos: List[np.ndarray], num_rows: int) -> np.ndarray:
    """Arrange same-sized ``(T, H, W, 3)`` videos in a grid; trailing videos that do not fill a row are dropped."""
    per_row = len(videos) // num_rows
    rows = [np.concatenate(videos[r * per_row:(r + 1) * per_row], axis=2) for r in range(num_rows)]
    return np.concatenate(rows, axis=1)
