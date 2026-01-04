import os
from PIL import Image
from typing import NamedTuple
import numpy as np

from utils.graphics_utils import getWorld2View2


class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    image: np.array
    depth: np.array
    image_path: str
    image_name: str
    width: int
    height: int
    timestamp: float = 0.0
    fl_x: float = -1.0
    fl_y: float = -1.0
    cx: float = -1.0
    cy: float = -1.0

def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center

    return {"translate": translate, "radius": radius}

def get_c2w_from_up_and_look_at(
    up,
    look_at,
    pos,
    opengl=False,
):
    up = up / np.linalg.norm(up)
    z = look_at - pos
    z = z / np.linalg.norm(z)
    y = -up
    x = np.cross(y, z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)

    c2w = np.zeros([4, 4], dtype=np.float32)
    c2w[:3, 0] = x
    c2w[:3, 1] = y
    c2w[:3, 2] = z
    c2w[:3, 3] = pos
    c2w[3, 3] = 1.0

    # opencv to opengl
    if opengl:
        c2w[..., 1:3] *= -1

    return c2w

def get_uniform_poses(num_frames, radius, elevation, opengl=False, azimuths=None):
    T = num_frames
    if azimuths is None:
        azimuths = np.deg2rad(np.linspace(0, 360, num_frames + 1)[1:] % 360)
    elevations = np.full_like(azimuths, np.deg2rad(elevation))
    cam_dists = np.full_like(azimuths, radius)

    campos = np.stack(
        [
            cam_dists * np.cos(elevations) * np.cos(azimuths),
            cam_dists * np.cos(elevations) * np.sin(azimuths),
            cam_dists * np.sin(elevations),
        ],
        axis=-1,
    )

    center = np.array([0, 0, 0], dtype=np.float32)
    up = np.array([0, 0, 1], dtype=np.float32)
    poses = []
    for t in range(T):
        poses.append(get_c2w_from_up_and_look_at(up, center, campos[t], opengl=opengl))

    return np.stack(poses, axis=0)

def contructVideoNVSInfor(
    path,
    num_views,
    radius,
    elevation,
    fov,
    reso,
    masks,
    num_pts=100_000,
    train=True,
    num_frames=21,
    fps=6,
    time_duration=None,
    render_high_fps=False
):
    poses = get_uniform_poses(num_views, radius, elevation)
    w2cs = np.linalg.inv(poses)
    train_cam_infos = []

    for view_idx, pose in enumerate(w2cs):
        for frame_idx in range(num_frames):
            timestamp = frame_idx / fps
            image_path = os.path.join(path, f"view_{view_idx:02d}", f"frame_{frame_idx:02d}.png")
            image = Image.open(image_path)

            train_cam_infos.append(
                CameraInfo(
                    uid=view_idx,
                    R=np.transpose(pose[:3, :3]),
                    T=pose[:3, 3],
                    FovY=np.deg2rad(fov),
                    FovX=np.deg2rad(fov),
                    image=image,
                    image_path=image_path,
                    image_name=f"{view_idx:02d}_{frame_idx:02d}",
                    width=reso,
                    height=reso,
                    depth=None,
                    timestamp=timestamp,
                )
            )

    test_cameras = train_cam_infos[::50]
    return train_cam_infos, test_cameras