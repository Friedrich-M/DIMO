"""Loading of multi-view motion videos.

Expected layout of an input folder::

    <input_folder>/
        info.json                       # optional: azimuths_deg, elevations_deg, input_videos
        <motion_name>/view_XX/FF.png    # XX = view index, FF = frame index

Foreground masks are computed with ``rembg`` on first use and cached next to each frame as
``FF_mask.npy`` (frames with an alpha channel use it directly).
"""

import itertools
import json
import os
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import tqdm
import torch.multiprocessing as mp

# --------------------------------------------------------------------------- frame loading

# Pinned so masks do not change with the rembg version (recent releases moved the default away from
# u2net). Override with $DIMO_REMBG_MODEL.
REMBG_MODEL = os.environ.get("DIMO_REMBG_MODEL", "u2net")
_REMBG_SESSION = None

# How a missing mask is computed. "rembg" runs the u2net matting network (~0.2 s per frame, so a
# 51 x 9 x 21 dataset costs about half an hour on one core; it is cached afterwards and the loader
# parallelises over `num_workers`). "white_bg" is for frames rendered on a white background -- every
# frame this pipeline produces -- and derives the alpha from the distance to white in a few
# milliseconds. Set with `mask_method=` or $DIMO_MASK_METHOD.
MASK_METHOD = os.environ.get("DIMO_MASK_METHOD", "rembg")
WHITE_BG_TOLERANCE = 12


def _rembg_session():
    global _REMBG_SESSION
    if _REMBG_SESSION is None:
        os.environ.setdefault("OMP_NUM_THREADS", "8")
        import rembg

        _REMBG_SESSION = rembg.new_session(REMBG_MODEL)
    return _REMBG_SESSION


def _init_worker():
    # The rembg session is created lazily (only when a mask is missing) so that machines
    # without internet access can load datasets whose masks are already cached.
    torch.set_num_threads(1)


def white_background_alpha(image: np.ndarray, tolerance: int = WHITE_BG_TOLERANCE) -> np.ndarray:
    """Foreground alpha ``(H, W, 1)`` for an object rendered on a white background.

    The alpha is the distance from white, ramped over ``2 * tolerance`` so that edges stay soft
    rather than aliased. Pixels that are white *and* reachable from the image border are then forced
    to zero, so a white part of the object (a shirt, a highlight) keeps its alpha instead of being
    punched out the way a plain threshold would.

    About 30x faster than the matting network, and exact for renders like these, whose background is
    a uniform white. It is not a general background remover: use the ``rembg`` method for
    photographs or any frame whose background is not white.
    """
    bgr = image[..., :3]
    distance = 255 - bgr.min(axis=2)                       # 0 on pure white

    # Background = near-white pixels reachable from the image border. Flood-filling from the corners
    # is what keeps a white shirt or a specular highlight inside the object opaque: it is white, but
    # the object around it walls the fill off, so it is never reached.
    white = (distance <= tolerance).astype(np.uint8)
    height, width = white.shape
    barrier = np.zeros((height + 2, width + 2), np.uint8)
    barrier[1:-1, 1:-1] = 1 - white                        # the object blocks the fill
    flooded = white.copy()
    for x, y in ((0, 0), (width - 1, 0), (0, height - 1), (width - 1, height - 1)):
        if white[y, x]:
            cv2.floodFill(flooded, barrier, (x, y), 2)
    background = flooded == 2

    # Opaque everywhere the object is, and the distance ramp applied only in the thin band touching
    # the background, so the silhouette gets an anti-aliased edge while light-coloured parts of the
    # object stay fully opaque instead of turning translucent.
    alpha = np.ones(distance.shape, np.float32)
    band = cv2.dilate(background.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=1).astype(bool)
    edge = band & ~background
    alpha[edge] = np.clip(distance[edge].astype(np.float32) / (2.0 * tolerance), 0.0, 1.0)
    alpha[background] = 0.0
    return alpha[..., None]


def _load_or_compute_mask(image: np.ndarray, mask_path: str, method: Optional[str] = None) -> np.ndarray:
    """Foreground alpha ``(H, W, 1)`` in [0, 1] for a BGR(A) uint8 image."""
    if image.shape[-1] == 4:
        alpha = image[..., 3:4].astype(np.float32) / 255.0
        if not os.path.exists(mask_path):
            np.save(mask_path, alpha)
        return alpha

    if os.path.exists(mask_path):
        try:
            mask = np.load(mask_path)
            if mask.ndim == 2:
                mask = mask[..., None]
            return mask.astype(np.float32)
        except Exception:
            os.remove(mask_path)

    method = method or MASK_METHOD
    if method == "white_bg":
        alpha = white_background_alpha(image)
    elif method == "rembg":
        import rembg

        # rembg is given RGB: cv2 returns BGR, and passing that swaps red and blue, which produces a
        # different matte than data_generation's precomputed masks. A dataset would then train on
        # different masks depending on whether the mask cache was populated here or by stage 5.
        rgba = rembg.remove(np.ascontiguousarray(image[..., ::-1]), session=_rembg_session())
        alpha = rgba[..., 3:4].astype(np.float32) / 255.0
    else:
        raise ValueError(f"unknown mask method {method!r}; use 'rembg' or 'white_bg'")
    np.save(mask_path, alpha)
    return alpha


def load_frame(path: str, ref_size: int, mask_method: Optional[str] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """Read one frame; returns ``image (3, S, S)`` and ``mask (1, S, S)`` as uint8 tensors."""
    image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(f"Failed to read image: {path}")
    mask = _load_or_compute_mask(image, os.path.splitext(path)[0] + "_mask.npy", mask_method)

    rgb = image[..., :3][..., ::-1].astype(np.float32) / 255.0
    image_t = torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1)[None]
    mask_t = torch.from_numpy(mask).permute(2, 0, 1)[None]
    if image.shape[0] != ref_size or image.shape[1] != ref_size:
        image_t = F.interpolate(image_t, (ref_size, ref_size), mode="bilinear", align_corners=False)
        mask_t = F.interpolate(mask_t, (ref_size, ref_size), mode="bilinear", align_corners=False)
    return _to_uint8(image_t[0]), _to_uint8(mask_t[0])


def _to_uint8(x: torch.Tensor) -> torch.Tensor:
    return (x.clamp(0, 1) * 255).round().to(torch.uint8)


def _load_frame_job(args):
    key, path, ref_size, mask_method = args
    image, mask = load_frame(path, ref_size, mask_method)
    return key, image, mask


def frame_path(root: str, motion: str, view: int, frame: int) -> str:
    return os.path.join(root, motion, f"view_{view:02d}", f"{frame:02d}.png")


def load_frames(jobs: Sequence[Tuple[tuple, str]], ref_size: int, num_workers: int = 16,
                desc: str = "Loading frames", mask_method: Optional[str] = None):
    """Load many frames in parallel. ``jobs`` is a list of ``(key, path)``; yields ``(key, image, mask)``.

    Workers are spawned (not forked), so they re-import the calling module. Any script that reaches
    this code must therefore live in a real file and keep its top-level work under
    ``if __name__ == "__main__":`` -- a snippet piped in on stdin cannot be re-imported and the
    workers will die trying. Set ``num_workers=1`` to load in-process and avoid both constraints.
    """
    if num_workers <= 1:  # in-process: usable from anywhere, and enough when masks are cached
        for key, path in tqdm.tqdm(jobs, desc=desc):
            image, mask = load_frame(path, ref_size, mask_method)
            yield key, image, mask
        return
    args = [(key, path, ref_size, mask_method) for key, path in jobs]
    num_workers = max(1, min(num_workers, os.cpu_count() or 1))
    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=num_workers, initializer=_init_worker) as pool:
        for item in tqdm.tqdm(pool.imap(_load_frame_job, args, chunksize=32), total=len(args), desc=desc):
            yield item


def load_motion(root: str, num_views: int, num_frames: int, ref_size: int,
                num_workers: int = 16, mask_method: Optional[str] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """All frames of one multi-view video under ``root/view_XX/FF.png``.

    Returns ``images (V, T, 3, S, S)`` and ``masks (V, T, 1, S, S)`` as uint8 tensors.
    """
    images = torch.zeros((num_views, num_frames, 3, ref_size, ref_size), dtype=torch.uint8)
    masks = torch.zeros((num_views, num_frames, 1, ref_size, ref_size), dtype=torch.uint8)
    jobs = [((v, f), os.path.join(root, f"view_{v:02d}", f"{f:02d}.png"))
            for v, f in itertools.product(range(num_views), range(num_frames))]
    for (v, f), image, mask in load_frames(jobs, ref_size, num_workers, mask_method=mask_method):
        images[v, f] = image
        masks[v, f] = mask
    return images, masks


def to_float(x: torch.Tensor, device) -> torch.Tensor:
    """uint8 ``(..., C, H, W)`` -> float ``(1, C, H, W)`` in [0, 1] on ``device``."""
    return x.to(device).float().div_(255.0)[None]


# --------------------------------------------------------------------------- scene description


def parse_video_list(value) -> Optional[List[str]]:
    if value is None:
        return None
    if isinstance(value, str):
        return [v for v in value.split(",") if v]
    return list(value)


class SceneInfo:
    """Camera layout and motion list of an input folder.

    The view count comes from the dataset itself, so folders produced by any of the multi-view
    models load unchanged: SV4D 1.0 and SV4D 2.0 ``sv4d2_8views`` give 9 views (1 input + 8 novel),
    SV4D 2.0 ``sv4d2`` gives 5. ``num_views`` / ``num_frames`` are only defaults for datasets whose
    ``info.json`` does not state them.
    """

    def __init__(self, root: str, num_views: int, num_frames: int, elevation: float = 0.0,
                 input_videos=None):
        self.root = root
        info_path = os.path.join(root, "info.json")
        requested = parse_video_list(input_videos)

        if os.path.exists(info_path):
            with open(info_path) as f:
                info = json.load(f)
            self.azimuths = list(info["azimuths_deg"])
            if len(self.azimuths) != num_views:
                print(f"[INFO] {info_path} declares {len(self.azimuths)} views (config said {num_views}); "
                      "using the dataset's layout")
            self.num_views = len(self.azimuths)
            self.num_frames = int(info.get("num_frames", num_frames))
            if self.num_frames != num_frames:
                print(f"[INFO] {info_path} declares {self.num_frames} frames (config said {num_frames}); "
                      "using the dataset's value")
            self.elevations = list(info.get("elevations_deg", [elevation] * self.num_views))
            if len(self.elevations) != self.num_views:
                raise ValueError(f"info.json has {len(self.elevations)} elevations for {self.num_views} views")
            print(f"[INFO] camera layout from {info_path}: {self.num_views} views x {self.num_frames} frames"
                  + (f" ({info['model']})" if info.get("model") else ""))
            self.motions = requested if requested is not None else info.get("input_videos")
            if self.motions is None:
                raise ValueError("No motion list: pass `input_videos` or add it to info.json")
        else:
            print("[INFO] no info.json found, using uniformly spaced azimuths")
            self.num_views, self.num_frames = num_views, num_frames
            self.azimuths = [360.0 / num_views * i for i in range(num_views)]
            self.elevations = [elevation] * num_views
            if requested is None:
                raise ValueError("No info.json in the input folder: `input_videos` must be given")
            self.motions = requested

        self.times = [i / self.num_frames for i in range(self.num_frames)]

    def __len__(self):
        return len(self.motions)

    def index(self, motion: str) -> int:
        return self.motions.index(motion)


class MotionDataset(SceneInfo):
    """All frames of every motion, kept on the CPU as uint8 and moved to the GPU on access."""

    def __init__(self, root: str, num_views: int, num_frames: int, ref_size: int, elevation: float = 0.0,
                 input_videos=None, num_workers: int = 16, mask_method: Optional[str] = None):
        super().__init__(root, num_views, num_frames, elevation, input_videos)
        self.ref_size = ref_size
        num_views, num_frames = self.num_views, self.num_frames
        shape = (num_views, num_frames, ref_size, ref_size)
        self.images: Dict[str, torch.Tensor] = {m: torch.zeros((*shape[:2], 3, *shape[2:]), dtype=torch.uint8) for m in self.motions}
        self.masks: Dict[str, torch.Tensor] = {m: torch.zeros((*shape[:2], 1, *shape[2:]), dtype=torch.uint8) for m in self.motions}

        jobs = [((m, v, f), frame_path(root, m, v, f))
                for m, v, f in itertools.product(self.motions, range(num_views), range(num_frames))]
        for (m, v, f), image, mask in load_frames(jobs, ref_size, num_workers, desc="Loading data",
                                                  mask_method=mask_method):
            self.images[m][v, f] = image
            self.masks[m][v, f] = mask
        print(f"[INFO] loaded {len(self.motions)} motions x {num_views} views x {num_frames} frames")

    def get(self, motion: str, view: int, frame: int, device) -> Tuple[torch.Tensor, torch.Tensor]:
        """``image (1, 3, S, S)`` and ``mask (1, 1, S, S)`` as floats on ``device``."""
        return to_float(self.images[motion][view, frame], device), to_float(self.masks[motion][view, frame], device)
