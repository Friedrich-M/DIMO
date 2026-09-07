"""Stage 5: assemble the DIMO training folder from the multi-view renders."""

import os
import shutil
from typing import Dict, List

import numpy as np
from PIL import Image

from datagen.io import ensure_dir, read_json, write_json


def _mask_from_png(path: str, method: str = "rembg") -> np.ndarray:
    """The mask ``dimo.data`` would compute for this frame, cached as ``FF_mask.npy``.

    ``method`` mirrors ``dimo.data.MASK_METHOD``: ``rembg`` runs the matting network, ``white_bg``
    derives the alpha from the distance to white, which is ~30x faster and exact for these renders.
    Whichever is used here must match the training config, or the cached masks and the ones DIMO
    would compute on the fly would disagree.
    """
    image = Image.open(path)
    if image.mode == "RGBA":
        return np.asarray(image)[..., 3:4].astype(np.float32) / 255.0
    if method == "white_bg":
        import cv2

        from datagen.image import white_background_alpha

        return white_background_alpha(cv2.cvtColor(np.asarray(image.convert("RGB")), cv2.COLOR_RGB2BGR))
    from datagen.image import remove_background

    return np.asarray(remove_background(image))[..., 3:4].astype(np.float32) / 255.0


def _complete(root: str, motion: str, num_views: int, num_frames: int) -> bool:
    """True if every view of ``motion`` holds every frame (not just the last one)."""
    return all(os.path.exists(os.path.join(root, motion, f"view_{v:02d}", f"{f:02d}.png"))
               for v in range(num_views) for f in range(num_frames))


def build_dataset(multiview_dir: str, output_dir: str, motions: List[str], captions: Dict[str, Dict],
                  num_views: int, num_frames: int, compute_masks: bool = True, link: bool = False,
                  force: bool = False, mask_method: str = "rembg"):
    """Copy (or symlink) ``multiview_dir/<motion>`` folders and write ``info.json`` / ``captions.json``.

    ``compute_masks`` precomputes the ``FF_mask.npy`` files DIMO's loader would otherwise create with
    rembg on first use, so that training can run on machines without internet access.

    A destination that already exists is reused only when it is complete. An interrupted copy, or a
    motion that stage 4 has since re-rendered, is replaced rather than kept forever, and ``force``
    replaces every motion.
    """
    ensure_dir(output_dir)
    info = read_json(os.path.join(multiview_dir, "info.json"))
    kept = []
    for motion in motions:
        src = os.path.join(multiview_dir, motion)
        dst = os.path.join(output_dir, motion)
        if not _complete(multiview_dir, motion, num_views, num_frames):
            print(f"[WARN] {motion}: incomplete multi-view render, skipped")
            continue
        stale = force or not _complete(output_dir, motion, num_views, num_frames)
        if os.path.islink(dst) and (stale or not os.path.exists(os.path.realpath(dst))):
            os.unlink(dst)
        elif os.path.isdir(dst) and stale:
            shutil.rmtree(dst)
        if os.path.islink(dst) or os.path.isdir(dst):
            pass
        elif link:
            os.symlink(os.path.abspath(src), dst)
        else:
            shutil.copytree(src, dst)
        if compute_masks:
            for v in range(num_views):
                for f in range(num_frames):
                    png = os.path.join(dst, f"view_{v:02d}", f"{f:02d}.png")
                    mask_path = os.path.splitext(png)[0] + "_mask.npy"
                    if not os.path.exists(mask_path):
                        np.save(mask_path, _mask_from_png(png, mask_method))
        kept.append(motion)

    write_json(os.path.join(output_dir, "info.json"), {
        "azimuths_deg": info["azimuths_deg"],
        "full_azimuths_deg": info.get("full_azimuths_deg", info["azimuths_deg"]),
        "elevations_deg": info["elevations_deg"],
        "input_videos": kept,
        "num_views": num_views,
        "num_frames": num_frames,
        "image_size": info.get("image_size"),
        "model": info.get("model"),
    })
    write_json(os.path.join(output_dir, "captions.json"), {m: captions[m] for m in kept if m in captions})
    validate_dataset(output_dir, expect_masks=compute_masks)
    print(f"[INFO] dataset with {len(kept)} motions written to {output_dir}")
    return kept


def validate_dataset(root: str, expect_masks: bool = True):
    """Check the folder against the contract ``dimo.data`` reads, so format drift fails here.

    ``dimo.data.SceneInfo`` needs ``azimuths_deg`` (its length defines the view count),
    ``elevations_deg`` of the same length and a non-empty ``input_videos``; ``dimo.data.frame_path``
    then asks for ``<motion>/view_XX/FF.png`` for every view and frame, with ``FF`` zero-padded to at
    least two digits, and ``dimo.data.load_frame`` reads ``FF_mask.npy`` beside each frame.
    """
    info = read_json(os.path.join(root, "info.json"))
    for key in ("azimuths_deg", "elevations_deg", "input_videos", "num_frames"):
        if key not in info:
            raise ValueError(f"{root}/info.json is missing {key!r}")
    num_views, num_frames = len(info["azimuths_deg"]), int(info["num_frames"])
    if len(info["elevations_deg"]) != num_views:
        raise ValueError(f"{root}/info.json: {len(info['elevations_deg'])} elevations for {num_views} views")
    if not info["input_videos"]:
        raise ValueError(f"{root}/info.json lists no motions")

    for motion in info["input_videos"]:
        for view in range(num_views):
            for frame in range(num_frames):
                png = os.path.join(root, motion, f"view_{view:02d}", f"{frame:02d}.png")
                if not os.path.exists(png):
                    raise FileNotFoundError(f"dataset is missing the frame dimo would read: {png}")
                if expect_masks and not os.path.exists(os.path.splitext(png)[0] + "_mask.npy"):
                    raise FileNotFoundError(f"dataset is missing the cached mask for {png}")
    print(f"[INFO] format check: {len(info['input_videos'])} motions x {num_views} views x {num_frames} frames"
          + (" (+ masks)" if expect_masks else ""))
