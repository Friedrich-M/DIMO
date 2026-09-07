"""Stage 2: generate one monocular video per motion caption.

    python generate_videos.py object.name=trump wan.ckpt_dir=/path/Wan2.2-TI2V-5B      # Wan2.2 (default)
    python generate_videos.py object.name=trump video.backend=cogvideox                # CogVideoX-5B-I2V

Reads ``<workdir>/<object>/motions.json`` and ``reference.png``; writes ``<workdir>/<object>/videos/<motion>.mp4``
(+ ``.json`` sidecar with the prompt and generation settings). Already generated motions are skipped.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from PIL import Image

from datagen.config import load_config, object_dir
from datagen.io import ensure_dir, read_json
from datagen.video_gen import build_backend, run_motion


def main():
    cfg = load_config(description=__doc__)
    out_dir = object_dir(cfg)
    motions = read_json(os.path.join(out_dir, "motions.json"))["motions"]
    reference = Image.open(os.path.join(out_dir, "reference.png")).convert("RGBA")
    backend = build_backend(cfg)
    video_dir = ensure_dir(os.path.join(out_dir, "videos"))

    selected = set(cfg.motions.split(",")) if cfg.motions else None
    todo = [m for m in motions if selected is None or m["name"] in selected]
    print(f"[INFO] generating {len(todo)} motion videos with {backend.name} "
          f"({backend.num_frames} frames @ {backend.fps} fps)")
    for i, motion in enumerate(todo):
        print(f"[INFO] [{i + 1}/{len(todo)}] {motion['name']}: {motion['short']}")
        record = run_motion(backend, motion, reference, video_dir, force=cfg.force)
        print(f"        -> {record['video']} ({record['num_frames']} frames @ {record['fps']} fps)")


if __name__ == "__main__":
    main()
