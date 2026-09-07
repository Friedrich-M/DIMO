"""Stage 5: assemble the DIMO training folder.

    python build_dataset.py object.name=trump [dataset.output_dir=../data]

Writes ``<dataset.output_dir>/<object>/<motion>/view_XX/FF.png`` (+ ``FF_mask.npy``), ``info.json`` and
``captions.json``, ready for ``python train.py input_folder=<dataset.output_dir>/<object>``.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from datagen.config import load_config, multiview_dir, object_dir
from datagen.dataset import build_dataset
from datagen.io import read_json


def main():
    cfg = load_config(description=__doc__)
    out_dir = object_dir(cfg)
    motions = {m["name"]: m for m in read_json(os.path.join(out_dir, "motions.json"))["motions"]}
    mv_dir = multiview_dir(cfg)
    info = read_json(os.path.join(mv_dir, "info.json"))
    report = read_json(os.path.join(out_dir, "filter_report.json"), default={})
    names = [m for m in info["input_videos"] if not report or report.get(m, {}).get("accepted")]
    # The frame count comes from the render, not the config: stage 4 recorded what it actually
    # produced, and taking `clips.num_frames` here would silently truncate every motion if the two
    # ever disagree (which they do as soon as the config is changed between stages).
    num_frames = int(info["num_frames"])
    if num_frames != int(cfg.clips.num_frames):
        print(f"[WARN] the render in {mv_dir} has {num_frames} frames per view but clips.num_frames "
              f"is {cfg.clips.num_frames}; using the render's {num_frames}")
    build_dataset(mv_dir, os.path.join(cfg.dataset.output_dir, cfg.dataset.name or cfg.object.name), names,
                  motions, num_views=info["num_views"], num_frames=num_frames,
                  compute_masks=cfg.dataset.compute_masks, link=cfg.dataset.symlink, force=cfg.force,
                  mask_method=cfg.dataset.mask_method)


if __name__ == "__main__":
    main()
