"""Stage 4: multi-view videos of every accepted clip with SV4D 1.0 / 2.0.

    python generate_multiview.py object.name=trump sv4d.repo=/path/generative-models sv4d.python=/path/env/bin/python

Writes ``<workdir>/<object>/multiview/<model>/<motion>/view_XX/FF.png`` plus an ``info.json`` with the
camera layout. The output folder is per model, so switching ``sv4d.model`` cannot reuse a render made
with a different view layout. Motions that are already complete are skipped, so the stage can be
re-run after adding clips.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from datagen.config import load_config, multiview_dir, object_dir
from datagen.io import read_json
from datagen.multiview import run_sv4d


def accepted_motions(out_dir: str, cfg) -> list:
    report = read_json(os.path.join(out_dir, "filter_report.json"), default={})
    motions = [m["name"] for m in read_json(os.path.join(out_dir, "motions.json"))["motions"]]
    if cfg.motions:
        return [m for m in motions if m in set(cfg.motions.split(","))]
    if not report:
        print("[WARN] no filter report found; using every clip")
        return [m for m in motions if os.path.isdir(os.path.join(out_dir, "clips", m))]
    return [m for m in motions if report.get(m, {}).get("accepted")]


def main():
    cfg = load_config(description=__doc__)
    out_dir = object_dir(cfg)
    motions = accepted_motions(out_dir, cfg)
    if not motions:
        raise SystemExit("no accepted clips to process; run filter_videos.py first")
    print(f"[INFO] {len(motions)} motions -> SV4D ({cfg.sv4d.model})")
    run_sv4d(cfg.sv4d, os.path.join(out_dir, "clips"), multiview_dir(cfg), cfg.clips.num_frames, motions,
             log_file=os.path.join(out_dir, "sv4d.log"))


if __name__ == "__main__":
    main()
