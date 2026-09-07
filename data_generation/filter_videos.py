"""Stage 3: extract object-centred clips (`clips.num_frames` frames) and filter them by motion magnitude and (optionally) a VLM judge.

    python filter_videos.py object.name=trump [filter.judge=vlm]

Writes ``<workdir>/<object>/clips/<motion>/FF.png`` (RGBA) and ``<workdir>/<object>/filter_report.json``
listing every clip's scores and whether it was accepted.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from PIL import Image

from datagen.config import load_config, object_dir
from datagen.filtering import ensure_clip, evaluate_clip
from datagen.io import ensure_dir, read_json, write_json
from datagen.llm import build_client


def main():
    cfg = load_config(description=__doc__)
    out_dir = object_dir(cfg)
    video_dir = os.path.join(out_dir, "videos")
    clip_dir = ensure_dir(os.path.join(out_dir, "clips"))
    motions = read_json(os.path.join(out_dir, "motions.json"))["motions"]
    selected = set(cfg.motions.split(",")) if cfg.motions else None
    records = [read_json(os.path.join(video_dir, f"{m['name']}.json")) for m in motions
               if (selected is None or m["name"] in selected)
               and os.path.exists(os.path.join(video_dir, f"{m['name']}.json"))]
    print(f"[INFO] {len(records)} generated videos to process")

    report_path = os.path.join(out_dir, "filter_report.json")
    report = read_json(report_path, default={}) if not cfg.force else {}
    client = build_client(cfg.llm) if cfg.filter.judge == "vlm" else None
    reference = Image.open(os.path.join(out_dir, "reference_white.png")).convert("RGB")

    for record in records:
        name = record["name"]
        folder = ensure_clip(record, clip_dir, cfg.clips, force=cfg.force, video_dir=video_dir)
        if name in report and not cfg.force:
            continue
        report[name] = evaluate_clip(folder, record, cfg.filter, reference=reference, client=client, device=cfg.device)
        write_json(report_path, report)
        status = "accepted" if report[name]["accepted"] else f"rejected ({'; '.join(report[name]['reasons'])})"
        print(f"[INFO] {name}: motion {report[name]['motion']:.2f} px -> {status}")

    if not report:
        raise SystemExit("no clips were evaluated; run generate_videos.py first")
    scores = np.array([r["motion"] for r in report.values()])
    accepted = [n for n, r in report.items() if r["accepted"]]
    print(f"[INFO] motion score percentiles (px): 10% {np.percentile(scores, 10):.2f}, 50% {np.median(scores):.2f}, "
          f"90% {np.percentile(scores, 90):.2f}; thresholds [{cfg.filter.min_motion}, {cfg.filter.max_motion}]")
    print(f"[INFO] {len(accepted)} / {len(report)} clips accepted; report at {report_path}")


if __name__ == "__main__":
    main()
