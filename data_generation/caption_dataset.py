"""Caption the motions of an existing dataset, so the text projector can be trained for it.

    python caption_dataset.py dataset_dir=../data/trump_n51_step20 [llm.backend=local llm.model=...]

Stage 1 already records a caption per motion, so this is only needed for datasets that arrived
without them, such as the released Trump data. For every motion it shows a strip of frames from the
reference view to the vision-language model and asks for the short imperative phrase that
``train_text_projector.py`` regresses onto the motion latent codes. The result is written as the
``captions.json`` that stage 5 would have produced, in the motion order of ``info.json``.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from PIL import Image

from datagen.config import load_config
from datagen.filtering import frame_strip
from datagen.io import read_json, write_json
from datagen.llm import build_client

DESCRIBE_MOTION_SYSTEM = """You are shown frames sampled from a short video of a single object, in temporal order from left to right. The object's identity and the camera never change; only its motion does.
Describe the motion, and nothing else, as a JSON object:
{
    "short": "the motion as a short imperative phrase of 2 to 5 words, naming the part that moves and the direction where it matters, e.g. 'lift the right hand', 'walk forward', 'shake head'",
    "caption": "one sentence describing the motion in more detail"
}
If the object barely moves, say so plainly (e.g. "stand almost still"). Answer with the JSON object only."""


def motion_strip(root: str, motion: str, num_frames: int, view: int = 0, num: int = 6) -> Image.Image:
    """Frame strip of one motion's reference view, as the judge prompt uses."""
    indices = np.linspace(0, num_frames - 1, num).round().astype(int)
    frames = []
    for i in indices:
        path = os.path.join(root, motion, f"view_{view:02d}", f"{i:02d}.png")
        frames.append(np.asarray(Image.open(path).convert("RGBA")))
    return frame_strip(np.stack(frames), num=num)


def main():
    cfg = load_config(description=__doc__, require_object=False)
    root = cfg.dataset_dir
    if not root:
        raise SystemExit("`dataset_dir` must point at a dataset folder containing info.json")
    info = read_json(os.path.join(root, "info.json"))
    motions = info["input_videos"]
    # The released datasets predate the `num_frames` key, so fall back to counting the frames on
    # disk rather than to `clips.num_frames`, whose default (41) does not match them (21).
    num_frames = int(info["num_frames"]) if "num_frames" in info else len(
        [f for f in os.listdir(os.path.join(root, motions[0], "view_00")) if f.endswith(".png")])
    print(f"[INFO] {num_frames} frames per view")
    out_path = os.path.join(root, "captions.json")
    captions = read_json(out_path, default={}) if not cfg.force else {}
    client = build_client(cfg.llm)

    print(f"[INFO] captioning {len(motions)} motions from {root}")
    for i, motion in enumerate(motions):
        if motion in captions:
            continue
        strip = motion_strip(root, motion, num_frames)
        result = client.chat_json([
            {"role": "system", "content": DESCRIBE_MOTION_SYSTEM},
            {"role": "user", "content": [{"type": "image", "image": strip}]},
        ], temperature=0.2)
        if isinstance(result, list):
            result = result[0]
        short = str(result.get("short", "")).strip().rstrip(".")
        if not short:
            raise RuntimeError(f"{motion}: model returned no short phrase ({result})")
        captions[motion] = {"name": motion, "short": short,
                            "caption": str(result.get("caption", "")).strip(), "source": "caption_dataset"}
        write_json(out_path, captions)
        print(f"[INFO] [{i + 1}/{len(motions)}] {motion}: {short}")

    missing = [m for m in motions if m not in captions]
    if missing:
        raise SystemExit(f"still missing captions for {len(missing)} motions: {missing[:5]}")
    print(f"[INFO] {len(captions)} captions written to {out_path}")


if __name__ == "__main__":
    main()
