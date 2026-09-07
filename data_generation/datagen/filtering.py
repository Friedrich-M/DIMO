"""Stage 3: turn generated clips into object-centred RGBA clips of `clips.num_frames` frames and filter them.

Following the paper, clips are rejected when the object moves too little or too much (optical-flow
magnitude inside the object mask, RAFT) and, optionally, when a vision-language judge scores the
visual quality / consistency / prompt alignment below a threshold.
"""

import os
import re
from typing import Dict, List, Optional

import numpy as np
import torch
from PIL import Image

from datagen.image import crop_frames_to_square, frames_to_rgba
from datagen.io import read_frames, read_video, select_frame_indices, write_frames
from datagen.llm import ChatClient
from datagen.prompts import QUALITY_JUDGE_SYSTEM

# ----------------------------------------------------------------------------- clip extraction


def extract_clip(video_path: str, num_frames: int, start: int, stride: Optional[int], image_ratio: float,
                 size: int) -> np.ndarray:
    """``(num_frames, size, size, 4)`` RGBA frames with rembg mattes, cropped around the object."""
    frames = read_video(video_path)
    indices = select_frame_indices(frames.shape[0], num_frames, start=start, stride=stride)
    rgba = frames_to_rgba(frames[indices])
    return crop_frames_to_square(rgba, image_ratio=image_ratio, size=size)


def resolve_video(record: Dict, video_dir: str) -> str:
    """Absolute path of a record's clip.

    The sidecar's ``video`` field was written relative to the working directory of the stage that
    generated it, so it only resolves if every later stage is run from that same directory. The
    stored path is tried first, then the same basename inside ``video_dir``, then the motion's own
    conventional name, so records from older runs keep working.
    """
    stored = record.get("video") or ""
    for candidate in (stored, os.path.join(video_dir, os.path.basename(stored)),
                      os.path.join(video_dir, record["name"] + ".mp4")):
        if candidate and os.path.exists(candidate):
            return os.path.abspath(candidate)
    raise FileNotFoundError(f"{record['name']}: no generated video at {stored!r} or in {video_dir}")


def ensure_clip(record: Dict, clip_dir: str, cfg, force: bool = False, video_dir: Optional[str] = None) -> str:
    """Write ``clip_dir/<motion>/FF.png`` (RGBA) for one generated video and return the folder.

    A cached folder counts as usable only if it holds *exactly* ``cfg.num_frames`` frames. Accepting
    "at least" that many would silently reuse a 41-frame extraction for a 21-frame run, and the
    motion score would then be measured over a different time span than the render uses.
    """
    folder = os.path.join(clip_dir, record["name"])
    existing = [f for f in os.listdir(folder) if f.endswith(".png")] if os.path.isdir(folder) else []
    if len(existing) == cfg.num_frames and not force:
        return folder
    for name in existing:  # a stale extraction at a different frame count must not survive
        os.remove(os.path.join(folder, name))
    video = resolve_video(record, video_dir) if video_dir else record["video"]
    clip = extract_clip(video, cfg.num_frames, cfg.frame_start, cfg.frame_stride, cfg.image_ratio, cfg.size)
    write_frames(folder, clip)
    return folder


# ----------------------------------------------------------------------------- motion score

_RAFT: Dict[str, torch.nn.Module] = {}


def _raft(device):
    device = str(device)
    if device not in _RAFT:
        from torchvision.models.optical_flow import Raft_Small_Weights, raft_small

        _RAFT[device] = raft_small(weights=Raft_Small_Weights.DEFAULT).to(device).eval()
    return _RAFT[device]


@torch.no_grad()
def motion_score(frames_rgba: np.ndarray, device: str = "cuda", flow_size: int = 384, top_fraction: float = 0.2) -> float:
    """Mean over frame pairs of the average of the largest ``top_fraction`` flow magnitudes inside the mask.

    Magnitudes are expressed in pixels at the clip resolution. This is the "motion amplitude" the paper
    uses to discard clips with minimal or excessive motion.
    """
    model = _raft(device)
    T, H = frames_rgba.shape[0], frames_rgba.shape[1]
    rgb = torch.from_numpy(frames_rgba[..., :3]).permute(0, 3, 1, 2).float().div(255)
    alpha = torch.from_numpy(frames_rgba[..., 3]).float().div(255)
    rgb = torch.nn.functional.interpolate(rgb, size=(flow_size, flow_size), mode="bilinear", align_corners=False)
    alpha = torch.nn.functional.interpolate(alpha[:, None], size=(flow_size, flow_size), mode="bilinear", align_corners=False)[:, 0]
    rgb = (rgb * 2 - 1).to(device)
    scale = H / flow_size

    scores = []
    for t in range(T - 1):
        flow = model(rgb[t:t + 1], rgb[t + 1:t + 2])[-1][0]  # (2, h, w)
        magnitude = flow.norm(dim=0).cpu() * scale
        mask = (alpha[t] > 0.5) | (alpha[t + 1] > 0.5)
        values = magnitude[mask]
        if values.numel() == 0:
            scores.append(0.0)
            continue
        k = max(1, int(values.numel() * top_fraction))
        scores.append(values.topk(k).values.mean().item())
    return float(np.mean(scores))


# ----------------------------------------------------------------------------- VLM judge


def frame_strip(frames_rgba: np.ndarray, num: int = 6, size: int = 256) -> Image.Image:
    """A horizontal strip of ``num`` frames on white, for the judge."""
    indices = np.linspace(0, frames_rgba.shape[0] - 1, num).round().astype(int)
    tiles = []
    for i in indices:
        rgba = frames_rgba[i].astype(np.float32) / 255
        rgb = rgba[..., :3] * rgba[..., 3:] + (1 - rgba[..., 3:])
        tiles.append(Image.fromarray((rgb * 255).astype(np.uint8)).resize((size, size), Image.LANCZOS))
    strip = Image.new("RGB", (size * num, size), (255, 255, 255))
    for i, tile in enumerate(tiles):
        strip.paste(tile, (i * size, 0))
    return strip


def judge_clip(client: ChatClient, reference: Image.Image, frames_rgba: np.ndarray, caption: str) -> Dict:
    strip = frame_strip(frames_rgba)
    messages = [
        {"role": "system", "content": QUALITY_JUDGE_SYSTEM},
        {"role": "user", "content": [
            {"type": "text", "text": "Reference image:"}, {"type": "image", "image": reference},
            {"type": "text", "text": "Frames sampled from the generated video, in temporal order (left to right):"},
            {"type": "image", "image": strip},
            {"type": "text", "text": f"Prompt used to generate the video: {caption}"},
        ]},
    ]
    result = client.chat_json(messages, temperature=0.1)
    if isinstance(result, list):
        result = result[0]
    return {k: result.get(k) for k in ("visual_quality", "consistency", "motion_alignment", "comment")}


# ----------------------------------------------------------------------------- decision


def _as_score(value) -> Optional[float]:
    """A judge score as a number, or None if the model did not return one.

    The prompt asks for integers, but models sometimes answer "4/5" or "four". A score that cannot be
    read is treated as absent rather than aborting the stage part-way through a run.
    """
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    match = re.search(r"-?\d+(?:\.\d+)?", str(value))
    if match is None:
        print(f"[WARN] ignoring unreadable judge score {value!r}")
        return None
    return float(match.group(0))


def evaluate_clip(folder: str, record: Dict, cfg, reference: Optional[Image.Image] = None,
                  client: Optional[ChatClient] = None, device: str = "cuda") -> Dict:
    frames = read_frames(folder)
    report = {"name": record["name"], "motion": motion_score(frames, device=device)}
    reasons: List[str] = []
    if report["motion"] < cfg.min_motion:
        reasons.append("too little motion")
    if report["motion"] > cfg.max_motion:
        reasons.append("too much motion")
    if client is not None and reference is not None:
        scores = judge_clip(client, reference, frames, record["caption"])
        report.update(scores)
        for key in ("visual_quality", "consistency", "motion_alignment"):
            value = _as_score(scores.get(key))
            if value is not None and value < cfg.min_score:
                reasons.append(f"{key} {value} < {cfg.min_score}")
    report["accepted"] = not reasons
    report["reasons"] = reasons
    return report
