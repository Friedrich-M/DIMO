"""Stage 2: one image-to-video clip per motion caption.

Two backends are supported, both driven through a subprocess so each can live in its own
environment (``<backend>.python`` in the config):

* ``wan`` (default): Wan2.2 **TI2V-5B**, 121 frames at 24 fps and 1280x704. It centre-crops its
  conditioning image to that aspect ratio, so the square reference canvas is padded to 1280:704
  first (nothing is cropped away) and the padding is removed from the generated frames afterwards.
* ``cogvideox``: **CogVideoX-5B-I2V**, the model used for the paper's data, 49 frames at 8 fps.
  It squashes the conditioning image to its native 720x480, so the square aspect ratio is restored
  by resizing the generated frames back.

Both write ``<motion>.mp4`` (square, object centred) plus a ``.json`` sidecar recording the prompt
and the generation settings.
"""

import os
import subprocess
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
from PIL import Image

from datagen.image import composite_on_white, pad_to_aspect
from datagen.io import ensure_dir, read_json, read_video, write_json, write_video

COGVIDEOX_RUNNER = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                               "cogvideox", "generate_video.py")


# ----------------------------------------------------------------------------- backends


@dataclass
class Backend:
    """What stage 2 needs to know about one video model."""

    name: str
    fps: int
    num_frames: int
    python: str = "python"
    cwd: Optional[str] = None
    seed: int = 42
    extra_args: str = ""

    def reference_image(self, reference_rgba: Image.Image) -> Image.Image:
        """The conditioning image to hand the model, on a white background."""
        return composite_on_white(reference_rgba)

    def reference_tag(self) -> str:
        """File name for the cached conditioning image."""
        return self.name

    def command(self, prompt: str, image_path: str, save_file: str) -> List[str]:
        raise NotImplementedError

    def postprocess(self, raw_path: str, reference_rgba: Image.Image) -> np.ndarray:
        """Generated frames as a square, object-centred ``(T, S, S, 3)`` uint8 array."""
        return read_video(raw_path)

    def settings(self) -> Dict:
        """Extra fields recorded in the sidecar."""
        return {}


@dataclass
class WanBackend(Backend):
    name: str = "wan"
    fps: int = 24
    num_frames: int = 121
    repo: str = "../references/Wan2.2"
    ckpt_dir: str = ""
    size: str = "1280*704"
    task: str = "ti2v-5B"
    sample_steps: Optional[int] = None
    sample_guide_scale: Optional[float] = None
    sample_shift: Optional[float] = None
    frame_num: Optional[int] = None
    offload_model: bool = True
    convert_model_dtype: bool = True
    t5_cpu: bool = True

    SIZES = ("1280*704", "704*1280")

    def __post_init__(self):
        if self.size not in self.SIZES:
            raise ValueError(f"wan.size {self.size!r} is not valid for {self.task}: {self.SIZES}")
        if self.frame_num:
            self.num_frames = self.frame_num
        self.cwd = self.repo

    def _padded(self, reference_rgba: Image.Image) -> Image.Image:
        width, height = (int(v) for v in self.size.split("*"))
        return pad_to_aspect(reference_rgba, width, height)

    def reference_image(self, reference_rgba: Image.Image) -> Image.Image:
        return composite_on_white(self._padded(reference_rgba))

    def reference_tag(self) -> str:
        return f"{self.task}_{self.size.replace('*', 'x')}"

    def command(self, prompt: str, image_path: str, save_file: str) -> List[str]:
        cmd = [
            self.python, "generate.py",
            "--task", self.task,
            "--size", self.size,
            "--ckpt_dir", os.path.abspath(self.ckpt_dir),
            "--image", os.path.abspath(image_path),
            "--prompt", prompt,
            "--save_file", os.path.abspath(save_file),
            "--base_seed", str(self.seed),
            "--offload_model", "True" if self.offload_model else "False",
        ]
        if self.convert_model_dtype:
            cmd.append("--convert_model_dtype")
        if self.t5_cpu:
            cmd.append("--t5_cpu")
        for flag, value in (("--sample_steps", self.sample_steps), ("--sample_guide_scale", self.sample_guide_scale),
                            ("--sample_shift", self.sample_shift), ("--frame_num", self.frame_num)):
            if value is not None:
                cmd += [flag, str(value)]
        return cmd + (self.extra_args.split() if self.extra_args else [])

    def postprocess(self, raw_path: str, reference_rgba: Image.Image) -> np.ndarray:
        """Crop back the square region that corresponds to the un-padded reference canvas."""
        frames = read_video(raw_path)
        H, W = frames.shape[1:3]
        padded = self._padded(reference_rgba)
        pw, ph = padded.size
        sw, sh = reference_rgba.size
        scale_x, scale_y = W / pw, H / ph
        left = int(round((pw - sw) / 2 * scale_x))
        top = int(round((ph - sh) / 2 * scale_y))
        side = int(round(min(sw * scale_x, sh * scale_y)))
        return frames[:, top:top + side, left:left + side]

    def settings(self) -> Dict:
        return {"task": self.task, "size": self.size, "ckpt_dir": self.ckpt_dir}


@dataclass
class CogVideoXBackend(Backend):
    name: str = "cogvideox"
    fps: int = 8
    num_frames: int = 49
    model: str = "zai-org/CogVideoX-5b-I2V"
    steps: int = 50
    guidance_scale: float = 6.0
    dtype: str = "bfloat16"
    offload: bool = False

    def command(self, prompt: str, image_path: str, save_file: str) -> List[str]:
        cmd = [
            self.python, COGVIDEOX_RUNNER,
            "--model", self.model,
            "--image", os.path.abspath(image_path),
            "--prompt", prompt,
            "--save_file", os.path.abspath(save_file),
            "--num_frames", str(self.num_frames),
            "--steps", str(self.steps),
            "--guidance_scale", str(self.guidance_scale),
            "--fps", str(self.fps),
            "--seed", str(self.seed),
            "--dtype", self.dtype,
        ]
        if self.offload:
            cmd.append("--offload")
        return cmd + (self.extra_args.split() if self.extra_args else [])

    def postprocess(self, raw_path: str, reference_rgba: Image.Image) -> np.ndarray:
        """CogVideoX squashes the square input to its native aspect; undo that by resizing back."""
        frames = read_video(raw_path)
        side = reference_rgba.size[0]
        return np.stack([np.asarray(Image.fromarray(f).resize((side, side), Image.LANCZOS)) for f in frames])

    def settings(self) -> Dict:
        return {"model": self.model, "steps": self.steps, "guidance_scale": self.guidance_scale}


BACKENDS = {"wan": WanBackend, "cogvideox": CogVideoXBackend}


def build_backend(cfg) -> Backend:
    """``cfg`` is the whole config: ``video.backend`` selects the section to read."""
    name = cfg.video.backend
    if name not in BACKENDS:
        raise ValueError(f"unknown video.backend {name!r}; use one of {list(BACKENDS)}")
    cls = BACKENDS[name]
    section = cfg[name]
    fields = {k: section.get(k) for k in cls.__dataclass_fields__ if k in section}
    backend = cls(**{k: v for k, v in fields.items() if v is not None})
    if name == "wan" and not backend.ckpt_dir:
        raise SystemExit("`wan.ckpt_dir` must point to the downloaded Wan2.2 checkpoint folder")
    return backend


# ----------------------------------------------------------------------------- generation


def run_motion(backend: Backend, motion: Dict, reference_rgba: Image.Image, out_dir: str,
               force: bool = False) -> Dict:
    """Generate one motion clip; returns the sidecar record (skips work if the clip already exists)."""
    name = motion["name"]
    raw_path = os.path.join(out_dir, "raw", f"{name}.mp4")
    clip_path = os.path.join(out_dir, f"{name}.mp4")
    record_path = os.path.join(out_dir, f"{name}.json")
    if os.path.exists(clip_path) and os.path.exists(record_path) and not force:
        return read_json(record_path)

    # Rewritten when `force` is set: the tag encodes only the backend and resolution, so a changed
    # reference image or crop would otherwise be conditioned on the previous run's cached copy.
    image_path = os.path.join(out_dir, "reference", f"{backend.reference_tag()}.png")
    if force or not os.path.exists(image_path):
        ensure_dir(os.path.dirname(image_path))
        backend.reference_image(reference_rgba).save(image_path)

    log_file = os.path.join(out_dir, f"{backend.name}.log")
    cmd = backend.command(motion["caption"], image_path, raw_path)
    ensure_dir(os.path.dirname(os.path.abspath(raw_path)))
    print(f"[INFO] {backend.name}: {' '.join(cmd[1:3])} ...")
    with open(log_file, "a") as log:
        subprocess.run(cmd, cwd=backend.cwd, check=True, stdout=log, stderr=subprocess.STDOUT)
    if not os.path.exists(raw_path):
        raise RuntimeError(f"{backend.name} did not write {raw_path}; see {log_file}")

    frames = backend.postprocess(raw_path, reference_rgba)
    write_video(clip_path, frames, fps=backend.fps)

    record = {
        "name": name, "caption": motion["caption"], "short": motion.get("short"),
        # Absolute, so a later stage started from a different directory can still find it.
        "video": os.path.abspath(clip_path), "raw_video": os.path.abspath(raw_path), "backend": backend.name,
        "seed": backend.seed, "fps": backend.fps, "num_frames": int(frames.shape[0]),
        "resolution": [int(frames.shape[2]), int(frames.shape[1])], **backend.settings(),
    }
    write_json(record_path, record)
    return record
