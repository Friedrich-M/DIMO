"""File helpers: videos, frames, JSON, naming."""

import json
import os
import re
import tempfile
from typing import Any, Iterable, List, Optional

import imageio.v2 as imageio
import numpy as np
from PIL import Image


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def read_json(path: str, default: Any = None) -> Any:
    if not os.path.exists(path):
        if default is not None:
            return default
        raise FileNotFoundError(path)
    with open(path) as f:
        return json.load(f)


def write_json(path: str, data: Any):
    """Write JSON atomically: a temporary file in the same directory, then ``os.replace``.

    These files are the pipeline's resume state, rewritten after every clip and every motion. A
    plain truncate-then-dump leaves a half-written file if the run is interrupted (Ctrl-C, OOM kill,
    Slurm preemption), and every later run would then fail to parse it until it was deleted by hand.
    """
    ensure_dir(os.path.dirname(path) or ".")
    directory = os.path.dirname(path) or "."
    fd, tmp = tempfile.mkstemp(dir=directory, prefix=os.path.basename(path) + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def slugify(text: str, max_words: int = 4, max_len: int = 32) -> str:
    """``"Lifts the right hand slowly"`` -> ``"lifts_the_right_hand"``."""
    words = re.sub(r"[^a-z0-9 ]+", " ", text.lower()).split()
    return "_".join(words[:max_words])[:max_len].strip("_") or "motion"


def motion_name(index: int, short_description: str) -> str:
    """Stable, readable motion folder name, e.g. ``003-raise_right_hand``."""
    return f"{index:03d}-{slugify(short_description)}"


def read_video(path: str) -> np.ndarray:
    """All frames of a video (or an animated gif) as ``(T, H, W, 3)`` uint8."""
    reader = imageio.get_reader(path)
    frames = [np.asarray(f)[..., :3] for f in reader]
    reader.close()
    return np.stack(frames)


def write_video(path: str, frames: Iterable[np.ndarray], fps: int = 8):
    ensure_dir(os.path.dirname(path) or ".")
    frames = [np.ascontiguousarray(np.asarray(f, dtype=np.uint8)) for f in frames]
    imageio.mimwrite(path, frames, fps=fps, quality=8, macro_block_size=1)


def write_frames(directory: str, frames: Iterable[np.ndarray], fmt: str = "{:02d}.png") -> List[str]:
    ensure_dir(directory)
    paths = []
    for i, frame in enumerate(frames):
        paths.append(os.path.join(directory, fmt.format(i)))
        Image.fromarray(np.asarray(frame, dtype=np.uint8)).save(paths[-1])
    return paths


def frame_names(directory: str, pattern: str = r"^\d+\.png$") -> List[str]:
    """Frame file names in temporal order, sorted numerically so 100.png follows 99.png."""
    names = [n for n in os.listdir(directory) if re.match(pattern, n)]
    return sorted(names, key=lambda n: int(os.path.splitext(n)[0]))


def read_frames(directory: str, pattern: str = r"^\d+\.png$") -> np.ndarray:
    names = frame_names(directory, pattern)
    return np.stack([np.asarray(Image.open(os.path.join(directory, n)).convert("RGBA")) for n in names])


def evenly_spaced_counts(num_available: int, minimum: int = 5) -> List[int]:
    """Frame counts that divide a ``num_available``-frame clip into exactly equal steps."""
    span = num_available - 1
    return sorted({span // s + 1 for s in range(1, span + 1) if span % s == 0 and span // s + 1 >= minimum})


def select_frame_indices(num_available: int, num_frames: int, start: int = 0, stride: Optional[int] = None,
                         warn: bool = True) -> List[int]:
    """Indices of ``num_frames`` frames from a clip of ``num_available`` frames.

    With ``stride=None`` the frames are spread over the whole clip, first and last frame included,
    which is the default: ``121 -> 21`` gives every 6th frame. When the count does not divide the
    clip evenly the steps cannot all be equal (``49 -> 21`` alternates 2 and 3 frames), so the
    frames are no longer equally spaced in time even though DIMO treats them as if they were; a
    warning then lists the counts that do divide evenly.

    With ``stride`` set, every ``stride``-th frame from ``start`` is taken instead, which is always
    evenly spaced but covers only part of the clip.
    """
    if num_available - start < num_frames:
        raise ValueError(f"clip has {num_available} frames, need {num_frames} from frame {start}")
    if stride is None:
        indices = [int(round(i)) for i in np.linspace(start, num_available - 1, num_frames)]
        gaps = {b - a for a, b in zip(indices, indices[1:])}
        if warn and len(gaps) > 1:
            print(f"[WARN] {num_frames} frames cannot be spaced evenly over {num_available}: steps alternate "
                  f"between {min(gaps)} and {max(gaps)} frames, so the clip's timestamps are slightly uneven. "
                  f"Evenly spaced counts for this clip: {evenly_spaced_counts(num_available)}")
        return indices
    indices = list(range(start, start + stride * num_frames, stride))
    if indices[-1] >= num_available:
        raise ValueError(f"stride {stride} x {num_frames} frames exceeds the {num_available}-frame clip")
    return indices
