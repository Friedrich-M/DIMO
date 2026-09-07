"""Stage 4: drive the SV4D sampler (``data_generation/sv4d/sample_multiview.py``) in its own environment."""

import os
import subprocess
from typing import List, Optional

SAMPLER = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "sv4d", "sample_multiview.py")

# Frames each model renders per window. Mirrors ``MODELS`` in sv4d/sample_multiview.py, which cannot
# be imported from here because it pulls in the SV4D environment; that module re-checks the count.
WINDOW_FRAMES = {"sv4d": 5, "sv4d2_8views": 5, "sv4d2": 12}


def check_frame_count(model: str, num_frames: int):
    """Fail before launching the sampler if the window schedule cannot cover ``num_frames``.

    Mirrors ``check_frame_count`` in the sampler, which re-checks after the SV4D environment loads.
    ``sv4d`` (SV4D 1.0) is pinned to 21 frames by its anchor pass; see that function for why.
    """
    if model not in WINDOW_FRAMES:
        raise SystemExit(f"unknown sv4d.model {model!r}; use one of {list(WINDOW_FRAMES)}")
    T = WINDOW_FRAMES[model]
    if model == "sv4d":
        exact = (T - 1) * T + 1
        if num_frames != exact:
            raise SystemExit(
                f"sv4d (SV4D 1.0) supports exactly {exact} frames, not {num_frames}: its anchor pass "
                f"conditions on {T} motion frames spaced {T - 1} apart, which only covers a "
                f"{exact}-frame clip. Run with `clips.num_frames={exact} sv4d.model=sv4d`, or keep the "
                f"default sv4d.model=sv4d2_8views, which handles any 4k+1 frame count."
            )
        return
    if num_frames < T:
        raise SystemExit(f"{model} renders {T} frames per window, so clips.num_frames must be >= {T} "
                         f"(got {num_frames})")
    if model != "sv4d2" and (num_frames - 1) % (T - 1) != 0:
        valid = ", ".join(str(1 + i * (T - 1)) for i in range(1, 7))
        raise SystemExit(f"{model} steps {T - 1} frames per window, so clips.num_frames must be "
                         f"{T - 1}k + 1 ({valid}, ...); got {num_frames}, which would leave the last "
                         f"{(num_frames - 1) % (T - 1)} frame(s) unrendered. `sv4d2` accepts any count >= 12.")


def run_sv4d(cfg, clips_dir: str, output_dir: str, num_frames: int, motions: Optional[List[str]] = None,
             log_file: Optional[str] = None):
    """``cfg`` is the ``sv4d`` section of the config; ``num_frames`` comes from ``clips.num_frames``."""
    check_frame_count(cfg.model, num_frames)
    cmd = [
        cfg.python, SAMPLER,
        "--clips_dir", os.path.abspath(clips_dir),
        "--output_dir", os.path.abspath(output_dir),
        "--model", cfg.model,
        "--checkpoints_dir", os.path.abspath(cfg.checkpoints_dir),
        "--sv3d_version", cfg.sv3d_version,
        "--num_steps", str(cfg.num_steps),
        "--img_size", str(cfg.img_size),
        "--n_frames", str(num_frames),
        "--elevation_deg", str(cfg.elevation),
        "--seed", str(cfg.seed),
        "--encoding_t", str(cfg.encoding_t),
        "--decoding_t", str(cfg.decoding_t),
    ]
    if motions:
        cmd += ["--motions", ",".join(motions)]
    env = dict(os.environ, PYTHONPATH=os.path.abspath(cfg.repo) + os.pathsep + os.environ.get("PYTHONPATH", ""))
    print(f"[INFO] sv4d: {' '.join(cmd[1:])}")
    with open(log_file, "a") if log_file else open(os.devnull, "w") as log:
        subprocess.run(cmd, cwd=cfg.repo, env=env, check=True, stdout=log, stderr=subprocess.STDOUT)
