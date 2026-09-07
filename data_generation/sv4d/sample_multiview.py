"""Multi-view video generation for DIMO with SV4D 1.0 or SV4D 2.0 (Stability generative-models).

Run from the root of the ``generative-models`` checkout, inside its environment::

    PYTHONPATH=. python /path/to/DIMO/data_generation/sv4d/sample_multiview.py \
        --clips_dir <object>/clips --output_dir <object>/multiview --model sv4d2_8views

Input: ``clips_dir/<motion>/FF.png`` — 21 object-centred RGBA (or white-background RGB) frames per
motion, all motions starting from the same reference frame. Output, per motion, in the layout
DIMO trains on::

    output_dir/<motion>/view_XX/FF.png     view_00 = input clip, view_01.. = novel views
    output_dir/info.json                   azimuths_deg / elevations_deg of the views (+ full SV3D orbit)

Models:
* ``sv4d2_8views``  SV4D 2.0 (default), 8 novel views in autoregressive 5-frame windows. Generates the
                    views directly, so no SV3D reference orbit is needed. Gives the 9-view layout DIMO
                    trains on.
* ``sv4d2``         SV4D 2.0, 4 novel views (12-frame windows); train DIMO with ``num_views=5``.
* ``sv4d``          SV4D 1.0: SV3D first renders a 21-view orbit of the reference frame (shared by all
                    motions of the object and cached in ``output_dir/canonical``); 8 of those views seed
                    anchor + dense sampling over the 21 frames (5 frames x 8 views).
"""

import json
import os
import sys
from glob import glob
from typing import List, Optional

import numpy as np
import torch
from fire import Fire
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, os.getcwd())  # the generative-models repo root
import scripts.demo.sv4d_helpers as sv4d_helpers  # noqa: E402
from scripts.demo.sv4d_helpers import (  # noqa: E402
    decode_latents,
    do_sample_per_step,
    initial_model_load,
    load_model,
    prepare_inputs,
    run_img2vid,
    sample_sv3d,
    save_img,
)
from sgm.modules.encoders.modules import VideoPredictionEmbedderWithEncoder  # noqa: E402

# `load_model` always builds the NSFW/watermark classifier, which pulls a ~1 GB CLIP checkpoint and
# sits in GPU memory. The SV4D sampling path never applies it, so it is stubbed out here.
sv4d_helpers.DeepFloydDataFiltering = lambda *args, **kwargs: None

UC_KEYS = ["cond_frames", "cond_frames_without_noise", "cond_view", "cond_motion"]
MODELS = {
    "sv4d": dict(T=5, V=8, config="scripts/sampling/configs/sv4d.yaml", ckpt="sv4d.safetensors",
                 options=dict(cfg=2.0, guider=5)),
    "sv4d2_8views": dict(T=5, V=8, config="scripts/sampling/configs/sv4d2_8views.yaml", ckpt="sv4d2_8views.safetensors",
                         options=dict(cfg=2.5, min_cfg=1.5, guider=5),
                         azimuths=[0, 30, 75, 120, 165, 210, 255, 300, 330]),
    "sv4d2": dict(T=12, V=4, config="scripts/sampling/configs/sv4d2.yaml", ckpt="sv4d2.safetensors",
                  options=dict(cfg=2.0, min_cfg=2.0, guider=2), azimuths=[0, 60, 120, 180, 240]),
}
SV3D_VIEWS = 21
SV3D_SUBSAMPLED_VIEWS = np.array([0, 2, 5, 7, 9, 12, 14, 16, 19])  # 9 roughly uniform views of the 21-view orbit


def check_frame_count(model: str, n_frames: int):
    """Reject frame counts the window schedule cannot cover.

    All variants generate ``T`` frames at a time in windows that overlap by one frame, so the windows
    start at ``0, T-1, 2(T-1), ...`` and the dense pass needs ``n_frames = (T-1)k + 1``. Beyond that
    the three variants differ:

    * ``sv4d`` (SV4D 1.0, T=5) needs **exactly 21 frames**. Before its dense pass it runs an anchor
      pass over ``frame_indices = T-1, 2(T-1), ..., n_frames``, and that list is fed to a
      conditioner declared with ``n_cond_frames=T``, so it must hold exactly ``T`` entries. Together
      with the ``(T-1)k + 1`` requirement that pins ``n_frames`` to ``(T-1)T + 1 = 21`` and nothing
      else, which is also the only value the upstream script and the reference implementation that
      produced the released data ever used.
    * ``sv4d2_8views`` (T=5) has no anchor pass, so any ``n_frames = 4k + 1`` works: 5, 9, ... 41,
      ... 121.
    * ``sv4d2`` (T=12) additionally clamps its last window to ``n_frames - T``, so any
      ``n_frames >= 12`` works.

    Longer sequences are fine for the 2.0 variants: the windows are autoregressive, each one
    conditioning on the views generated for its first frame.
    """
    T = MODELS[model]["T"]
    if n_frames < T:
        raise ValueError(f"{model} generates {T} frames per window, so n_frames must be >= {T} (got {n_frames})")
    if model == "sv4d":
        exact = (T - 1) * T + 1
        if n_frames != exact:
            raise ValueError(
                f"sv4d (SV4D 1.0) supports exactly {exact} frames, not {n_frames}: its anchor pass "
                f"conditions on {T} motion frames spaced {T - 1} apart, which only covers a "
                f"{exact}-frame clip. Use `clips.num_frames={exact}` with sv4d.model=sv4d, or keep "
                f"the default sv4d.model=sv4d2_8views, which handles any 4k+1 frame count."
            )
        return
    if model == "sv4d2":
        return
    if (n_frames - 1) % (T - 1) != 0:
        valid = [T - 1 + 1 + i * (T - 1) for i in range(4)]
        raise ValueError(
            f"{model} steps {T - 1} frames per window, so n_frames must be {T - 1}k + 1 "
            f"(e.g. {', '.join(str(v) for v in valid)}, ...); got {n_frames}, which would leave "
            f"the last {(n_frames - 1) % (T - 1)} frame(s) unrendered"
        )


def load_clip(folder: str, n_frames: int, size: int, device: str) -> List[torch.Tensor]:
    """RGBA/RGB frames -> list of ``(1, 3, H, W)`` tensors in [-1, 1] on white."""
    # Numeric sort: lexicographic order would put 100.png before 11.png.
    names = sorted((f for f in os.listdir(folder) if f.endswith(".png")),
                   key=lambda f: int(os.path.splitext(f)[0]))[:n_frames]
    if len(names) < n_frames:
        raise ValueError(f"{folder} has {len(names)} frames, need {n_frames}")
    frames = []
    for name in names:
        image = Image.open(os.path.join(folder, name)).convert("RGBA").resize((size, size), Image.LANCZOS)
        rgba = np.asarray(image).astype(np.float32) / 255
        rgb = rgba[..., :3] * rgba[..., 3:] + (1 - rgba[..., 3:])
        frames.append(torch.from_numpy(rgb).permute(2, 0, 1)[None].to(device) * 2 - 1)
    return frames


def save_view(folder: str, frames: List[torch.Tensor]):
    for t, frame in enumerate(frames):
        save_img(os.path.join(folder, f"{t:02d}.png"), frame)


def write_info(path: str, info: dict):
    """Write ``info.json`` atomically; it is rewritten after every motion and is the resume state."""
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(info, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def motion_done(folder: str, n_views: int, n_frames: int) -> bool:
    """True only if every view holds exactly the expected frames.

    Counting "at least" ``n_frames`` would report a stale 41-frame render as a finished 21-frame one
    and record it in ``info.json``, so the frame count must match exactly.
    """
    for v in range(n_views):
        view = os.path.join(folder, f"view_{v:02d}")
        if len(glob(os.path.join(view, "*.png"))) != n_frames:
            return False
        if not all(os.path.exists(os.path.join(view, f"{t:02d}.png")) for t in range(n_frames)):
            return False
    return True


def canonical_orbit(reference: torch.Tensor, cache_dir: str, sv3d_version: str, checkpoints_dir: str, num_steps: int,
                    decoding_t: int, polars_rad, azimuths_rad, device: str, verbose: bool) -> torch.Tensor:
    """21 SV3D views of the reference frame, index 0 = the reference itself; cached as PNGs.

    Only used by SV4D 1.0; SV4D 2.0 generates the novel views without a reference orbit.
    """
    cached = sorted(glob(os.path.join(cache_dir, "*.png")),
                    key=lambda p: int(os.path.splitext(os.path.basename(p))[0]))
    if len(cached) == SV3D_VIEWS:
        print(f"[INFO] reusing canonical orbit from {cache_dir}")
        views = [torch.from_numpy(np.asarray(Image.open(p).convert("RGB")).astype(np.float32) / 255).permute(2, 0, 1) for p in cached]
        return (torch.stack(views) * 2 - 1).to(device)
    # Load SV3D explicitly so that the checkpoint comes from `checkpoints_dir` rather than the
    # repo-relative path baked into the config.
    ckpt = os.path.join(checkpoints_dir, f"{sv3d_version}.safetensors")
    if not os.path.exists(ckpt):
        raise FileNotFoundError(f"SV4D 1.0 needs the SV3D checkpoint at {ckpt} (accept the licence on the Hub first)")
    sv3d, _ = load_model(f"scripts/sampling/configs/{sv3d_version}.yaml", device, SV3D_VIEWS, num_steps, verbose, ckpt)
    images_t0 = sample_sv3d(reference, SV3D_VIEWS, num_steps, sv3d_version, 6, 127, 1e-5, decoding_t, device,
                            polars_rad, azimuths_rad, verbose, sv3d_model=sv3d)
    del sv3d
    torch.cuda.empty_cache()
    images_t0 = torch.roll(images_t0, 1, 0)  # the conditioning image becomes view 0
    save_view(cache_dir, [v[None] for v in images_t0])
    return images_t0


def sample(
    clips_dir: str,
    output_dir: str,
    model: str = "sv4d2_8views",
    checkpoints_dir: str = "checkpoints",
    motions: Optional[List[str]] = None,
    sv3d_version: str = "sv3d_u",
    num_steps: int = 20,
    img_size: int = 576,
    n_frames: int = 21,
    elevation_deg: float = 0.0,
    seed: int = 23,
    encoding_t: int = 8,
    decoding_t: int = 8,
    device: str = "cuda",
    verbose: bool = False,
):
    if model not in MODELS:
        raise ValueError(f"model must be one of {list(MODELS)}")
    spec = MODELS[model]
    T, V = spec["T"], spec["V"]
    check_frame_count(model, n_frames)
    n_views = V + 1
    H = W = img_size
    C, F = 4, 8
    version_dict = {
        "T": T * V, "H": H, "W": W, "C": C, "f": F,
        "options": {
            "discretization": 1, "num_views": V, "sigma_min": 0.002, "sigma_max": 700.0, "rho": 7.0,
            "num_steps": num_steps, "force_uc_zero_embeddings": UC_KEYS,
            "additional_guider_kwargs": {"additional_cond_keys": ["cond_view", "cond_motion"]},
            **spec["options"],
        },
    }
    torch.manual_seed(seed)
    os.makedirs(output_dir, exist_ok=True)

    motion_names = motions if motions else sorted(d for d in os.listdir(clips_dir) if os.path.isdir(os.path.join(clips_dir, d)))
    if isinstance(motion_names, str):
        motion_names = motion_names.split(",")
    todo = [m for m in motion_names if not motion_done(os.path.join(output_dir, m), n_views, n_frames)]
    print(f"[INFO] {len(motion_names)} motions, {len(todo)} to generate with {model}")

    # Camera layout. Azimuths are relative to the input view (view 0).
    if model == "sv4d":
        # SV3D samples the orbit at 360*(j+1)/21 degrees for output frame j. Rolling the stack by one
        # puts the conditioning image at index 0, so orbit image k sits at azimuth 360*k/21.
        full_azimuths = np.roll(np.linspace(0, 360, SV3D_VIEWS + 1)[1:] % 360, 1)  # [0, 17.14, ...]
        sv3d_azimuths = np.linspace(0, 360, SV3D_VIEWS + 1)[1:] % 360
        polars_rad = np.deg2rad(90 - np.full(SV3D_VIEWS, elevation_deg))
        view_azimuths = full_azimuths[SV3D_SUBSAMPLED_VIEWS]
        subsampled_views = SV3D_SUBSAMPLED_VIEWS
        # SV4D is conditioned on angles relative to view 0 (the `- azimuths_rad[v0]` below), so indexing
        # the unrolled SV3D azimuths here is equivalent to using the rolled ones, as in the reference scripts.
        sv3d_azimuths_rad = np.deg2rad(sv3d_azimuths)
        azimuths_rad = sv3d_azimuths_rad
    else:
        view_azimuths = np.array(spec["azimuths"], dtype=float)
        polars_rad = np.deg2rad(90 - np.full(n_views, elevation_deg))
        azimuths_rad = np.deg2rad((view_azimuths - view_azimuths[-1]) % 360)
        subsampled_views = np.arange(n_views)
        full_azimuths = view_azimuths
        sv3d_azimuths_rad = azimuths_rad

    info_path = os.path.join(output_dir, "info.json")
    info = json.load(open(info_path)) if os.path.exists(info_path) else {}
    info.update({
        "azimuths_deg": view_azimuths.tolist(),
        "elevations_deg": [float(elevation_deg)] * n_views,
        "full_azimuths_deg": full_azimuths.tolist(),
        "num_views": n_views, "num_frames": n_frames, "image_size": img_size,
        "model": model, "sv3d_version": sv3d_version if model == "sv4d" else None, "num_steps": num_steps, "seed": seed,
    })
    info["input_videos"] = sorted(set(info.get("input_videos", [])) | set(m for m in motion_names if m not in todo))
    write_info(info_path, info)
    if not todo:
        return

    # SV4D 1.0 needs the multi-view images of the reference frame, shared by all motions of the object.
    images_t0 = None
    if model == "sv4d":
        reference = load_clip(os.path.join(clips_dir, motion_names[0]), 1, img_size, device)[0]
        images_t0 = canonical_orbit(reference, os.path.join(output_dir, "canonical"), sv3d_version, checkpoints_dir,
                                    num_steps, decoding_t, polars_rad, sv3d_azimuths_rad, device, verbose)

    sv4d, _ = load_model(spec["config"], device, version_dict["T"], num_steps, verbose,
                         os.path.join(checkpoints_dir, spec["ckpt"]))
    if model == "sv4d":
        sv4d = initial_model_load(sv4d)
    for emb in sv4d.conditioner.embedders:
        if isinstance(emb, VideoPredictionEmbedderWithEncoder):
            emb.en_and_decode_n_samples_a_time = encoding_t
    sv4d.en_and_decode_n_samples_a_time = decoding_t

    v0 = 0
    view_indices = np.arange(V) + 1
    polars = polars_rad[subsampled_views[1:]][None].repeat(T, 0).flatten()
    azims = azimuths_rad[subsampled_views[1:]][None].repeat(T, 0).flatten()
    azims = (azims - azimuths_rad[v0]) % (2 * np.pi)
    if model != "sv4d":
        polars = (polars - polars_rad[v0] + np.pi / 2) % (2 * np.pi)

    for motion in tqdm(todo, desc="motions"):
        motion_dir = os.path.join(output_dir, motion)
        images_v0 = load_clip(os.path.join(clips_dir, motion), n_frames, img_size, device)
        img_matrix = [[None] * n_views for _ in range(n_frames)]
        if model == "sv4d":
            for i, v in enumerate(subsampled_views):
                img_matrix[0][i] = images_t0[v][None]
        else:
            for i in range(1, n_views):
                img_matrix[0][i] = torch.zeros(1, 3, H, W, device=device)
        # Written last, and deliberately so: `images_t0` is the cached SV3D orbit of whichever motion
        # came first in the run, and its view 0 would otherwise replace this clip's own frame 0 with a
        # differently framed image (the square crop is derived per clip). Upstream orders it the same way.
        for t in range(n_frames):
            img_matrix[t][0] = images_v0[t]
        if model == "sv4d":
            _sample_sv4d1(sv4d, version_dict, img_matrix, n_frames, T, V, C, H, W, F, view_indices, v0, polars, azims, seed, num_steps, decoding_t)
        else:
            _sample_sv4d2(sv4d, version_dict, img_matrix, n_frames, T, V, H, W, view_indices, v0, polars, azims, seed, decoding_t, model)

        for v in range(n_views):
            save_view(os.path.join(motion_dir, f"view_{v:02d}"), [img_matrix[t][v] for t in range(n_frames)])
        info["input_videos"] = sorted(set(info["input_videos"]) | {motion})
        write_info(info_path, info)
        torch.cuda.empty_cache()


def _sample_sv4d1(model, version_dict, img_matrix, n_frames, T, V, C, H, W, F, view_indices, v0, polars, azims,
                  seed, num_steps, decoding_t):
    """Anchor frames first, then dense 5-frame windows denoised with alternating forward/backward conditioning."""
    t0 = 0
    frame_indices = np.arange(T - 1, n_frames, T - 1)  # [4, 8, 12, 16, 20]
    cond_motion = torch.cat([img_matrix[t][v0] for t in frame_indices], 0)
    cond_view = torch.cat([img_matrix[t0][v] for v in view_indices], 0)
    samples = run_img2vid(version_dict, model, img_matrix[t0][v0], seed, polars, azims, cond_motion, cond_view, decoding_t)
    samples = samples.view(T, V, 3, H, W)
    for i, t in enumerate(frame_indices):
        for j, v in enumerate(view_indices):
            img_matrix[t][v] = samples[i, j][None] * 2 - 1

    for t0 in np.arange(0, n_frames - 1, T - 1):  # [0, 4, 8, 12, 16]
        frame_indices = t0 + np.arange(T)
        latent_matrix = torch.randn(n_frames, len(img_matrix[0]), C, H // F, W // F, device="cuda")
        forward, forward_idx, backward, backward_idx = prepare_inputs(frame_indices, img_matrix, v0, view_indices, model, version_dict, seed, polars, azims)
        for step in range(num_steps):
            c, uc, extra, sampler = forward if step % 2 == 1 else backward
            indices = forward_idx if step % 2 == 1 else backward_idx
            noisy = latent_matrix[indices][:, view_indices].flatten(0, 1)
            out = do_sample_per_step(model, sampler, noisy, c, uc, step, extra).view(T, V, C, H // F, W // F)
            for i, t in enumerate(indices):
                for j, v in enumerate(view_indices):
                    latent_matrix[t, v] = out[i, j]
        decode_latents(model, latent_matrix, img_matrix, indices, view_indices, T)


def _sample_sv4d2(model, version_dict, img_matrix, n_frames, T, V, H, W, view_indices, v0, polars, azims, seed,
                  decoding_t, variant):
    """Autoregressive windows; the previous window's last frame conditions the next one."""
    t0_list = range(0, n_frames, T - 1) if variant == "sv4d2" else range(0, n_frames - T + 1, T - 1)
    for t0 in t0_list:
        if t0 + T > n_frames:
            t0 = n_frames - T
        frame_indices = t0 + np.arange(T)
        cond_motion = torch.cat([img_matrix[t][v0] for t in frame_indices], 0)
        cond_view = torch.cat([img_matrix[t0][v] for v in view_indices], 0)
        samples = run_img2vid(version_dict, model, img_matrix[t0][v0], seed, polars, azims, cond_motion, cond_view,
                              decoding_t, cond_mv=t0 != 0)
        samples = samples.view(T, V, 3, H, W)
        for i, t in enumerate(frame_indices):
            for j, v in enumerate(view_indices):
                img_matrix[t][v] = samples[i, j][None] * 2 - 1


if __name__ == "__main__":
    Fire(sample)
