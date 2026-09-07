"""Inference: 4D rendering, key-point trajectories and the latent-space applications.

Modes (``mode=`` in the config):

* ``render``               render the training motions (videos, trajectories, optional grids)
* ``interpolation``        render the motion half-way between two training motions in latent space
* ``language``             text prompt -> BERT -> projector -> latent code -> motion
* ``fit_motion``           fit a latent code to a new multi-view video (motion reconstruction)
* ``fit_unaligned_motion`` like ``fit_motion`` but also fine-tunes the motion decoder
* ``benchmark``            rendering speed
"""

import math
import os
import random
import time
from typing import Dict, Iterable, List, Optional, Sequence, Union

from omegaconf import ListConfig

import cv2
import lpips
import numpy as np
import torch
import torch.nn.functional as F
import tqdm
from tensorboardX import SummaryWriter

from dimo.cameras import OrbitIntrinsics
from dimo.data import SceneInfo, frame_path, load_motion, parse_video_list, to_float
from dimo.losses import ssim
from dimo.models.gaussian_model import STAGE_1, STAGE_2, TEXT_PROJECTOR_NAME
from dimo.models.renderer import Renderer
from dimo.models.text_encoder import MLPEncoder, encode_text
from dimo.trainer import resolution_for_step
from dimo.utils.general import BasicPointCloud, seed_everything
from dimo.utils.io import ensure_dir, side_by_side, tensor_to_uint8, write_frames, write_image, write_video
from dimo.utils.sh import SH2RGB
from dimo.utils.visualization import (
    downscale_video,
    draw_trajectories,
    draw_trajectory_frames,
    overlay,
    plot_3d_tracks,
    plot_3d_tracks_image,
    tile_videos,
    trajectory_colors,
)

KEY_POINT_COLOR = 0.1


class Tester:
    def __init__(self, opt):
        self.opt = opt
        self.device = torch.device("cuda")
        seed_everything(opt.seed)

        self.scene = SceneInfo(opt.input_folder, opt.num_views, opt.num_frames, opt.elevation, opt.input_videos)
        self.motions: List[str] = self.scene.motions
        self.num_views, self.num_frames = self.scene.num_views, self.scene.num_frames
        self.intrinsics = OrbitIntrinsics(opt.W, opt.H, opt.radius, opt.fovy)
        self.stage = opt.test_stage
        self.out_dir = ensure_dir(opt.video_save_dir)

        self.renderer = self._new_renderer(len(self.motions))
        self.render_videos = parse_video_list(opt.render_videos) or self.motions
        self.writer: Optional[SummaryWriter] = None
        self.lpips = None
        torch.cuda.empty_cache()

    @property
    def gaussians(self):
        return self.renderer.gaussians

    def _new_renderer(self, num_latent_codes: int) -> Renderer:
        opt = self.opt
        return Renderer(
            sh_degree=opt.sh_degree,
            num_latent_codes=num_latent_codes,
            latent_dim=opt.latent_code_dim,
            add_normal=opt.add_normal,
            vae_latent=opt.vae_latent,
        )

    # ----------------------------------------------------------------------------- loading

    def load(self):
        opt, g = self.opt, self.gaussians
        g.load_checkpoint(os.path.join(opt.save_path, self.stage), self.stage, opt.ckpt_step)
        if len(g.latents) != len(self.motions):
            raise ValueError(f"checkpoint has {len(g.latents)} latent codes but {len(self.motions)} motions are listed")
        if self.stage == STAGE_2:
            g.find_knn(k=opt.knn)
        g.eval()

    # ----------------------------------------------------------------------------- rendering helpers

    def orbit_azimuths(self) -> List[float]:
        return [360.0 / self.num_frames * i for i in range(self.num_frames)]

    @torch.no_grad()
    def render_frames(self, latent_index: int, azimuths: Union[float, Sequence[float]], stage: Optional[str] = None,
                      renderer: Optional[Renderer] = None) -> np.ndarray:
        """Render all frames of a motion from a fixed azimuth or a per-frame list of azimuths. ``(T, H, W, 3)`` uint8."""
        opt = self.opt
        renderer = renderer or self.renderer
        stage = stage or self.stage
        if not isinstance(azimuths, (list, tuple, np.ndarray)):
            azimuths = [azimuths] * self.num_frames
        frames = []
        for i, azimuth in enumerate(azimuths):
            camera = self.intrinsics.view(opt.elevation, azimuth)
            out = renderer.render(camera, time=self.scene.times[i], stage=stage, latent_index=latent_index)
            frames.append(tensor_to_uint8(out["image"]))
        return np.stack(frames)

    @torch.no_grad()
    def key_point_trajectories(self, latent_index: int, azimuth: float) -> Dict[str, np.ndarray]:
        """Render the key points as spheres and track them in 2D (``(N, T, 2)``) and 3D (``(T, N, 3)``)."""
        opt = self.opt
        kp_renderer = self.renderer.key_point_renderer()
        color = torch.full((kp_renderer.gaussians.num_points, 3), KEY_POINT_COLOR, device=self.device)
        frames, traj_2d, traj_3d = [], [], []
        moved = 0.0
        previous = None
        for i in range(self.num_frames):
            camera = self.intrinsics.view(opt.elevation, azimuth)
            out = kp_renderer.render(camera, time=self.scene.times[i], stage=STAGE_1, latent_index=latent_index, override_color=color)
            frames.append(tensor_to_uint8(out["image"]))
            key_points = out["cpts_t"]
            if previous is not None:
                moved += torch.dist(key_points, previous).item()
            previous = key_points
            traj_3d.append(key_points.cpu().numpy())
            traj_2d.append(camera.project(key_points).cpu().numpy())
        print(f"[INFO] total key-point displacement: {moved:.4f}")
        return {"frames": np.stack(frames), "traj_2d": np.stack(traj_2d, axis=1), "traj_3d": np.stack(traj_3d)}

    def _gt_frame(self, motion: str, frame: int, view: int = 0) -> Optional[np.ndarray]:
        path = frame_path(self.opt.input_folder, motion, view, frame)
        if not os.path.exists(path):
            return None
        image = cv2.cvtColor(cv2.imread(path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        return cv2.resize(image, (self.opt.W, self.opt.H))

    def export_motion(self, name: str, latent_index: int, stage: Optional[str] = None, orbit: bool = True,
                      per_view: bool = False, gt_last_frame: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
        """Render one motion and write every visualisation for it into the output directory."""
        opt = self.opt
        out_dir = self.out_dir
        H, W = opt.H, opt.W
        stage = stage or self.stage

        frames = self.render_frames(latent_index, opt.test_azi, stage=stage)
        write_video(os.path.join(out_dir, f"{name}_ref.mp4"), frames)
        if opt.save_frames:
            write_frames(os.path.join(out_dir, name, "ref"), frames)
        if orbit:
            write_video(os.path.join(out_dir, f"{name}_orbit.mp4"), self.render_frames(latent_index, self.orbit_azimuths(), stage=stage))
        if per_view:
            for view, azimuth in enumerate(self.scene.azimuths):
                view_frames = self.render_frames(latent_index, azimuth, stage=stage)
                write_frames(os.path.join(out_dir, name, f"view_{view:02d}"), view_frames)
                write_video(os.path.join(out_dir, f"{name}_view_{view:02d}.mp4"), view_frames, verbose=False)

        # Key-point trajectories: 2D overlays and 3D plots.
        tracks = self.key_point_trajectories(latent_index, opt.test_azi)
        colors = trajectory_colors(tracks["traj_2d"].shape[0], opt.traj_cmap)
        full_traj = draw_trajectories(tracks["traj_2d"], H, W, colors)
        write_image(os.path.join(out_dir, f"{name}_trajectory.png"), full_traj, verbose=True)
        last_frame = gt_last_frame if gt_last_frame is not None else frames[-1]
        write_image(os.path.join(out_dir, f"{name}_last_trajectory.png"), overlay(last_frame, full_traj), verbose=True)

        traces = draw_trajectory_frames(tracks["traj_2d"], H, W, colors, trace_length=opt.traj_trace_length)
        blend = np.stack([overlay(frame, trace) for frame, trace in zip(frames, traces)])
        write_video(os.path.join(out_dir, f"{name}_blend.mp4"), blend)

        traj_3d_video = plot_3d_tracks(tracks["traj_3d"], trace_length=opt.traj_trace_length, cmap=opt.traj_cmap)
        write_video(os.path.join(out_dir, f"{name}_traj_3d.mp4"), traj_3d_video)
        write_image(os.path.join(out_dir, f"{name}_trajectory_3d.png"), plot_3d_tracks_image(tracks["traj_3d"], cmap=opt.traj_cmap))
        if opt.save_traj_html:
            from dimo.utils.scenepic import interactive_3d_trajectories

            with open(os.path.join(out_dir, f"{name}_trajectory_3d.html"), "w") as f:
                f.write(interactive_3d_trajectories(tracks["traj_3d"], fov_y=opt.fovy))

        return {"frames": frames, "blend": blend, "traj_3d": traj_3d_video}

    # ----------------------------------------------------------------------------- modes

    def run(self):
        mode = self.opt.mode
        runner = {
            "render": self.run_render,
            "interpolation": self.run_interpolation,
            "language": self.run_language,
            "fit_motion": self.run_fit_motion,
            "fit_unaligned_motion": self.run_fit_unaligned_motion,
            "benchmark": self.run_benchmark,
        }.get(mode)
        if runner is None:
            raise ValueError(f"Unknown mode {mode!r}")
        if mode in ("interpolation", "language", "fit_motion", "fit_unaligned_motion") and self.stage != STAGE_2:
            raise ValueError(f"mode={mode} renders with the key-point motion model, so it needs "
                             f"test_stage={STAGE_2}, not {self.stage!r}")
        runner()

    def run_render(self):
        """Render the training motions listed in ``render_videos`` (all by default)."""
        opt = self.opt
        self.load()
        print(f"[INFO] rendering {len(self.render_videos)} motions: {self.render_videos}")
        # Only the downscaled tiles are retained: the full-resolution renders of every motion would
        # cost several GB of host memory for the 51-motion dataset, and are already written to disk.
        keys = ("frames", "blend", "traj_3d")
        tiles: List[dict] = []
        for index, motion in enumerate(self.motions):
            if motion not in self.render_videos:
                continue
            gt_last = self._gt_frame(motion, self.num_frames - 1)
            result = self.export_motion(motion, index, per_view=opt.render_views, gt_last_frame=gt_last)
            if opt.save_grid:
                tiles.append({key: downscale_video(result[key], opt.grid_tile_size) for key in keys})

        if opt.save_grid and len(tiles) > 1:
            num_rows = max(1, math.floor(math.sqrt(len(tiles))))
            for key in keys:
                write_video(os.path.join(self.out_dir, f"grid_{key}.mp4"),
                            tile_videos([t[key] for t in tiles], num_rows))

    def run_interpolation(self):
        """Render the latent-space path between two training motions.

        ``interp_alpha`` may be a single value or a list; the default sweep renders both endpoints
        and the midpoint, which is the comparison the paper shows, and the renders are also tiled
        into one side-by-side video so the path can be read at a glance.
        """
        opt = self.opt
        self.load()
        name_a, name_b = parse_video_list(opt.interp_videos)
        index_a, index_b = self.scene.index(name_a), self.scene.index(name_b)
        alphas = opt.interp_alpha if isinstance(opt.interp_alpha, (list, tuple, ListConfig)) else [float(opt.interp_alpha)]
        print(f"[INFO] interpolating {name_a} -> {name_b} at alpha {list(alphas)}")

        # All codes are read before the first write: `set_codes` replaces the whole table, so
        # interpolating again after it would index a table that now holds a single code.
        codes = [(float(a), self.gaussians.latents.interpolate(index_a, index_b, float(a))) for a in alphas]

        results = []
        for alpha, code in codes:
            self.gaussians.latents.set_codes(code[None])
            self.gaussians.eval()
            # Endpoints keep the motion's own name; anything else is tagged with its alpha, so
            # extrapolating alphas (1.5, 2.0, ...) cannot overwrite each other's outputs.
            name = (name_a if alpha == 0 else name_b) + "_endpoint" if alpha in (0.0, 1.0) else \
                   f"intp_{name_a}_{name_b}_a{alpha:.2f}".replace(".", "p", 1)
            results.append(self.export_motion(name, 0, stage=STAGE_2, orbit=len(alphas) <= 3))

        if len(results) > 1:
            for key in ("frames", "blend"):
                write_video(os.path.join(self.out_dir, f"intp_{name_a}_{name_b}_sweep_{key}.mp4"),
                            np.concatenate([r[key] for r in results], axis=2))

    def run_language(self):
        """Language-guided motion generation with a pre-trained text -> latent projector."""
        opt = self.opt
        prompt = opt.test_text_prompt
        if not prompt:
            raise ValueError("`test_text_prompt` must be set for the language mode")
        print(f"[INFO] text prompt: {prompt!r}")
        # Defaults to where train_text_projector.py writes it, so the mode works straight after
        # training without having to repeat the path.
        ckpt = opt.text_encoder_ckpt or os.path.join(opt.save_path, self.stage, TEXT_PROJECTOR_NAME)
        if not os.path.exists(ckpt):
            raise FileNotFoundError(
                f"no text projector at {ckpt}. This mode needs one trained for this model, and the "
                f"released checkpoint does not include it. Train it with:\n"
                f"    python data_generation/train_text_projector.py --config data_generation/configs/default.yaml \\\n"
                f"        object.name={os.path.basename(os.path.normpath(opt.input_folder))} "
                f"dataset.output_dir={os.path.dirname(os.path.normpath(opt.input_folder)) or '.'} \\\n"
                f"        projector.checkpoint_dir={os.path.join(opt.save_path, self.stage)}\n"
                f"which writes {os.path.join(opt.save_path, self.stage, TEXT_PROJECTOR_NAME)}, the "
                f"location this mode uses when `text_encoder_ckpt` is left empty. The dataset needs a "
                f"captions.json; data_generation/caption_dataset.py writes one for datasets that lack it.")
        projector = MLPEncoder(output_size=opt.latent_code_dim).to(self.device).eval()
        projector.load_state_dict(torch.load(ckpt, map_location=self.device))
        print(f"[INFO] loaded text projector from {ckpt}")
        with torch.no_grad():
            code = projector(encode_text([prompt], cache_dir=opt.bert_cache_dir).to(self.device))

        self.load()
        self.gaussians.latents.set_codes(code)
        self.gaussians.eval()
        self.export_motion(prompt.replace(" ", "_"), 0, stage=STAGE_2)

    def run_fit_motion(self):
        """Fit a new latent code to a new multi-view video, keeping geometry and motion decoder fixed."""
        opt = self.opt
        self.load()
        data = self._load_test_motion()
        g = self.gaussians
        g.latents.reset_single()
        self._fit(self.renderer, STAGE_2, groups=g.latents.GROUP_NAMES, steps=opt.fit_steps, data=data, tag="fit")
        g.save_checkpoint(self.out_dir, STAGE_2)
        g.eval()
        self.export_motion("fit", 0, stage=STAGE_2, per_view=True)

    def run_fit_unaligned_motion(self):
        """Fit a new motion whose key-point layout is not aligned with the training ones.

        First the latent code and the translation head are fitted on a key-point-only model
        (stage-1 rendering) with the decoder trunk frozen, then the latent code and the whole motion
        decoder are refined on the full model. The second phase deliberately trains the trunk and
        both heads: freezing the trunk and translation head there would leave the per-point
        translations stuck at their stage-1 values, since `translation_parameters()` is exactly trunk
        plus translation head, and the residual translation error would be unrecoverable.
        """
        opt = self.opt
        self.load()
        data = self._load_test_motion()
        g = self.gaussians

        # Key-point-only model sharing the pre-trained motion decoder.
        control = self._new_renderer(1)
        control.gaussians.deform_net.load_state_dict(g.deform_net.state_dict())
        gc = control.gaussians
        c_xyz, c_log_radius = g._c_xyz.detach(), g._c_radius.detach()
        n = c_xyz.shape[0]
        gc.create_from_pcd(BasicPointCloud(points=c_xyz.cpu().numpy(), colors=SH2RGB(np.random.random((n, 3)) / 255.0), normals=np.zeros((n, 3))))
        gc._xyz = torch.nn.Parameter(c_xyz.clone())
        gc._scaling = torch.nn.Parameter(c_log_radius.expand(-1, 3).clone())
        gc.latents.reset_single()
        self._fit(control, STAGE_1, groups=(*gc.latents.GROUP_NAMES, "deform", "opacity", "f_dc", "f_rest"),
                  steps=opt.fit_unaligned_steps_s1, data=data, freeze=[gc.deform_net.deformnet], tag="fit_s1")

        # Transfer to the full model and refine the whole motion decoder.
        if hasattr(gc.latents, "log_var"):
            g.latents.set_codes(gc.latents.mu.detach(), gc.latents.log_var.detach())
        else:
            g.latents.set_codes(gc.latents.codes.detach())
        g.deform_net.load_state_dict(gc.deform_net.state_dict())
        self._fit(self.renderer, STAGE_2, groups=(*g.latents.GROUP_NAMES, "deform", "deform_rot"),
                  steps=opt.fit_steps, data=data, tag="fit_s2")
        g.save_model(self.out_dir)
        g.eval()
        self.export_motion("fit_unaligned", 0, stage=STAGE_2, orbit=False)

    @torch.no_grad()
    def run_benchmark(self, resolution: int = 512):
        self.load()
        camera = self.intrinsics.view(self.opt.elevation, 0.0, resolution)
        self.renderer.render(camera, time=0.0, stage=self.stage)
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(self.opt.benchmark_rounds):
            self.renderer.render(camera, time=0.0, stage=self.stage)
        torch.cuda.synchronize()
        print(f"[INFO] {self.opt.benchmark_rounds / (time.time() - t0):.1f} FPS at {resolution}x{resolution}")

    # ----------------------------------------------------------------------------- test-time fitting

    def _load_test_motion(self):
        opt = self.opt
        if not opt.test_motion_data:
            raise ValueError("`test_motion_data` must point to the multi-view video to fit")
        images, masks = load_motion(opt.test_motion_data, self.num_views, self.num_frames, opt.ref_size,
                                    opt.num_workers, mask_method=opt.mask_method)
        return {"images": images, "masks": masks}

    def _fit(self, renderer: Renderer, stage: str, groups: Iterable[str], steps: int, data, tag: str,
             freeze: Optional[List[torch.nn.Module]] = None, arap: bool = False):
        """Optimise the given parameter groups of ``renderer`` to reproduce ``data``."""
        opt = self.opt
        g = renderer.gaussians
        if self.writer is None:
            self.writer = SummaryWriter(log_dir=os.path.join(self.out_dir, "tb"))
        if self.lpips is None:
            self.lpips = lpips.LPIPS(net="vgg").to(self.device)

        g.training_setup(opt, stage, max_steps=opt.finetune_lr_max_steps)
        g.set_trainable_groups(groups)
        g.train()
        for module in freeze or []:
            module.requires_grad_(False)

        for step in tqdm.trange(1, steps + 1, desc=f"Fitting ({tag})"):
            g.update_learning_rate(step, stage)
            if stage == STAGE_2:
                g.find_knn(k=opt.knn)
            resolution = resolution_for_step(step, opt.finetune_resolution_steps, opt.render_resolutions)

            # The reference view is always included; the rest are sampled from what the dataset has.
            views = [0] + random.sample(range(1, self.num_views), min(opt.batch_size, self.num_views - 1))
            frames = random.sample(range(self.num_frames), min(opt.batch_size, self.num_frames))
            images, gt_images, masks, gt_masks = [], [], [], []
            for view in views:
                camera = self.intrinsics.view(opt.elevation, self.scene.azimuths[view], resolution)
                for frame in frames:
                    out = renderer.render(camera, time=self.scene.times[frame], stage=stage, latent_index=0)
                    images.append(out["image"][None])
                    masks.append(out["alpha"][None])
                    gt_image = to_float(data["images"][view, frame], self.device)
                    gt_mask = to_float(data["masks"][view, frame], self.device)
                    gt_images.append(F.interpolate(gt_image, (resolution, resolution), mode="bilinear", align_corners=False))
                    gt_masks.append(F.interpolate(gt_mask, (resolution, resolution), mode="bilinear", align_corners=False))
            images, gt_images = torch.cat(images), torch.cat(gt_images)
            masks, gt_masks = torch.cat(masks), torch.cat(gt_masks)

            mse_loss = F.mse_loss(images, gt_images)
            lpips_loss = self.lpips(images, gt_images).mean()
            ssim_loss = 1 - ssim(images, gt_images)
            mask_loss = F.mse_loss(masks, gt_masks)
            loss = opt.lambda_mse * mse_loss + opt.lambda_lpips * lpips_loss + opt.lambda_ssim * ssim_loss + opt.lambda_mask * mask_loss
            self.writer.add_scalar(f"{tag}/loss_mse", mse_loss.item(), step)
            self.writer.add_scalar(f"{tag}/loss_lpips", lpips_loss.item(), step)
            self.writer.add_scalar(f"{tag}/loss_ssim", ssim_loss.item(), step)
            self.writer.add_scalar(f"{tag}/loss_mask", mask_loss.item(), step)
            if arap and opt.use_arap and step < opt.arap_end_iter_s2:
                arap_term, _ = renderer.arap_loss(stage=stage, latent_index=0)
                loss = loss + opt.lambda_arap * arap_term
                self.writer.add_scalar(f"{tag}/loss_arap", arap_term.item(), step)

            if step % opt.tb_image_interval == 0:
                self.writer.add_images(f"{tag}/render", images[:1], step)
                self.writer.add_images(f"{tag}/gt", gt_images[:1], step)
                self.writer.add_histogram(f"{tag}/latent_code", g.latents.numpy(0), step)
                write_image(os.path.join(self.out_dir, "debug", f"{tag}_{step}.png"),
                            side_by_side(tensor_to_uint8(gt_images[0]), tensor_to_uint8(images[0])))

            loss.backward()
            g.optimizer.step()
            g.optimizer.zero_grad()
            torch.cuda.empty_cache()

        for module in freeze or []:
            module.requires_grad_(True)
        torch.cuda.synchronize()
