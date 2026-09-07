"""Two-stage training of DIMO.

Stage 1 (motion pre-training): a few hundred isotropic Gaussians act as key points and are
deformed directly by the latent-conditioned motion decoder, giving a coarse motion basis and the
shared latent space. Stage 2 (joint refinement): the stage-1 Gaussians become the key points,
dense Gaussians are spawned around them and everything is optimised jointly, with the stage-1
key-point trajectories used as a geometry anchor.
"""

import os
import random
from typing import Dict, List, Sequence

import lpips
import numpy as np
import torch
import torch.nn.functional as F
import tqdm
from tensorboardX import SummaryWriter

from dimo.cameras import OrbitIntrinsics
from dimo.config import save_config
from dimo.data import MotionDataset
from dimo.losses import (
    bilateral_normal_smoothness_loss,
    chamfer_forward,
    edge_aware_depth_smoothness_loss,
    ssim,
)
from dimo.models.gaussian_model import STAGE_1, STAGE_2
from dimo.models.renderer import Renderer
from dimo.utils.general import random_point_cloud, seed_everything
from dimo.utils.io import ensure_dir, side_by_side, tensor_to_uint8, write_image

SCENE_EXTENT = 4.0
MAX_SCREEN_SIZE = 1


def resolution_for_step(step: int, thresholds: Sequence[int], resolutions: Sequence[int]) -> int:
    """Coarse-to-fine rendering resolution: ``resolutions[i]`` while ``step < thresholds[i]``."""
    for threshold, resolution in zip(thresholds, resolutions):
        if step < threshold:
            return resolution
    return resolutions[-1]


class Trainer:
    def __init__(self, opt):
        self.opt = opt
        self.device = torch.device("cuda")
        seed_everything(opt.seed)

        self.save_dir = ensure_dir(opt.save_path)
        save_config(opt, os.path.join(self.save_dir, "config.yaml"))
        self.writer = SummaryWriter(log_dir=os.path.join(self.save_dir, "tb"))

        self.data = MotionDataset(opt.input_folder, opt.num_views, opt.num_frames, opt.ref_size,
                                  elevation=opt.elevation, input_videos=opt.input_videos,
                                  num_workers=opt.num_workers, mask_method=opt.mask_method)
        self.motions: List[str] = self.data.motions
        self.num_views, self.num_frames = self.data.num_views, self.data.num_frames
        self.intrinsics = OrbitIntrinsics(opt.W, opt.H, opt.radius, opt.fovy)

        self.renderer = Renderer(
            sh_degree=opt.sh_degree,
            num_latent_codes=len(self.motions),
            latent_dim=opt.latent_code_dim,
            add_normal=opt.add_normal,
            vae_latent=opt.vae_latent,
        )
        if opt.vae_latent:
            print("[INFO] latent codes follow a Gaussian distribution (KL regularised)")
        self.lpips = lpips.LPIPS(net="vgg").to(self.device)

        self.stage = STAGE_1
        self.step = 0
        # Stage-1 key-point trajectories, cached at the start of stage 2 for the anchoring loss.
        self.key_point_trajectories: Dict[str, List[torch.Tensor]] = {}
        torch.cuda.empty_cache()

    @property
    def gaussians(self):
        return self.renderer.gaussians

    # ----------------------------------------------------------------------------- stages

    def run(self):
        opt = self.opt
        if opt.load_stage == STAGE_1:
            load_dir = os.path.join(opt.load_path or opt.save_path, STAGE_1)
            print(f"[INFO] resuming from stage-1 checkpoint {load_dir}")
            self.gaussians.load_checkpoint(load_dir, STAGE_1, opt.ckpt_step)
            if len(self.gaussians.latents) != len(self.motions):
                raise ValueError(
                    f"{load_dir} holds {len(self.gaussians.latents)} latent codes but "
                    f"{len(self.motions)} motions are listed: row i of the table belongs to motion i, "
                    f"so resuming with a different `input_videos` would train the wrong codes")
        elif opt.load_stage:
            raise ValueError(f"load_stage must be '' or '{STAGE_1}', got {opt.load_stage!r}")
        else:
            self.renderer.initialize(opt.num_cpts, radius=opt.init_radius)
            self.train_stage1()
        self.prepare_stage2()
        self.train_stage2()

    def train_stage1(self):
        opt, g = self.opt, self.gaussians
        self.stage, self.step = STAGE_1, 0
        g.training_setup(opt, STAGE_1)
        g.active_sh_degree = g.max_sh_degree
        g.train()

        for _ in tqdm.trange(opt.iters_s1, desc="Stage 1"):
            self.train_step()

        g.prune(min_opacity=opt.prune_opacity_s1_end, extent=SCENE_EXTENT, max_screen_size=None)
        print(f"[INFO] {g.num_points} key points after stage 1")
        g.save_checkpoint(os.path.join(self.save_dir, STAGE_1), STAGE_1, motions=self.motions)

    def prepare_stage2(self):
        """Freeze the stage-1 Gaussians as key points and spawn the dense Gaussians."""
        opt, g = self.opt, self.gaussians
        with torch.no_grad():
            c_xyz = g._xyz.detach().clone()
            c_log_radius = torch.log(g.get_scaling).mean(dim=1, keepdim=True)
        g.set_key_points(c_xyz, c_log_radius)
        g.clear_shared_radius()

        if opt.init_type == "normal":
            self.renderer.initialize_from_pcd(random_point_cloud(opt.num_pts, opt.init_radius))
        elif opt.init_type == "ag":  # adaptive Gaussian initialisation around each key point
            self.renderer.initialize_around_key_points(c_xyz, torch.exp(c_log_radius), opt.num_pts_per_cpt, opt.init_ratio)
        else:
            raise ValueError(f"Unsupported init_type {opt.init_type!r}")

        self.stage, self.step = STAGE_2, 0
        g.training_setup(opt, STAGE_2)
        g.active_sh_degree = g.max_sh_degree
        g.train()

        # Cache the stage-1 key-point trajectories used by the geometry-anchoring loss.
        with torch.no_grad():
            for index, motion in enumerate(self.motions):
                self.key_point_trajectories[motion] = [
                    self.renderer.deform_key_points(t, index).detach() for t in self.data.times
                ]

    def train_stage2(self):
        opt, g = self.opt, self.gaussians
        for _ in tqdm.trange(opt.iters_s2, desc="Stage 2"):
            self.train_step()
        g.save_checkpoint(os.path.join(self.save_dir, STAGE_2), STAGE_2, motions=self.motions)

    # ----------------------------------------------------------------------------- one step

    def sample_batch(self):
        """Random views / frames / motions for one step (paper Sec. 3.3).

        The counts are capped by what the dataset provides, so a 5-view dataset (SV4D 2.0 ``sv4d2``)
        trains with the same `batch_size` as a 9-view one.
        """
        opt = self.opt
        frames = random.sample(range(self.num_frames), min(opt.batch_size, self.num_frames))
        views = random.sample(range(self.num_views), min(opt.batch_size, self.num_views))
        num_motions = min(2 * opt.batch_size, len(self.motions))
        motion_indices = np.random.choice(len(self.motions), num_motions, replace=False)
        return views, frames, [(int(i), self.motions[i]) for i in motion_indices]

    def train_step(self):
        opt, g = self.opt, self.gaussians

        # Stage 1: periodically anneal the key points with farthest point sampling.
        if self.stage == STAGE_1 and self.step % opt.FPS_iter == 0:
            g.farthest_point_downsample(opt.num_cpts)

        self.step += 1
        g.update_learning_rate(self.step, self.stage, warmup_steps=opt.position_lr_warmup_s2 if self.stage == STAGE_2 else 0)
        if self.stage == STAGE_2:
            g.find_knn(k=opt.knn)

        resolution = resolution_for_step(self.step, opt.render_resolution_steps, opt.render_resolutions)
        views, frames, motions = self.sample_batch()

        loss = 0.0
        for latent_index, motion in motions:
            batch = self._render_batch(motion, latent_index, views, frames, resolution)
            loss = loss + batch["anchor_loss"]
            loss = loss + self._compute_losses(motion, latent_index, batch, views, frames)
            self._log_images(motion, latent_index, batch, views)
            out = batch["last_render"]

        loss.backward()
        g.optimizer.step()
        g.optimizer.zero_grad()

        if self.step % opt.save_inter == 0:
            g.save_checkpoint(os.path.join(self.save_dir, self.stage), self.stage, self.step, motions=self.motions)

        self._densify_and_prune(out)
        del loss, batch, out
        torch.cuda.empty_cache()

    def _render_batch(self, motion: str, latent_index: int, views, frames, resolution: int):
        """Render every (view, frame) pair of the batch for one motion and gather the targets."""
        opt = self.opt
        images, gt_images, masks, gt_masks, depths, normals = [], [], [], [], [], []
        anchor_loss = 0.0
        for view in views:
            camera = self.intrinsics.view(opt.elevation, self.data.azimuths[view], resolution)
            for frame in frames:
                gt_image, gt_mask = self.data.get(motion, view, frame, self.device)
                out = self.renderer.render(camera, time=self.data.times[frame], stage=self.stage, latent_index=latent_index)

                if self.stage == STAGE_2 and opt.add_ga:
                    anchor_loss = anchor_loss + self._anchor_loss(out["cpts_t"], self.key_point_trajectories[motion][frame])

                images.append(out["image"][None])
                masks.append(out["alpha"][None])
                depths.append(out["depth"][None])
                if out["normal"] is not None:
                    normals.append(out["normal"][None])
                gt_images.append(F.interpolate(gt_image, (resolution, resolution), mode="bilinear", align_corners=False))
                gt_masks.append(F.interpolate(gt_mask, (resolution, resolution), mode="bilinear", align_corners=False))

        if self.stage == STAGE_2 and opt.add_ga:
            self.writer.add_scalar(f"{self.stage}/{motion}/loss_ga", float(anchor_loss), self.step)
        return {
            "images": torch.cat(images), "gt_images": torch.cat(gt_images),
            "masks": torch.cat(masks), "gt_masks": torch.cat(gt_masks),
            "depths": torch.cat(depths), "normals": torch.cat(normals) if normals else None,
            "anchor_loss": anchor_loss, "last_render": out,
        }

    def _anchor_loss(self, key_points_t: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        """Keep the stage-2 key-point trajectory close to the stage-1 one (paper Sec. 3.3)."""
        if self.opt.ga_chamfer:
            return self.opt.lambda_ga1 * chamfer_forward(key_points_t, reference)
        return self.opt.lambda_ga2 * (key_points_t - reference).abs().mean()

    def _compute_losses(self, motion: str, latent_index: int, batch, views, frames) -> torch.Tensor:
        opt, g = self.opt, self.gaussians
        tag = f"{self.stage}/{motion}"
        log = lambda name, value: self.writer.add_scalar(f"{tag}/{name}", float(value), self.step)  # noqa: E731
        images, gt_images = batch["images"], batch["gt_images"]
        loss = 0.0

        # Pixel loss; the reference view / frame gets full weight, the others half.
        mse_total = 0.0
        num_frames = len(frames)
        for i, view in enumerate(views):
            for j, frame in enumerate(frames):
                mse = F.mse_loss(images[i * num_frames + j], gt_images[i * num_frames + j])
                weight = 1.0 if (view == 0 or frame == 0) else 0.5
                loss = loss + opt.lambda_mse * weight * mse
                mse_total = mse_total + mse.detach()
        mse_mean = mse_total / (len(views) * num_frames)
        log("loss_mse", mse_mean)
        log("psnr", 10 * torch.log10(1 / mse_mean))

        lpips_loss = self.lpips(images, gt_images).mean()
        ssim_loss = 1 - ssim(images, gt_images)
        mask_loss = F.mse_loss(batch["masks"], batch["gt_masks"])
        loss = loss + opt.lambda_lpips * lpips_loss + opt.lambda_ssim * ssim_loss + opt.lambda_mask * mask_loss
        log("loss_lpips", lpips_loss)
        log("loss_ssim", ssim_loss)
        log("loss_mask", mask_loss)

        kl = g.latents.kl_loss(latent_index)
        if kl is not None:
            loss = loss + opt.lambda_kl * kl
            log("loss_kl", kl)

        if opt.add_depth and self.step > opt.depth_reg_start_iter:
            depth_loss = edge_aware_depth_smoothness_loss(batch["depths"].permute(0, 2, 3, 1), images.permute(0, 2, 3, 1))
            loss = loss + opt.lambda_smooth * depth_loss
            log("loss_edge_aware_smooth", depth_loss)

        if opt.add_normal and batch["normals"] is not None and self.step > opt.normal_reg_start_iter:
            normal_loss = bilateral_normal_smoothness_loss(batch["normals"].permute(0, 2, 3, 1), images.permute(0, 2, 3, 1))
            loss = loss + opt.lambda_bilateral * normal_loss
            log("loss_bilateral_normal_smooth", normal_loss)

        use_arap = opt.use_arap and (
            (self.stage == STAGE_1 and self.step > opt.arap_start_iter_s1)
            or (self.stage == STAGE_2 and self.step < opt.arap_end_iter_s2)
        )
        if use_arap:
            arap, _ = self.renderer.arap_loss(stage=self.stage, latent_index=latent_index)
            loss = loss + opt.lambda_arap * arap
            log("loss_arap", arap)

        log("loss_total", loss)
        return loss

    def _log_images(self, motion: str, latent_index: int, batch, views):
        if self.step % self.opt.debug_image_interval == 0:
            path = os.path.join(self.save_dir, "debug", motion, f"image_{self.stage}_{self.step}.png")
            write_image(path, side_by_side(tensor_to_uint8(batch["gt_images"][0]), tensor_to_uint8(batch["images"][0])))
        if self.step % self.opt.tb_image_interval == 0:
            tag = f"{self.stage}/view{views[0]}_{motion}"
            self.writer.add_images(f"{tag}/gt", batch["gt_images"][:1], self.step)
            self.writer.add_images(f"{tag}/render", batch["images"][:1], self.step)
            self.writer.add_histogram(f"{self.stage}/{motion}/latent_code", self.gaussians.latents.numpy(latent_index), self.step)

    def _densify_and_prune(self, out):
        opt, g = self.opt, self.gaussians
        if self.stage == STAGE_1:
            # Densify between two farthest-point-sampling rounds; `out` is the last render of the step.
            if self.step % opt.FPS_iter >= opt.density_start_iter and self.step <= opt.density_end_iter:
                visible, radii = out["visibility_filter"], out["radii"]
                g.max_radii2D[visible] = torch.max(g.max_radii2D[visible], radii[visible])
                g.add_densification_stats(out["viewspace_points"], visible)
                if self.step % opt.densification_interval == 0:
                    g.densify_and_prune(opt.densify_grad_threshold, opt.densify_opacity_threshold_s1, SCENE_EXTENT, MAX_SCREEN_SIZE)
                    print(f"[INFO] {g.num_points} Gaussians after densification")
                if self.step % opt.opacity_reset_interval == 0:
                    g.reset_opacity()
        elif self.step < opt.density_end_iter_s2 and self.step % opt.densification_interval_s2 == 0 and opt.init_type == "ag":
            g.prune(opt.densify_opacity_threshold_s2, SCENE_EXTENT, MAX_SCREEN_SIZE)
            print(f"[INFO] {g.num_points} Gaussians after pruning")
