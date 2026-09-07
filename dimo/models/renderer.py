"""Differentiable rendering of the deformed Gaussians at a given time and motion."""

import math
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F

from dimo.cameras import Camera
from dimo.losses.arap import arap_loss
from dimo.models.gaussian_model import STAGE_1, STAGE_2, GaussianModel
from dimo.utils.general import BasicPointCloud, build_rotation, quat_mul, random_point_cloud, sample_ball
from dimo.utils.sh import SH2RGB


class Renderer:
    """Owns a :class:`GaussianModel` and rasterises it.

    Stage 1 deforms every Gaussian directly with the motion decoder. Stage 2 deforms the key
    points and transfers their rigid motion to each Gaussian by blending its ``k`` nearest key
    points (weights fall off with the key-point radius).
    """

    def __init__(self, sh_degree: int = 0, white_background: bool = True, num_latent_codes: int = 1,
                 latent_dim: int = 32, add_normal: bool = False, vae_latent: bool = False, device: str = "cuda"):
        self.sh_degree = sh_degree
        self.add_normal = add_normal
        self.device = device
        self.gaussians = GaussianModel(sh_degree, num_latent_codes, latent_dim, vae_latent=vae_latent, device=device)
        self.bg_color = torch.tensor([1, 1, 1] if white_background else [0, 0, 0], dtype=torch.float32, device=device)

        if add_normal:
            from diff_gauss import GaussianRasterizationSettings, GaussianRasterizer
        else:
            from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
        self._settings_cls = GaussianRasterizationSettings
        self._rasterizer_cls = GaussianRasterizer

    # ----------------------------------------------------------------------------- initialisation

    def initialize(self, num_points: int, radius: float = 0.5):
        """Random Gaussians inside a ball (stage-1 start), sharing one radius parameter."""
        self.gaussians.create_from_pcd(random_point_cloud(num_points, radius))
        self.gaussians.init_shared_radius()

    def initialize_from_pcd(self, pcd: BasicPointCloud):
        self.gaussians.create_from_pcd(pcd)

    def initialize_around_key_points(self, c_xyz: torch.Tensor, c_radius: torch.Tensor,
                                     num_points_per_key_point: int = 200, init_ratio: float = 1.0):
        """Adaptive Gaussian initialisation: spawn ``num_points_per_key_point`` Gaussians in a ball
        around every key point, with radius ``mean(c_radius) * init_ratio``."""
        num_key_points = c_xyz.shape[0]
        offsets = sample_ball(num_points_per_key_point, c_radius.mean().item() * init_ratio)  # (P, 3)
        offsets = torch.tensor(offsets)[None].repeat(num_key_points, 1, 1).flatten(0, 1)
        centers = c_xyz.detach().cpu()[:, None].repeat(1, num_points_per_key_point, 1).flatten(0, 1)
        xyz = (offsets + centers).numpy()
        n = xyz.shape[0]
        shs = np.random.random((n, 3)) / 255.0
        self.gaussians.create_from_pcd(BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((n, 3))))

    def key_point_renderer(self, xyz: Optional[torch.Tensor] = None, log_radius: float = -5.0) -> "Renderer":
        """A renderer that draws the key points (or ``xyz``) as small dark spheres, sharing this
        renderer's motion decoder and latent codes."""
        if xyz is None:
            xyz = self.gaussians._c_xyz if self.gaussians.num_key_points > 0 else self.gaussians._xyz
        other = Renderer(self.sh_degree, add_normal=self.add_normal, device=self.device)
        other.gaussians.set_points_for_visualisation(xyz, log_radius=log_radius)
        other.gaussians.latents = self.gaussians.latents
        other.gaussians.deform_net = self.gaussians.deform_net
        return other

    # ----------------------------------------------------------------------------- regularisers

    def arap_loss(self, stage: str, latent_index: int = 0, num_times: int = 8, K: int = 10, radius: float = 0.1):
        """ARAP energy of the key-point trajectories sampled at ``num_times`` random time stamps."""
        g = self.gaussians
        nodes = (g._xyz if stage == STAGE_1 else g._c_xyz)[None]  # (1, M, 3)
        times = torch.rand(num_times, device=nodes.device)[:, None, None].expand(num_times, nodes.shape[1], 1)
        translation, _ = g.deform_net(nodes, times, g.latents(latent_index))  # (T, M, 3)
        nodes_t = nodes.expand(num_times, -1, -1).detach() + translation
        return arap_loss(nodes_t, K=K, radius=radius)

    # ----------------------------------------------------------------------------- rendering

    def deform_key_points(self, time: float, latent_index: int = 0):
        """Deformed key-point positions ``(M, 3)`` at ``time`` (stage-2 models)."""
        g = self.gaussians
        translation, _ = g.deform(g._c_xyz, time, latent_index)
        return g._c_xyz + translation

    def render(self, camera: Camera, time: float = 0.0, stage: str = STAGE_1, latent_index: int = 0,
               bg_color: Optional[torch.Tensor] = None, override_color: Optional[torch.Tensor] = None,
               scaling_modifier: float = 1.0) -> Dict[str, torch.Tensor]:
        g = self.gaussians

        # Screen-space positions; their gradients drive densification.
        screenspace_points = torch.zeros_like(g.get_xyz, requires_grad=True) + 0
        try:
            screenspace_points.retain_grad()
        except Exception:
            pass

        raster_settings = self._settings_cls(
            image_height=camera.image_height,
            image_width=camera.image_width,
            tanfovx=math.tan(camera.FoVx * 0.5),
            tanfovy=math.tan(camera.FoVy * 0.5),
            bg=self.bg_color if bg_color is None else bg_color,
            scale_modifier=scaling_modifier,
            viewmatrix=camera.world_view_transform,
            projmatrix=camera.full_proj_transform,
            sh_degree=g.active_sh_degree,
            campos=camera.camera_center,
            prefiltered=False,
            debug=False,
        )
        rasterizer = self._rasterizer_cls(raster_settings=raster_settings)

        means3D = g.get_xyz
        latent = g.latents(latent_index)
        if stage == STAGE_2:
            c_xyz = g.get_c_xyz
            c_translation, c_rotation = g.deform_net(c_xyz, time, latent)
            key_points_t = c_xyz + c_translation
            means3D, rotations = self._blend_key_point_motion(means3D, c_xyz, c_translation, c_rotation)
        elif stage == STAGE_1:
            translation, _ = g.deform_net(means3D, time, latent)
            means3D = means3D + translation
            key_points_t = means3D
            rotations = g._rotation
        else:
            raise ValueError(f"Unknown stage {stage!r}")
        rotations = g.rotation_activation(rotations)

        shs, colors_precomp = (g.get_features, None) if override_color is None else (None, override_color)
        common = dict(
            means3D=means3D,
            means2D=screenspace_points,
            shs=shs,
            colors_precomp=colors_precomp,
            opacities=g.get_opacity,
            scales=g.get_scaling,
            rotations=rotations,
        )
        if self.add_normal:
            image, depth, normal, alpha, radii, _ = rasterizer(**common, cov3Ds_precomp=None, extra_attrs=None)
        else:
            image, radii, depth, alpha = rasterizer(**common, cov3D_precomp=None)
            normal = None

        return {
            "image": image.clamp(0, 1),
            "depth": depth,
            "normal": normal,
            "alpha": alpha,
            "viewspace_points": screenspace_points,
            "visibility_filter": radii > 0,
            "radii": radii,
            "pts_t": means3D,
            "cpts_t": key_points_t,
        }

    def _blend_key_point_motion(self, means3D, c_xyz, c_translation, c_rotation, eps: float = 1e-7):
        """Move each Gaussian with the rigid motions of its cached nearest key points."""
        g = self.gaussians
        if g.neighbor_indices is None:
            raise RuntimeError("call GaussianModel.find_knn() before rendering in stage 2")
        idx, dists = g.neighbor_indices, g.neighbor_dists  # (N, k)
        radius_n = g.get_c_radius[idx][:, :, 0]  # (N, k)
        w = torch.exp(-(dists**2) / (2.0 * radius_n**2)) + eps
        w = F.normalize(w, p=1)[..., None]  # (N, k, 1)

        c_n = c_xyz[idx]  # (N, k, 3)
        t_n = c_translation[idx]  # (N, k, 3)
        q_n = c_rotation[idx]  # (N, k, 4)
        local = (build_rotation(q_n) @ (means3D[:, None] - c_n)[..., None]).squeeze(-1)  # rotate in the key point's frame
        positions = (w * (local + c_n + t_n)).sum(dim=1)
        rotations = quat_mul((w * q_n).sum(dim=1), g._rotation)
        return positions, rotations
