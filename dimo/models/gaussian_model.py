"""Canonical 3D Gaussians + sparse key points + motion decoder.

Tensor naming follows the 3D Gaussian Splatting convention:

* ``_xyz, _features_dc, _features_rest, _scaling, _rotation, _opacity`` are the per-Gaussian
  parameters (positions, SH colour, log-scales, quaternions, opacity logits).
* ``_c_xyz`` / ``_c_radius`` are the key points (control nodes) and their log-radii. They only
  exist from stage 2 on; in stage 1 the Gaussians themselves play the role of key points.
* ``_r`` is a single shared log-radius used in stage 1 so that all key points are isotropic
  spheres of the same size.
"""

import json
import os
from typing import Dict, Iterable, List, Optional, Set, Tuple

import numpy as np
import torch
from plyfile import PlyData, PlyElement
from pytorch3d.ops import knn_points
from torch import nn

from dimo.models.deform_net import DeformNet
from dimo.models.latent import LatentCodes, build_latent_codes
from dimo.utils.general import (
    BasicPointCloud,
    build_rotation,
    build_scaling_rotation,
    get_expon_lr_func,
    inverse_sigmoid,
    strip_symmetric,
)
from dimo.utils.io import ensure_dir, resolve_step_path
from dimo.utils.sh import RGB2SH

STAGE_1, STAGE_2 = "s1", "s2"
PLY_NAME, KEY_POINT_PLY_NAME, DEFORM_NET_NAME = "point_cloud.ply", "point_cloud_c.ply", "timenet.pth"
MOTION_ORDER_NAME = "motion_order.json"
TEXT_PROJECTOR_NAME = "mlp_encoder.pth"


def mean_knn_sq_distance(points: torch.Tensor, k: int = 3) -> torch.Tensor:
    """Mean squared distance of every point to its ``k`` nearest neighbours (excluding itself)."""
    dists = knn_points(points[None], points[None], K=k + 1).dists[0, :, 1:]
    return dists.mean(dim=1)


class GaussianModel:
    # Parameter groups that hold one row per Gaussian and must be pruned / grown together.
    PER_POINT_GROUPS = ("xyz", "f_dc", "f_rest", "opacity", "scaling", "rotation")

    def __init__(self, sh_degree: int, num_latent_codes: int = 1, latent_dim: int = 32,
                 vae_latent: bool = False, device: str = "cuda"):
        self.device = device
        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree

        empty = lambda: torch.empty(0, device=device)  # noqa: E731
        self._xyz = empty()
        self._features_dc = empty()
        self._features_rest = empty()
        self._scaling = empty()
        self._rotation = empty()
        self._opacity = empty()
        self.max_radii2D = empty()
        self.xyz_gradient_accum = empty()
        self.denom = empty()

        self._c_xyz = empty()
        self._c_radius = empty()
        self._r = empty()
        self.neighbor_indices: Optional[torch.Tensor] = None
        self.neighbor_dists: Optional[torch.Tensor] = None

        self.latents: LatentCodes = build_latent_codes(num_latent_codes, latent_dim, vae_latent, device)
        self.deform_net = DeformNet(latent_dim=latent_dim).to(device)

        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.trainable_groups: Optional[Set[str]] = None
        self.percent_dense = 0.0

        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log
        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid
        self.rotation_activation = torch.nn.functional.normalize

    # ----------------------------------------------------------------------------- properties

    @property
    def num_points(self) -> int:
        return self._xyz.shape[0]

    @property
    def num_key_points(self) -> int:
        return self._c_xyz.shape[0]

    @property
    def has_shared_radius(self) -> bool:
        return self._r.numel() == 1

    @property
    def get_xyz(self) -> torch.Tensor:
        return self._xyz

    @property
    def get_scaling(self) -> torch.Tensor:
        if self.has_shared_radius:
            return self.scaling_activation(self._r.reshape(1, 1).expand(self.num_points, 3))
        return self.scaling_activation(self._scaling)

    @property
    def get_rotation(self) -> torch.Tensor:
        return self.rotation_activation(self._rotation)

    @property
    def get_opacity(self) -> torch.Tensor:
        return self.opacity_activation(self._opacity)

    @property
    def get_features(self) -> torch.Tensor:
        return torch.cat((self._features_dc, self._features_rest), dim=1)

    @property
    def get_c_xyz(self) -> torch.Tensor:
        return self._c_xyz

    @property
    def get_c_radius(self) -> torch.Tensor:
        return torch.exp(self._c_radius)

    def get_covariance(self, scaling_modifier: float = 1.0) -> torch.Tensor:
        L = build_scaling_rotation(scaling_modifier * self.get_scaling, self._rotation)
        return strip_symmetric(L @ L.transpose(1, 2))

    # ----------------------------------------------------------------------------- motion

    def deform(self, points: torch.Tensor, t, latent_index: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
        """Translation and rotation of ``points`` at time ``t`` for the given motion."""
        return self.deform_net(points, t, self.latents(latent_index))

    def train(self, mode: bool = True):
        self.latents.train(mode)
        self.deform_net.train(mode)
        return self

    def eval(self):
        return self.train(False)

    # ----------------------------------------------------------------------------- initialisation

    def create_from_pcd(self, pcd: BasicPointCloud):
        """(Re-)initialise the Gaussians from a point cloud; key points and motion are untouched."""
        device = self.device
        xyz = torch.tensor(np.asarray(pcd.points)).float().to(device)
        color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().to(device))
        features = torch.zeros((xyz.shape[0], 3, (self.max_sh_degree + 1) ** 2), device=device)
        features[:, :3, 0] = color
        print(f"[INFO] initialised {xyz.shape[0]} Gaussians")

        dist2 = torch.clamp_min(mean_knn_sq_distance(xyz), 1e-7)
        scales = torch.log(torch.sqrt(dist2))[..., None].repeat(1, 3)
        rots = torch.zeros((xyz.shape[0], 4), device=device)
        rots[:, 0] = 1
        opacities = inverse_sigmoid(0.05 * torch.ones((xyz.shape[0], 1), device=device))

        self._xyz = nn.Parameter(xyz.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:, :, 0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:, :, 1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((xyz.shape[0]), device=device)

    def init_shared_radius(self):
        """Stage 1: one shared log-radius for all key-point Gaussians, initialised to the mean scale."""
        r = self._scaling.detach().mean() * torch.ones((1, 1), device=self.device)
        self._r = nn.Parameter(r.requires_grad_(True))

    def clear_shared_radius(self):
        self._r = torch.empty(0, device=self.device)

    def set_key_points(self, c_xyz: torch.Tensor, c_log_radius: torch.Tensor):
        self._c_xyz = nn.Parameter(c_xyz.detach().clone().to(self.device).requires_grad_(True))
        self._c_radius = nn.Parameter(c_log_radius.detach().clone().to(self.device).requires_grad_(True))

    def set_points_for_visualisation(self, xyz: torch.Tensor, log_radius: float = -5.0, opacity_logit: float = 2.0):
        """Turn the model into small uniform spheres at ``xyz`` (used to render key points)."""
        n = xyz.shape[0]
        device = self.device
        self._xyz = xyz.detach().clone()
        self._features_dc = torch.zeros((n, 1, 3), device=device)
        self._features_rest = torch.zeros((n, (self.max_sh_degree + 1) ** 2 - 1, 3), device=device)
        self._scaling = torch.full((n, 3), log_radius, device=device)
        self._rotation = torch.zeros((n, 4), device=device)
        self._rotation[:, 0] = 1
        self._opacity = torch.full((n, 1), opacity_logit, device=device)
        self._r = torch.full((1, 1), log_radius, device=device)
        self.max_radii2D = torch.zeros((n,), device=device)
        self.active_sh_degree = self.max_sh_degree

    def find_knn(self, k: int = 4):
        """Cache, for every Gaussian, its ``k`` nearest key points (needed by stage-2 rendering)."""
        dists, idx, _ = knn_points(self._xyz.detach()[None], self._c_xyz.detach()[None], K=k)
        self.neighbor_dists = dists[0].sqrt()
        self.neighbor_indices = idx[0]

    # ----------------------------------------------------------------------------- optimisation

    def training_setup(self, opt, stage: str, max_steps: Optional[int] = None):
        """Build the optimiser and learning-rate schedules for ``stage`` (``"s1"`` or ``"s2"``)."""
        self.percent_dense = opt.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.num_points, 1), device=self.device)
        self.denom = torch.zeros((self.num_points, 1), device=self.device)
        self.trainable_groups = None

        if stage == STAGE_1:
            pos_init, pos_final = opt.position_lr_init, opt.position_lr_final
            max_steps = max_steps or opt.position_lr_max_steps_s1
        else:
            pos_init, pos_final = opt.position_lr_init_s2, opt.position_lr_final_s2
            max_steps = max_steps or opt.iters_s2
        self._stage_pos_lr_init = pos_init

        groups = [
            {"params": [self._xyz], "lr": pos_init, "name": "xyz"},
            {"params": [self._features_dc], "lr": opt.feature_lr, "name": "f_dc"},
            {"params": [self._features_rest], "lr": opt.feature_lr / 20.0, "name": "f_rest"},
            {"params": [self._opacity], "lr": opt.opacity_lr, "name": "opacity"},
            {"params": [self._scaling], "lr": opt.scaling_lr, "name": "scaling"},
            {"params": [self._rotation], "lr": opt.rotation_lr, "name": "rotation"},
            *self.latents.param_groups(opt.latent_code_lr_init),
            {"params": self.deform_net.translation_parameters(), "lr": opt.deform_lr_init, "name": "deform"},
            {"params": self.deform_net.rotation_parameters(), "lr": opt.deform_lr_init, "name": "deform_rot"},
        ]
        if self.has_shared_radius and isinstance(self._r, nn.Parameter):
            groups.append({"params": [self._r], "lr": opt.r_lr, "name": "r"})
        if self.num_key_points > 0 and isinstance(self._c_xyz, nn.Parameter):
            groups.append({"params": [self._c_xyz], "lr": opt.c_position_lr_init, "name": "c_xyz"})
            groups.append({"params": [self._c_radius], "lr": opt.c_radius_lr, "name": "c_radius"})
        self.optimizer = torch.optim.Adam(groups, lr=0.0, eps=1e-15)

        self.xyz_scheduler = get_expon_lr_func(pos_init, pos_final, lr_delay_mult=opt.position_lr_delay_mult, max_steps=max_steps)
        self.c_xyz_scheduler = get_expon_lr_func(opt.c_position_lr_init, opt.c_position_lr_final, lr_delay_mult=opt.c_position_lr_delay_mult, max_steps=max_steps)
        self.latent_scheduler = get_expon_lr_func(opt.latent_code_lr_init, opt.latent_code_lr_final, lr_delay_mult=opt.position_lr_delay_mult, max_steps=max_steps)
        self.deform_scheduler = get_expon_lr_func(opt.deform_lr_init, opt.deform_lr_final, lr_delay_mult=opt.position_lr_delay_mult, max_steps=max_steps)

    def set_trainable_groups(self, names: Optional[Iterable[str]]):
        """Restrict optimisation to the given parameter groups (``None`` = all)."""
        self.trainable_groups = None if names is None else set(names)
        if self.trainable_groups is not None:
            for group in self.optimizer.param_groups:
                if group["name"] not in self.trainable_groups:
                    group["lr"] = 0.0

    def update_learning_rate(self, step: int, stage: str, warmup_steps: int = 0):
        """Per-step learning-rate schedule. Motion parameters are only scheduled from stage 2 on."""
        scheduled = {"xyz": self.xyz_scheduler}
        if stage == STAGE_2:
            scheduled.update({
                "c_xyz": self.c_xyz_scheduler,
                "deform": self.deform_scheduler,
                "deform_rot": self.deform_scheduler,
                **{name: self.latent_scheduler for name in self.latents.GROUP_NAMES},
            })
        for group in self.optimizer.param_groups:
            name = group["name"]
            if self.trainable_groups is not None and name not in self.trainable_groups:
                group["lr"] = 0.0
            elif name in scheduled:
                if name == "xyz" and step < warmup_steps:
                    group["lr"] = self._stage_pos_lr_init
                else:
                    group["lr"] = scheduled[name](step)

    # ----------------------------------------------------------------------------- checkpoints

    def _ply_attributes(self):
        names = ["x", "y", "z", "nx", "ny", "nz"]
        names += [f"f_dc_{i}" for i in range(self._features_dc.shape[1] * self._features_dc.shape[2])]
        names += [f"f_rest_{i}" for i in range(self._features_rest.shape[1] * self._features_rest.shape[2])]
        names.append("opacity")
        names += [f"scale_{i}" for i in range(3)]
        names += [f"rot_{i}" for i in range(self._rotation.shape[1])]
        return names

    @torch.no_grad()
    def save_ply(self, path: str, key_point_path: Optional[str] = None):
        ensure_dir(os.path.dirname(path))
        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = torch.log(self.get_scaling).detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        dtype = [(name, "f4") for name in self._ply_attributes()]
        elements = np.empty(xyz.shape[0], dtype=dtype)
        elements[:] = list(map(tuple, np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)))
        PlyData([PlyElement.describe(elements, "vertex")]).write(path)

        if key_point_path is not None:
            ensure_dir(os.path.dirname(key_point_path))
            c_xyz = self._c_xyz.detach().cpu().numpy()
            c_radius = self._c_radius.detach().cpu().numpy()
            dtype = [(name, "f4") for name in ("c_x", "c_y", "c_z", "c_radius")]
            elements = np.empty(c_xyz.shape[0], dtype=dtype)
            elements[:] = list(map(tuple, np.concatenate((c_xyz, c_radius), axis=1)))
            PlyData([PlyElement.describe(elements, "vertex")]).write(key_point_path)

    def load_ply(self, path: str, key_point_path: Optional[str] = None):
        vertex = PlyData.read(path).elements[0]
        xyz = np.stack((np.asarray(vertex["x"]), np.asarray(vertex["y"]), np.asarray(vertex["z"])), axis=1)
        opacities = np.asarray(vertex["opacity"])[..., np.newaxis]
        print(f"[INFO] loaded {xyz.shape[0]} Gaussians from {path}")

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        for i in range(3):
            features_dc[:, i, 0] = np.asarray(vertex[f"f_dc_{i}"])
        rest_names = sorted((p.name for p in vertex.properties if p.name.startswith("f_rest_")), key=lambda n: int(n.split("_")[-1]))
        assert len(rest_names) == 3 * (self.max_sh_degree + 1) ** 2 - 3, "SH degree of the checkpoint does not match"
        features_rest = np.stack([np.asarray(vertex[n]) for n in rest_names], axis=1) if rest_names else np.zeros((xyz.shape[0], 0))
        features_rest = features_rest.reshape((xyz.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))
        scales = np.stack([np.asarray(vertex[f"scale_{i}"]) for i in range(3)], axis=1)
        rot_names = [p.name for p in vertex.properties if p.name.startswith("rot")]
        rots = np.stack([np.asarray(vertex[n]) for n in rot_names], axis=1)

        to_param = lambda a: nn.Parameter(torch.tensor(a, dtype=torch.float, device=self.device).requires_grad_(True))  # noqa: E731
        self._xyz = to_param(xyz)
        self._features_dc = to_param(features_dc.transpose(0, 2, 1).copy())
        self._features_rest = to_param(features_rest.transpose(0, 2, 1).copy())
        self._opacity = to_param(opacities)
        self._scaling = to_param(scales)
        self._rotation = to_param(rots)
        self.max_radii2D = torch.zeros((xyz.shape[0]), device=self.device)
        self.active_sh_degree = self.max_sh_degree
        self.clear_shared_radius()

        if key_point_path is not None:
            vertex = PlyData.read(key_point_path).elements[0]
            c_xyz = np.stack((np.asarray(vertex["c_x"]), np.asarray(vertex["c_y"]), np.asarray(vertex["c_z"])), axis=1)
            c_radius = np.asarray(vertex["c_radius"])[..., np.newaxis]
            print(f"[INFO] loaded {c_xyz.shape[0]} key points from {key_point_path}")
            self._c_xyz = to_param(c_xyz)
            self._c_radius = to_param(c_radius)

    @torch.no_grad()
    def save_model(self, directory: str, step: Optional[int] = None):
        """Save the latent codes and the motion decoder."""
        ensure_dir(directory)
        self.latents.save(directory, step)
        torch.save(self.deform_net.state_dict(), resolve_step_path(os.path.join(directory, DEFORM_NET_NAME), step))

    def load_model(self, directory: str, step: Optional[int] = None):
        print(f"[INFO] loading motion model from {directory}")
        self.latents.load(directory, step)
        state = torch.load(resolve_step_path(os.path.join(directory, DEFORM_NET_NAME), step), map_location=self.device)
        self.deform_net.load_state_dict(state)
        self.deform_net.to(self.device)

    def save_checkpoint(self, directory: str, stage: str, step: Optional[int] = None,
                        motions: Optional[List[str]] = None):
        """Gaussians (+ key points from stage 2), latent codes and motion decoder.

        ``motions`` records which motion each row of the latent table belongs to. Row *i* means
        ``motions[i]`` and nothing else, so anything that reads the codes back (the text projector,
        a resumed run) can check the mapping instead of assuming the dataset's order is unchanged.
        """
        key_point_path = resolve_step_path(os.path.join(directory, KEY_POINT_PLY_NAME), step) if stage == STAGE_2 else None
        self.save_ply(resolve_step_path(os.path.join(directory, PLY_NAME), step), key_point_path)
        self.save_model(directory, step)
        if motions is not None:
            ensure_dir(directory)
            with open(os.path.join(directory, MOTION_ORDER_NAME), "w") as f:
                json.dump(list(motions), f, indent=2)

    def load_checkpoint(self, directory: str, stage: str, step: Optional[int] = None):
        key_point_path = resolve_step_path(os.path.join(directory, KEY_POINT_PLY_NAME), step) if stage == STAGE_2 else None
        self.load_ply(resolve_step_path(os.path.join(directory, PLY_NAME), step), key_point_path)
        self.load_model(directory, step)

    # ----------------------------------------------------------------------------- densification

    def _per_point_groups(self):
        for group in self.optimizer.param_groups:
            if group["name"] in self.PER_POINT_GROUPS:
                yield group

    def _replace_tensor_in_optimizer(self, tensor: torch.Tensor, name: str) -> torch.Tensor:
        for group in self.optimizer.param_groups:
            if group["name"] != name:
                continue
            stored_state = self.optimizer.state.get(group["params"][0], None)
            if stored_state is not None:
                # `state` is a defaultdict, so `.get` does not insert; deleting unconditionally
                # would raise KeyError for a parameter Adam has not stepped yet.
                del self.optimizer.state[group["params"][0]]
            group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
            if stored_state is not None:
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)
                self.optimizer.state[group["params"][0]] = stored_state
            return group["params"][0]
        raise KeyError(name)

    def _prune_optimizer(self, mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Keep only the rows selected by ``mask`` in every per-point parameter (and its Adam state)."""
        tensors = {}
        for group in self._per_point_groups():
            stored_state = self.optimizer.state.get(group["params"][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]
                del self.optimizer.state[group["params"][0]]
            group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
            if stored_state is not None:
                self.optimizer.state[group["params"][0]] = stored_state
            tensors[group["name"]] = group["params"][0]
        return tensors

    def _cat_tensors_to_optimizer(self, extensions: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        tensors = {}
        for group in self._per_point_groups():
            extension = extensions[group["name"]]
            stored_state = self.optimizer.state.get(group["params"][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension)), dim=0)
                del self.optimizer.state[group["params"][0]]
            group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension), dim=0).requires_grad_(True))
            if stored_state is not None:
                self.optimizer.state[group["params"][0]] = stored_state
            tensors[group["name"]] = group["params"][0]
        return tensors

    def _assign(self, tensors: Dict[str, torch.Tensor]):
        self._xyz = tensors["xyz"]
        self._features_dc = tensors["f_dc"]
        self._features_rest = tensors["f_rest"]
        self._opacity = tensors["opacity"]
        self._scaling = tensors["scaling"]
        self._rotation = tensors["rotation"]

    def prune_points(self, remove_mask: torch.Tensor):
        """Remove the Gaussians flagged in the boolean ``remove_mask``."""
        keep = ~remove_mask
        self._assign(self._prune_optimizer(keep))
        self.xyz_gradient_accum = self.xyz_gradient_accum[keep]
        self.denom = self.denom[keep]
        self.max_radii2D = self.max_radii2D[keep]

    def keep_points(self, indices: torch.Tensor):
        """Keep only the Gaussians with the given indices."""
        remove = torch.ones(self.num_points, dtype=torch.bool, device=self.device)
        remove[indices] = False
        self.prune_points(remove)

    def farthest_point_downsample(self, num_points: int):
        """Sub-sample the Gaussians with farthest point sampling (stage-1 key-point annealing)."""
        from pytorch3d.ops import sample_farthest_points

        if self.num_points <= num_points:
            return
        _, idx = sample_farthest_points(points=self._xyz.detach()[None], K=num_points)
        self.keep_points(idx[0])

    def _densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation):
        self._assign(self._cat_tensors_to_optimizer({
            "xyz": new_xyz,
            "f_dc": new_features_dc,
            "f_rest": new_features_rest,
            "opacity": new_opacity,
            "scaling": new_scaling,
            "rotation": new_rotation,
        }))
        self.xyz_gradient_accum = torch.zeros((self.num_points, 1), device=self.device)
        self.denom = torch.zeros((self.num_points, 1), device=self.device)
        self.max_radii2D = torch.zeros((self.num_points), device=self.device)

    def densify_and_split(self, grads, grad_threshold, scene_extent, N: int = 2):
        n_init_points = self.num_points
        padded_grad = torch.zeros((n_init_points), device=self.device)
        padded_grad[: grads.shape[0]] = grads.squeeze()
        selected = padded_grad >= grad_threshold
        selected = torch.logical_and(selected, torch.max(self.get_scaling, dim=1).values > self.percent_dense * scene_extent)

        stds = self.get_scaling[selected].repeat(N, 1)
        samples = torch.normal(mean=torch.zeros((stds.size(0), 3), device=self.device), std=stds)
        rots = build_rotation(self._rotation[selected]).repeat(N, 1, 1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected].repeat(N, 1) / (0.8 * N))
        self._densification_postfix(
            new_xyz,
            self._features_dc[selected].repeat(N, 1, 1),
            self._features_rest[selected].repeat(N, 1, 1),
            self._opacity[selected].repeat(N, 1),
            new_scaling,
            self._rotation[selected].repeat(N, 1),
        )
        prune_filter = torch.cat((selected, torch.zeros(N * selected.sum(), device=self.device, dtype=torch.bool)))
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        selected = torch.norm(grads, dim=-1) >= grad_threshold
        selected = torch.logical_and(selected, torch.max(self.get_scaling, dim=1).values <= self.percent_dense * scene_extent)
        self._densification_postfix(
            self._xyz[selected],
            self._features_dc[selected],
            self._features_rest[selected],
            self._opacity[selected],
            self._scaling[selected],
            self._rotation[selected],
        )

    def _prune_mask(self, min_opacity, extent, max_screen_size) -> torch.Tensor:
        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        return prune_mask

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0
        self.densify_and_clone(grads, max_grad, extent)
        self.densify_and_split(grads, max_grad, extent)
        self.prune_points(self._prune_mask(min_opacity, extent, max_screen_size))
        torch.cuda.empty_cache()

    def prune(self, min_opacity, extent, max_screen_size=None):
        self.prune_points(self._prune_mask(min_opacity, extent, max_screen_size))
        torch.cuda.empty_cache()

    def reset_opacity(self):
        opacities_new = inverse_sigmoid(torch.min(self.get_opacity, torch.ones_like(self.get_opacity) * 0.01))
        self._opacity = self._replace_tensor_in_optimizer(opacities_new, "opacity")

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter, :2], dim=-1, keepdim=True)
        self.denom[update_filter] += 1
