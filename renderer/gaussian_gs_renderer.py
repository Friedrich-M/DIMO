import os
import math
import numpy as np
from typing import NamedTuple
from plyfile import PlyData, PlyElement
import torch
from torch import nn
import torch.nn.init as init
from diff_gaussian_rasterization import (
    GaussianRasterizationSettings,
    GaussianRasterizer,
)
from diff_gauss import (
    GaussianRasterizationSettings as GaussianRasterizationSettingsNormal,
    GaussianRasterizer as GaussianRasterizerNormal,
)
from simple_knn._C import distCUDA2
from utils.sh_utils import eval_sh, SH2RGB, RGB2SH
from src.helpers import o3d_knn
from src.pos_enc import get_embedder
import torch.nn.functional as F
from utils.deform_utils import cal_connectivity_from_points, cal_connectivity_from_points_v2, cal_arap_error, arap_deformation_loss
import pytorch3d
from pytorch3d.transforms import quaternion_to_matrix

def inverse_sigmoid(x):
    return torch.log(x/(1-x))

def get_expon_lr_func(
    lr_init, lr_final, lr_delay_steps=0, lr_delay_mult=1.0, max_steps=1000000
):
    
    def helper(step):
        if lr_init == lr_final:
            # constant lr, ignore other params
            return lr_init
        if step < 0 or (lr_init == 0.0 and lr_final == 0.0):
            # Disable this parameter
            return 0.0
        if lr_delay_steps > 0:
            # A kind of reverse cosine decay.
            delay_rate = lr_delay_mult + (1 - lr_delay_mult) * np.sin(
                0.5 * np.pi * np.clip(step / lr_delay_steps, 0, 1)
            )
        else:
            delay_rate = 1.0
        t = np.clip(step / max_steps, 0, 1)
        log_lerp = np.exp(np.log(lr_init) * (1 - t) + np.log(lr_final) * t)
        return delay_rate * log_lerp

    return helper


def strip_lowerdiag(L):
    uncertainty = torch.zeros((L.shape[0], 6), dtype=torch.float, device="cuda")

    uncertainty[:, 0] = L[:, 0, 0]
    uncertainty[:, 1] = L[:, 0, 1]
    uncertainty[:, 2] = L[:, 0, 2]
    uncertainty[:, 3] = L[:, 1, 1]
    uncertainty[:, 4] = L[:, 1, 2]
    uncertainty[:, 5] = L[:, 2, 2]
    return uncertainty

def strip_symmetric(sym):
    return strip_lowerdiag(sym)

def gaussian_3d_coeff(xyzs, covs):
    # xyzs: [N, 3]
    # covs: [N, 6]
    x, y, z = xyzs[:, 0], xyzs[:, 1], xyzs[:, 2]
    a, b, c, d, e, f = covs[:, 0], covs[:, 1], covs[:, 2], covs[:, 3], covs[:, 4], covs[:, 5]

    # eps must be small enough !!!
    inv_det = 1 / (a * d * f + 2 * e * c * b - e**2 * a - c**2 * d - b**2 * f + 1e-24)
    inv_a = (d * f - e**2) * inv_det
    inv_b = (e * c - b * f) * inv_det
    inv_c = (e * b - c * d) * inv_det
    inv_d = (a * f - c**2) * inv_det
    inv_e = (b * c - e * a) * inv_det
    inv_f = (a * d - b**2) * inv_det

    power = -0.5 * (x**2 * inv_a + y**2 * inv_d + z**2 * inv_f) - x * y * inv_b - x * z * inv_c - y * z * inv_e

    power[power > 0] = -1e10 # abnormal values... make weights 0
        
    return torch.exp(power)

def build_rotation(r):
    norm = torch.sqrt(r[:,0]*r[:,0] + r[:,1]*r[:,1] + r[:,2]*r[:,2] + r[:,3]*r[:,3])

    q = r / norm[:, None]

    R = torch.zeros((q.size(0), 3, 3), device='cuda')

    r = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]

    R[:, 0, 0] = 1 - 2 * (y*y + z*z)
    R[:, 0, 1] = 2 * (x*y - r*z)
    R[:, 0, 2] = 2 * (x*z + r*y)
    R[:, 1, 0] = 2 * (x*y + r*z)
    R[:, 1, 1] = 1 - 2 * (x*x + z*z)
    R[:, 1, 2] = 2 * (y*z - r*x)
    R[:, 2, 0] = 2 * (x*z - r*y)
    R[:, 2, 1] = 2 * (y*z + r*x)
    R[:, 2, 2] = 1 - 2 * (x*x + y*y)
    return R

def build_rotation_3d(r):
    norm = torch.sqrt(r[:,:,0]*r[:,:,0] + r[:,:,1]*r[:,:,1] + r[:,:,2]*r[:,:,2] + r[:,:,3]*r[:,:,3])

    q = r / norm[:, :, None]

    R = torch.zeros((q.size(0), q.size(1), 3, 3), device='cuda')

    r = q[:, :, 0]
    x = q[:, :, 1]
    y = q[:, :, 2]
    z = q[:, :, 3]

    R[:, :, 0, 0] = 1 - 2 * (y*y + z*z)
    R[:, :, 0, 1] = 2 * (x*y - r*z)
    R[:, :, 0, 2] = 2 * (x*z + r*y)
    R[:, :, 1, 0] = 2 * (x*y + r*z)
    R[:, :, 1, 1] = 1 - 2 * (x*x + z*z)
    R[:, :, 1, 2] = 2 * (y*z - r*x)
    R[:, :, 2, 0] = 2 * (x*z - r*y)
    R[:, :, 2, 1] = 2 * (y*z + r*x)
    R[:, :, 2, 2] = 1 - 2 * (x*x + y*y)
    return R

def quat_mul(q1, q2):
    q = torch.zeros((q1.size(0), 4), device='cuda')

    r1, r2 = q1[:, 0], q2[:, 0]
    x1, x2 = q1[:, 1], q2[:, 1]
    y1, y2 = q1[:, 2], q2[:, 2]
    z1, z2 = q1[:, 3], q2[:, 3]

    q[:, 0] = r1*r2 - x1*x2 - y1*y2 - z1*z2
    q[:, 1] = r1*x2 + x1*r2 + y1*z2 - z1*y2
    q[:, 2] = r1*y2 - x1*z2 + y1*r2 + z1*x2
    q[:, 3] = r1*z2 + x1*y2 - y1*x2 + z1*r2
    return q

def build_scaling_rotation(s, r):
    L = torch.zeros((s.shape[0], 3, 3), dtype=torch.float, device="cuda")
    R = build_rotation(r)

    L[:,0,0] = s[:,0]
    L[:,1,1] = s[:,1]
    L[:,2,2] = s[:,2]

    L = R @ L
    return L


class BasicPointCloud(NamedTuple):
    points: np.array
    colors: np.array
    normals: np.array

def initialize_weights(m):
    if isinstance(m, nn.Linear):
        init.xavier_uniform_(m.weight,gain=1)
        if m.bias is not None:
            init.xavier_uniform_(m.weight,gain=1)

def initialize_weights_zero(m):
    if isinstance(m, nn.Linear):
        init.constant_(m.weight, 0)
        if m.bias is not None:
            init.constant_(m.bias, 0)

def initialize_weights_one(m):
    if isinstance(m, nn.Linear):
        init.constant_(m.weight, 0)
        if m.bias is not None:
            m.bias.data = torch.tensor([1., 0., 0., 0.])

class TimeNet(nn.Module):
    def __init__(self, D=8, W=256, skips=[4], latent_code_dim=32, device="cuda"):
        super(TimeNet, self).__init__()
        self.pts_ch = 10
        self.times_ch = 6
        self.pts_emb_fn, pts_out_dims = get_embedder(self.pts_ch, 3)
        self.times_emb_fn, times_out_dims = get_embedder(self.times_ch, 1)
        self.input_ch = pts_out_dims + times_out_dims + latent_code_dim
        self.skips = skips
        self.deformnet = nn.ModuleList(
            [nn.Linear(self.input_ch, W)] + \
            [nn.Linear(W, W) if i not in self.skips else nn.Linear(W + self.input_ch, W) for i in range(D-1)])
        self.pts_layers = nn.Sequential(nn.Linear(W, W), nn.ReLU(), nn.Linear(W, 3))
        self.rot_layers = nn.Sequential(nn.Linear(W, W), nn.ReLU(), nn.Linear(W, 4))
        self.device = device
        self.deformnet.apply(initialize_weights)
        self.pts_layers.apply(initialize_weights)
        self.rot_layers.apply(initialize_weights)
        self.pts_layers[-1].apply(initialize_weights_zero)
        self.rot_layers[-1].apply(initialize_weights_one)

    def forward(self, pts, t, latent_code, nobatch=False, t_apply=False):
        """
        Args: 
            pts: (N, 3), number of control points
            t: float, timestamp
            latent_code: (latent_code_dim, ), latent code for each scene
        """
        if len(pts.shape) == 2: # 1
            nobatch = True
            pts = pts.unsqueeze(0) # 1 * N * 3
        if t_apply:
            times = t
            pts = pts.repeat(times.shape[0], 1, 1)
        else: # 1
            times = torch.tensor([t])[:, None, None].repeat(1, pts.shape[1], 1).to(self.device) # 1 * N * 1
        if len(latent_code.shape) == 1: # 1
            latent_code = latent_code[None, None, :].repeat(pts.shape[0], pts.shape[1], 1) # 1 * N * l

        pts_emb = self.pts_emb_fn(pts) # 1 * N * p
        times_emb = self.times_emb_fn(times) # 1 * N * t
        pts_emb = torch.cat([pts_emb, times_emb, latent_code], dim=-1) # B * N * (p + t + l)
        h = pts_emb
        for i, l in enumerate(self.deformnet):
            h = self.deformnet[i](h)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([pts_emb, h], dim=-1) # B * N * W
        pts_t, rot_t = self.pts_layers(h), self.rot_layers(h) # B * N * 3, B * N * 4
        if nobatch:
            pts_t, rot_t = pts_t[0], rot_t[0]
        return pts_t, rot_t
    
    def get_mlp_parameters(self):
        parameter_list = []
        parameter_list_rot = [] 
        for name, param in self.named_parameters():
            if name.split('.')[0] == "rot_layers":
                parameter_list_rot.append(param)
            else:
                parameter_list.append(param)
        return parameter_list, parameter_list_rot


class GaussianModel:

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm
        
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize


    def __init__(self, sh_degree:int, num_latent_code:int = 1, latent_code_dim:int = 32):
        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree  
        self.num_latent_code = num_latent_code
        self.latent_code_dim = latent_code_dim
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.setup_functions()
        # learnable latent codes (gaussian distribution)
        mu = torch.zeros((self.num_latent_code, self.latent_code_dim), device="cuda") # mean=0
        log_var = torch.zeros((self.num_latent_code, self.latent_code_dim), device="cuda") # variance=1
        self._mu = nn.Parameter(mu.requires_grad_(True))
        self._log_var = nn.Parameter(log_var.requires_grad_(True))

        # deformation network
        self._timenet = TimeNet(latent_code_dim=latent_code_dim)
        # control points
        self._c_xyz = torch.empty(0) # position of control points
        self._c_radius = torch.empty(0) # radius of control points
        self._r = torch.empty(0) # radius of all points

    def capture(self):
        return (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
            self._mu, 
            self._log_var,
            self._timenet.state_dict(),
            self._c_xyz,
            self._c_radius,
            self._r,
        )
    
    def restore(self, model_args, training_args):
        (self.active_sh_degree, 
        self._xyz, 
        self._features_dc, 
        self._features_rest,
        self._scaling, 
        self._rotation, 
        self._opacity,
        self.max_radii2D, 
        xyz_gradient_accum, 
        denom,
        opt_dict, 
        self.spatial_lr_scale, 
        self._mu,
        self._log_var,
        self._timenet,
        self._c_xyz,
        self._c_radius,
        self._r,) = model_args
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)

    @property
    def get_scaling(self):
        if len(self._r) == 0:
            return self.scaling_activation(self._scaling)
        elif self._r.shape[0] != self._xyz.shape[0]:
            return self.scaling_activation(self._r.repeat(self._xyz.shape[0], 3))
        elif self._r.shape[0] == self._xyz.shape[0] and self._r.shape[1] == 1:
            return self.scaling_activation(self._r.repeat(1, 3))
        elif self._r.shape == self._xyz.shape:
            return self.scaling_activation(self._r)
        else:
            raise ValueError("Shape of _r is not supported.")

    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)
    
    @property
    def get_xyz(self):
        return self._xyz
    
    @property
    def get_mu(self, index=None):
        if index is not None:
            return self._mu[index]
        return self._mu
    
    @property
    def get_log_var(self, index=None):
        if index is not None:
            return self._log_var[index]
        return self._log_var
    
    @property
    def get_c_xyz(self):
        return self._c_xyz
    
    @property
    def get_c_rotation(self):
        return self.rotation_activation(self._c_rotation)
    
    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)
    
    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)
    
    def get_rotation_matrix(self):
        return quaternion_to_matrix(self.get_rotation)
    
    def get_smallest_axis(self, return_idx=False):
        rotation_matrices = self.get_rotation_matrix()
        smallest_axis_idx = self.get_scaling.min(dim=-1)[1][..., None, None].expand(-1, 3, -1)
        smallest_axis = rotation_matrices.gather(2, smallest_axis_idx)
        if return_idx:
            return smallest_axis.squeeze(dim=2), smallest_axis_idx[..., 0, 0]
        return smallest_axis.squeeze(dim=2)
    
    def get_normal(self, view_cam):
        normal_global = self.get_smallest_axis()
        gaussian_to_cam_global = view_cam.camera_center - self._xyz
        neg_mask = (normal_global * gaussian_to_cam_global).sum(-1) < 0.0
        normal_global[neg_mask] = -normal_global[neg_mask]
        return normal_global
    
    def get_c_radius(self, stage="s2"):
        if stage < "s2":
            return torch.exp(self._r.repeat(self._xyz.shape[0], 1))
        else:
            return torch.exp(self._c_radius)
    
    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def create_from_pcd(self, pcd : BasicPointCloud, pcd2 : BasicPointCloud, spatial_lr_scale : float = 1, r_type="1*1", only_init_gaussians=False):
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0 ] = fused_color
        features[:, 3:, 1:] = 0.0

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = inverse_sigmoid(0.05 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

        if not only_init_gaussians:
            # learnable latent codes
            self._timenet = self._timenet.to("cuda")
            fused_point_cloud = torch.tensor(np.asarray(pcd2.points)).float().cuda()
            c_radius = scales[:, :1]
            print("Number of control points at initialisation : ", fused_point_cloud.shape[0])
            self._c_xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
            self._c_radius= nn.Parameter(c_radius.requires_grad_(True))
            
            r = scales.clone().mean() * torch.ones((1, 1), device="cuda")
            self._r = nn.Parameter(r.requires_grad_(True))

    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        
        self._timenet = self._timenet.to("cuda")

        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"},
            {'params': [self._mu], 'lr': training_args.latent_code_lr_init, "name": "latent_code_mu"},
            {'params': [self._log_var], 'lr': training_args.latent_code_lr_init, "name": "latent_code_log_var"},
            {'params': list(self._timenet.get_mlp_parameters()[0]), 'lr': training_args.deform_lr_init, "name": "deform"},
            {'params': list(self._timenet.get_mlp_parameters()[1]), 'lr': training_args.deform_lr_init, "name": "deform_rot"},
            {'params': [self._c_xyz], 'lr': training_args.c_position_lr_init * self.spatial_lr_scale, "name": "c_xyz"},
            {'params': [self._c_radius], 'lr': training_args.c_radius_lr, "name": "c_radius"},
            {'params': [self._r], 'lr': training_args.r_lr, "name": "r"},
        ]

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.lr_setup(training_args)
    
    def lr_setup(self, training_args):
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)
        self.c_xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.c_position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.c_position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.c_position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)
        self.latent_code_scheduler_args = get_expon_lr_func(lr_init=training_args.latent_code_lr_init,  
                                                    lr_final=training_args.latent_code_lr_final,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)
        self.deform_scheduler_args = get_expon_lr_func(lr_init=training_args.deform_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.deform_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)
        self.deform_rot_scheduler_args = self.deform_scheduler_args

    def update_learning_rate(self, iteration, stage):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
            if stage >= "s2":
                if param_group["name"] == "c_xyz":
                    lr = self.c_xyz_scheduler_args(iteration)
                    param_group['lr'] = lr
                elif param_group["name"] == "latent_code_mu":
                    lr = self.latent_code_scheduler_args(iteration)
                    param_group['lr'] = lr
                elif param_group["name"] == "latent_code_log_var":
                    lr = self.latent_code_scheduler_args(iteration)
                    param_group['lr'] = lr
                elif param_group["name"] == "deform":
                    lr = self.deform_scheduler_args(iteration)
                    param_group['lr'] = lr
                elif param_group["name"] == "deform_rot":
                    lr = self.deform_scheduler_args(iteration)
                    param_group['lr'] = lr

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l
    
    def construct_list_of_attributes_c(self):
        l = ['c_x', 'c_y', 'c_z']
        # All channels except the 3 DC
        l.append('c_radius')
        return l
    
    @torch.no_grad()
    def save_ply(self, path1, path2=None):
        os.makedirs(os.path.dirname(path1), exist_ok=True)

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        if len(self._r) > 0:
            scale = self._r.expand_as(self._xyz).detach().cpu().numpy()
        else:
            scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path1)

        if path2 is not None:
            os.makedirs(os.path.dirname(path2), exist_ok=True)
            c_xyz = self._c_xyz.detach().cpu().numpy()
            c_radius = self._c_radius.detach().cpu().numpy()
            dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes_c()]
            elements = np.empty(c_xyz.shape[0], dtype=dtype_full)
            attributes = np.concatenate((c_xyz, c_radius), axis=1)
            elements[:] = list(map(tuple, attributes))
            el = PlyElement.describe(elements, 'vertex')
            PlyData([el]).write(path2)

    def reset_opacity(self):
        opacities_new = inverse_sigmoid(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def load_ply(self, path1, path2=None):
        plydata = PlyData.read(path1)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        print("Number of points at loading : ", xyz.shape[0])

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))

        self.active_sh_degree = self.max_sh_degree
        
        if path2 is not None:
            plydata = PlyData.read(path2)
            c_xyz = np.stack((np.asarray(plydata.elements[0]["c_x"]),
                            np.asarray(plydata.elements[0]["c_y"]),
                            np.asarray(plydata.elements[0]["c_z"])),  axis=1)
            c_radius = np.asarray(plydata.elements[0]["c_radius"])[..., np.newaxis]
            print("Number of control points at loading : ", c_xyz.shape[0])
            self._c_xyz = nn.Parameter(torch.tensor(c_xyz, dtype=torch.float, device="cuda").requires_grad_(True))
            self._c_radius = nn.Parameter(torch.tensor(c_radius, dtype=torch.float, device="cuda").requires_grad_(True))

    @torch.no_grad()
    def save_model(self, path, step=None):
        if not step:
            torch.save(self._mu, os.path.join(path, "mu.pth"))
            torch.save(self._log_var, os.path.join(path, "log_var.pth"))
            torch.save(self._timenet.state_dict(), os.path.join(path, "timenet.pth"))
        else:
            torch.save(self._mu, os.path.join(path, "mu_{}.pth".format(step)))
            torch.save(self._log_var, os.path.join(path, "log_var_{}.pth".format(step)))
            torch.save(self._timenet.state_dict(), os.path.join(path, "timenet_{}.pth".format(step)))

    def load_model(self, path, step=None):
        print("loading model from exists{}".format(path))
        if not step:
            mu_weight = torch.load(os.path.join(path, "mu.pth"),map_location="cuda")
            log_var_weight = torch.load(os.path.join(path, "log_var.pth"),map_location="cuda")
            weight_dict = torch.load(os.path.join(path, "timenet.pth"),map_location="cuda")
        else:
            mu_weight = torch.load(os.path.join(path, "mu_{}.pth".format(step)),map_location="cuda")
            log_var_weight = torch.load(os.path.join(path, "log_var_{}.pth".format(step)),map_location="cuda")
            weight_dict = torch.load(os.path.join(path, "timenet_{}.pth".format(step)),map_location="cuda")
        self._mu = nn.Parameter(mu_weight.requires_grad_(True)).to("cuda")
        self._log_var = nn.Parameter(log_var_weight.requires_grad_(True)).to("cuda")
        self._mu = self._mu.to("cuda") # (num_latent_code, latent_code_dim)
        self._log_var = self._log_var.to("cuda") # (num_latent_code, latent_code_dim)
        num_latent_code, latent_code_dim = self._mu.shape
        print(f"Load {num_latent_code} latent codes with {latent_code_dim} dimensions")
        self._timenet.load_state_dict(weight_dict)
        self._timenet = self._timenet.to("cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if len(group["params"]) > 1:
                continue
            elif group["name"] == 'latent_code_mu':
                continue
            elif group["name"] == 'latent_code_log_var':
                continue
            elif group["name"] == 'c_xyz':
                continue
            elif group["name"] == 'c_radius':
                continue
            elif group["name"] == 'r' and group["params"][0].shape[0]<=1:
                continue
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors
    
    def _prune_optimizer_s1_end(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if len(group["params"]) > 1:
                continue
            elif group["name"] == 'r' and group["params"][0].shape[0]<=1:
                continue
            elif group["name"] == 'latent_code_mu':
                continue
            elif group["name"] == 'latent_code_log_var':
                continue
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        if self._r.shape[0] > 1:
            self._r = optimizable_tensors["r"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]
    
    def easy_prune_points(self, mask):
        valid_points_mask = ~mask
        valid_points_mask = valid_points_mask.cpu()

        self._xyz = self._xyz[valid_points_mask]
        self._features_dc = self._features_dc[valid_points_mask]
        self._features_rest = self._features_rest[valid_points_mask]
        self._opacity = self._opacity[valid_points_mask]
        self._scaling = self._scaling[valid_points_mask]
        self._rotation = self._rotation[valid_points_mask]
    
    def prune_points_s1_end(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer_s1_end(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        self._c_xyz = optimizable_tensors["c_xyz"]
        self._c_radius = optimizable_tensors["c_radius"]
        if self._r.shape[0] > 1:
            self._r = optimizable_tensors["r"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if len(group["params"]) > 1:
                continue
            elif group["name"] == 'latent_code_mu':
                continue
            elif group["name"] == 'latent_code_log_var':
                continue
            elif group["name"] == 'c_xyz': 
                continue
            elif group["name"] == 'c_radius': 
                continue
            elif group["name"] == 'r' and group["params"][0].shape[0]<=1: 
                continue
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, new_r=None):
        d = {
            "xyz": new_xyz,
            "f_dc": new_features_dc,
            "f_rest": new_features_rest,
            "opacity": new_opacities,
            "scaling": new_scaling,
            "rotation": new_rotation,
        }

        if new_r is not None:
            d["r"] = new_r

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        if new_r is not None:
            self._r = optimizable_tensors["r"]

        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
            torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent
        )

        stds = self.get_scaling[selected_pts_mask].repeat(N,1)
        means =torch.zeros((stds.size(0), 3),device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1,1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N,1,1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)
        
        new_r = None
        if self._r.shape[0] == n_init_points:
            new_r = new_scaling[:,:self._r.shape[1]]

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation, new_r)

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
            torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent
        )
        
        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]

        new_r = None
        if self._r.shape[0] == self._xyz.shape[0]:
            new_r = self._r[selected_pts_mask]

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, new_r)

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        self.densify_and_clone(grads, max_grad, extent)
        self.densify_and_split(grads, max_grad, extent)

        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        self.prune_points(prune_mask)

        torch.cuda.empty_cache()

    def prune(self, min_opacity, extent, max_screen_size=None):

        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        self.prune_points(prune_mask)

        torch.cuda.empty_cache()
    
    def easy_prune(self, min_opacity, extent, max_screen_size):

        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        self.easy_prune_points(prune_mask)

        torch.cuda.empty_cache()
    
    def prune_s1_end(self, min_opacity, extent, max_screen_size):

        prune_mask = (self.get_opacity < min_opacity).squeeze()
        self.prune_points_s1_end(prune_mask)

        torch.cuda.empty_cache()


    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter,:2], dim=-1, keepdim=True)
        self.denom[update_filter] += 1


def getProjectionMatrix(znear, zfar, fovX, fovY):
    tanHalfFovY = math.tan((fovY / 2))
    tanHalfFovX = math.tan((fovX / 2))

    P = torch.zeros(4, 4) 

    z_sign = 1.0
    
    P[0, 0] = 1 / tanHalfFovX 
    P[1, 1] = 1 / tanHalfFovY 
    P[3, 2] = z_sign 
    P[2, 2] = z_sign * zfar / (zfar - znear) 
    P[2, 3] = -(zfar * znear) / (zfar - znear) 
    return P


class MiniCam:
    def __init__(self, c2w, width, height, fovy, fovx, znear, zfar):
        # c2w (pose) should be in NeRF convention. 

        self.image_width = width # 512
        self.image_height = height # 512
        self.FoVy = fovy
        self.FoVx = fovx
        self.znear = znear # 0.01
        self.zfar = zfar # 100

        w2c = np.linalg.inv(c2w) # world to camera

        # rectify...
        w2c[1:3, :3] *= -1 # flip y and z axis
        w2c[:3, 3] *= -1 # flip x axis

        self.world_view_transform = torch.tensor(w2c).transpose(0, 1).cuda() # 4x4 world to camera

        self.projection_matrix = (
            getProjectionMatrix(
                znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy
            )
            .transpose(0, 1)
            .cuda()
        ) # 4x4 project from camera to clip space
        self.full_proj_transform = self.world_view_transform @ self.projection_matrix # 4x4 project from world to clip space
        self.camera_center = -torch.tensor(c2w[:3, 3]).cuda()


class Renderer:
    def __init__(self, sh_degree=3, white_background=True, radius=1, delta_t=1/32, num_latent_code=1, latent_code_dim=32, add_normal=False):
        
        self.sh_degree = sh_degree
        self.white_background = white_background
        self.radius = radius

        self.gaussians = GaussianModel(
            sh_degree=sh_degree,
            num_latent_code=num_latent_code,
            latent_code_dim=latent_code_dim,
        )

        self.bg_color = torch.tensor(
            [1, 1, 1] if white_background else [0, 0, 0],
            dtype=torch.float32,
            device="cuda",
        )

        self.delta_t = delta_t
        self.add_normal = add_normal

    def initialize(self, input=None, num_pts=5000, num_cpts=512, radius=0.5, radius2=0.5, only_init_gaussians=False):
        # load checkpoint
        if input is None:
            # init from random point cloud
            phis = np.random.random((num_pts,)) * 2 * np.pi # (num_pts,) in [0, 2pi]
            costheta = np.random.random((num_pts,)) * 2 - 1 # (num_pts,) in [-1, 1]
            thetas = np.arccos(costheta) # (num_pts,) in [0, pi]
            mu = np.random.random((num_pts,)) # (num_pts,) in [0, 1]
            radius = radius * np.cbrt(mu) # (num_pts,) in [0, radius]
            x = radius * np.sin(thetas) * np.cos(phis) # (num_pts,) in [-radius, radius]
            y = radius * np.sin(thetas) * np.sin(phis) # (num_pts,) in [-radius, radius]
            z = radius * np.cos(thetas) # (num_pts,) in [-radius, radius]
            xyz = np.stack((x, y, z), axis=1) # (num_pts, 3)
            # xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3

            shs = np.random.random((num_pts, 3)) / 255.0 # (num_pts, 3) in [0, 1]
            pcd = BasicPointCloud(
                points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3))
            )
            ###
            phis = np.random.random((num_cpts,)) * 2 * np.pi
            costheta = np.random.random((num_cpts,)) * 2 - 1
            thetas = np.arccos(costheta)
            mu = np.random.random((num_cpts,))
            radius = radius2 * np.cbrt(mu)
            x = radius * np.sin(thetas) * np.cos(phis)
            y = radius * np.sin(thetas) * np.sin(phis)
            z = radius * np.cos(thetas)
            xyz = np.stack((x, y, z), axis=1)
            # xyz = np.random.random((num_cpts, 3)) * 2.6 - 1.3

            shs = np.random.random((num_cpts, 3)) / 255.0
            pcd2 = BasicPointCloud(
                points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_cpts, 3))
            )
            self.gaussians.create_from_pcd(pcd, pcd2, 1, only_init_gaussians=only_init_gaussians)
        elif isinstance(input, BasicPointCloud):
            # load from a provided pcd
            self.gaussians.create_from_pcd(input, input, 1)
        else:
            assert ValueError("Unsupported initialization type!!!")
        
    # AG initialization
    def initialize_ag(self, c_xyz, c_radius, num_cpts=512, num_pts_per_cpt=200, init_ratio=1):
        phis = np.random.random((num_pts_per_cpt,)) * 2 * np.pi # (num_pts_per_cpt,) in [0, 2pi]
        costheta = np.random.random((num_pts_per_cpt,)) * 2 - 1 # (num_pts_per_cpt,) in [-1, 1]
        thetas = np.arccos(costheta) # (num_pts_per_cpt,) in [0, pi]
        mu = np.random.random((num_pts_per_cpt,)) # (num_pts_per_cpt,) in [0, 1]
        radius = c_radius.mean().item() * init_ratio * np.cbrt(mu)
        x = radius * np.sin(thetas) * np.cos(phis)
        y = radius * np.sin(thetas) * np.sin(phis)
        z = radius * np.cos(thetas)
        xyz = np.stack((x, y, z), axis=1) # (num_pts_per_cpt, 3)
        # xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        
        xyz = torch.tensor(xyz)[None].repeat(num_cpts, 1, 1).flatten(0, 1) # (num_cpst*num_pts_per_cpt, 3)
        c_xyz = c_xyz.cpu().data[:, None].repeat(1, num_pts_per_cpt, 1).flatten(0, 1) 
        xyz = (xyz + c_xyz).numpy() # (num_cpst*num_pts_per_cpt, 3)

        shs = np.random.random((num_pts_per_cpt*num_cpts, 3)) / 255.0 # (num_cpst*num_pts_per_cpt, 3) in [0, 1]
        pcd = BasicPointCloud(
            points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts_per_cpt*num_cpts, 3))
        )
        self.gaussians.create_from_pcd(pcd, pcd, 1, only_init_gaussians=True)

    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor):
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [N x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [N x D]
        :return: (Tensor) [N x D]
        """
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std).to(mu.device)
        return eps * std + mu
    
    def arap_loss(self, t=None, delta_t=0.05, t_samp_num=2, stage="s1", latent_index=0):
        t = torch.rand([]).cuda() if t is None else t.squeeze() + delta_t * (torch.rand([]).cuda() - .5)
        t_samp = torch.rand(t_samp_num).cuda() * delta_t + t - .5 * delta_t
        
        t_samp = torch.zeros([2]).cuda()
        
        if stage >= "s2":
            means3D = self.gaussians.get_c_xyz
        else:
            means3D = self.gaussians._xyz
        node_num = means3D.shape[0]
        t_samp = t_samp[None, :, None].expand(node_num, t_samp_num, 1)  # M, T, 1
        mu = self.gaussians._mu[latent_index] # 32
        log_var = self.gaussians._log_var[latent_index] # 32
        latent_code = self.reparameterize(mu, log_var)
        node_trans, _ = self.gaussians._timenet(means3D[None], t_samp.permute(1, 0, 2), latent_code,t_apply=True)
        node_trans = node_trans.permute(1, 0, 2)
        nodes_t = means3D[:,None].repeat(1, t_samp_num, 1).detach() + node_trans 
        hyper_nodes = nodes_t[:,0]  # M, 3
        ii, jj, nn, weight = cal_connectivity_from_points(hyper_nodes, K=10)  # connectivity of control nodes
        error = cal_arap_error(nodes_t.permute(1,0,2), ii, jj, nn)
        return error, (ii, jj, nn, weight)

    def arap_loss_v2(self, delta_t=0.05, t_samp_num=8, stage="s1", latent_index=0):
        q_times = torch.rand(t_samp_num).to("cuda") 
        if stage == "s1":
            means3D = self.gaussians._xyz[None] # 1, M, 3
        else:
            means3D = self.gaussians._c_xyz[None]
        q_times = q_times[:, None, None].repeat(1, means3D.shape[1], 1).to(means3D.device) # T, M, 1
        mu = self.gaussians._mu[latent_index] # 32
        log_var = self.gaussians._log_var[latent_index]
        latent_code = self.reparameterize(mu, log_var)
        means3D_deform, _ = self.gaussians._timenet(means3D, q_times, latent_code, t_apply=True) # T, M, 3
        means3D_t = means3D.repeat(t_samp_num, 1, 1).detach() + means3D_deform # T, M, 3
        # Ball query
        ii, jj, nn, _ = cal_connectivity_from_points_v2(means3D_t, K=10) 
        error = cal_arap_error(means3D_t, ii, jj, nn)
        return error, (ii, jj, nn, _)

    def render(
        self,
        viewpoint_camera,
        scaling_modifier=1.0,
        bg_color=None,
        override_color=None,
        compute_cov3D_python=False,
        convert_SHs_python=False,
        time=0.0,
        stage="s1",
        rot_as_res=True,
        xyz_detach=False,
        local_frame=True,
        direct_deform=False,
        vertices_deform=None,
        latent_index=0,
    ):
        # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
        screenspace_points = (
            torch.zeros_like(
                self.gaussians.get_xyz,
                dtype=self.gaussians.get_xyz.dtype,
                requires_grad=True,
                device=self.gaussians.get_xyz.device,
            )
            + 0
        )
        try:
            screenspace_points.retain_grad()
        except:
            pass

        # Set up rasterization configuration
        tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
        tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

        if self.add_normal:
            raster_settings = GaussianRasterizationSettingsNormal(
                image_height=int(viewpoint_camera.image_height),
                image_width=int(viewpoint_camera.image_width),
                tanfovx=tanfovx,
                tanfovy=tanfovy,
                bg=self.bg_color if bg_color is None else bg_color,
                scale_modifier=scaling_modifier,
                viewmatrix=viewpoint_camera.world_view_transform,
                projmatrix=viewpoint_camera.full_proj_transform,
                sh_degree=self.gaussians.active_sh_degree,
                campos=viewpoint_camera.camera_center,
                prefiltered=False,
                debug=False,
            )
            rasterizer = GaussianRasterizerNormal(raster_settings=raster_settings)
        else:
            raster_settings = GaussianRasterizationSettings(
                image_height=int(viewpoint_camera.image_height),
                image_width=int(viewpoint_camera.image_width),
                tanfovx=tanfovx,
                tanfovy=tanfovy,
                bg=self.bg_color if bg_color is None else bg_color,
                scale_modifier=scaling_modifier,
                viewmatrix=viewpoint_camera.world_view_transform,
                projmatrix=viewpoint_camera.full_proj_transform,
                sh_degree=self.gaussians.active_sh_degree,
                campos=viewpoint_camera.camera_center,
                prefiltered=False,
                debug=False,
            )
            rasterizer = GaussianRasterizer(raster_settings=raster_settings)

        means3D = self.gaussians.get_xyz 
        means2D = screenspace_points
        opacity = self.gaussians.get_opacity
        c_means3D = self.gaussians.get_c_xyz if stage >= "s2" else means3D
        mu = self.gaussians._mu[latent_index]
        log_var = self.gaussians._log_var[latent_index]
        latent_code = self.reparameterize(mu, log_var)
        
        if stage >= "s2":
            c_means3D = self.gaussians.get_c_xyz
            means3D_deform, rots_deform = self.gaussians._timenet(c_means3D, time, latent_code) # (10, 3), (10, 4)
            cpts_t = c_means3D + means3D_deform
        else:
            means3D_deform, rots_deform = self.gaussians._timenet(means3D, time, latent_code)
            cpts_t = means3D + means3D_deform
        
        # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
        # scaling / rotation by the rasterizer.
        scales = None
        rotations = None
        cov3D_precomp = None
        if compute_cov3D_python:
            cov3D_precomp = self.gaussians.get_covariance(scaling_modifier)
        else:
            scales = self.gaussians.get_scaling
            # rotations = self.gaussians.get_rotation
            rotations = self.gaussians._rotation
        
        if stage >= "s2":
            eps = 1e-7
            c_radius = self.gaussians.get_c_radius(stage)
            neighbor_dists = self.gaussians.neighbor_dists
            neighbor_indices = self.gaussians.neighbor_indices
            c_radius_n = c_radius[neighbor_indices]
            w = torch.exp(-1.0*neighbor_dists**2/(2.*(c_radius_n[:,:,0]**2)))
            w = w + eps
            w = F.normalize(w, p=1)
            means3D_n = c_means3D[neighbor_indices] # N*4*3
            means3D_n_deform = means3D_deform[neighbor_indices] # N*4*3
            rots3D_n_deform = rots_deform[neighbor_indices] # N*4*4
            if local_frame:
                pts3D = (w[...,None] * ((build_rotation_3d(rots3D_n_deform) @ (means3D[:, None] - means3D_n)[..., None]).squeeze(-1) + means3D_n + means3D_n_deform)).sum(dim=1) 
            else:
                pts3D = means3D + (w[..., None] * means3D_n_deform).sum(dim=1)
            rots3D = (w[..., None] * rots3D_n_deform).sum(dim=1)
            means3D = pts3D   
            rotations = quat_mul(rots3D, rotations)
            # rotations = rotations + rots3D
        elif stage == "s1":
            means3D = means3D + means3D_deform
        else:
            assert ValueError("Nonexistent stage!!!")
            
        if xyz_detach:
            means3D = means3D.detach()
        
        rotations = self.gaussians.rotation_activation(rotations)

        # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
        # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
        shs = None
        colors_precomp = None
        # if colors_precomp is None:
        if override_color is None:
            if convert_SHs_python:
                shs_view = self.gaussians.get_features.transpose(1, 2).view(
                    -1, 3, (self.gaussians.max_sh_degree + 1) ** 2
                )
                dir_pp = self.gaussians.get_xyz - viewpoint_camera.camera_center.repeat(
                    self.gaussians.get_features.shape[0], 1
                )
                dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
                sh2rgb = eval_sh(
                    self.gaussians.active_sh_degree, shs_view, dir_pp_normalized
                )
                colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
            else:
                shs = self.gaussians.get_features # [N, 1, 3]
        else:
            colors_precomp = override_color

        # global_normal = self.gaussians.get_normal(viewpoint_camera) # [N, 3]
        # local_normal = global_normal @ viewpoint_camera.world_view_transform[:3,:3]
        # pts_in_cam = means3D @ viewpoint_camera.world_view_transform[:3,:3] + viewpoint_camera.world_view_transform[3,:3]
        # depth_z = pts_in_cam[:, 2]
        # local_distance = (local_normal * pts_in_cam).sum(-1).abs()
        # input_all_map = torch.zeros((means3D.shape[0], 5)).cuda().float()
        # input_all_map[:, :3] = local_normal
        # input_all_map[:, 3] = 1.0
        # input_all_map[:, 4] = local_distance

        # Rasterize visible Gaussians to image, obtain their radii (on screen).
        if self.add_normal:
            rendered_image, rendered_depth, rendered_norm, rendered_alpha, radii, extra = rasterizer(
                means3D = means3D,
                means2D = means2D,
                shs = shs,
                colors_precomp = colors_precomp,
                opacities = opacity,
                scales = scales,
                rotations = rotations,
                cov3Ds_precomp = cov3D_precomp,
                extra_attrs = None,
            )
        else:
            rendered_image, radii, rendered_depth, rendered_alpha = rasterizer(
                means3D=means3D,
                means2D=means2D,
                shs=shs,
                colors_precomp=colors_precomp,
                opacities=opacity,
                scales=scales,
                rotations=rotations,
                cov3D_precomp=cov3D_precomp,
            )

        rendered_image = rendered_image.clamp(0, 1)

        # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
        # They will be excluded from value updates used in the splitting criteria.
        return {
            "image": rendered_image,
            "depth": rendered_depth,
            "normal": rendered_norm,
            "alpha": rendered_alpha,
            "viewspace_points": screenspace_points,
            "visibility_filter": radii > 0,
            "radii": radii,
            "pts_t": means3D,
            "cpts_t": cpts_t,
        }
