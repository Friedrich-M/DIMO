import os
import cv2
import tqdm
import torch
import imageio
from torch.multiprocessing import Pool
import itertools
import json
from PIL import Image
import numpy as np
import torch.nn.functional as F
import torchvision.transforms as T
from matplotlib import cm

import warnings; warnings.filterwarnings("ignore")

from utils.cam_utils import orbit_camera, OrbitCamera
from utils.vis_utils import get_interactive_3d_visualization
from utils.load_utils import parallel_loader, init_rembg_worker
from renderer.latent_gs_renderer import Renderer, MiniCam
from renderer.gaussian_gs_renderer import Renderer as VAERenderer
from torchvision.utils import save_image
from tensorboardX import SummaryWriter
from knn_cuda import KNN
import pytorch3d.ops as ops
from chamferdist import ChamferDistance
from src.loss import (
    compute_tv_norm, ssim, 
    compute_edge_aware_smoothness_loss,
    compute_bilateral_normal_smoothness_loss,
)
from src.helpers import plot_3d_tracks
import lpips
import random
import torch.multiprocessing as mp
mp.set_start_method('spawn', force=True)


class GUI:
    def __init__(self, opt):
        self.opt = opt  # shared with the trainer's opt to support in-place modification of rendering parameters.
        self.W = opt.W
        self.H = opt.H
        self.num_frames = opt.get("num_frames", 21)
        self.num_views = opt.get("num_views", 9)
        self.cam = OrbitCamera(opt.W, opt.H, r=opt.radius, fovy=opt.fovy)

        # tensorboard
        if opt.train_dynamic:
            tb_dir = os.path.join(opt.save_path, "tb")
            os.makedirs(tb_dir, exist_ok=True)
            self.tb_writer = SummaryWriter(log_dir=tb_dir)

            with open(os.path.join(opt.save_path, "config.yaml"), "w") as f:
                OmegaConf.save(opt, f)

        self.seed = opt.seed
        self.seed_everything()

        # models
        self.device = torch.device("cuda")

        if 'info.json' in os.listdir(self.opt.input_folder):
            with open(os.path.join(self.opt.input_folder, 'info.json'), 'r') as f:
                info = json.load(f)
            print(f'[INFO] load info from {self.opt.input_folder}/info.json...')
            self.azimuths = info["azimuths_deg"]
            self.full_azimuths = info["full_azimuths_deg"]
            self.elevations = info["elevations_deg"]
            if self.opt.input_videos is not None:
                self.input_videos = self.opt.input_videos if isinstance(self.opt.input_videos, list) else self.opt.input_videos.split(",")
            elif "input_videos" in info:
                self.input_videos = info["input_videos"]
            else:
                raise ValueError("Input videos list not provided!!!")
            assert len(self.azimuths) == self.num_views
            assert len(self.full_azimuths) == 21
        else:
            print(f'Using default azimuths and elevations...')
            self.azimuths = [360/self.num_views*i for i in range(self.num_views)]
            self.full_azimuths = [360/self.num_frames*i for i in range(self.num_frames)]
            self.elevations = [self.opt.elevation for i in range(self.num_views)]

        # renderer
        if self.opt.vae_latent:
            self.renderer = VAERenderer(
                sh_degree=self.opt.sh_degree, # degree of spherical harmonics
                num_latent_code=len(self.input_videos), # number of latent codes
                latent_code_dim=self.opt.latent_code_dim, # dimension of latent code
                add_normal=self.opt.add_normal, # whether to render normal
            )
            print("[INFO] Using Gaussian distribution latent codes.")
        else:
            self.renderer = Renderer(
                sh_degree=self.opt.sh_degree, # degree of spherical harmonics
                num_latent_code=len(self.input_videos), # number of latent codes
                latent_code_dim=self.opt.latent_code_dim, # dimension of latent code
                add_normal=self.opt.add_normal, # whether to render normal
            )

        # gt data storage (images and masks)
        self.source_images = {motion_video_name: torch.zeros((self.num_views, self.num_frames, 1, 3, self.opt.ref_size, self.opt.ref_size)) for motion_video_name in self.input_videos}
        self.source_masks = {motion_video_name: torch.zeros((self.num_views, self.num_frames, 1, 1, self.opt.ref_size, self.opt.ref_size)) for motion_video_name in self.input_videos}
        self.source_time = [i/self.num_frames for i in range(self.num_frames)]
        
        # load all training data
        if self.opt.train_dynamic:
            ref_size = self.opt.ref_size
            input_folder = self.opt.input_folder
            args_list = [
                (
                    os.path.join(input_folder, motion_video_name, f"view_{view_idx:02d}", f"{frame_idx:02d}.png"),
                    motion_video_name, view_idx, frame_idx, ref_size
                )
                for motion_video_name, view_idx, frame_idx in itertools.product(
                    self.input_videos, range(self.num_views), range(self.num_frames)
                )
            ]

            with Pool(processes=min(16, os.cpu_count()), initializer=init_rembg_worker) as p:
                iterator = p.imap(parallel_loader, args_list, chunksize=32)
                for motion_video_name, view_idx, frame_idx, input_img_torch, input_mask_torch in tqdm.tqdm(
                    iterator, total=len(args_list), desc="Loading data"
                ):
                    self.source_images[motion_video_name][view_idx][frame_idx] = input_img_torch
                    self.source_masks[motion_video_name][view_idx][frame_idx] = input_mask_torch
            
            print(f'[INFO] loaded {len(self.source_images)} multi-view videos with {len(self.source_images[motion_video_name])} views and {len(self.source_images[motion_video_name][0])} frames each.')
        
        # training stuff
        self.optimizer = None
        self.step = 0
        self.train_steps = 1  # steps per rendering loop
        self.stage = "s1"

        # initialize renderer
        if self.opt.train_dynamic:
            # override if provide a checkpoint
            if self.opt.load_stage == "s1":
                save_path = os.path.join(self.opt.save_path, "s1/point_cloud.ply")
                g = self.renderer.gaussians
                g.load_ply(save_path)
                self.renderer.initialize(num_pts=g._xyz.shape[0], num_cpts=g._xyz.shape[0])           
            else:
                # initialize gaussians to a blob
                self.renderer.initialize(num_pts=self.opt.num_cpts, num_cpts=self.opt.num_cpts)
        
        # loss functions
        self.chamferDist = ChamferDistance()
        self.lpips_loss = lpips.LPIPS(net='vgg').to(self.device)

        self.cpts_s1 = {motion_video_name: [] for motion_video_name in self.input_videos}
        torch.cuda.empty_cache()

    def seed_everything(self):
        try:
            seed = int(self.seed)
        except:
            seed = np.random.randint(0, 1000000)

        os.environ["PYTHONHASHSEED"] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True

        self.last_seed = seed

    def train_dynamic(self, iters_s1=1500, iters_s2=5000, load_stage=""): 
        g = self.renderer.gaussians
        iters_s1 = iters_s1 if load_stage < "s1" else 0
        iters_s2 = iters_s2 if load_stage < "s2" else 0
        if load_stage != "": 
            print("Loading from stage {}...".format(load_stage[-1]))
            # Loading pretrained model
            self.load_model(g=g)
            if load_stage >= "s1":
                g._r.data = g._scaling
                if load_stage == "s1":
                    with torch.no_grad():
                        g._c_xyz.copy_(g._xyz) 
                        g._c_radius.copy_(g._scaling.mean(dim=1, keepdim=True))
                    if self.opt.init_type == "normal":
                        self.renderer.initialize(num_pts=self.opt.num_pts, only_init_gaussians=True)
                    elif self.opt.init_type == "ag":
                        self.renderer.initialize_ag(g._c_xyz, g.get_c_radius(stage="s2"), num_cpts=g._c_xyz.shape[0], num_pts_per_cpt=200, init_ratio=self.opt.init_ratio)
                    else:
                        raise ValueError("Unsupported init type!!!")
        
        self.opt.save_path = self.opt.save_path if self.opt.save_path_new is None else self.opt.save_path_new

        ### Stage 1: motion pretraining -> coarse motion basis and latent space 
        self.prepare_train_s1()
        self.renderer.gaussians.lr_setup(self.opt)
        if iters_s1 > 0:
            for i in tqdm.trange(iters_s1, desc="Stage 1"):
                self.train_step()
            # prune points at s1 end
            self.renderer.gaussians.prune_s1_end(min_opacity=0.01, extent=4, max_screen_size=1)
            print("Num of cpts after s1: ", self.renderer.gaussians._c_xyz.shape[0])
            save_path = os.path.join(self.opt.save_path, "s1")
            # save the coarse-level canonical key points
            g.save_ply(os.path.join(save_path, "point_cloud.ply"))
            # save the initial latent codes and motion decoder
            g.save_model(save_path)

        ### Stage 2: motion refinement and geometry optimization -> joint optimization of motion and geometry
        self.prepare_train_s2()
        self.renderer.gaussians.lr_setup(self.opt)
        if iters_s2 > 0:
            for i in tqdm.trange(iters_s2, desc="Stage 2"):
                self.train_step()
            save_path = os.path.join(self.opt.save_path, "s2")
            # save the fine-level canonical key points and canonical 3D Gaussians
            g.save_ply(os.path.join(save_path, "point_cloud.ply"), os.path.join(save_path, "point_cloud_c.ply")) 
            # save the final latent codes and motion decoder
            g.save_model(save_path)


    def train_step(self):
        starter = torch.cuda.Event(enable_timing=True)
        ender = torch.cuda.Event(enable_timing=True)
        starter.record()
        
        # downsample the key points using FPS as an annealing process (paper sec 3.3
        if self.stage == "s1" and self.step % self.opt.FPS_iter == 0:
            self.FPS(num_pts=self.opt.num_cpts)

        # inherit the canonical shape (distribution) modeled in the first stage (paper sec 3.3
        if self.stage == "s2" and self.step == 0: 
            c_means3D = self.renderer.gaussians._c_xyz # (n_nodes, 3)
            # cache the key point trajectories for regularization
            for latent_index, motion_video_name in enumerate(self.input_videos):
                if self.opt.vae_latent:
                    # VAE reparameterization trick to sample latent code (paper sec 3.2.2.
                    mu = self.renderer.gaussians._mu[latent_index] 
                    log_var = self.renderer.gaussians._log_var[latent_index]
                    latent_code = self.renderer.reparameterize(mu, log_var)
                else:
                    latent_code = self.renderer.gaussians._latent_codes[latent_index]
                for t in self.source_time:
                    means3D_deform, _ = self.renderer.gaussians._timenet(c_means3D, t, latent_code) # (n_nodes, 3)
                    self.cpts_s1[motion_video_name].append(c_means3D + means3D_deform) # num_frames x (n_nodes, 3)

        for _ in range(self.train_steps):
            self.step += 1
            
            self.renderer.gaussians.update_learning_rate(self.step, self.stage)
            
            if self.stage == "s2" and self.step < 1000:
                for param_group in self.optimizer.param_groups:
                    if param_group["name"] == "xyz":
                        param_group['lr'] = 0.0002 
            
            # find knn
            if self.stage >= "s2":
                self.find_knn(g=self.renderer.gaussians, k=4) 
        
            loss = 0

            # gradually increase render resolution from 128×128 to 512×512 (paper sec 3.3.
            render_resolution = 128 if self.step < 300 else (256 if self.step < 450 else 512)
            
            # randomly sample 4 motions × 3 views × 2 frames within a batch (paper sec 3.3.
            batch_frame_indexs = random.sample(range(self.num_frames), self.opt.batch_size)
            # batch_view_indexs = [0] + random.sample(range(1, self.num_views), self.opt.batch_size) # including reference view
            batch_view_indexs = random.sample(range(self.num_views), self.opt.batch_size)
            random_selected_indexs = np.random.choice(len(self.input_videos), min(int(self.opt.batch_size*2), len(self.input_videos)), replace=False)
            index_video_pair = [(latent_index, self.input_videos[latent_index]) for latent_index in random_selected_indexs]

            # render all selected views and frames
            batch_render_images, batch_gt_images = {}, {}
            batch_render_masks, batch_gt_masks = {}, {}
            batch_render_depths, batch_render_normals = {}, {}
            for latent_index, motion_video_name in index_video_pair:
                render_images, gt_images = [], []
                render_masks, gt_masks = [], []
                render_depths, render_normals = [], []
                
                for view_index in batch_view_indexs:
                    for frame_index in batch_frame_indexs:
                        gt_image = self.source_images[motion_video_name][view_index][frame_index].to(self.device) # [1, 3, H, W]
                        gt_mask = self.source_masks[motion_video_name][view_index][frame_index].to(self.device) # [1, 1, H, W]

                        pose = orbit_camera(self.opt.elevation, self.azimuths[view_index], self.opt.radius)
                        cur_cam = MiniCam(pose, render_resolution, render_resolution, self.cam.fovy, self.cam.fovx, self.cam.near, self.cam.far)
                        timestamp = self.source_time[frame_index] # [0,1)
            
                        if self.stage == "s2":
                            cpts_ori = self.cpts_s1[motion_video_name][frame_index]

                        out = self.renderer.render(cur_cam, time=timestamp, stage=self.stage, latent_index=latent_index)

                        if self.opt.add_ga and self.stage == "s2":
                            cpts_ori = cpts_ori.detach() # (n_nodes, 3)
                            cpts = out["cpts_t"] # (n_nodes, 3)
                            if self.opt.ga_chamfer:
                                dist_forward = self.chamferDist(cpts[None, ...], cpts_ori[None, ...]) # the order is important!!!
                                loss = loss + self.opt.lambda_ga1 * dist_forward
                            else:
                                loss = loss + self.opt.lambda_ga2 * (cpts - cpts_ori).abs().mean()
                            self.tb_writer.add_scalar(f'{self.stage}/{motion_video_name}/loss_ga', loss.item(), self.step)

                        render_image = out["image"].unsqueeze(0) # (1, 3, rH, rW)
                        render_images.append(render_image)
                        gt_image = F.interpolate(gt_image, (render_resolution, render_resolution), mode="bilinear", align_corners=False) # (1, 3, rH, rW)
                        gt_images.append(gt_image)

                        render_mask = out["alpha"].unsqueeze(0) # (1, 1, rH, rW)
                        render_masks.append(render_mask)
                        gt_mask = F.interpolate(gt_mask, (render_resolution, render_resolution), mode="bilinear", align_corners=False) # (1, 1, rH, rW)
                        gt_masks.append(gt_mask)

                        render_depth = out["depth"].unsqueeze(0) # (1, 1, rH, rW)
                        render_depths.append(render_depth)
                        render_normal = out["normal"].unsqueeze(0) # (1, 3, rH, rW)
                        render_normals.append(render_normal)

                batch_render_images[motion_video_name] = torch.cat(render_images, dim=0) # b^2, 3, rH, rW
                batch_gt_images[motion_video_name] = torch.cat(gt_images, dim=0) # b^2, 3, rH, rW
                batch_render_masks[motion_video_name] = torch.cat(render_masks, dim=0)  # b^2, 1, rH, rW
                batch_gt_masks[motion_video_name] = torch.cat(gt_masks, dim=0) # b^2, 1, rH, rW
                batch_render_depths[motion_video_name] = torch.cat(render_depths, dim=0) # b^2, 1, rH, rW
                batch_render_normals[motion_video_name] = torch.cat(render_normals, dim=0) # b^2, 3, rH, rW

            # Compute loss
            for latent_index, motion_video_name in index_video_pair:
                # Image-level loss
                ## pixel-level MSE loss
                for i, view_index in enumerate(batch_view_indexs):
                    for j, frame_index in enumerate(batch_frame_indexs):
                        mse_loss = F.mse_loss(batch_render_images[motion_video_name][i * len(batch_frame_indexs) + j], batch_gt_images[motion_video_name][i * len(batch_frame_indexs) + j])
                        if view_index == 0 or frame_index == 0: # reference view and frame
                            loss = loss + self.opt.lambda_mse * mse_loss
                        else:
                            loss = loss + self.opt.lambda_mse * 0.5 * mse_loss
                        
                ## LPIPS loss
                lpips_loss = self.lpips_loss(batch_render_images[motion_video_name], batch_gt_images[motion_video_name]).mean()
                loss = loss + self.opt.lambda_lpips * lpips_loss
                ## SSIM loss
                ssim_loss = 1 - ssim(batch_render_images[motion_video_name], batch_gt_images[motion_video_name])
                loss = loss + self.opt.lambda_ssim * ssim_loss
                self.tb_writer.add_scalar(f'{self.stage}/{motion_video_name}/loss_lpips', lpips_loss.item(), self.step)
                self.tb_writer.add_scalar(f'{self.stage}/{motion_video_name}/loss_ssim', ssim_loss.item(), self.step)
                self.tb_writer.add_scalar(f'{self.stage}/{motion_video_name}/loss_mse', mse_loss.item(), self.step)
                
                # Mask loss
                mask_loss = F.mse_loss(batch_render_masks[motion_video_name], batch_gt_masks[motion_video_name])
                loss = loss + self.opt.lambda_mask * mask_loss
                self.tb_writer.add_scalar(f'{self.stage}/{motion_video_name}/loss_mask', mask_loss.item(), self.step)

                # KL divergence loss for VAE latent code
                if self.opt.vae_latent:
                    mu = self.renderer.gaussians._mu[latent_index]
                    log_var = self.renderer.gaussians._log_var[latent_index]
                    kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
                    loss = loss + self.opt.lambda_kl * kl_loss
                    self.tb_writer.add_scalar(f'{self.stage}/{motion_video_name}/loss_kl', kl_loss.item(), self.step)

                # Edge-aware depth smoothness loss
                if self.opt.add_depth and self.step > self.opt.depth_reg_start_iter: # 200
                    edge_aware_smooth_loss = compute_edge_aware_smoothness_loss(batch_render_depths[motion_video_name].permute(0, 2, 3, 1), batch_render_images[motion_video_name].permute(0, 2, 3, 1))
                    loss = loss + self.opt.lambda_smooth * edge_aware_smooth_loss
                    self.tb_writer.add_scalar(f'{self.stage}/{motion_video_name}/loss_edge_aware_smooth', edge_aware_smooth_loss.item(), self.step)
                
                # Bilateral normal smoothness loss
                if self.opt.add_normal and self.step > self.opt.normal_reg_start_iter: # 200
                    bilateral_normal_smooth_loss = compute_bilateral_normal_smoothness_loss(batch_render_normals[motion_video_name].permute(0, 2, 3, 1), batch_render_images[motion_video_name].permute(0, 2, 3, 1))
                    loss = loss + self.opt.lambda_bilateral * bilateral_normal_smooth_loss
                    self.tb_writer.add_scalar(f'{self.stage}/{motion_video_name}/loss_bilateral_normal_smooth', bilateral_normal_smooth_loss.item(), self.step)

                # ARAP loss in s1
                if self.opt.use_arap and self.stage == "s1" and self.step > self.opt.arap_start_iter_s1: # 1000
                    loss_arap, conns = self.renderer.arap_loss_v2(stage=self.stage, latent_index=latent_index)
                    loss += self.opt.lambda_arap * loss_arap
                    self.tb_writer.add_scalar(f'{self.stage}/{motion_video_name}/loss_arap', loss_arap.item(), self.step)
                
                # ARAP loss in s2
                if self.opt.use_arap and self.stage == "s2" and self.step < self.opt.arap_end_iter_s2: # 2000
                    loss_arap, conns = self.renderer.arap_loss_v2(stage=self.stage, latent_index=latent_index)
                    loss += self.opt.lambda_arap * loss_arap
                    self.tb_writer.add_scalar(f'{self.stage}/{motion_video_name}/loss_arap', loss_arap.item(), self.step)

                # total loss
                self.tb_writer.add_scalar(f'{self.stage}/{motion_video_name}/loss_total', loss.item(), self.step)
                # psnr result
                psnr = 10 * torch.log10(1 / mse_loss)
                self.tb_writer.add_scalar(f'{self.stage}/{motion_video_name}/psnr', psnr.item(), self.step)

                # debug image saving
                if self.step % 100 == 0:
                    debug_path = os.path.join(self.opt.save_path, f"debug_{motion_video_name}")
                    os.makedirs(debug_path, exist_ok=True)
                    gt_img_show = (batch_gt_images[motion_video_name][0].permute(1,2,0).clamp(0,1)[:,:,[2,1,0]]*255).detach().cpu().numpy().astype(np.uint8)
                    img_show = (batch_render_images[motion_video_name][0].permute(1,2,0).clamp(0,1)[:,:,[2,1,0]]*255).detach().cpu().numpy().astype(np.uint8)
                    os.makedirs(debug_path, exist_ok=True)
                    image_to_show = np.concatenate([gt_img_show, img_show], axis=1)
                    cv2.imwrite(os.path.join(debug_path, f"image_{self.stage}_{self.step}.png"), image_to_show)

                # image and latent code distribution visualization 
                if self.step % 10 == 0:
                    # render and gt image comparison
                    self.tb_writer.add_images(f'{self.stage}/view{batch_view_indexs[0]}_{motion_video_name}/gt', batch_gt_images[motion_video_name][:1], self.step)
                    self.tb_writer.add_images(f'{self.stage}/view{batch_view_indexs[0]}_{motion_video_name}/render', batch_render_images[motion_video_name][:1], self.step)
                    # visualize the latent code 
                    if self.opt.vae_latent:
                        vis_latent_code = self.renderer.reparameterize(self.renderer.gaussians._mu[latent_index], self.renderer.gaussians._log_var[latent_index]).detach().cpu().numpy()
                    else:
                        vis_latent_code = self.renderer.gaussians._latent_codes[latent_index].detach().cpu().numpy()
                    self.tb_writer.add_histogram(f'{self.stage}/{motion_video_name}/latent_code', vis_latent_code, self.step)
               
            # optimize step
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

            if self.step % self.opt.save_inter == 0: # 500
                save_path = os.path.join(self.opt.save_path, self.stage)
                path2 = os.path.join(save_path, "point_cloud_c_{}.ply".format(self.step)) if self.stage >= "s2" else None
                self.renderer.gaussians.save_ply(os.path.join(save_path, "point_cloud_{}.ply".format(self.step)), path2)
                self.renderer.gaussians.save_model(save_path, step=self.step)

            # densify and prune
            if self.stage == "s1":
                # if self.step >= self.opt.density_start_iter and self.step <= self.opt.density_end_iter: # [100, 1000]
                if self.step % self.opt.FPS_iter >= self.opt.density_start_iter and self.step <= self.opt.density_end_iter: # [100, 1000]
                    viewspace_point_tensor, visibility_filter, radii = out["viewspace_points"], out["visibility_filter"], out["radii"]
                    self.renderer.gaussians.max_radii2D[visibility_filter] = torch.max(self.renderer.gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                    self.renderer.gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                    if self.step % self.opt.densification_interval == 0:
                        self.renderer.gaussians.densify_and_prune(self.opt.densify_grad_threshold, min_opacity=self.opt.densify_opacity_threshold_s1, extent=4, max_screen_size=1)
                        print("Num of gaussians: ", self.renderer.gaussians._xyz.shape[0])

                    if self.step % self.opt.opacity_reset_interval == 0:
                        self.renderer.gaussians.reset_opacity()
            
            if self.stage == "s2" and self.step < self.opt.density_end_iter_s2: # 5000
                if self.step % self.opt.densification_interval_s2 == 0 and self.opt.init_type == "ag": # 1000
                    self.renderer.gaussians.prune(min_opacity=self.opt.densify_opacity_threshold_s2, extent=4, max_screen_size=1)
                    print("Num of gaussians after pruning: ", self.renderer.gaussians._xyz.shape[0])

            # clear memory
            del loss, batch_render_images, batch_gt_images, batch_render_masks, batch_gt_masks, batch_render_depths, batch_render_normals
            torch.cuda.empty_cache()

        ender.record()
        torch.cuda.synchronize()
        t = starter.elapsed_time(ender)
    
    def prepare_train_s1(self):
        self.step = 0
        self.stage = "s1"
        self.opt.position_lr_max_steps = 500

        # setup training
        self.renderer.gaussians.training_setup(self.opt)
        # do not do progressive sh-level
        self.renderer.gaussians.active_sh_degree = self.renderer.gaussians.max_sh_degree # 0
        self.optimizer = self.renderer.gaussians.optimizer

        if self.stage == "s1":
            for param_group in self.optimizer.param_groups:
                if param_group["name"] == "c_radius":
                    param_group['lr'] = 0.0
                if param_group["name"] == "c_xyz":
                    param_group['lr'] = 0.0
    
    def prepare_train_s2(self):
        self.stage = "s2"
        self.step = 0
        g = self.renderer.gaussians
        if self.opt.load_stage == "":
            with torch.no_grad():
                g._c_xyz.copy_(g._xyz) # (n_nodes, 3)
                g._scaling.copy_(g._r.expand_as(g._xyz)) # (n_nodes, 1)
                g._c_radius.copy_(g._r.expand_as(g._c_radius)) # (n_nodes, 1)
            if self.opt.init_type == "normal":
                self.renderer.initialize(num_pts=self.opt.num_pts, only_init_gaussians=True)
            elif self.opt.init_type == "ag": # 1 -> Adaptive Gaussian Initialization
                # consider each key point as a sphere with a radius of s, and randomly initialize K Gaussians
                self.renderer.initialize_ag(g._c_xyz, g.get_c_radius(stage="s2"), num_cpts=g._c_xyz.shape[0], num_pts_per_cpt=200, init_ratio=self.opt.init_ratio)
            else:
                raise ValueError("Unsupported init type!!!")
            # update training
            self.renderer.gaussians.training_setup(self.opt)
            self.renderer.gaussians.active_sh_degree = self.renderer.gaussians.max_sh_degree # 0
            self.optimizer = self.renderer.gaussians.optimizer
        g._r = torch.tensor([], device="cuda")
        r_id = 0
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "r":
                param_group['lr'] = 0.0
                self.optimizer.param_groups.pop(r_id)
            r_id += 1
        self.opt.position_lr_max_steps = self.opt.iters_s2
        self.opt.position_lr_init = 0.0002
        self.opt.position_lr_final = 0.000002

    def find_knn(self, g, k=4):
        key_pts = g._c_xyz.detach()
        gaussian_pts = g._xyz.detach()
        knn = KNN(k=k, transpose_mode=True)
        dist, indx = knn(key_pts.unsqueeze(0), gaussian_pts.unsqueeze(0))  # num_frames x 50 x 10
        dist, indx = dist[0], indx[0]  
        g.neighbor_dists = dist             
        g.neighbor_indices = indx

    def FPS(self, num_pts):
        g = self.renderer.gaussians
        _, idxs = ops.sample_farthest_points(points=g._xyz.unsqueeze(0), K=num_pts)
        idxs = idxs[0]
        g.prune_points(idxs)
    
    def load_model(self, g):
        load_stage = self.opt.load_stage or self.opt.test_stage
        path1 = "{}/{}/point_cloud.ply".format(self.opt.save_path, load_stage)
        path2 = "{}/{}/point_cloud_c.ply".format(self.opt.save_path, load_stage)
        model_dir = "{}/{}".format(self.opt.save_path, load_stage)
        if self.opt.test_step:
            path1 = path1.split('.')[0] + "_{}".format(self.opt.test_step) + '.ply'
            if load_stage > "s1":
                path2 = path2.split('.')[0] + "_{}".format(self.opt.test_step) + '.ply'
        if load_stage < "s2":
            path2 = None
        g.load_ply(path1, path2)
        g.load_model(model_dir, self.opt.test_step)

    def test(self, test_cpts=True, render_type="fixed"):
        video_save_dir = self.opt.video_save_dir
        test_stage = self.opt.test_stage
        if not os.path.exists(video_save_dir):
            os.makedirs(video_save_dir)
        g = self.renderer.gaussians
        self.load_model(g=g)
        if self.opt.vae_latent:
            assert len(self.renderer.gaussians._mu) == len(self.renderer.gaussians._log_var) == len(self.input_videos), "Number of latent codes does not match the number of input videos!!!"
        else:
            assert len(self.renderer.gaussians._latent_codes) == len(self.input_videos), "Number of latent codes does not match the number of input videos!!!"
        if test_stage >= "s2":
            self.find_knn(g) # KNN for key points

        all_traj_imgs = []
        all_traj_imgs_3d = []
        all_imgs = []
        for video_index, motion_video_name in enumerate(self.input_videos):
            if test_cpts: # 1
                traj_imgs, traj_imgs_3d = self.test_cpts(test_stage=self.opt.test_stage, render_type=render_type, latent_index=video_index, motion_video_name=motion_video_name)
                all_traj_imgs_3d.append(traj_imgs_3d)
            frames = []
            for i in range(self.num_frames): # 21
                if render_type == "fixed":
                    test_azi = self.opt.test_azi # 0
                else:
                    test_azi = 360/self.num_frames*i
                pose = orbit_camera(
                    elevation=self.opt.elevation, # 0 
                    azimuth=test_azi, 
                    radius=self.opt.radius, # 2.0
                ) # (4, 4)
                cur_cam = MiniCam(
                    pose,
                    self.W,
                    self.H,
                    self.cam.fovy,
                    self.cam.fovx,
                    self.cam.near,
                    self.cam.far,
                )
                out = self.renderer.render(cur_cam, time=i/self.num_frames, stage=test_stage, latent_index=video_index)
                img = out["image"].detach().cpu().permute(1, 2, 0).numpy() * 255 # (H, W, 3)
                img = img.astype('uint8')
                frames.append(img)
            all_imgs.append(np.stack(frames, axis=0)) # (num_frames, H, W, 3)
            save_name = self.opt.save_path.split("/")[-1].split(".")[0]
            if render_type == "fixed":
                video_name = video_save_dir + '/{}_{}_{}_fixed.mp4'.format(save_name, motion_video_name, test_stage)
            else:
                video_name = video_save_dir + '/{}_{}_{}_circle.mp4'.format(save_name, motion_video_name, test_stage)
            imageio.mimwrite(video_name, frames, fps=8, quality=8, macro_block_size=1)

            if test_cpts and render_type == "fixed":
                blend_imgs = []
                for i in range(self.num_frames):
                    blender_img = cv2.addWeighted(frames[i], 0.4, traj_imgs[i], 0.6, 0)
                    blend_imgs.append(blender_img)
                save_path = os.path.join(video_save_dir, f'{save_name}_{motion_video_name}_blend.mp4')
                imageio.mimwrite(save_path, blend_imgs, fps=8, quality=8, macro_block_size=1)
                print(f'[INFO] trajectory video saved to {os.path.abspath(save_path)}')

                all_traj_imgs.append(np.stack(traj_imgs, axis=0)) # (num_frames, H, W, 3)

        if test_cpts:   
            all_row_traj_imgs, all_row_traj_imgs_3d, all_row_imgs = [], [], []
            n_rows = 4
            rows_len = len(all_traj_imgs_3d) // n_rows
            for row_index in range(n_rows):
                row_traj_imgs = np.concatenate(all_traj_imgs[row_index*rows_len:(row_index+1)*rows_len], axis=2) # (num_frames, H, rows_len*W, 3)
                all_row_traj_imgs.append(row_traj_imgs) 
                row_traj_imgs_3d = np.concatenate(all_traj_imgs_3d[row_index*rows_len:(row_index+1)*rows_len], axis=2) # (num_frames, H, rows_len*W, 3)
                all_row_traj_imgs_3d.append(row_traj_imgs_3d) 
                row_imgs = np.concatenate(all_imgs[row_index*rows_len:(row_index+1)*rows_len], axis=2) # (num_frames, H, rows_len*W, 3)
                all_row_imgs.append(row_imgs)
            all_row_traj_imgs = np.concatenate(all_row_traj_imgs, axis=1) # (num_frames, H, 5*rows_len*W, 3)
            save_path = 'all_traj_imgs.mp4'
            imageio.mimwrite(save_path, all_row_traj_imgs, fps=8, quality=8, macro_block_size=1)
            print(f'[INFO] trajectory video saved to {os.path.abspath(save_path)}')
            all_row_traj_imgs_3d = np.concatenate(all_row_traj_imgs_3d, axis=1) # (num_frames, H, 5*rows_len*W, 3)
            save_path = 'all_traj_imgs_3d.mp4'
            imageio.mimwrite(save_path, all_row_traj_imgs_3d, fps=8, quality=8, macro_block_size=1)
            all_row_imgs = np.concatenate(all_row_imgs, axis=1) # (num_frames, H, 5*rows_len*W, 3)
            print(f'[INFO] 3d trajectory video saved to {os.path.abspath(save_path)}')
            save_path = 'all_imgs.mp4'
            imageio.mimwrite(save_path, all_row_imgs, fps=8, quality=8, macro_block_size=1)
            print(f'[INFO] video saved to {os.path.abspath(save_path)}')


    def test_cpts(self, test_stage="s1", render_type="fixed", sh_degree=0, latent_index=0, motion_video_name=None):
        video_save_dir = self.opt.video_save_dir
        renderer = Renderer(sh_degree=sh_degree, add_normal=self.opt.add_normal)
        if test_stage > "s1":
            renderer.initialize(num_pts=self.renderer.gaussians._c_xyz.shape[0]) 
            renderer.gaussians._xyz = self.renderer.gaussians._c_xyz # (n_nodes, 3)
        else:
            renderer.initialize(num_pts=self.renderer.gaussians._xyz.shape[0])
            renderer.gaussians._xyz = self.renderer.gaussians._xyz
        renderer.gaussians._r = torch.ones((1), device="cuda", requires_grad=True) * -5.0
        if self.opt.vae_latent:
            renderer.gaussians._mu = self.renderer.gaussians._mu
            renderer.gaussians._log_var = self.renderer.gaussians._log_var
        else:
            renderer.gaussians._latent_codes = self.renderer.gaussians._latent_codes
        renderer.gaussians._timenet = self.renderer.gaussians._timenet
        num_pts = renderer.gaussians._xyz.shape[0]
        device = renderer.gaussians._xyz.device
        renderer.gaussians._scaling = torch.ones((num_pts, 3), device=device, requires_grad=True) * -5.0 
        renderer.gaussians._opacity = torch.ones((num_pts, 1), device=device, requires_grad=True) * 2.0
        color = torch.ones((num_pts, 3), device=device) * 0.1
        frames = []
        traj_pts_3d = []
        traj_pts = []
        cpts_tra = 0
        for i in range(self.num_frames):
            if render_type == "fixed":
                test_azi = self.opt.test_azi # 0
            else:
                test_azi = 360/self.num_frames*i
            pose = orbit_camera(
                elevation=self.opt.elevation, 
                azimuth=test_azi, 
                radius=self.opt.radius
            )
            cur_cam = MiniCam(
                pose, # (4, 4)
                self.W, # 800
                self.H, # 800
                self.cam.fovy, # 0.5916666164260777
                self.cam.fovx, # 0.5916666164260777
                self.cam.near, # 0.01
                self.cam.far, # 100
            )
            out = renderer.render(cur_cam, override_color=color, time=i/self.num_frames, stage="s1", latent_index=latent_index) # dict_keys(['image', 'depth', 'alpha', 'viewspace_points', 'visibility_filter', 'radii', 'pts_t', 'cpts_t'])
            img = out["image"].detach().cpu().permute(1, 2, 0).numpy() * 255 # (H, W, 3)
            img = img.astype('uint8')
            frames.append(img)
            if i == 0:
                cpts_tmp = out["cpts_t"]
            cpts_t = out["cpts_t"] # (n_nodes, 3)
            cpts_tra += torch.dist(cpts_t, cpts_tmp, p=2)
            cpts_tmp = cpts_t
            traj_pts_3d.append(cpts_t)

            cpts_3d = torch.cat([cpts_t, torch.ones_like(cpts_t[..., :1])], dim=-1) # (n_nodes, 4)
            cpts_2d = cpts_3d @ cur_cam.full_proj_transform # (n_nodes, 4) @ (4, 4) -> (n_nodes, 4) 
            cpts_2d = cpts_2d[..., :2] / cpts_2d[..., -1:] # (n_nodes, 2)
            cpts_2d = (cpts_2d + 1) / 2 * torch.tensor([cur_cam.image_height, cur_cam.image_width], device=device)
            traj_pts.append(cpts_2d)

        print("cpts average moving length: ", cpts_tra.item()) 
        save_name = self.opt.save_path.split("/")[-1].split(".")[0]
        if render_type == "fixed":
            video_name = video_save_dir + '/{}_{}_cpts_{}.mp4'.format(save_name, motion_video_name, self.opt.test_azi)
        else:
            video_name = video_save_dir + '/{}_{}_cpts_circle.mp4'.format(save_name, motion_video_name)
        imageio.mimwrite(video_name, frames, fps=8, quality=8, macro_block_size=1)

        traj_pts = torch.stack(traj_pts, dim=1).detach().cpu().numpy() # (n_nodes, num_frames, 2)
        traj_imgs_3d = []
        if render_type == "fixed":
            gs_num = traj_pts.shape[0]
            color_map = cm.get_cmap("jet")
            colors = np.array([np.array(color_map(i/max(1, float(gs_num - 1)))[:3]) * 255 for i in range(gs_num)], dtype=np.int32)
            alpha_img = np.zeros([cur_cam.image_height, cur_cam.image_width, 3]) 
            traj_img = np.zeros([cur_cam.image_height, cur_cam.image_width, 3]) 
            for i in range(gs_num):            
                alpha_img = cv2.polylines(img=alpha_img, pts=[traj_pts[i].astype(np.int32)], isClosed=False, color=[1, 1, 1], thickness=1)
                color = colors[i] / 255
                traj_img = cv2.polylines(img=traj_img, pts=[traj_pts[i].astype(np.int32)], isClosed=False, color=[float(color[0]), float(color[1]), float(color[2])], thickness=1)
            traj_img = np.concatenate([traj_img, alpha_img[..., :1]], axis=-1) * 255
            traj_img_save_path = os.path.join(video_save_dir, f'trajectory_{motion_video_name}.png')
            Image.fromarray(traj_img.astype('uint8')).save(traj_img_save_path)
            print(f'[INFO] save trajectory image to {os.path.abspath(traj_img_save_path)}...')

            traj_pts_3d = torch.stack(traj_pts_3d, dim=0).detach().cpu().numpy() # (num_frames, n_nodes, 3)
            visibles = np.ones((traj_pts_3d.shape[0], traj_pts_3d.shape[1]), dtype=bool) # (num_frames, n_nodes)
            # html_3d_traj = get_interactive_3d_visualization(traj_pts_3d)
            # with open(f'trajectory_3d_{motion_video_name}.html', 'w') as f:
            #     f.write(html_3d_traj)
            traj_vis_3d = plot_3d_tracks(traj_pts_3d, visibles=visibles, tracks_leave_trace=8) # (num_frames, H, W, 3)
            traj_imgs_3d.append(traj_vis_3d)
            traj_vis_3d_save_path = os.path.join(video_save_dir, f'trajectory_3d_{motion_video_name}.mp4')
            imageio.mimwrite(traj_vis_3d_save_path, traj_vis_3d, fps=8, quality=8, macro_block_size=1)
            print(f'[INFO] save 3D trajectory video to {os.path.abspath(traj_vis_3d_save_path)}...')

        # blend the trajectory with the rendered image
        traj_imgs = []
        if render_type == "fixed":
            for frame_idx, frame in enumerate(frames):
                traj_img_cur = np.zeros([cur_cam.image_height, cur_cam.image_width, 3])
                for i in range(gs_num):            
                    color = colors[i] / 255
                    traj_img_cur = cv2.polylines(img=traj_img_cur, pts=[traj_pts[i, :frame_idx+1].astype(np.int32)], isClosed=False, color=[float(color[0]), float(color[1]), float(color[2])], thickness=2)
                    traj_img_cur = cv2.circle(traj_img_cur, tuple(traj_pts[i, frame_idx].astype(np.int32)), 2, [float(color[0]), float(color[1]), float(color[2])], -1, lineType=cv2.LINE_AA)
                traj_img_cur = (traj_img_cur * 255).astype('uint8')
                traj_imgs.append(traj_img_cur)

        return traj_imgs, traj_imgs_3d[0]



if __name__ == "__main__":
    import argparse
    from omegaconf import OmegaConf

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="./configs/train_config.yaml", type=str, help="path to the yaml config file")
    args, extras = parser.parse_known_args()

    # override default config from cli
    opt = OmegaConf.merge(OmegaConf.load(args.config), OmegaConf.from_cli(extras))

    gui = GUI(opt)

    if opt.train_dynamic:
        gui.train_dynamic(opt.iters_s1, opt.iters_s2, opt.load_stage)
    else:
        gui.test(render_type=opt.render_type)
