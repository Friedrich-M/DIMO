import os
import cv2
import tqdm
import torch
import imageio
import json
from PIL import Image
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
from matplotlib import cm

import warnings; warnings.filterwarnings("ignore")

import rembg
import time

from src.text_embs import MLPEncoder, get_motion_embs
from utils.cam_utils import orbit_camera, OrbitCamera
from utils.vis_utils import get_interactive_3d_visualization
from renderer.latent_gs_renderer import MiniCam
from renderer.latent_gs_renderer import Renderer as LatentRenderer
from renderer.gaussian_gs_renderer import Renderer as GaussianRenderer
from knn_cuda import KNN
import pytorch3d.ops as ops
from src.helpers import plot_3d_tracks, plot_singel_3d_tracks
import math
from chamferdist import ChamferDistance
from fused_ssim import fused_ssim
import lpips
import random


class GUI:
    def __init__(self, opt):
        self.opt = opt  # shared with the trainer's opt to support in-place modification of rendering parameters.
        self.W = opt.W
        self.H = opt.H
        self.num_frames = opt.get("num_frames", 21)
        self.num_views = opt.get("num_views", 9)
        self.cam = OrbitCamera(opt.W, opt.H, r=opt.radius, fovy=opt.fovy)

        self.seed = opt.seed
        self.seed_everything()

        # models
        self.device = torch.device("cuda")
        self.bg_remover = None
        self.use_gaussian_renderer = opt.vae_latent

        if 'info.json' in os.listdir(self.opt.input_folder):
            with open(os.path.join(self.opt.input_folder, 'info.json'), 'r') as f:
                info = json.load(f)
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
            self.azimuths = [360/self.num_views*i for i in range(self.num_views)]
            self.full_azimuths = [360/self.num_frames*i for i in range(self.num_frames)]
            self.elevations = [self.opt.elevation for i in range(self.num_views)]

        if self.opt.test_motion or self.opt.test_unaligned_motion:
            assert self.opt.test_motion_data is not None, "Please provide the test data folder!!!"
            self.test_motion_imgs = torch.zeros((self.num_views, self.num_frames, 1, 3, self.opt.ref_size, self.opt.ref_size)) # (num_views, num_frames, 1, 3, H, W)
            self.test_motion_masks = torch.zeros((self.num_views, self.num_frames, 1, 1, self.opt.ref_size, self.opt.ref_size)) # (num_views, num_frames, 1, 1, H, W)
            self.test_motion_times = [i/self.num_frames for i in range(self.num_frames)]
            for view_idx in range(self.num_views):
                for frame_idx in range(self.num_frames):
                    image_path = os.path.join(self.opt.test_motion_data, f"view_{view_idx:02d}", f"{frame_idx:02d}.png")
                    img_torch, mask_torch = self.load_input(image_path, self.opt.ref_size)
                    self.test_motion_imgs[view_idx, frame_idx] = img_torch
                    self.test_motion_masks[view_idx, frame_idx] = mask_torch

        # renderer  
        if self.use_gaussian_renderer:
            self.renderer = GaussianRenderer(
                sh_degree=self.opt.sh_degree, # degree of spherical harmonics
                num_latent_code=len(self.input_videos), # number of latent codes
                latent_code_dim=self.opt.latent_code_dim, # dimension of latent code
                add_normal=self.opt.add_normal, # whether to render normal
            )
            if self.opt.test_unaligned_motion:
                self.renderer_control = GaussianRenderer(
                    sh_degree=self.opt.sh_degree, # degree of spherical harmonics
                    latent_code_dim=self.opt.latent_code_dim, # dimension of latent code
                    add_normal=self.opt.add_normal, # whether to render normal
                )
        else:
            self.renderer = LatentRenderer(
                sh_degree=self.opt.sh_degree, # degree of spherical harmonics
                num_latent_code=len(self.input_videos), # number of latent codes
                latent_code_dim=self.opt.latent_code_dim, # dimension of latent code
                add_normal=self.opt.add_normal, # whether to render normal
            )
            if self.opt.test_unaligned_motion:
                self.renderer_control = LatentRenderer(
                    sh_degree=self.opt.sh_degree, # degree of spherical harmonics
                    latent_code_dim=self.opt.latent_code_dim, # dimension of latent code
                    add_normal=self.opt.add_normal, # whether to render normal
                )

        if self.opt.test_motion or self.opt.test_unaligned_motion:
            tb_dir = os.path.join(opt.video_save_dir, "tb")
            os.makedirs(tb_dir, exist_ok=True)
            from torch.utils.tensorboard import SummaryWriter
            self.tb_writer = SummaryWriter(log_dir=tb_dir)

            self.optimizer = None
            self.step = 0
            self.train_steps = 1
            self.stage = "s2"
            # initialize loss functions
            self.chamferDist = ChamferDistance()
            self.lpips_loss = lpips.LPIPS(net='vgg').to(self.device)

        if self.opt.test_language:
            self.mlp_proj = MLPEncoder()
            try:
                ckpt_path = 'mlp_encoder.pth'
                self.mlp_proj.load_state_dict(torch.load(ckpt_path))
            except:
                ckpt_path = '/scratch/bbsh/linzhan/log/motion_embedding_caches/trump/mlp_encoder.pth'
                self.mlp_proj.load_state_dict(torch.load(ckpt_path))
            print(f"[INFO] load projector encoder from {ckpt_path}")
            self.mlp_proj.to(self.device)
            self.mlp_proj.eval()
        else:
            self.render_videos = self.opt.render_videos if self.opt.render_videos is not None else self.input_videos
            self.render_videos = self.render_videos if isinstance(self.render_videos, list) else self.render_videos.split(",")
            print(f"[INFO] render {len(self.render_videos)} videos: {self.render_videos}")

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

    def load_pcd(self, g):
        load_stage = self.opt.load_stage or self.opt.test_stage
        path1 = "{}/{}/point_cloud.ply".format(self.opt.save_path, load_stage)
        path2 = "{}/{}/point_cloud_c.ply".format(self.opt.save_path, load_stage)
        if self.opt.test_step:
            path1 = path1.split('.')[0] + "_{}".format(self.opt.test_step) + '.ply'
            if load_stage > "s1":
                path2 = path2.split('.')[0] + "_{}".format(self.opt.test_step) + '.ply'
        if load_stage < "s2":
            path2 = None
        g.load_ply(path1, path2)
    
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


    def test(self, test_cpts=True, render_type="fixed", downsample_indexs=None):
        video_save_dir = self.opt.video_save_dir
        os.makedirs(video_save_dir, exist_ok=True)
        test_stage = self.opt.test_stage
        self.load_model(g=self.renderer.gaussians)
        if self.use_gaussian_renderer:
            assert len(self.renderer.gaussians._mu) == len(self.input_videos) and len(self.renderer.gaussians._log_var) == len(self.input_videos), "Number of latent codes does not match the number of input videos!!!"
        else:
            assert len(self.renderer.gaussians._latent_codes) == len(self.input_videos), "Number of latent codes does not match the number of input videos!!!"
        
        ## downsample the point cloud
        # downsample_indexs = ops.sample_farthest_points(points=self.renderer.gaussians._c_xyz.unsqueeze(0), K=250)[1][0]

        if test_stage >= "s2":
            self.find_knn(g=self.renderer.gaussians)  
        
        all_imgs, all_traj_imgs, all_traj_imgs_3d = [], [], []
        len_latents = len(self.renderer.gaussians._mu) if self.use_gaussian_renderer else len(self.renderer.gaussians._latent_codes)
        for video_index, motion_video_name in enumerate(self.input_videos[:len_latents]):
            if motion_video_name not in self.render_videos:
                continue
            traj_imgs = []
            if test_cpts: # 1
                trajs, traj_imgs, traj_imgs_3d = self.test_cpts(test_stage=self.opt.test_stage, render_type=render_type, latent_index=video_index, motion_video_name=motion_video_name, downsample_indexs=downsample_indexs)
                if len(traj_imgs_3d) > 0:
                    all_traj_imgs_3d.append(np.stack(traj_imgs_3d, axis=0).squeeze()) # (num_frames, H, W, 3)
            frames = []
            if render_type == "fixed":
                for i in range(self.num_frames): # 21
                    test_azi = self.opt.test_azi # 0
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
            else:
                for view_idx, azi in enumerate(self.azimuths):
                    view_frames = []
                    for frame_idx in range(self.num_frames):
                        save_view_dir = os.path.join(video_save_dir, motion_video_name, f'view_{view_idx:02d}')
                        if not os.path.exists(save_view_dir):
                            os.makedirs(save_view_dir)
                        # os.makedirs(save_view_dir, exist_ok=True)
                        pose = orbit_camera(
                            elevation=self.opt.elevation, 
                            azimuth=azi, 
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
                        out = self.renderer.render(cur_cam, time=frame_idx/self.num_frames, stage=test_stage, latent_index=video_index)
                        img = out["image"].detach().cpu().permute(1, 2, 0).numpy() * 255
                        img = img.astype('uint8')
                        img_save_path = os.path.join(save_view_dir, f"{frame_idx:02d}.png")
                        Image.fromarray(img).save(img_save_path)
                        frames.append(img)
                        view_frames.append(img)
                    video_name = video_save_dir + '/{}_{}_{}_view_{}.mp4'.format(self.opt.save_path.split("/")[-1].split(".")[0], motion_video_name, test_stage, view_idx)
                    imageio.mimwrite(video_name, view_frames, fps=8, quality=8, macro_block_size=1)
                
            all_imgs.append(np.stack(frames, axis=0)) # (num_frames, H, W, 3)
            save_name = self.opt.save_path.split("/")[-1].split(".")[0]
            if render_type == "fixed":
                video_name = video_save_dir + '/{}_{}_{}_fixed.mp4'.format(save_name, motion_video_name, test_stage)
            else:
                video_name = video_save_dir + '/{}_{}_{}_circle.mp4'.format(save_name, motion_video_name, test_stage)
            # imageio.mimwrite(video_name, frames, fps=8, quality=8, macro_block_size=1)
            # save the frames

            if render_type == "fixed":
                save_frames_dir = os.path.join(video_save_dir, f'{save_name}_{motion_video_name}_{test_stage}')
                os.makedirs(save_frames_dir, exist_ok=True)
                for i, frame in enumerate(frames):
                    frame_path = os.path.join(save_frames_dir, f'{i}.png')
                    Image.fromarray(frame.astype('uint8')).save(frame_path)

            if test_cpts and render_type == "fixed":
                # save the 2d traj
                # last_frame = frames[-1] # (H, W, 3)
                # previous_motion_blende_frames_idxs = [1, 5, 8, 12, 14]
                # previous_motion_blende_frames = []
                # for pre_idx in previous_motion_blende_frames_idxs:
                #     gt_frame_path = os.path.join(self.opt.input_folder, motion_video_name, "view_00", f"{pre_idx:02d}.png")
                #     gt_frame = cv2.imread(gt_frame_path, cv2.IMREAD_COLOR)
                #     gt_frame = cv2.cvtColor(gt_frame, cv2.COLOR_BGR2RGB)
                #     gt_frame = cv2.resize(gt_frame, (self.W, self.H))
                #     previous_motion_blende_frames.append(gt_frame)
                last_gt_frame_path = os.path.join(self.opt.input_folder, motion_video_name, "view_00", f"{self.num_frames-1:02d}.png")
                last_frame = cv2.imread(last_gt_frame_path, cv2.IMREAD_COLOR)
                last_frame = cv2.cvtColor(last_frame, cv2.COLOR_BGR2RGB)
                last_frame = cv2.resize(last_frame, (self.W, self.H))   

                # 对last frame和previous frames进行alpha blend
                # blende_frame = previous_motion_blende_frames[0]
                # for i, pre_frame in enumerate(previous_motion_blende_frames[1:]):
                #     blende_frame += pre_frame
                # blende_frame = blende_frame / (len(previous_motion_blende_frames) + 1)
                # blende_frame = blende_frame.astype(np.int8)
                # last_frame = cv2.addWeighted(blende_frame, 0.3, last_frame, 0.7, 0)
                # save the last frame
                # last_frame_save_path = 'test_last_frame.png'
                # Image.fromarray(last_frame.astype('uint8')).save(last_frame_save_path)

                # last_frame = last_frame.astype(np.float32) / 255

                last_frame = cv2.cvtColor(last_frame.astype(np.uint8), cv2.COLOR_RGB2GRAY)[..., None]
                mask = trajs[-1][..., -1:] / 255
                # traj_img = last_frame * (1 - mask) + trajs[-1][..., :3] * mask
                traj_img = last_frame * (1 - mask) + trajs[-1][..., :3] * mask
                traj_img_save_path = os.path.join(video_save_dir, f'last_trajectory_{motion_video_name}_{test_stage}.png')
                Image.fromarray(traj_img.astype('uint8')).save(traj_img_save_path)
                print(f'[INFO] save last trajectory image to {os.path.abspath(traj_img_save_path)}...')

                blend_imgs = []
                for i in range(self.num_frames):
                    blender_img = cv2.addWeighted(frames[i], 0.3, traj_imgs[i], 0.7, 0)
                    blend_imgs.append(blender_img)

                save_path = os.path.join(video_save_dir, f'{save_name}_{motion_video_name}_blend.mp4')
                # imageio.mimwrite(save_path, blend_imgs, fps=8, quality=8, macro_block_size=1)
                # print(f'[INFO] trajectory video saved to {os.path.abspath(save_path)}')

                all_traj_imgs.append(np.stack(traj_imgs, axis=0)) # (num_frames, H, W, 3)
            
        if test_cpts and len(traj_imgs) > 0 and len(traj_imgs_3d) > 0:
            all_row_traj_imgs, all_row_traj_imgs_3d, all_row_render_imgs = [], [], []
            n_rows = math.floor(math.sqrt(len(self.input_videos)))
            rows_len = len(all_traj_imgs_3d) // n_rows
            for row_index in range(n_rows):
                row_traj_imgs = np.concatenate(all_traj_imgs[row_index*rows_len:(row_index+1)*rows_len], axis=2).squeeze() # (num_frames, H, rows_len*W, 3)
                all_row_traj_imgs.append(row_traj_imgs) 
                row_traj_imgs_3d = np.concatenate(all_traj_imgs_3d[row_index*rows_len:(row_index+1)*rows_len], axis=2).squeeze() # (num_frames, H, rows_len*W, 3)
                all_row_traj_imgs_3d.append(row_traj_imgs_3d) 
                row_imgs = np.concatenate(all_imgs[row_index*rows_len: (row_index+1)*rows_len], axis=2).squeeze() # (num_frames, H, rows_len*W, 3)
                all_row_render_imgs.append(row_imgs)
            all_row_traj_imgs = np.concatenate(all_row_traj_imgs, axis=1).squeeze() 
            save_path = 'all_traj_imgs.mp4'
            imageio.mimwrite(save_path, all_row_traj_imgs, fps=8, quality=8, macro_block_size=1)
            print(f'[INFO] all-in-one trajectory video saved to {os.path.abspath(save_path)}')
            all_row_traj_imgs_3d = np.concatenate(all_row_traj_imgs_3d, axis=1).squeeze()
            save_path = 'all_traj_imgs_3d.mp4'
            imageio.mimwrite(save_path, all_row_traj_imgs_3d, fps=8, quality=8, macro_block_size=1)
            print(f'[INFO] all-in-one 3d trajectory video saved to {os.path.abspath(save_path)}')
            all_row_render_imgs = np.concatenate(all_row_render_imgs, axis=1).squeeze()
            save_path = 'all_render_imgs.mp4'
            imageio.mimwrite(save_path, all_row_render_imgs, fps=8, quality=8, macro_block_size=1)
            print(f'[INFO] all-in-one video saved to {os.path.abspath(save_path)}')


    def test_cpts(self, test_stage="s1", render_type="fixed", sh_degree=0, latent_index=0, motion_video_name=None, downsample_indexs=None, tracks_leave_trace=5):
        video_save_dir = self.opt.video_save_dir
        if self.use_gaussian_renderer:
            renderer = GaussianRenderer(sh_degree=sh_degree, add_normal=self.opt.add_normal)
        else:
            renderer = LatentRenderer(sh_degree=sh_degree, add_normal=self.opt.add_normal)

        if test_stage > "s1":
            if downsample_indexs is not None:
                renderer.initialize(num_pts=downsample_indexs.shape[0], num_cpts=downsample_indexs.shape[0])
                renderer.gaussians._xyz = self.renderer.gaussians._c_xyz[downsample_indexs]
            else:
                renderer.initialize(num_pts=self.renderer.gaussians._c_xyz.shape[0], num_cpts=self.renderer.gaussians._c_xyz.shape[0])
                renderer.gaussians._xyz = self.renderer.gaussians._c_xyz # (n_nodes, 3)
        else:
            if downsample_indexs is not None:
                renderer.initialize(num_pts=downsample_indexs.shape[0], num_cpts=downsample_indexs.shape[0])
                renderer.gaussians._xyz = self.renderer.gaussians._xyz[downsample_indexs]
            else:
                renderer.initialize(num_pts=self.renderer.gaussians._xyz.shape[0], num_cpts=self.renderer.gaussians._c_xyz.shape[0])
                renderer.gaussians._xyz = self.renderer.gaussians._xyz
        renderer.gaussians._r = torch.ones((1), device="cuda", requires_grad=True) * -5.0
        if self.use_gaussian_renderer:
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
            traj_pts_3d.append(cpts_t) # 3d points
            cpts_tra += torch.dist(cpts_t, cpts_tmp, p=2)
            cpts_tmp = cpts_t

            cpts_3d = torch.cat([cpts_t, torch.ones_like(cpts_t[..., :1])], dim=-1) # (n_nodes, 4)
            cpts_2d = cpts_3d @ cur_cam.full_proj_transform # (n_nodes, 4) @ (4, 4) -> (n_nodes, 4) 
            cpts_2d = cpts_2d[..., :2] / cpts_2d[..., -1:] # (n_nodes, 2)
            cpts_2d = (cpts_2d + 1) / 2 * torch.tensor([cur_cam.image_height, cur_cam.image_width], device=device)
            traj_pts.append(cpts_2d)

        print("cpts average moving length: ", cpts_tra.item()) 
        # save_name = self.opt.save_path.split("/")[-1].split(".")[0]
        # if render_type == "fixed":
        #     video_name = video_save_dir + '/{}_{}_cpts_{}.mp4'.format(save_name, motion_video_name, self.opt.test_azi)
        # else:
        #     video_name = video_save_dir + '/{}_{}_cpts_circle.mp4'.format(save_name, motion_video_name)
        # imageio.mimwrite(video_name, frames, fps=8, quality=8, macro_block_size=1)

        traj_pts = torch.stack(traj_pts, dim=1).detach().cpu().numpy() # (n_nodes, num_frames, 2)
        trajs, traj_imgs_3d = [], []
        if render_type == "fixed":
            gs_num = traj_pts.shape[0]
            # color_map = cm.get_cmap("jet")
            color_map = cm.get_cmap("hsv")
            colors = np.array([np.array(color_map(i/max(1, float(gs_num - 1)))[:3]) * 255 for i in range(gs_num)], dtype=np.int32)
            alpha_img = np.zeros([cur_cam.image_height, cur_cam.image_width, 3]) 
            traj_img = np.zeros([cur_cam.image_height, cur_cam.image_width, 3]) 
            for i in range(gs_num):            
                alpha_img = cv2.polylines(img=alpha_img, pts=[traj_pts[i].astype(np.int32)], isClosed=False, color=[1, 1, 1], thickness=2)
                color = colors[i] / 255
                traj_img = cv2.polylines(img=traj_img, pts=[traj_pts[i].astype(np.int32)], isClosed=False, color=[float(color[0]), float(color[1]), float(color[2])], thickness=2)
            traj_img = np.concatenate([traj_img, alpha_img[..., :1]], axis=-1) * 255 # (H, W, 4)
            trajs.append(traj_img)
            traj_img_save_path = os.path.join(video_save_dir, f'trajectory_{motion_video_name}.png')
            Image.fromarray(traj_img.astype('uint8')).save(traj_img_save_path)
            print(f'[INFO] save trajectory image to {os.path.abspath(traj_img_save_path)}...')

            traj_pts_3d = torch.stack(traj_pts_3d, dim=0).detach().cpu().numpy() # (num_frames, n_nodes, 3)
            visibles = np.ones((traj_pts_3d.shape[0], traj_pts_3d.shape[1]), dtype=bool) # (num_frames, n_nodes)
            traj_3d_img = plot_singel_3d_tracks(traj_pts_3d, visibles=visibles) # (H, W, 4)
            traj_3d_img_save_path = os.path.join(video_save_dir, f'trajectory_3d_{motion_video_name}.png')
            Image.fromarray(traj_3d_img.astype('uint8')).save(traj_3d_img_save_path)
            print(f'[INFO] save 3D trajectory image to {os.path.abspath(traj_3d_img_save_path)}...')

            # html_3d_traj = get_interactive_3d_visualization(traj_pts_3d)
            # with open(f'trajectory_3d_{motion_video_name}.html', 'w') as f:
            #     f.write(html_3d_traj)

            traj_vis_3d = plot_3d_tracks(traj_pts_3d, visibles=visibles, tracks_leave_trace=tracks_leave_trace) # (num_frames, H, W, 3)
            traj_imgs_3d.append(traj_vis_3d)
            # traj_vis_3d_save_path = os.path.join(video_save_dir, f'trajectory_3d_{motion_video_name}.mp4')
            # imageio.mimwrite(traj_vis_3d_save_path, traj_vis_3d, fps=8, quality=8, macro_block_size=1)
            # print(f'[INFO] save 3D trajectory video to {os.path.abspath(traj_vis_3d_save_path)}...')

        # blend the trajectory with the rendered image
        traj_imgs = []
        if render_type == "fixed":
            for frame_idx, frame in enumerate(frames):
                traj_img_cur = np.zeros([cur_cam.image_height, cur_cam.image_width, 3])
                alpha_img_cur = np.zeros([cur_cam.image_height, cur_cam.image_width, 3]) 
                for i in range(gs_num):      
                    color = colors[i] / 255
                    # traj_img_cur = cv2.polylines(img=traj_img_cur, pts=[traj_pts[i, :frame_idx+1].astype(np.int32)], isClosed=False, color=[float(color[0]), float(color[1]), float(color[2])], thickness=2)
                    traj_img_cur = cv2.polylines(img=traj_img_cur, pts=[traj_pts[i, max(0, frame_idx-tracks_leave_trace):frame_idx+1].astype(np.int32)], isClosed=False, color=[float(color[0]), float(color[1]), float(color[2])], thickness=2)
                    alpha_img_cur = cv2.polylines(img=alpha_img_cur, pts=[traj_pts[i, max(0, frame_idx-tracks_leave_trace):frame_idx+1].astype(np.int32)], isClosed=False, color=[1, 1, 1], thickness=2)
                    traj_img_cur = cv2.circle(traj_img_cur, tuple(traj_pts[i, frame_idx].astype(np.int32)), 2, [float(color[0]), float(color[1]), float(color[2])], -1, lineType=cv2.LINE_AA)
                    alpha_img_cur = cv2.circle(alpha_img_cur, tuple(traj_pts[i, frame_idx].astype(np.int32)), 2, [1, 1, 1], -1, lineType=cv2.LINE_AA)
                traj_img_cur = np.concatenate([traj_img_cur, alpha_img_cur[..., :1]], axis=-1) * 255 # (H, W, 4)
                traj_imgs.append(traj_img_cur)

        return trajs, traj_imgs, traj_imgs_3d   


    def test_interpolation(self, render_type="fixed", downsample_indexs=None):
        video_save_dir = self.opt.video_save_dir
        os.makedirs(video_save_dir, exist_ok=True)
        test_stage = self.opt.test_stage
        self.load_model(g=self.renderer.gaussians)
        if self.use_gaussian_renderer:
            assert len(self.renderer.gaussians._mu) == len(self.input_videos) and len(self.renderer.gaussians._log_var) == len(self.input_videos), "Number of latent codes does not match the number of input videos!!!"
        else:
            assert len(self.renderer.gaussians._latent_codes) == len(self.input_videos), "Number of latent codes does not match the number of input videos!!!"
        
        video_name_1 = '04-032041'
        video_name_2 = '11-raise'
        latent_index_1 = self.input_videos.index(video_name_1)
        latent_index_2 = self.input_videos.index(video_name_2)
        self.renderer.gaussians._latent_codes = ((self.renderer.gaussians._latent_codes[latent_index_1] + self.renderer.gaussians._latent_codes[latent_index_2]) / 2).unsqueeze(0).repeat(len(self.input_videos), 1) # linear interpolation
        motion_video_name = f'intp_{video_name_1}_{video_name_2}'

        # downsample the point cloud
        # downsample_indexs = ops.sample_farthest_points(points=self.renderer.gaussians._c_xyz.unsqueeze(0), K=250)[1][0]

        self.find_knn(g=self.renderer.gaussians) 
        trajs, traj_imgs, traj_imgs_3d = self.test_cpts(test_stage=test_stage, render_type=render_type, latent_index=0, motion_video_name=motion_video_name, downsample_indexs=downsample_indexs)
        if len(traj_imgs_3d) > 0:
            imageio.mimwrite(f'{video_save_dir}/traj_3d_{motion_video_name}.mp4', traj_imgs_3d[-1], fps=8, quality=8, macro_block_size=1)

        frames = []
        for frame_idx in range(self.num_frames):
            test_azi = self.opt.test_azi # 0
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
            out = self.renderer.render(cur_cam, time=frame_idx/self.num_frames, stage='s2', latent_index=0)
            img = out["image"].detach().cpu().permute(1, 2, 0).numpy() * 255 # (H, W, 3)
            img = img.astype('uint8')
            frames.append(img)
        
        # save the frames
        save_frames_dir = os.path.join(video_save_dir, f'{motion_video_name}')
        os.makedirs(save_frames_dir, exist_ok=True)
        for i, frame in enumerate(frames):
            frame_path = os.path.join(save_frames_dir, f'{i}.png')
            Image.fromarray(frame.astype('uint8')).save(frame_path)
        imageio.mimwrite(os.path.join(video_save_dir, f'{motion_video_name}.mp4'), frames, fps=8, quality=8, macro_block_size=1)

        blended_imgs = []
        for traj_idx, traj in enumerate(traj_imgs):
            mask = traj[..., -1:] / 255 # (H, W, 1)
            blender_img = frames[traj_idx][..., :3].copy() # (H, W, 3)
            blender_img = cv2.cvtColor(blender_img, cv2.COLOR_RGB2GRAY)[..., None] # (H, W, 1)
            blender_img = blender_img * (1 - mask) + traj[..., :3] * mask # (H, W, 3)
            blended_imgs.append(blender_img.astype('uint8'))
        imageio.mimwrite(os.path.join(video_save_dir, f'{motion_video_name}_blend.mp4'), blended_imgs, fps=8, quality=8, macro_block_size=1)

        last_frame = img
        mask = trajs[-1][..., -1:] / 255
        traj_img = last_frame * (1 - mask) + trajs[-1][..., :3] * mask
        traj_img_save_path = os.path.join(video_save_dir, f'{motion_video_name}_last_trajectory.png')
        Image.fromarray(traj_img.astype('uint8')).save(traj_img_save_path)
        print(f'[INFO] save last trajectory image to {os.path.abspath(traj_img_save_path)}...')


    def test_language(self, render_type="fixed"):
        video_save_dir = self.opt.video_save_dir
        os.makedirs(video_save_dir, exist_ok=True)
        text_prompt = self.opt.test_text_prompt
        print(f"[INFO] text prompt: {text_prompt}")
        motion_embedding = get_motion_embs(descriptions=[text_prompt])
        motion_embedding = motion_embedding.to(self.device)
        latent_code = self.mlp_proj(motion_embedding)

        self.load_model(g=self.renderer.gaussians)
        if self.use_gaussian_renderer:
            assert len(self.renderer.gaussians._mu) == len(self.input_videos) and len(self.renderer.gaussians._log_var) == len(self.input_videos), "Number of latent codes does not match the number of input videos!!!"
        else:
            assert len(self.renderer.gaussians._latent_codes) == len(self.input_videos), "Number of latent codes does not match the number of input videos!!!"
        self.renderer.gaussians._latent_codes = latent_code

        self.find_knn(g=self.renderer.gaussians)
        trajs, traj_imgs, traj_imgs_3d = self.test_cpts(test_stage=self.opt.test_stage, render_type=render_type, latent_index=0)
        if len(traj_imgs_3d) > 0:
            imageio.mimwrite(os.path.join(video_save_dir, f'{text_prompt}_3d.mp4'), traj_imgs_3d[-1], fps=8, quality=8, macro_block_size=1)

        frames = []
        for frame_idx in range(self.num_frames):
            test_azi = self.opt.test_azi # 0
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
            out = self.renderer.render(cur_cam, time=frame_idx/self.num_frames, stage='s2', latent_index=0)
            img = out["image"].detach().cpu().permute(1, 2, 0).numpy() * 255 # (H, W, 3)
            img = img.astype('uint8')
            frames.append(img)
        # save the frames
        save_dir = os.path.join(video_save_dir, f'{text_prompt}')
        os.makedirs(save_dir, exist_ok=True)
        for i, frame in enumerate(frames):
            frame_path = os.path.join(save_dir, f'{i}.png')
            Image.fromarray(frame.astype('uint8')).save(frame_path)
        imageio.mimwrite(os.path.join(video_save_dir, f'{text_prompt}.mp4'), frames, fps=8, quality=8, macro_block_size=1)

        blended_imgs = []
        for traj_idx, traj in enumerate(traj_imgs):
            mask = traj[..., -1:] / 255 # (H, W, 1)
            blender_img = frames[traj_idx][..., :3].copy() # (H, W, 3)
            blender_img = cv2.cvtColor(blender_img, cv2.COLOR_RGB2GRAY)[..., None] # (H, W, 1)
            blender_img = blender_img * (1 - mask) + traj[..., :3] * mask # (H, W, 3)
            blended_imgs.append(blender_img.astype('uint8'))
        imageio.mimwrite(os.path.join(video_save_dir, f'{text_prompt}_blend.mp4'), blended_imgs, fps=8, quality=8, macro_block_size=1)

        last_frame = img
        # last_frame = cv2.cvtColor(last_frame.astype(np.uint8), cv2.COLOR_RGB2GRAY)[..., None]
        mask = trajs[-1][..., -1:] / 255
        # traj_img = last_frame * (1 - mask) + trajs[-1][..., :3] * mask
        traj_img = last_frame * (1 - mask) + trajs[-1][..., :3] * mask
        traj_img_save_path = os.path.join(video_save_dir, f'last_trajectory.png')
        Image.fromarray(traj_img.astype('uint8')).save(traj_img_save_path)
        print(f'[INFO] save last trajectory image to {os.path.abspath(traj_img_save_path)}...')


    def test_motion(self):
        video_save_dir = self.opt.video_save_dir
        os.makedirs(video_save_dir, exist_ok=True)
        self.load_model(g=self.renderer.gaussians)
        # Loading pretrained model
        if self.use_gaussian_renderer:
            assert len(self.renderer.gaussians._mu) == len(self.input_videos) and len(self.renderer.gaussians._log_var) == len(self.input_videos), "Number of latent codes does not match the number of input videos!!!"
        else:
            assert len(self.renderer.gaussians._latent_codes) == len(self.input_videos), "Number of latent codes does not match the number of input videos!!!"

        # finetuning the latent code 
        if self.use_gaussian_renderer:
            mu = torch.zeros_like(self.renderer.gaussians._mu[:1])
            log_var = torch.zeros_like(self.renderer.gaussians._log_var[:1])
            mu = nn.Parameter(mu.requires_grad_(True))
            log_var = nn.Parameter(log_var.requires_grad_(True))
            self.renderer.gaussians._mu = mu
            self.renderer.gaussians._log_var = log_var
        else:
            latent_codes = torch.randn_like(self.renderer.gaussians._latent_codes[:1])
            latent_codes = nn.Parameter(latent_codes.requires_grad_(True))
            self.renderer.gaussians._latent_codes = latent_codes 
        self.prepare_ft_latent()
        self.renderer.gaussians.lr_setup(self.opt)
        for _ in tqdm.trange(1000, desc="Finetuning latent code"):
            self.finetune_latent()

        # if self.use_gaussian_renderer:
        #     torch.save(self.renderer.gaussians._mu, os.path.join(self.opt.video_save_dir, f"mu.pth"))
        #     torch.save(self.renderer.gaussians._log_var, os.path.join(self.opt.video_save_dir, f"log_var.pth"))  
        # else:
        #     torch.save(self.renderer.gaussians._latent_codes, os.path.join(self.opt.video_save_dir, f"latent_codes.pth"))
        self.renderer.gaussians.save_ply(os.path.join(self.opt.video_save_dir, "point_cloud.ply"), os.path.join(self.opt.video_save_dir, "point_cloud_c.ply"))
        self.renderer.gaussians.save_model(self.opt.video_save_dir)

        trajs, traj_imgs, traj_imgs_3d = self.test_cpts(test_stage=self.opt.test_stage)
        if len(traj_imgs_3d) > 0:
            imageio.mimwrite(os.path.join(video_save_dir, f'3d.mp4'), traj_imgs_3d[-1], fps=8, quality=8, macro_block_size=1)
        frames, frame_diag = [], []
        for i in range(self.num_frames): # 21
            test_azi = self.opt.test_azi # 0
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
            out = self.renderer.render(cur_cam, time=i/self.num_frames, stage="s2")
            img = out["image"].detach().cpu().permute(1, 2, 0).numpy() * 255 # (H, W, 3)
            img = img.astype('uint8')
            frames.append(img)

            test_diag_azi = 360/self.num_frames*i
            pose_diag = orbit_camera(
                elevation=self.opt.elevation, 
                azimuth=test_diag_azi, 
                radius=self.opt.radius
            )
            cur_cam_diag = MiniCam(
                pose_diag, # (4, 4)
                self.W, # 800
                self.H, # 800
                self.cam.fovy, # 0.5916666164260777
                self.cam.fovx, # 0.5916666164260777
                self.cam.near, # 0.01
                self.cam.far, # 100
            )
            out_diag = self.renderer.render(cur_cam_diag, time=i/self.num_frames, stage="s2")
            img_diag = out_diag["image"].detach().cpu().permute(1, 2, 0).numpy() * 255
            img_diag = img_diag.astype('uint8')
            frame_diag.append(img_diag)

        # save the frames
        frames = np.stack(frames, axis=0) # (num_frames, H, W, 3)
        imgs_save_path = os.path.join(self.opt.video_save_dir, f'render_images.mp4')
        imageio.mimwrite(imgs_save_path, frames, fps=8, quality=8, macro_block_size=1)

        # save the diagonal frames
        frame_diag = np.stack(frame_diag, axis=0) # (num_frames, H, W, 3)
        imgs_diag_save_path = os.path.join(self.opt.video_save_dir, f'render_images_diag.mp4')
        imageio.mimwrite(imgs_diag_save_path, frame_diag, fps=8, quality=8, macro_block_size=1)

        mask = trajs[-1][..., -1:] / 255
        traj_img = img * (1 - mask) + trajs[-1][..., :3] * mask
        traj_img_save_path = os.path.join(self.opt.video_save_dir, f'last_trajectory.png')
        Image.fromarray(traj_img.astype('uint8')).save(traj_img_save_path)
        print(f'[INFO] save last trajectory image to {os.path.abspath(traj_img_save_path)}...')

        blended_imgs = []
        for traj_idx, traj in enumerate(traj_imgs):
            mask = traj[..., -1:] / 255 # (H, W, 1)
            blender_img = frames[traj_idx][..., :3].copy() # (H, W, 3)
            blender_img = cv2.cvtColor(blender_img, cv2.COLOR_RGB2GRAY)[..., None] # (H, W, 1)
            blender_img = blender_img * (1 - mask) + traj[..., :3] * mask # (H, W, 3)
            blended_imgs.append(blender_img.astype('uint8'))
        imageio.mimwrite(os.path.join(video_save_dir, f'blend.mp4'), blended_imgs, fps=8, quality=8, macro_block_size=1)

        frames = []
        test_azis = []
        for view_idx, azi in enumerate(self.azimuths):
            for frame_idx in range(self.num_frames):
                save_view_dir = os.path.join(video_save_dir, 'recon', f'view_{view_idx:02d}')
                if not os.path.exists(save_view_dir):
                    os.makedirs(save_view_dir)
                # os.makedirs(save_view_dir, exist_ok=True)
                pose = orbit_camera(
                    elevation=self.opt.elevation, 
                    azimuth=azi, 
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
                out = self.renderer.render(cur_cam, time=frame_idx/self.num_frames, stage="s2")
                img = out["image"].detach().cpu().permute(1, 2, 0).numpy() * 255
                img = img.astype('uint8')
                img_save_path = os.path.join(save_view_dir, f"{frame_idx:02d}.png")
                Image.fromarray(img).save(img_save_path)
                frames.append(img)

    
    def test_paper(self, render_type="fixed", downsample_indexs=None):
        video_save_dir = self.opt.video_save_dir
        os.makedirs(video_save_dir, exist_ok=True)
        self.load_model(g=self.renderer.gaussians)
        if self.use_gaussian_renderer:
            assert len(self.renderer.gaussians._mu) == len(self.input_videos) and len(self.renderer.gaussians._log_var) == len(self.input_videos), "Number of latent codes does not match the number of input videos!!!"
        else:
            assert len(self.renderer.gaussians._latent_codes) == len(self.input_videos), "Number of latent codes does not match the number of input videos!!!"
        
        self.find_knn(g=self.renderer.gaussians)  
        len_latents = len(self.renderer.gaussians._mu) if self.use_gaussian_renderer else len(self.renderer.gaussians._latent_codes)
        for video_index, motion_video_name in enumerate(self.input_videos[:len_latents]):
            if motion_video_name not in self.render_videos:
                continue
            trajs, traj_imgs, traj_imgs_3d = self.test_cpts(test_stage=self.opt.test_stage, render_type=render_type, latent_index=video_index, motion_video_name=motion_video_name, downsample_indexs=downsample_indexs)
            if len(traj_imgs_3d) > 0:
                imageio.mimwrite(os.path.join(video_save_dir, f'{motion_video_name}_traj_3d.mp4'), traj_imgs_3d[-1], fps=8, quality=8, macro_block_size=1)
                print(f'[INFO] save key point 3D trajectory video to {os.path.abspath(os.path.join(video_save_dir, f"{motion_video_name}_traj_3d.mp4"))}...')
            
            frames = []
            for frame_idx in range(self.num_frames):
                gt_frame_path = os.path.join(self.opt.input_folder, motion_video_name, "view_00", f"{frame_idx:02d}.png")
                frame = cv2.imread(gt_frame_path, cv2.IMREAD_COLOR)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, (self.W, self.H)) 
                frames.append(frame) 

            frames, frames_diag = [], []
            for i in range(self.num_frames): # 21
                test_azi = self.opt.test_azi # 0
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
                out = self.renderer.render(cur_cam, time=i/self.num_frames, stage=self.opt.test_stage, latent_index=video_index)
                img = out["image"].detach().cpu().permute(1, 2, 0).numpy() * 255 # (H, W, 3)
                img = img.astype('uint8')
                frames.append(img)

                test_diag_azi = 360/self.num_frames*i
                pose_diag = orbit_camera(
                    elevation=self.opt.elevation, 
                    azimuth=test_diag_azi, 
                    radius=self.opt.radius
                )
                cur_cam_diag = MiniCam(
                    pose_diag, # (4, 4)
                    self.W, # 800
                    self.H, # 800
                    self.cam.fovy, # 0.5916666164260777
                    self.cam.fovx, # 0.5916666164260777
                    self.cam.near, # 0.01
                    self.cam.far, # 100
                )
                out_diag = self.renderer.render(cur_cam_diag, time=i/self.num_frames, stage=self.opt.test_stage, latent_index=video_index)
                img_diag = out_diag["image"].detach().cpu().permute(1, 2, 0).numpy() * 255
                img_diag = img_diag.astype('uint8')
                frames_diag.append(img_diag)

            # save reference frames 
            frames = np.stack(frames, axis=0) # (num_frames, H, W, 3)
            image_save_path = os.path.join(video_save_dir, f'{motion_video_name}_refer_images.mp4')
            imageio.mimwrite(image_save_path, frames, fps=8, quality=8, macro_block_size=1)
            print(f'[INFO] save reference frames to {os.path.abspath(image_save_path)}...')

            # save diagonal frames
            frames_diag = np.stack(frames_diag, axis=0) # (num_frames, H, W, 3)
            image_diag_save_path = os.path.join(video_save_dir, f'{motion_video_name}_diag_images.mp4')
            imageio.mimwrite(image_diag_save_path, frames_diag, fps=8, quality=8, macro_block_size=1)
            print(f'[INFO] save diagonal frames to {os.path.abspath(image_diag_save_path)}...')

            blended_imgs = []
            for traj_idx, traj in enumerate(traj_imgs):
                mask = traj[..., -1:] / 255
                blender_img = frames[traj_idx][..., :3].copy()
                blender_img = cv2.cvtColor(blender_img, cv2.COLOR_RGB2GRAY)[..., None]
                blender_img = blender_img * (1 - mask) + traj[..., :3] * mask
                blended_imgs.append(blender_img.astype('uint8'))
            imageio.mimwrite(os.path.join(video_save_dir, f'{motion_video_name}_blend.mp4'), blended_imgs, fps=8, quality=8, macro_block_size=1)
            print(f'[INFO] save blended video to {os.path.abspath(os.path.join(video_save_dir, f"{motion_video_name}_blend.mp4"))}...')
            

    def test_fps(self, round=500):
        self.load_model(g=self.renderer.gaussians)
        self.find_knn(g=self.renderer.gaussians)
        test_pose = orbit_camera(
            elevation=self.opt.elevation, 
            azimuth=0,
            radius=self.opt.radius
        )
        test_cam = MiniCam(
            test_pose, # (4, 4)
            512, # 800
            512, # 800
            self.cam.fovy, # 0.5916666164260777
            self.cam.fovx, # 0.5916666164260777
            self.cam.near, # 0.01
            self.cam.far, # 100
        )
        self.renderer.render(test_cam, time=0, stage="s2")
        t0 = time.time()
        for i in range(round):
            self.renderer.render(test_cam, time=0, stage="s2")
        t1 = time.time()
        print(f"[INFO] FPS: {round/(t1-t0)}")

    
    def prepare_ft_latent(self):
        self.renderer.gaussians.training_setup(self.opt)
        self.optimizer = self.renderer.gaussians.optimizer
        # fix the learning rate of other parameters
        for param_group in self.optimizer.param_groups:
            if self.use_gaussian_renderer:
                if param_group["name"] != "latent_code_mu" and param_group["name"] != "latent_code_log_var":
                    param_group["lr"] = 0.0
            else:
                if param_group["name"] != "latent_code":
                    param_group["lr"] = 0.0 

    def finetune_latent(self):
        for _ in range(self.train_steps):
            self.step += 1
            
            for param_group in self.optimizer.param_groups:
                if self.use_gaussian_renderer:
                    if param_group["name"] == "latent_code_mu":
                        param_group['lr'] = self.renderer.gaussians.latent_code_scheduler_args(self.step)
                    elif param_group["name"] == "latent_code_log_var":
                        param_group['lr'] = self.renderer.gaussians.latent_code_scheduler_args(self.step)
                    else:
                        param_group['lr'] = 0.0
                else:
                    if param_group["name"] == "latent_code":
                        param_group['lr'] = self.renderer.gaussians.latent_code_scheduler_args(self.step)
                    else:
                        param_group['lr'] = 0.0
            
            # find knn
            if self.stage >= "s2":
                self.find_knn(g=self.renderer.gaussians, k=4) 
        
            loss = 0
            # gradually increasing the rendering resolution from 128x128 to 512x512.
            render_resolution = 128 if self.step < 100 else (256 if self.step < 200 else 512)
            
            # randomly sample {1+batch_size} frames x {batch_size} views for training
            batch_view_indexs = [0] + random.sample(range(1, self.num_views), self.opt.batch_size)
            batch_frame_indexs = random.sample(range(self.num_frames), self.opt.batch_size)

            gt_images = []
            gt_masks = []
            render_images = []
            render_masks = []
            # manually sample {batch_size} frames x {batch_size} views for training
            for view_index in batch_view_indexs:
                for frame_index in batch_frame_indexs:
                    gt_image = self.test_motion_imgs[view_index][frame_index].to(self.device) # [1, 3, H, W]
                    gt_mask = self.test_motion_masks[view_index][frame_index].to(self.device) # [1, 1, H, W]

                    pose = orbit_camera(self.opt.elevation, self.azimuths[view_index], self.opt.radius)
                    cur_cam = MiniCam(pose, render_resolution, render_resolution, self.cam.fovy, self.cam.fovx, self.cam.near, self.cam.far)
                    timestamp = self.test_motion_times[frame_index] # [0, 1)

                    out = self.renderer.render(cur_cam, time=timestamp, stage=self.stage)

                    render_image = out["image"].unsqueeze(0) # (1, 3, rH, rW)
                    render_images.append(render_image)
                    gt_image = F.interpolate(gt_image, (render_resolution, render_resolution), mode="bilinear", align_corners=False) # (1, 3, rH, rW)
                    gt_images.append(gt_image)

                    render_mask = out["alpha"].unsqueeze(0) # (1, 1, rH, rW)
                    render_masks.append(render_mask)
                    gt_mask = F.interpolate(gt_mask, (render_resolution, render_resolution), mode="bilinear", align_corners=False) # (1, 1, rH, rW)
                    gt_masks.append(gt_mask)

            render_images = torch.cat(render_images, dim=0) # (batch_size, 3, rH, rW)
            gt_images = torch.cat(gt_images, dim=0) # (batch_size, 3, rH, rW)
            render_masks = torch.cat(render_masks, dim=0) # (batch_size, 1, rH, rW)
            gt_masks = torch.cat(gt_masks, dim=0) # (batch_size, 1, rH, rW)
        
            ## pixel-level MSE loss
            mse_loss = F.mse_loss(render_images, gt_images)
            self.tb_writer.add_scalar("MSE Loss", mse_loss, self.step)
            loss = loss + self.opt.lambda_mse * mse_loss
            ## LPIPS loss
            lpips_loss = self.lpips_loss(render_images, gt_images).mean()
            self.tb_writer.add_scalar("LPIPS Loss", lpips_loss, self.step)
            loss = loss + self.opt.lambda_lpips * lpips_loss
            ## SSIM loss
            ssim_loss = 1 - fused_ssim(render_images, gt_images)
            self.tb_writer.add_scalar("SSIM Loss", ssim_loss, self.step)
            loss = loss + self.opt.lambda_ssim * ssim_loss
            ## Mask loss
            mask_loss = F.mse_loss(render_masks, gt_masks)
            self.tb_writer.add_scalar("Mask Loss", mask_loss, self.step)
            loss = loss + self.opt.lambda_mask * mask_loss

            self.tb_writer.add_images("Render Images", render_images[:1], self.step)
            self.tb_writer.add_images("GT Images", gt_images[:1], self.step)
            vis_latent_code = self.renderer.gaussians._latent_codes.detach().cpu().numpy()
            self.tb_writer.add_histogram("Latent Codes", vis_latent_code, self.step)

            if self.step % 10 == 0:
                debug_path = os.path.join(self.opt.video_save_dir, f"debug")
                os.makedirs(debug_path, exist_ok=True)
                gt_img_show = (gt_images[0].permute(1,2,0).clamp(0,1)[:,:,[2,1,0]]*255).detach().cpu().numpy().astype(np.uint8)
                img_show = (render_images[0].permute(1,2,0).clamp(0,1)[:,:,[2,1,0]]*255).detach().cpu().numpy().astype(np.uint8)
                os.makedirs(debug_path, exist_ok=True)
                image_to_show = np.concatenate([gt_img_show, img_show], axis=1)
                cv2.imwrite(os.path.join(debug_path, f"image_{self.stage}_{self.step}.png"), image_to_show)

            # optimize step
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

            # clear memory
            torch.cuda.empty_cache()

        torch.cuda.synchronize()

    
    def test_unaligned_motion(self):
        video_save_dir = self.opt.video_save_dir
        if not os.path.exists(video_save_dir):
            os.makedirs(video_save_dir)
        # Loading pretrained model
        self.load_model(g=self.renderer.gaussians)
        if self.use_gaussian_renderer:
            assert len(self.renderer.gaussians._mu) == len(self.input_videos) and len(self.renderer.gaussians._log_var) == len(self.input_videos), "Number of latent codes does not match the number of input videos!!!"
        else:
            assert len(self.renderer.gaussians._latent_codes) == len(self.input_videos), "Number of latent codes does not match the number of input videos!!!"

        latent_codes = torch.randn_like(self.renderer.gaussians._latent_codes[:1])
        latent_codes = nn.Parameter(latent_codes.requires_grad_(True))

        # finetuning the deformation of the key points
        self.load_model(g=self.renderer_control.gaussians)
        self.renderer_control.gaussians._latent_codes = latent_codes
        self.prepare_ft_latent_deformation_stage1()
        self.renderer_control.gaussians.lr_setup(self.opt)
        self.renderer_control.gaussians._timenet.deformnet.requires_grad_(False)
        self.cpts_s1 = []
        for _ in tqdm.trange(400, desc="Finetuning latent code"):
            self.finetune_latent_deformation_stage1()
        pretrained_latent_codes = self.renderer_control.gaussians._latent_codes.detach()
        # finetuning the latent code and deformation
        pretrained_latent_codes = nn.Parameter(pretrained_latent_codes.requires_grad_(True))
        self.renderer.gaussians._latent_codes = pretrained_latent_codes.to(self.device)
        pretrained_deformation_weights = self.renderer_control.gaussians._timenet.state_dict()
        self.renderer.gaussians._timenet.load_state_dict(pretrained_deformation_weights)
        self.renderer.gaussians._timenet.requires_grad_(True)
        self.renderer.gaussians._timenet.to(self.device)
        self.prepare_ft_latent_deformation()
        self.renderer.gaussians.lr_setup(self.opt)
        for _ in tqdm.trange(1000, desc="Finetuning latent code and deformation"):
            self.finetune_latent_deformation()

        if self.use_gaussian_renderer:
            torch.save(self.renderer.gaussians._mu, os.path.join(self.opt.video_save_dir, f"mu.pth"))
            torch.save(self.renderer.gaussians._log_var, os.path.join(self.opt.video_save_dir, f"log_var.pth"))
        else:
            torch.save(self.renderer.gaussians._latent_codes, os.path.join(self.opt.video_save_dir, f"latent_codes.pth"))
        torch.save(self.renderer.gaussians._timenet.state_dict(), os.path.join(self.opt.video_save_dir, "timenet.pth"))

        trajs, traj_imgs, traj_imgs_3d = self.test_cpts(test_stage=self.opt.test_stage)
        frames = []
        for i in range(self.num_frames): # 21
            test_azi = self.opt.test_azi # 0
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
            out = self.renderer.render(cur_cam, time=i/self.num_frames, stage="s2")
            img = out["image"].detach().cpu().permute(1, 2, 0).numpy() * 255 # (H, W, 3)
            img = img.astype('uint8')
            frames.append(img)

        frames = np.stack(frames, axis=0) # (num_frames, H, W, 3)
        imgs_save_path = os.path.join(self.opt.video_save_dir, f'render_images.mp4')
        imageio.mimwrite(imgs_save_path, frames, fps=8, quality=8, macro_block_size=1)

        mask = trajs[-1][..., -1:] / 255
        traj_img = img * (1 - mask) + trajs[-1][..., :3] * mask
        traj_img_save_path = os.path.join(self.opt.video_save_dir, f'last_trajectory.png')
        Image.fromarray(traj_img.astype('uint8')).save(traj_img_save_path)
        print(f'[INFO] save last trajectory image to {os.path.abspath(traj_img_save_path)}...')
    

    def prepare_ft_latent_deformation_stage1(self):
        self.stage = "s1"
        self.step = 0
        self.renderer_control.initialize(num_pts=self.renderer_control.gaussians._c_xyz.shape[0])
        self.renderer_control.gaussians._xyz = self.renderer_control.gaussians._c_xyz.detach().clone()
        self.renderer_control.gaussians._r = self.renderer_control.gaussians._c_radius.detach().clone()
        
        self.renderer_control.gaussians.training_setup(self.opt)
        self.optimizer_control = self.renderer_control.gaussians.optimizer
        # only optimize the latent code and deformation
        for param_group in self.optimizer_control.param_groups:
            # fix the learning rate of other parameters 
            if self.use_gaussian_renderer:
                if param_group["name"] != "latent_code_mu" and param_group["name"] != "latent_code_log_var" and param_group["name"] != "deform" and param_group["name"] != "scaling" and param_group["name"] != "opacity" and param_group["name"] != "f_dc" and param_group["name"] != "f_rest":
                    param_group["lr"] = 0.0
            else:
                if param_group["name"] != "latent_code" and param_group["name"] != "deform" and param_group["name"] != "scaling" and param_group["name"] != "opacity" and param_group["name"] != "f_dc" and param_group["name"] != "f_rest":
                    param_group["lr"] = 0.0 

    def finetune_latent_deformation_stage1(self):
        for _ in range(self.train_steps):
            self.step += 1
            
            loss = 0
            # gradually increasing the rendering resolution from 128x128 to 512x512.
            render_resolution = 128 if self.step < 100 else (256 if self.step < 200 else 512)
            
            # randomly sample {1+batch_size} frames x {batch_size} views for training
            batch_view_indexs = [0] + random.sample(range(1, self.num_views), self.opt.batch_size)
            batch_frame_indexs = random.sample(range(self.num_frames), self.opt.batch_size)

            gt_images = []
            gt_masks = []
            render_images = []
            render_masks = []
            # manually sample {batch_size} frames x {batch_size} views for training
            for view_index in batch_view_indexs:
                for frame_index in batch_frame_indexs:
                    gt_image = self.test_motion_imgs[view_index][frame_index].to(self.device) # [1, 3, H, W]
                    gt_mask = self.test_motion_masks[view_index][frame_index].to(self.device) # [1, 1, H, W]

                    pose = orbit_camera(self.opt.elevation, self.azimuths[view_index], self.opt.radius)
                    cur_cam = MiniCam(pose, render_resolution, render_resolution, self.cam.fovy, self.cam.fovx, self.cam.near, self.cam.far)
                    timestamp = self.test_motion_times[frame_index] # [0, 1)

                    out = self.renderer_control.render(cur_cam, time=timestamp, stage=self.stage)

                    render_image = out["image"].unsqueeze(0) # (1, 3, rH, rW)
                    render_images.append(render_image)
                    gt_image = F.interpolate(gt_image, (render_resolution, render_resolution), mode="bilinear", align_corners=False) # (1, 3, rH, rW)
                    gt_images.append(gt_image)

                    render_mask = out["alpha"].unsqueeze(0) # (1, 1, rH, rW)
                    render_masks.append(render_mask)
                    gt_mask = F.interpolate(gt_mask, (render_resolution, render_resolution), mode="bilinear", align_corners=False) # (1, 1, rH, rW)
                    gt_masks.append(gt_mask)

            render_images = torch.cat(render_images, dim=0) # (batch_size, 3, rH, rW)
            gt_images = torch.cat(gt_images, dim=0) # (batch_size, 3, rH, rW)
            render_masks = torch.cat(render_masks, dim=0) # (batch_size, 1, rH, rW)
            gt_masks = torch.cat(gt_masks, dim=0) # (batch_size, 1, rH, rW)
        
            ## pixel-level MSE loss
            mse_loss = F.mse_loss(render_images, gt_images)
            self.tb_writer.add_scalar("Stage1/MSE Loss", mse_loss, self.step)
            loss = loss + self.opt.lambda_mse * mse_loss
            ## LPIPS loss
            lpips_loss = self.lpips_loss(render_images, gt_images).mean()
            self.tb_writer.add_scalar("Stage1/LPIPS Loss", lpips_loss, self.step)
            loss = loss + self.opt.lambda_lpips * lpips_loss
            ## SSIM loss
            ssim_loss = 1 - fused_ssim(render_images, gt_images)
            self.tb_writer.add_scalar("Stage1/SSIM Loss", ssim_loss, self.step)
            loss = loss + self.opt.lambda_ssim * ssim_loss
            ## Mask loss
            mask_loss = F.mse_loss(render_masks, gt_masks)
            self.tb_writer.add_scalar("Stage1/Mask Loss", mask_loss, self.step)
            loss = loss + self.opt.lambda_mask * mask_loss

            self.tb_writer.add_images("Stage1/Render Images", render_images[:1], self.step)
            self.tb_writer.add_images("Stage1/GT Images", gt_images[:1], self.step)
            vis_latent_code = self.renderer_control.gaussians._latent_codes.detach().cpu().numpy()
            self.tb_writer.add_histogram("Stage1/Latent Codes", vis_latent_code, self.step)

            if self.step % 10 == 0:
                debug_path = os.path.join(self.opt.video_save_dir, f"debug")
                os.makedirs(debug_path, exist_ok=True)
                gt_img_show = (gt_images[0].permute(1,2,0).clamp(0,1)[:,:,[2,1,0]]*255).detach().cpu().numpy().astype(np.uint8)
                img_show = (render_images[0].permute(1,2,0).clamp(0,1)[:,:,[2,1,0]]*255).detach().cpu().numpy().astype(np.uint8)
                os.makedirs(debug_path, exist_ok=True)
                image_to_show = np.concatenate([gt_img_show, img_show], axis=1)
                cv2.imwrite(os.path.join(debug_path, f"image_{self.stage}_{self.step}.png"), image_to_show)

            # optimize step
            loss.backward()
            self.optimizer_control.step()
            self.optimizer_control.zero_grad()

            # clear memory
            torch.cuda.empty_cache()

        torch.cuda.synchronize()

    def prepare_ft_latent_deformation(self):
        self.stage = "s2"
        self.step = 0
        self.renderer.gaussians.training_setup(self.opt)
        self.optimizer = self.renderer.gaussians.optimizer
        # only optimize the latent code and deformation
        for param_group in self.optimizer.param_groups:
            # fix the learning rate of other parameters 
            if self.use_gaussian_renderer:
                if param_group["name"] != "latent_code_mu" and param_group["name"] != "latent_code_log_var" and param_group["name"] != "deform" and param_group["name"] != "deform_rot":
                    param_group["lr"] = 0.0
            else:
                if param_group["name"] != "latent_code" and param_group["name"] != "deform" and param_group["name"] != "deform_rot":
                    param_group["lr"] = 0.0 

    def finetune_latent_deformation(self):
        for _ in range(self.train_steps):
            self.step += 1
            
            for param_group in self.optimizer.param_groups:
                # fix the lr of other parameters, only optimize the latent code and deformation
                if self.use_gaussian_renderer:
                    if param_group["name"] == "latent_code_mu" or param_group["name"] == "latent_code_log_var":
                        param_group['lr'] = self.renderer.gaussians.latent_code_scheduler_args(self.step)
                    elif param_group["name"] == "deform" or param_group["name"] == "deform_rot":
                        param_group['lr'] = self.renderer.gaussians.deform_scheduler_args(self.step)
                    else:
                        param_group['lr'] = 0.0
                else:
                    if param_group["name"] == "latent_code":
                        param_group['lr'] = self.renderer.gaussians.latent_code_scheduler_args(self.step)
                    elif param_group["name"] == "deform" or param_group["name"] == "deform_rot":
                        param_group['lr'] = self.renderer.gaussians.deform_scheduler_args(self.step)
                    else:
                        param_group['lr'] = 0.0
            
            # find knn
            if self.stage >= "s2":
                self.find_knn(g=self.renderer.gaussians, k=4) 
        
            loss = 0
            # gradually increasing the rendering resolution from 128x128 to 512x512.
            render_resolution = 128 if self.step < 100 else (256 if self.step < 200 else 512)
            
            # randomly sample {1+batch_size} frames x {batch_size} views for training
            batch_view_indexs = [0] + random.sample(range(1, self.num_views), self.opt.batch_size)
            batch_frame_indexs = random.sample(range(self.num_frames), self.opt.batch_size)

            gt_images = []
            gt_masks = []
            render_images = []
            render_masks = []
            # manually sample {batch_size} frames x {batch_size} views for training
            for view_index in batch_view_indexs:
                for frame_index in batch_frame_indexs:
                    gt_image = self.test_motion_imgs[view_index][frame_index].to(self.device) # [1, 3, H, W]
                    gt_mask = self.test_motion_masks[view_index][frame_index].to(self.device) # [1, 1, H, W]

                    pose = orbit_camera(self.opt.elevation, self.azimuths[view_index], self.opt.radius)
                    cur_cam = MiniCam(pose, render_resolution, render_resolution, self.cam.fovy, self.cam.fovx, self.cam.near, self.cam.far)
                    timestamp = self.test_motion_times[frame_index] # [0, 1)

                    out = self.renderer.render(cur_cam, time=timestamp, stage=self.stage)

                    if self.step < 200:
                        self.renderer.gaussians._timenet.deformnet.requires_grad_(False)
                        self.renderer.gaussians._timenet.pts_layers.requires_grad_(False)

                    render_image = out["image"].unsqueeze(0) # (1, 3, rH, rW)
                    render_images.append(render_image)
                    gt_image = F.interpolate(gt_image, (render_resolution, render_resolution), mode="bilinear", align_corners=False) # (1, 3, rH, rW)
                    gt_images.append(gt_image)

                    render_mask = out["alpha"].unsqueeze(0) # (1, 1, rH, rW)
                    render_masks.append(render_mask)
                    gt_mask = F.interpolate(gt_mask, (render_resolution, render_resolution), mode="bilinear", align_corners=False) # (1, 1, rH, rW)
                    gt_masks.append(gt_mask)

            render_images = torch.cat(render_images, dim=0) # (batch_size, 3, rH, rW)
            gt_images = torch.cat(gt_images, dim=0) # (batch_size, 3, rH, rW)
            render_masks = torch.cat(render_masks, dim=0) # (batch_size, 1, rH, rW)
            gt_masks = torch.cat(gt_masks, dim=0) # (batch_size, 1, rH, rW)
        
            ## pixel-level MSE loss
            mse_loss = F.mse_loss(render_images, gt_images)
            self.tb_writer.add_scalar("Stage2/MSE Loss", mse_loss, self.step)
            loss = loss + self.opt.lambda_mse * mse_loss
            ## LPIPS loss
            lpips_loss = self.lpips_loss(render_images, gt_images).mean()
            self.tb_writer.add_scalar("Stage2/LPIPS Loss", lpips_loss, self.step)
            loss = loss + self.opt.lambda_lpips * lpips_loss
            ## SSIM loss
            ssim_loss = 1 - fused_ssim(render_images, gt_images)
            self.tb_writer.add_scalar("Stage2/SSIM Loss", ssim_loss, self.step)
            loss = loss + self.opt.lambda_ssim * ssim_loss
            ## Mask loss
            mask_loss = F.mse_loss(render_masks, gt_masks)
            self.tb_writer.add_scalar("Stage2/Mask Loss", mask_loss, self.step)
            loss = loss + self.opt.lambda_mask * mask_loss

            # ARAP loss in s2
            if self.opt.use_arap and self.stage == "s2" and self.step < self.opt.arap_end_iter_s2: # 2000
                loss_arap, conns = self.renderer.arap_loss_v2(stage=self.stage)
                loss += self.opt.lambda_arap * loss_arap
                self.tb_writer.add_scalar(f'Stage2/ARAP Loss', loss_arap.item(), self.step)

            self.tb_writer.add_images("Stage2/Render Images", render_images[:1], self.step)
            self.tb_writer.add_images("Stage2/GT Images", gt_images[:1], self.step)
            vis_latent_code = self.renderer.gaussians._latent_codes.detach().cpu().numpy()
            self.tb_writer.add_histogram("Stage2/Latent Codes", vis_latent_code, self.step)

            if self.step % 10 == 0:
                debug_path = os.path.join(self.opt.video_save_dir, f"debug")
                os.makedirs(debug_path, exist_ok=True)
                gt_img_show = (gt_images[0].permute(1,2,0).clamp(0,1)[:,:,[2,1,0]]*255).detach().cpu().numpy().astype(np.uint8)
                img_show = (render_images[0].permute(1,2,0).clamp(0,1)[:,:,[2,1,0]]*255).detach().cpu().numpy().astype(np.uint8)
                os.makedirs(debug_path, exist_ok=True)
                image_to_show = np.concatenate([gt_img_show, img_show], axis=1)
                cv2.imwrite(os.path.join(debug_path, f"image_{self.stage}_{self.step}.png"), image_to_show)

            # optimize step
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

            # clear memory
            torch.cuda.empty_cache()

        torch.cuda.synchronize()

    def load_input(self, file, ref_size):
        print(f'[INFO] load image from {file}...')
        orig_img = cv2.imread(file, cv2.IMREAD_UNCHANGED)
        orig_size = orig_img.shape[:2]
        mask_path = file.replace(".png", "_mask.npy")
        if not os.path.exists(mask_path):
            if orig_img.shape[-1] == 3:
                bg_remover = rembg.new_session()
                img = rembg.remove(orig_img, session=bg_remover)
            input_mask = img[..., 3:].astype(np.float32) / 255.0 # (H, W, 1)
            np.save(mask_path, input_mask)
        else:
            input_mask = np.load(mask_path) # (H, W, 1)
            save_mask_path = file.replace(".png", "_mask.png")
            if not os.path.exists(save_mask_path):
                cv2.imwrite(save_mask_path, (input_mask * 255).astype(np.uint8))
        input_img = orig_img.astype(np.float32) / 255.0 # (H, W, 3)
        input_img = input_img[..., ::-1].copy() # bgr to rgb
        
        # to torch tensors
        input_img_torch = torch.from_numpy(input_img).permute(2, 0, 1).unsqueeze(0)
        input_mask_torch = torch.from_numpy(input_mask).permute(2, 0, 1).unsqueeze(0)
        if orig_size[0] != ref_size or orig_size[1] != ref_size:
            input_img_torch = F.interpolate(input_img_torch, (ref_size, ref_size), mode="bilinear", align_corners=False)
            input_mask_torch = F.interpolate(input_mask_torch, (ref_size, ref_size), mode="bilinear", align_corners=False)

        return input_img_torch, input_mask_torch
    

if __name__ == "__main__":
    import argparse
    from omegaconf import OmegaConf

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/test_config.yaml", type=str, help="path to the yaml config file")
    parser.add_argument("--test_motion", help="whether to test motion")
    parser.add_argument("--test_motion_data", help="test data folder")
    parser.add_argument("--test_language", help="whether to test motion")
    parser.add_argument("--test_interpolation", help="whether to test interpolation")
    parser.add_argument("--test_paper", help="whether to test paper")
    parser.add_argument("--test_text_prompt", help="test text prompt")
    parser.add_argument("--test_unaligned_motion", help="whether to test unaligned motion")
    parser.add_argument("--test_unaligned_motion_data", help="test unaligned data folder")
    args, extras = parser.parse_known_args()

    # override default config from cli
    opt = OmegaConf.merge(OmegaConf.load(args.config), OmegaConf.from_cli(extras))
    gui = GUI(opt)

    if opt.test_motion:
        gui.test_motion()
    elif opt.test_unaligned_motion:
        gui.test_unaligned_motion()
    elif opt.test_language:
        gui.test_language()
    elif opt.test_interpolation:
        gui.test_interpolation()
    elif opt.test_paper:
        gui.test_paper()
    else:
        gui.test(render_type=opt.render_type)
