import torch
import os
import math
import open3d as o3d
import numpy as np
from diff_gaussian_rasterization import GaussianRasterizationSettings as Camera
import matplotlib
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg


def build_rotation(q):
    norm = torch.sqrt(q[:, 0] * q[:, 0] + q[:, 1] * q[:, 1] + q[:, 2] * q[:, 2] + q[:, 3] * q[:, 3])
    q = q / norm[:, None]
    rot = torch.zeros((q.size(0), 3, 3), device='cuda')
    r = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]
    rot[:, 0, 0] = 1 - 2 * (y * y + z * z)
    rot[:, 0, 1] = 2 * (x * y - r * z)
    rot[:, 0, 2] = 2 * (x * z + r * y)
    rot[:, 1, 0] = 2 * (x * y + r * z)
    rot[:, 1, 1] = 1 - 2 * (x * x + z * z)
    rot[:, 1, 2] = 2 * (y * z - r * x)
    rot[:, 2, 0] = 2 * (x * z - r * y)
    rot[:, 2, 1] = 2 * (y * z + r * x)
    rot[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return rot

def setup_camera(w, h, k, w2c, near=0.01, far=100):
    fx, fy, cx, cy = k[0][0], k[1][1], k[0][2], k[1][2]
    w2c = torch.tensor(w2c).cuda().float()
    cam_center = torch.inverse(w2c)[:3, 3]
    w2c = w2c.unsqueeze(0).transpose(1, 2)
    opengl_proj = torch.tensor([[2 * fx / w, 0.0, -(w - 2 * cx) / w, 0.0],
                                [0.0, 2 * fy / h, -(h - 2 * cy) / h, 0.0],
                                [0.0, 0.0, far / (far - near), -(far * near) / (far - near)],
                                [0.0, 0.0, 1.0, 0.0]]).cuda().float().unsqueeze(0).transpose(1, 2)
    full_proj = w2c.bmm(opengl_proj)
    cam = Camera(
        image_height=h,
        image_width=w,
        tanfovx=w / (2 * fx),
        tanfovy=h / (2 * fy),
        bg=torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda"),
        scale_modifier=1.0,
        viewmatrix=w2c,
        projmatrix=full_proj,
        sh_degree=0,
        campos=cam_center,
        prefiltered=False
    )
    return cam


def params2rendervar(params):
    rendervar = {
        'means3D': params['means3D'],
        'colors_precomp': params['rgb_colors'],
        'rotations': torch.nn.functional.normalize(params['unnorm_rotations']),
        'opacities': torch.sigmoid(params['logit_opacities']),
        'scales': torch.exp(params['log_scales']),
        'means2D': torch.zeros_like(params['means3D'], requires_grad=True, device="cuda") + 0
    }
    return rendervar


def l1_loss_v1(x, y):
    return torch.abs((x - y)).mean()


def l1_loss_v2(x, y):
    return (torch.abs(x - y).sum(-1)).mean()


def weighted_l2_loss_v1(x, y, w):
    return torch.sqrt(((x - y) ** 2) * w + 1e-20).mean()


def weighted_l2_loss_v2(x, y, w):
    return torch.sqrt(((x - y) ** 2).sum(-1) * w + 1e-20).mean()


def quat_mult(q1, q2):
    w1, x1, y1, z1 = q1.T
    w2, x2, y2, z2 = q2.T
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return torch.stack([w, x, y, z]).T


def o3d_knn(pts, num_knn):
    indices = []
    sq_dists = []
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.ascontiguousarray(pts, np.float64))
    pcd_tree = o3d.geometry.KDTreeFlann(pcd)
    for p in pcd.points:
        [_, i, d] = pcd_tree.search_knn_vector_3d(p, num_knn + 1)
        indices.append(i[1:])
        sq_dists.append(d[1:])
    return np.array(sq_dists), np.array(indices)

### find knn between two point clouds
def o3d_knn_c(pts1, pts2, num_knn):
    indices = []
    sq_dists = []
    pcd1 = o3d.geometry.PointCloud() 
    pcd1.points = o3d.utility.Vector3dVector(np.ascontiguousarray(pts1, np.float64))
    pcd2 = o3d.geometry.PointCloud() 
    pcd2.points = o3d.utility.Vector3dVector(np.ascontiguousarray(pts2, np.float64))
    kdtree1 = o3d.geometry.KDTreeFlann(pcd1)
    for p in pcd2.points:
        [_, i, d] = kdtree1.search_knn_vector_3d(p, num_knn + 1)
        indices.append(i[1:])
        sq_dists.append(d[1:])
    return np.array(sq_dists), np.array(indices)

def params2cpu(params, is_initial_timestep):
    if is_initial_timestep:
        res = {k: v.detach().cpu().contiguous().numpy() for k, v in params.items()}
    else:
        res = {k: v.detach().cpu().contiguous().numpy() for k, v in params.items() if
               k in ['means3D', 'rgb_colors', 'unnorm_rotations']}
    return res


def save_params(output_params, seq, exp):
    to_save = {}
    for k in output_params[0].keys():
        if k in output_params[1].keys():
            to_save[k] = np.stack([params[k] for params in output_params])
        else:
            to_save[k] = output_params[0][k]
    os.makedirs(f"./output/{exp}/{seq}", exist_ok=True)
    np.savez(f"./output/{exp}/{seq}/params", **to_save)


def plot_3d_tracks(points, visibles, tracks_leave_trace=16):
    """Visualize 3D point trajectories."""
    num_frames, num_points = points.shape[0:2]

    color_map = matplotlib.colormaps.get_cmap('hsv')
    # color_map = matplotlib.colormaps.get_cmap('cool')
    cmap_norm = matplotlib.colors.Normalize(vmin=0, vmax=num_points - 1)

    x_min, x_max = np.min(points[visibles, 0]), np.max(points[visibles, 0])
    y_min, y_max = np.min(points[visibles, 2]), np.max(points[visibles, 2])
    z_min, z_max = np.min(points[visibles, 1]), np.max(points[visibles, 1])

    interval = np.max([x_max - x_min, y_max - y_min, z_max - z_min])
    x_min = (x_min + x_max) / 2 - interval / 2
    x_max = x_min + interval
    y_min = (y_min + y_max) / 2 - interval / 2
    y_max = y_min + interval
    z_min = (z_min + z_max) / 2 - interval / 2
    z_max = z_min + interval

    frames = []
    for t in range(num_frames):
        fig = Figure(figsize=(6.4, 4.8))
        canvas = FigureCanvasAgg(fig)
        ax = fig.add_subplot(111, projection='3d', computed_zorder=False)

        ax.set_xlim([x_min, x_max])
        ax.set_ylim([y_min, y_max])
        ax.set_zlim([z_min, z_max])

        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])

        # ax.invert_zaxis()
        # ax.invert_xaxis()
        ax.invert_yaxis()
        ax.view_init()

        for i in range(num_points):
            color = color_map(cmap_norm(i))
            line = points[max(0, t - tracks_leave_trace) : t + 1, i]
            ax.plot(xs=line[:, 0], ys=line[:, 2], zs=line[:, 1], color=color, linewidth=1)
            end_point = points[t, i]
            ax.scatter(xs=end_point[0], ys=end_point[2], zs=end_point[1], color=color, s=3)

        fig.subplots_adjust(left=-0.05, right=1.05, top=1.05, bottom=-0.05)
        fig.canvas.draw()
        frames.append(canvas.buffer_rgba())
        
    return np.array(frames)[..., :3]


def plot_singel_3d_tracks(points, visibles):
    """Visualize 3D point trajectories."""
    num_frames, num_points = points.shape[0:2]

    color_map = matplotlib.colormaps.get_cmap('hsv')
    # color_map = matplotlib.colormaps.get_cmap('cool')
    cmap_norm = matplotlib.colors.Normalize(vmin=0, vmax=num_points - 1)

    x_min, x_max = np.min(points[visibles, 0]), np.max(points[visibles, 0])
    y_min, y_max = np.min(points[visibles, 2]), np.max(points[visibles, 2])
    z_min, z_max = np.min(points[visibles, 1]), np.max(points[visibles, 1])

    interval = np.max([x_max - x_min, y_max - y_min, z_max - z_min])
    x_min = (x_min + x_max) / 2 - interval / 2
    x_max = x_min + interval
    y_min = (y_min + y_max) / 2 - interval / 2
    y_max = y_min + interval
    z_min = (z_min + z_max) / 2 - interval / 2
    z_max = z_min + interval

    fig = Figure(figsize=(6.4, 4.8))
    canvas = FigureCanvasAgg(fig)
    ax = fig.add_subplot(111, projection='3d', computed_zorder=False)

    ax.set_xlim([x_min, x_max])
    ax.set_ylim([y_min, y_max])
    ax.set_zlim([z_min, z_max])

    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])

    ax.invert_yaxis()
    ax.view_init()

    for i in range(num_points):
        color = color_map(cmap_norm(i))
        line = points[:, i]
        ax.plot(xs=line[:, 0], ys=line[:, 2], zs=line[:, 1], color=color, linewidth=1)
        # end_point = points[-1, i]
        # ax.scatter(xs=end_point[0], ys=end_point[2], zs=end_point[1], color=color, s=3)

    fig.subplots_adjust(left=-0.05, right=1.05, top=1.05, bottom=-0.05)
    fig.canvas.draw()
    frame = canvas.buffer_rgba()
    
    return np.array(frame)[..., :3]