from torch.autograd import Variable
import torch
import torch.nn.functional as F
import math
import numpy as np
from math import exp


def compute_pearson_depth_loss(render_depth, gt_depth, gt_mask=None):
    """
    Args:
        render_depth: [H, W, 1]
        gt_depth: [H, W, 1]
    """
    if gt_mask is not None:
        render_depth = render_depth[gt_mask]
        gt_depth = gt_depth[gt_mask]
        valid_mask = (render_depth > 0.1) & (gt_depth > 0.1)
        render_depth = render_depth[valid_mask]
        gt_depth = gt_depth[valid_mask]
        
    src = render_depth - render_depth.mean()
    target = gt_depth - gt_depth.mean()

    src = src / (src.std() + 1e-6)
    target = target / (target.std() + 1e-6)

    co = (src * target).mean()

    assert not torch.any(torch.isnan(co))
    return 1 - co


def compute_local_pearson_loss(render_depth, gt_depth, box_p=128, p_corr=0.5, gt_mask=None):
    """
    Args:
        render_depth: [H, W]
        gt_depth: [H, W]
        box_p: size of the patch
        p_corr: percentage of the total number of patches to be considered
        gt_mask: mask for the ground truth depth
    Randomly select patch, 
    top left corner of the patch (x_0, y_0) has to be 0 <= x_0 <= max_h, 0 <= y_0 <= max_w
    """
    num_box_h = math.floor(render_depth.shape[0] / box_p) # number of boxes in height
    num_box_w = math.floor(render_depth.shape[1] / box_p) # number of boxes in width
    max_h = render_depth.shape[0] - box_p
    max_w = render_depth.shape[1] - box_p
    _loss = torch.tensor(0.0,device='cuda')
    n_corr = int(p_corr * num_box_h * num_box_w)
    x_0 = torch.randint(0, max_h, size=(n_corr,), device = 'cuda')
    y_0 = torch.randint(0, max_w, size=(n_corr,), device = 'cuda')
    x_1 = x_0 + box_p
    y_1 = y_0 + box_p
    _loss = torch.tensor(0.0,device='cuda')
    for i in range(len(x_0)):
        if gt_mask is not None:
            _loss += compute_pearson_depth_loss(render_depth[x_0[i]:x_1[i],y_0[i]:y_1[i]].reshape(-1), gt_depth[x_0[i]:x_1[i],y_0[i]:y_1[i]].reshape(-1), gt_mask[x_0[i]:x_1[i],y_0[i]:y_1[i]].reshape(-1))
        else:
            _loss += compute_pearson_depth_loss(render_depth[x_0[i]:x_1[i],y_0[i]:y_1[i]].reshape(-1), gt_depth[x_0[i]:x_1[i],y_0[i]:y_1[i]].reshape(-1))
    return _loss/n_corr


def compute_edge_aware_smoothness_loss(depth, rgb):
    """
    Args:
        depth: [batch, H, W, 1]
        rgb: [batch, H, W, 3]
    """
    grad_depth_x = torch.abs(depth[..., :, :-1, :] - depth[..., :, 1:, :])
    grad_depth_y = torch.abs(depth[..., :-1, :, :] - depth[..., 1:, :, :])

    grad_img_x = torch.mean(
        torch.abs(rgb[..., :, :-1, :] - rgb[..., :, 1:, :]), -1, keepdim=True
    )
    grad_img_y = torch.mean(
        torch.abs(rgb[..., :-1, :, :] - rgb[..., 1:, :, :]), -1, keepdim=True
    )

    grad_depth_x *= torch.exp(-grad_img_x)
    grad_depth_y *= torch.exp(-grad_img_y)

    return grad_depth_x.mean() + grad_depth_y.mean()


def compute_bilateral_normal_smoothness_loss(normal, rgb):
    """
    Args:
        depth: [batch, H, W, 3]
        rgb: [batch, H, W, 3]
    """
    # encourage the gradients of rendered normal rn to be smooth if (and only if) the input image gradients rI are smooth. The loss can be written as L bilateral = e^(-3 gradient_I) sqrt(1 + ||gradient_n||^2)
    grad_normal_x = torch.abs(normal[..., :, :-1, :] - normal[..., :, 1:, :])
    grad_normal_y = torch.abs(normal[..., :-1, :, :] - normal[..., 1:, :, :])

    grad_img_x = torch.mean(
        torch.abs(rgb[..., :, :-1, :] - rgb[..., :, 1:, :]), -1, keepdim=True
    )
    grad_img_y = torch.mean(
        torch.abs(rgb[..., :-1, :, :] - rgb[..., 1:, :, :]), -1, keepdim=True
    )

    grad_normal_x *= torch.exp(-3 * grad_img_x)
    grad_normal_y *= torch.exp(-3 * grad_img_y)

    return torch.sqrt(1 + grad_normal_x ** 2).mean() + torch.sqrt(1 + grad_normal_y ** 2).mean()


def compute_tv_norm(values, losstype='l2'):
    """Computes TV norm for input values, based on RegNeRF.
    Args:
        values: [batch, H, W, C] tensor
        losstype: l2 or l1
    Returns:
        loss: [batch, H-1, W-1] tensor
    """
    #   v00 = values[:-1, :-1]
    #   v01 = values[:-1, 1:]
    #   v10 = values[1:, :-1]
    v00 = values[..., :-1, :-1, :]
    v01 = values[..., :-1, 1:, :]
    v10 = values[..., 1:, :-1, :]
    if losstype == 'l2':
        loss = ((v00 - v01) ** 2) + ((v00 - v10) ** 2)
    elif losstype == 'l1':
        loss = np.abs(v00 - v01) + np.abs(v00 - v10)
    else:
        raise ValueError(f'losstype must be l2 or l1 but is {losstype}')
    return loss


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window


def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)
    
    
def l1_loss(network_output, gt):
    return torch.abs((network_output - gt)).mean()

