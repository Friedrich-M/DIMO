import os
import cv2
import torch
import torch.nn.functional as F
import numpy as np


# --------- per-process global cache ----------
REMBG_SESSION = None


def init_rembg_worker(model_name=None):
    """
    Pool initializer: called once per worker process.
    """
    global REMBG_SESSION
    import rembg
    REMBG_SESSION = rembg.new_session(model_name) if model_name else rembg.new_session()


def _compute_or_load_mask(orig_img, mask_path):
    """
    orig_img: uint8, shape (H, W, 3) or (H, W, 4) in BGR(A)
    return: input_mask float32, shape (H, W, 1), range [0, 1]
    """
    if orig_img.shape[-1] == 4:
        alpha = orig_img[..., 3:4].astype(np.float32) / 255.0
        if not os.path.exists(mask_path):
            np.save(mask_path, alpha)
        return alpha

    if os.path.exists(mask_path):
        try:
            m = np.load(mask_path)
            if m.ndim == 2:
                m = m[..., None]
            return m.astype(np.float32)
        except Exception:
            try:
                os.remove(mask_path)
            except:
                pass

    global REMBG_SESSION
    if REMBG_SESSION is None:
        import rembg
        REMBG_SESSION = rembg.new_session()

    import rembg
    rgba = rembg.remove(orig_img, session=REMBG_SESSION)
    alpha = rgba[..., 3:4].astype(np.float32) / 255.0
    np.save(mask_path, alpha)
    return alpha


def load_input(file, motion_video_name, view_idx, frame_idx, ref_size):
    orig_img = cv2.imread(file, cv2.IMREAD_UNCHANGED)
    if orig_img is None:
        raise FileNotFoundError(f"Failed to read image: {file}")

    orig_size = orig_img.shape[:2]
    mask_path = file.replace(".png", "_mask.npy")

    input_mask = _compute_or_load_mask(orig_img, mask_path)  # (H, W, 1)

    # image: BGR(A) -> RGB, float32
    if orig_img.shape[-1] == 4:
        bgr = orig_img[..., :3]
    else:
        bgr = orig_img
    input_img = bgr.astype(np.float32) / 255.0
    input_img = input_img[..., ::-1].copy()

    # to torch (still CPU tensors here)
    input_img_torch = torch.from_numpy(input_img).permute(2, 0, 1).unsqueeze(0)   # (1, 3, H, W)
    input_mask_torch = torch.from_numpy(input_mask).permute(2, 0, 1).unsqueeze(0) # (1, 1, H, W)

    if orig_size[0] != ref_size or orig_size[1] != ref_size:
        input_img_torch = F.interpolate(input_img_torch, (ref_size, ref_size), mode="bilinear", align_corners=False)
        input_mask_torch = F.interpolate(input_mask_torch, (ref_size, ref_size), mode="bilinear", align_corners=False)

    return motion_video_name, view_idx, frame_idx, input_img_torch, input_mask_torch
    

def parallel_loader(args):
    torch.set_num_threads(1)
    file, motion_video_name, view_idx, frame_idx, ref_size = args
    return load_input(
        file=file,
        motion_video_name=motion_video_name,
        view_idx=view_idx,
        frame_idx=frame_idx,
        ref_size=ref_size,
    )