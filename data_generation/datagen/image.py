"""Reference-image and frame preprocessing: background removal, object-centred square crops, padding."""

import os
from typing import Optional, Tuple

import numpy as np
from PIL import Image, ImageOps

# Matting model, pinned so that results do not change with the rembg version (recent releases moved
# the default away from u2net) and so that only one model has to be cached for offline machines.
# Override with $DIMO_REMBG_MODEL; `dimo.data` uses the same model for the masks it caches.
REMBG_MODEL = os.environ.get("DIMO_REMBG_MODEL", "u2net")
_REMBG_SESSION = None


def rembg_session():
    global _REMBG_SESSION
    if _REMBG_SESSION is None:
        # onnxruntime sizes its thread pool from the machine's core count and then fails noisily to pin
        # threads when the process is restricted to a subset of cores (e.g. under a job scheduler).
        os.environ.setdefault("OMP_NUM_THREADS", "8")
        import rembg

        _REMBG_SESSION = rembg.new_session(REMBG_MODEL)
    return _REMBG_SESSION


def white_background_alpha(bgr: np.ndarray, tolerance: int = 12) -> np.ndarray:
    """Foreground alpha ``(H, W, 1)`` for an object rendered on a white background.

    Identical to ``dimo.data.white_background_alpha`` -- the two packages are independent, so the
    implementation is duplicated rather than imported; keep them in step.
    """
    import cv2

    distance = 255 - bgr[..., :3].min(axis=2)                       # 0 on pure white

    # Background = near-white pixels reachable from the image border. Flood-filling from the corners
    # is what keeps a white shirt or a specular highlight inside the object opaque: it is white, but
    # the object around it walls the fill off, so it is never reached.
    white = (distance <= tolerance).astype(np.uint8)
    height, width = white.shape
    barrier = np.zeros((height + 2, width + 2), np.uint8)
    barrier[1:-1, 1:-1] = 1 - white                        # the object blocks the fill
    flooded = white.copy()
    for x, y in ((0, 0), (width - 1, 0), (0, height - 1), (width - 1, height - 1)):
        if white[y, x]:
            cv2.floodFill(flooded, barrier, (x, y), 2)
    background = flooded == 2

    # Opaque everywhere the object is, and the distance ramp applied only in the thin band touching
    # the background, so the silhouette gets an anti-aliased edge while light-coloured parts of the
    # object stay fully opaque instead of turning translucent.
    alpha = np.ones(distance.shape, np.float32)
    band = cv2.dilate(background.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=1).astype(bool)
    edge = band & ~background
    alpha[edge] = np.clip(distance[edge].astype(np.float32) / (2.0 * tolerance), 0.0, 1.0)
    alpha[background] = 0.0
    return alpha[..., None]


def remove_background(image: Image.Image) -> Image.Image:
    """RGBA image with a rembg alpha matte."""
    import rembg

    return rembg.remove(image.convert("RGB"), session=rembg_session()).convert("RGBA")


def load_rgba(path: str, remove_bg: bool = True) -> Image.Image:
    """Load an image as RGBA; images without alpha get a rembg matte when ``remove_bg`` is set."""
    image = ImageOps.exif_transpose(Image.open(path))
    if image.mode == "RGBA" and np.asarray(image)[..., 3].min() < 255:
        return image
    if remove_bg:
        return remove_background(image)
    return image.convert("RGBA")


def alpha_bbox(rgba: np.ndarray, threshold: int = 10) -> Tuple[int, int, int, int]:
    """``(x0, y0, x1, y1)`` of the pixels whose alpha exceeds ``threshold``."""
    ys, xs = np.where(rgba[..., 3] > threshold)
    if len(xs) == 0:
        return 0, 0, rgba.shape[1], rgba.shape[0]
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def square_crop_around_object(rgba: Image.Image, image_ratio: float = 0.8, object_centered: bool = True,
                              bbox: Optional[Tuple[int, int, int, int]] = None, multiple: int = 16) -> Image.Image:
    """Square RGBA canvas in which the object occupies ``image_ratio`` of the side length.

    ``object_centered`` keeps the object's bounding-box centre in the middle of the canvas
    (otherwise the canvas is centred on the image centre, as in the original pipeline).
    """
    array = np.asarray(rgba.convert("RGBA"))
    x0, y0, x1, y1 = bbox or alpha_bbox(array)
    if object_centered:
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        box = max(x1 - x0, y1 - y0)
    else:
        cx, cy = array.shape[1] / 2, array.shape[0] / 2
        box = 2 * max(cx - x0, x1 - cx, cy - y0, y1 - cy)
    # Rounded *up* to a multiple: flooring makes `side < box` whenever the object is nearly as wide
    # as the canvas (at image_ratio=0.95 that is any box under ~273 px), and `paste` would then clip
    # the object's edges silently.
    side = max(multiple, -(-int(np.ceil(box / image_ratio)) // multiple) * multiple)

    canvas = Image.new("RGBA", (side, side), (0, 0, 0, 0))
    left, top = int(round(cx - side / 2)), int(round(cy - side / 2))
    canvas.paste(rgba.convert("RGBA"), (-left, -top))
    return canvas


def composite_on_white(rgba: Image.Image) -> Image.Image:
    array = np.asarray(rgba.convert("RGBA")).astype(np.float32) / 255.0
    rgb = array[..., :3] * array[..., 3:] + (1 - array[..., 3:])
    return Image.fromarray((rgb * 255).round().astype(np.uint8))


def pad_to_aspect(rgba: Image.Image, width: int, height: int) -> Image.Image:
    """Pad a square RGBA canvas (transparent) to the aspect ratio ``width:height`` without scaling."""
    w, h = rgba.size
    target_w, target_h = w, h
    if w / h < width / height:
        target_w = int(round(h * width / height))
    else:
        target_h = int(round(w * height / width))
    canvas = Image.new("RGBA", (target_w, target_h), (0, 0, 0, 0))
    canvas.paste(rgba, ((target_w - w) // 2, (target_h - h) // 2))
    return canvas


def prepare_reference(path: str, image_ratio: float = 0.8, object_centered: bool = True, remove_bg: bool = True,
                      size: Optional[int] = None) -> Image.Image:
    """Object on a transparent square canvas, optionally resized to ``size``."""
    rgba = load_rgba(path, remove_bg=remove_bg)
    rgba = square_crop_around_object(rgba, image_ratio=image_ratio, object_centered=object_centered)
    if size is not None:
        rgba = rgba.resize((size, size), Image.LANCZOS)
    return rgba


def frames_to_rgba(frames: np.ndarray, batch_session=None) -> np.ndarray:
    """rembg mattes for ``(T, H, W, 3)`` uint8 frames -> ``(T, H, W, 4)``."""
    import rembg

    session = batch_session or rembg_session()
    out = [np.asarray(rembg.remove(Image.fromarray(f), session=session).convert("RGBA")) for f in frames]
    return np.stack(out)


def crop_frames_to_square(frames_rgba: np.ndarray, image_ratio: float = 0.9, size: int = 576,
                          composite_white: bool = False) -> np.ndarray:
    """Crop a video ``(T, H, W, 4)`` with one square window around the union of all object masks.

    Returns ``(T, size, size, 4)`` (or 3 channels on white when ``composite_white``).
    """
    union = frames_rgba[..., 3].max(axis=0)
    ys, xs = np.where(union > 10)
    x0, y0, x1, y1 = (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1) if len(xs) else (0, 0, union.shape[1], union.shape[0])
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    side = int(max(x1 - x0, y1 - y0) / image_ratio)
    left, top = int(round(cx - side / 2)), int(round(cy - side / 2))

    out = []
    for frame in frames_rgba:
        canvas = Image.new("RGBA", (side, side), (0, 0, 0, 0))
        canvas.paste(Image.fromarray(frame, "RGBA"), (-left, -top))
        canvas = canvas.resize((size, size), Image.LANCZOS)
        out.append(np.asarray(composite_on_white(canvas) if composite_white else canvas))
    return np.stack(out)
