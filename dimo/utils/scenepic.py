"""Interactive HTML visualisation of 3D trajectories with ScenePic (optional dependency)."""

import numpy as np


def _create_axis_planes(sp, scene, n_lines=10, min_x=-2, max_x=2, min_y=-1.5, max_y=1.5, min_z=1, max_z=5):
    gray = 0.2 * np.ones((3, 1))

    def lines(name, starts, ends):
        mesh = scene.create_mesh(name)
        mesh.add_lines(np.concatenate(starts, axis=0), np.concatenate(ends, axis=0), color=gray)
        return mesh

    ones = np.ones(n_lines)
    x_plane = lines(
        "xplane",
        [np.stack((min_x * ones, min_y * ones, np.linspace(min_z, max_z, n_lines)), -1),
         np.stack((min_x * ones, np.linspace(min_y, max_y, n_lines), min_z * ones), -1)],
        [np.stack((min_x * ones, max_y * ones, np.linspace(min_z, max_z, n_lines)), -1),
         np.stack((min_x * ones, np.linspace(min_y, max_y, n_lines), max_z * ones), -1)],
    )
    y_plane = lines(
        "yplane",
        [np.stack((min_x * ones, max_y * ones, np.linspace(min_z, max_z, n_lines)), -1),
         np.stack((np.linspace(min_x, max_x, n_lines), max_y * ones, min_z * ones), -1)],
        [np.stack((max_x * ones, max_y * ones, np.linspace(min_z, max_z, n_lines)), -1),
         np.stack((np.linspace(min_x, max_x, n_lines), max_y * ones, max_z * ones), -1)],
    )
    z_plane = lines(
        "zplane",
        [np.stack((np.linspace(min_x, max_x, n_lines), min_y * ones, max_z * ones), -1),
         np.stack((min_x * ones, np.linspace(min_y, max_y, n_lines), max_z * ones), -1)],
        [np.stack((np.linspace(min_x, max_x, n_lines), max_y * ones, max_z * ones), -1),
         np.stack((max_x * ones, np.linspace(min_y, max_y, n_lines), max_z * ones), -1)],
    )
    return x_plane, y_plane, z_plane


def interactive_3d_trajectories(points: np.ndarray, h: int = 512, w: int = 512, fov_y: float = 33.9,
                                framerate: int = 8, cmap: str = "cividis") -> str:
    """Self-contained HTML page animating 3D points ``(T, N, 3)`` (drag to rotate, wheel to zoom)."""
    import matplotlib.pyplot as plt
    import scenepic as sp

    n_frames, n_points = points.shape[:2]
    points = points[:, np.argsort(points[0, :, 1]), :]
    colors = plt.get_cmap(cmap)(np.linspace(0, 1, n_points))[:, :3]

    scene = sp.Scene()
    scene.framerate = framerate
    camera = sp.Camera(center=np.array([0, 0, 2]), aspect_ratio=w / h, fov_y_degrees=fov_y,
                       look_at=np.array([0.0, 0.0, -1.0]), up_dir=np.array([0.0, 1.0, 0.0]))
    canvas = scene.create_canvas_3d(width=w, height=h, shading=sp.Shading(bg_color=sp.Colors.White), camera=camera)

    planes = _create_axis_planes(sp, scene)
    frustum = scene.create_mesh("frustum")
    frustum.add_camera_frustum(camera, sp.Colors.Red, depth=0.5, thickness=0.002)
    spheres = scene.create_mesh("spheres")
    spheres.add_sphere(sp.Colors.White, transform=sp.Transforms.Scale(0.03))
    spheres.enable_instancing(points[0], colors=colors)

    for i in range(n_frames - 1):
        frame = canvas.create_frame()
        frame.add_mesh(frustum)
        frame.add_mesh(scene.update_instanced_mesh("spheres", points[i], colors=colors))
        for plane in planes:
            frame.add_mesh(plane)
    scene.quantize_updates()

    script = scene.get_script().replace("window.onload = function()", "function scenepic_main_function()")
    return (
        "<!DOCTYPE html><html lang=\"en\"><head><meta charset=\"utf-8\"><title>DIMO trajectories</title>"
        f"<script>{sp.js_lib_src()}</script><script>{script} scenepic_main_function();</script></head>"
        "<body onload=\"scenepic_main_function()\"></body></html>"
    )
