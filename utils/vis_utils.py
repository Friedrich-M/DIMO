import cv2
import numpy as np
import IPython

import matplotlib
from matplotlib import cm
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg

import mediapy as media
import scenepic as sp


def project_points_to_video_frame(camera_pov_points3d, camera_intrinsics, height, width):
  """Project 3d points to 2d image plane."""
  u_d = camera_pov_points3d[..., 0] / (camera_pov_points3d[..., 2] + 1e-8)
  v_d = camera_pov_points3d[..., 1] / (camera_pov_points3d[..., 2] + 1e-8)

  f_u, f_v, c_u, c_v = camera_intrinsics

  u_d = u_d * f_u + c_u
  v_d = v_d * f_v + c_v

  # Mask of points that are in front of the camera and within image boundary
  masks = (camera_pov_points3d[..., 2] >= 1)
  masks = masks & (u_d >= 0) & (u_d < width) & (v_d >= 0) & (v_d < height)
  return np.stack([u_d, v_d], axis=-1), masks

def plot_2d_tracks(video, points, visibles, infront_cameras=None, tracks_leave_trace=16, show_occ=False):
  """Visualize 2D point trajectories."""
  num_frames, num_points = points.shape[:2]

  # Precompute colormap for points
  color_map = matplotlib.colormaps.get_cmap('hsv')
  cmap_norm = matplotlib.colors.Normalize(vmin=0, vmax=num_points - 1)
  point_colors = np.zeros((num_points, 3))
  for i in range(num_points):
    point_colors[i] = np.array(color_map(cmap_norm(i)))[:3] * 255

  if infront_cameras is None:
    infront_cameras = np.ones_like(visibles).astype(bool)

  frames = []
  for t in range(num_frames):
    frame = video[t].copy()

    # Draw tracks on the frame
    line_tracks = points[max(0, t - tracks_leave_trace) : t + 1]
    line_visibles = visibles[max(0, t - tracks_leave_trace) : t + 1]
    line_infront_cameras = infront_cameras[max(0, t - tracks_leave_trace) : t + 1]
    for s in range(line_tracks.shape[0] - 1):
      img = frame.copy()

      for i in range(num_points):
        if line_visibles[s, i] and line_visibles[s + 1, i]:  # visible
          x1, y1 = int(round(line_tracks[s, i, 0])), int(round(line_tracks[s, i, 1]))
          x2, y2 = int(round(line_tracks[s + 1, i, 0])), int(round(line_tracks[s + 1, i, 1]))
          cv2.line(frame, (x1, y1), (x2, y2), point_colors[i], 1, cv2.LINE_AA)
        elif show_occ and line_infront_cameras[s, i] and line_infront_cameras[s + 1, i]:  # occluded
          x1, y1 = int(round(line_tracks[s, i, 0])), int(round(line_tracks[s, i, 1]))
          x2, y2 = int(round(line_tracks[s + 1, i, 0])), int(round(line_tracks[s + 1, i, 1]))
          cv2.line(frame, (x1, y1), (x2, y2), point_colors[i], 1, cv2.LINE_AA)

      alpha = (s + 1) / (line_tracks.shape[0] - 1)
      frame = cv2.addWeighted(frame, alpha, img, 1 - alpha, 0)

    # Draw end points on the frame
    for i in range(num_points):
      if visibles[t, i]:  # visible
        x, y = int(round(points[t, i, 0])), int(round(points[t, i, 1]))
        cv2.circle(frame, (x, y), 2, point_colors[i], -1)
      elif show_occ and infront_cameras[t, i]:  # occluded
        x, y = int(round(points[t, i, 0])), int(round(points[t, i, 1]))
        cv2.circle(frame, (x, y), 2, point_colors[i], 1)

    frames.append(frame)
  frames = np.stack(frames)
  return frames


def create_axis(scene, n_lines=10, min_x = -2, max_x = 2, min_y = -1.5, max_y = 1.5, min_z = 1, max_z = 5):
  x_plane = scene.create_mesh("xplane")
  vert_start_XYZ = np.stack((min_x*np.ones(n_lines), min_y * np.ones(n_lines), np.linspace(min_z,max_z,n_lines)), axis=-1)
  vert_end_XYZ = np.stack((min_x*np.ones(n_lines), max_y * np.ones(n_lines), np.linspace(min_z,max_z,n_lines)), axis=-1)
  horiz_start_XYZ = np.stack((min_x*np.ones(n_lines), np.linspace(min_y,max_y,n_lines), min_z*np.ones(n_lines)), axis=-1)
  horiz_end_XYZ = np.stack((min_x*np.ones(n_lines), np.linspace(min_y,max_y,n_lines), max_z*np.ones(n_lines)), axis=-1)
  x_plane.add_lines(np.concatenate((vert_start_XYZ,horiz_start_XYZ), axis=0), np.concatenate((vert_end_XYZ,horiz_end_XYZ), axis=0), color=0.2*np.ones((3,1)))

  y_plane = scene.create_mesh("yplane")
  vert_start_XYZ = np.stack((min_x*np.ones(n_lines), max_y * np.ones(n_lines), np.linspace(min_z,max_z,n_lines)), axis=-1)
  vert_end_XYZ = np.stack((max_x*np.ones(n_lines), max_y * np.ones(n_lines), np.linspace(min_z,max_z,n_lines)), axis=-1)
  horiz_start_XYZ = np.stack((np.linspace(min_x,max_x,n_lines), max_y * np.ones(n_lines), min_z*np.ones(n_lines)), axis=-1)
  horiz_end_XYZ = np.stack((np.linspace(min_x,max_x,n_lines), max_y * np.ones(n_lines), max_z*np.ones(n_lines)), axis=-1)
  y_plane.add_lines(np.concatenate((vert_start_XYZ,horiz_start_XYZ), axis=0), np.concatenate((vert_end_XYZ,horiz_end_XYZ), axis=0), color=0.2*np.ones((3,1)))

  z_plane = scene.create_mesh("zplane")
  vert_start_XYZ = np.stack((np.linspace(min_x,max_x,n_lines), min_y * np.ones(n_lines), max_z*np.ones(n_lines)), axis=-1)
  vert_end_XYZ = np.stack((np.linspace(min_x,max_x,n_lines), max_y * np.ones(n_lines), max_z*np.ones(n_lines)), axis=-1)
  horiz_start_XYZ = np.stack((min_x * np.ones(n_lines), np.linspace(min_y,max_y,n_lines), max_z*np.ones(n_lines)), axis=-1)
  horiz_end_XYZ = np.stack((max_x * np.ones(n_lines), np.linspace(min_y,max_y,n_lines), max_z*np.ones(n_lines)), axis=-1)
  z_plane.add_lines(np.concatenate((vert_start_XYZ,horiz_start_XYZ), axis=0), np.concatenate((vert_end_XYZ,horiz_end_XYZ), axis=0), color=0.2*np.ones((3,1)))

  return x_plane, y_plane, z_plane

def get_interactive_3d_visualization(XYZ, h=512, w=512, fov_y=33.9, framerate=8):
  n_frames, n_points = XYZ.shape[:2]
  # fov_y = (np.arctan2(h/2, fy) * 180 / np.pi) * 2

  # compute colors
  # cm = plt.get_cmap('hsv')
  cm = plt.get_cmap('cividis')
  # XYZ[:,:,2] = 2-XYZ[:,:,2]
  # XYZ[:,:,0] = -XYZ[:,:,0]
  Y = XYZ[0,:,1]
  XYZ = XYZ[:,np.argsort(Y),:]
  colors = cm(np.linspace(0,1,n_points))[:,:3]

  # create scene
  scene = sp.Scene()
  scene.framerate = framerate
  camera = sp.Camera(center=np.array([0, 0, 2]), aspect_ratio=w/h, fov_y_degrees=fov_y, look_at=np.array([0.,0.,-1.]), up_dir=np.array([0.,1.,0.]))
  # camera = sp.Camera(center=np.array([0, 0, 0]), aspect_ratio=w/h, fov_y_degrees=fov_y, look_at=np.array([0.,0.,1.]), up_dir=np.array([0.,1.,0.]))
  canvas = scene.create_canvas_3d(width=w, height=h, shading=sp.Shading(bg_color=sp.Colors.White), camera=camera)

  # create axis and frustrum
  x_plane, y_plane, z_plane = create_axis(scene)
  frustrum = scene.create_mesh("frustrum")
  frustrum.add_camera_frustum(camera, sp.Colors.Red, depth=0.5, thickness=0.002)

  # create track spheres
  spheres = scene.create_mesh("spheres")
  spheres.add_sphere(sp.Colors.White, transform=sp.Transforms.Scale(0.03))
  spheres.enable_instancing(XYZ[0], colors=colors)

  # create track trails
  lines_t = []
  for t in range(1, n_frames):
    start_XYZ = XYZ[t-1]
    end_XYZ = XYZ[t]
    start_colors = colors
    end_colors = colors
    mesh = scene.create_mesh(f"lines_{t}")
    mesh.add_lines(np.concatenate((start_XYZ, start_colors), axis=-1), np.concatenate((end_XYZ, end_colors), axis=-1))
    lines_t.append(mesh)

  # create scene frames
  for i in range(n_frames-1):
    frame = canvas.create_frame()
    frame.add_mesh(frustrum)
    # for j in range(max(0, i-10), i):
    #   frame.add_mesh(lines_t[j])
    spheres_updated = scene.update_instanced_mesh("spheres", XYZ[i], colors=colors)
    frame.add_mesh(spheres_updated)
    frame.add_mesh(x_plane)
    frame.add_mesh(y_plane)
    frame.add_mesh(z_plane)

  scene.quantize_updates()

  # generate html
  SP_LIB = sp.js_lib_src()
  SP_SCRIPT = scene.get_script().replace(
      'window.onload = function()', 'function scenepic_main_function()'
  )
  HTML_string = (
      '<!DOCTYPE html>'
      '<html lang="en">'
      '<head>'
        '<meta charset="utf-8">'
        '<title>ScenePic </title>'
        f'<script>{SP_LIB}</script>'
        # '<script type="text/javascript" src="scenepic.min.js"></script>'
        f'<script>{SP_SCRIPT} scenepic_main_function();</script>'
      '</head>'
      f'<body onload="scenepic_main_function()"></body>'
      '</html>'
  ) 
  # save to file
  return HTML_string
  
#   html_object = IPython.display.HTML(HTML_string)
#   IPython.display.display(html_object)
#   print('Press PLAY ▶ to start animation')
#   print(' - Drag with mouse to rotate')
#   print(' - Use mouse-wheel for zoom')
#   print(' - Shift to pan')
#   print(' - Use camera button 📷 to restore camera view')


def get_single_3d_visualization(XYZ, h=512, w=512, fov_y=33.9, framerate=8):
  # XYZ: (n_points, 3)
  n_points = XYZ.shape[0]
  # fov_y = (np.arctan2(h/2, fy) * 180 / np.pi) * 2

  # compute colors
  # cm = plt.get_cmap('hsv')
  # 棕色
  # cm = plt.get_cmap('cividis')
  # 偏紫色
  cm = plt.get_cmap('viridis')

  # XYZ[..., 2] = 2-XYZ[..., 2]
  # XYZ[..., 0] = -XYZ[..., 0]
  Y = XYZ[:,1]
  XYZ = XYZ[np.argsort(Y),:]
  colors = cm(np.linspace(0,1,n_points))[:,:3]

  # create scene
  scene = sp.Scene()
  scene.framerate = framerate
  camera = sp.Camera(center=np.array([0, 0, 2]), aspect_ratio=w/h, fov_y_degrees=fov_y, look_at=np.array([0.,0.,-1.]), up_dir=np.array([0.,1.,0.]))
  # camera = sp.Camera(center=np.array([0, 0, 0]), aspect_ratio=w/h, fov_y_degrees=fov_y, look_at=np.array([0.,0.,1.]), up_dir=np.array([0.,1.,0.]))
  canvas = scene.create_canvas_3d(width=w, height=h, shading=sp.Shading(bg_color=sp.Colors.White), camera=camera)
  
  # create axis and frustrum
  x_plane, y_plane, z_plane = create_axis(scene)
  frustrum = scene.create_mesh("frustrum")
  frustrum.add_camera_frustum(camera, sp.Colors.Purple, depth=0.5, thickness=0.004)
   
  # create track spheres
  spheres = scene.create_mesh("spheres")
  spheres.add_sphere(sp.Colors.White
  , transform=sp.Transforms.Scale(0.02))
  spheres.enable_instancing(XYZ, colors=colors)

  # create scene frame
  frame = canvas.create_frame()
  frame.add_mesh(frustrum)
  frame.add_mesh(spheres)
  frame.add_mesh(x_plane)
  frame.add_mesh(y_plane)
  frame.add_mesh(z_plane)

  scene.quantize_updates()

  # generate html
  SP_LIB = sp.js_lib_src()
  SP_SCRIPT = scene.get_script().replace(
      'window.onload = function()', 'function scenepic_main_function()'
  )
  HTML_string = (
      '<!DOCTYPE html>'
      '<html lang="en">'
      '<head>'
        '<meta charset="utf-8">'
        '<title>ScenePic </title>'
        f'<script>{SP_LIB}</script>'
        # '<script type="text/javascript" src="scenepic.min.js"></script>'
        f'<script>{SP_SCRIPT} scenepic_main_function();</script>'
      '</head>'
      f'<body onload="scenepic_main_function()"></body>'
      '</html>'
  ) 
  # save to file
  return HTML_string


def plot_3d_tracks(points, visibles, infront_cameras=None, tracks_leave_trace=16, show_occ=False):
  """Visualize 3D point trajectories."""
  num_frames, num_points = points.shape[0:2]

  color_map = matplotlib.colormaps.get_cmap('hsv')
  cmap_norm = matplotlib.colors.Normalize(vmin=0, vmax=num_points - 1)

  if infront_cameras is None:
    infront_cameras = np.ones_like(visibles).astype(bool)

  if show_occ:
    x_min, x_max = np.min(points[infront_cameras, 0]), np.max(points[infront_cameras, 0])
    y_min, y_max = np.min(points[infront_cameras, 2]), np.max(points[infront_cameras, 2])
    z_min, z_max = np.min(points[infront_cameras, 1]), np.max(points[infront_cameras, 1])
  else:
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

    ax.invert_zaxis()
    ax.view_init()

    for i in range(num_points):
      if visibles[t, i] or (show_occ and infront_cameras[t, i]):
        color = color_map(cmap_norm(i))
        line = points[max(0, t - tracks_leave_trace) : t + 1, i]
        ax.plot(xs=line[:, 0], ys=line[:, 2], zs=line[:, 1], color=color, linewidth=1)
        end_point = points[t, i]
        ax.scatter(xs=end_point[0], ys=end_point[2], zs=end_point[1], color=color, s=3)

    fig.subplots_adjust(left=-0.05, right=1.05, top=1.05, bottom=-0.05)
    fig.canvas.draw()
    frames.append(canvas.buffer_rgba())
  return np.array(frames)[..., :3]