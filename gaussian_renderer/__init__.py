#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import math
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene.gaussian_model import GaussianModel
from utils.sh_utils import eval_sh

import torch

# def check_same_device(*args):
#     """
#     检查输入的所有张量是否都在同一设备上。
    
#     参数:
#         *args: 任意数量的张量或其他对象（忽略非张量）。
    
#     返回:
#         True: 如果所有张量在同一设备上，或者所有张量为 None。
#         False: 如果设备不一致。
#         如果不一致，会打印出不同的设备集合。
#     """
#     devices = set(arg.device for arg in args if arg is not None and isinstance(arg, torch.Tensor))
#     if len(devices) > 1:
#         print(f"设备不一致，发现以下设备: {devices}")
#         return False
#     return True


def render(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, separate_sh = False, override_color = None, use_trained_exp=False):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        # debug=pipe.debug,
        debug=True,  # debug
        antialiasing=pipe.antialiasing
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    # # 验证投影位置是否正确

    # W = int(viewpoint_camera.image_width)
    # H = int(viewpoint_camera.image_height)
    # projmatrix = viewpoint_camera.full_proj_transform
    # pts = pc.get_xyz
    
    # import numpy as np
    # import cv2

    # def ndc2pix(v, S):
    #     """Convert normalized device coordinates (NDC) to pixel coordinates."""
    #     return ((v + 1.0) * S - 1.0) * 0.5

    # def project_to_image(projmatrix, p_orig, W, H, E, P):
    #     """
    #     Projects multiple 3D points to 2D screen space using a projection matrix.

    #     Args:
    #         projmatrix (numpy.ndarray): 4x4 projection matrix.
    #         p_orig (numpy.ndarray): (n, 3) array of 3D points.
    #         W (int): Image width.
    #         H (int): Image height.

    #     Returns:
    #         numpy.ndarray: (n, 2) array of 2D projected points.
    #     """
    #     n = p_orig.shape[0]
    #     p_hom = np.hstack([p_orig, np.ones((n, 1))])  # Convert to homogeneous coordinates
        
    #     # Project to normalized device coordinates (NDC)
    #     p_proj = (projmatrix @ p_hom.T).T  # (n, 4)
    #     p_proj[:, :3] /= (p_proj[:, 3:4] + 1e-7)  # Perspective divide
        
    #     # Convert NDC to pixel coordinates
    #     points_image = np.stack([
    #         ndc2pix(p_proj[:, 0], W),
    #         ndc2pix(p_proj[:, 1], H)
    #     ], axis=-1)
        
        
    #     p_hom = np.hstack([p_orig, np.ones((n, 1))])  # Convert to homogeneous coordinates
    #     # Project to normalized device coordinates (NDC)
    #     p_cam = (E.T @ p_hom.T).T  # (n, 4)
    #     b = (P.T @ p_cam.T).T
    #     b[:, :3] /= (b[:, 3:4] + 1e-7)  # Perspective divide
    #     points_image2 = np.stack([
    #         ndc2pix(b[:, 0], W),
    #         ndc2pix(b[:, 1], H)
    #     ], axis=-1)
        
    #     return points_image.astype(int), points_image2.astype(int)

    # def visualize_and_save(points_2d, W, H, output_path="projected_image.png"):
    #     """Visualizes projected points on an image and saves the image."""
    #     image = np.zeros((H, W, 3), dtype=np.uint8)
        
    #     for (x, y) in points_2d:
    #         if 0 <= x < W and 0 <= y < H:  # Ensure points are within image bounds
    #             cv2.circle(image, (x, y), radius=3, color=(0, 255, 0), thickness=-1)
        
    #     cv2.imwrite(output_path, image)
    #     print(f"Projected image saved to {output_path}")
    # points_2d_c, points_2d_s = project_to_image(projmatrix.transpose(0,1).cpu().numpy(), pts.cpu().detach().numpy(), W, H, viewpoint_camera.world_view_transform.cpu().detach().numpy(), viewpoint_camera.projection_matrix.cpu().detach().numpy())
    # visualize_and_save(points_2d_c, W, H, output_path="./proj_c.png")
    # visualize_and_save(points_2d_s, W, H, output_path="./proj_s.png")
    
    # def build_offaxis_projection_matrix(
    #     fx, fy, cx, cy, W, H, znear, zfar, 
    #     x0_is_left=True, y0_is_top=True
    # ):
    #     """
    #     根据相机内参构造“非对称视锥”透视投影矩阵，以与针孔模型投影一致。

    #     参数：
    #     ----------
    #     fx, fy : float
    #         相机焦距（单位：像素）。对应 K[0, 0], K[1, 1].
    #     cx, cy : float
    #         相机主点（光心）在图像坐标系下的 (x, y) 像素坐标，对应 K[0,2], K[1,2].
    #     W, H : int
    #         图像宽度和高度（像素）。
    #     znear, zfar : float
    #         近裁面、远裁面（相机坐标系下 Z>0 的距离）。
    #     x0_is_left, y0_is_top : bool
    #         指明图像坐标(0,0)是否在左上角。若 y0_is_top=True，则需要稍后在投影到屏幕时做 Y 翻转。
    #         若你在 pipeline 里做过其他处理，也可自行修改此处公式符号。

    #     返回：
    #     ----------
    #     P : (4, 4) numpy.ndarray
    #         右手坐标系、+Z 向前的透视投影矩阵（OpenGL 风格），clip-space 范围 [-1,1] x [-1,1] x [-1,1]。
    #     """

    #     # 计算：在相机坐标系的近裁面 (Z=znear) 上，
    #     # 左侧、右侧、底部、顶部 (l, r, b, t) 的坐标（单位：米或与z相同单位）。
    #     # 假设：相机坐标 X 右、Y 上、Z 前
        
    #     # 像素 (0,0) 在左上角：x 从左到右增大，y 从上到下增大
    #     # 对应到相机坐标：x_c = (u - cx)/fx * z, y_c = (v - cy)/fy * z  (z>0)
    #     # => 当 u=0 时, x_c = -cx/fx * z;    当 u=W 时, x_c = (W - cx)/fx * z
    #     # => 当 v=0 时, y_c = -cy/fy * z;    当 v=H 时, y_c = (H - cy)/fy * z
    #     # 但注意 "y 轴向上" vs "图像行数向下" 的差异，需要在 top/bottom 符号上相应处理。

    #     l = -cx / fx * znear
    #     r = (W - cx) / fx * znear

    #     if y0_is_top:
    #         # v=0 在图像最上方 => 在相机坐标系中是 "y=-(cy)/fy * z"
    #         t = -cy / fy * znear
    #         b = (H - cy) / fy * znear
    #     else:
    #         # 若 y0_is_top=False，说明 y=0 在图像最下方(类似OpenGL左下角原点)
    #         # 可根据实际情况自行调整
    #         t = (H - cy) / fy * znear
    #         b = -cy / fy * znear
        
    #     # 构造 4x4 投影矩阵 (OpenGL right-handed, z in [-1,1])
    #     # 参考典型的 "glFrustum(l, r, b, t, znear, zfar)" 矩阵形式:
    #     #
    #     # [ 2n/(r-l)     0          (r+l)/(r-l)       0     ]
    #     # [     0     2n/(t-b)      (t+b)/(t-b)       0     ]
    #     # [     0        0        -(f+n)/(f-n)   -2fn/(f-n) ]
    #     # [     0        0           -1               0     ]
    #     #
    #     # 这里 n=znear, f=zfar, x,y 正方向同OpenGL(摄像机面向 -Z)。 
    #     # 若我们想 +Z 向前，需要改动一下第三行符号。这里示例给出“+Z向前”的版本，
    #     # 并让 z_{ndc} = -1 对应 z=znear, z_{ndc}=+1 对应 z=zfar (同OpenGL但反转Z)。
    #     #
    #     # 如果希望 z_{ndc}=[0,1] 也可自行改写。
        
    #     # 根据“+Z 朝前”并与 OpenGL [-1,1] 区间兼容的一个方案：
    #     # 第三行可以使用: z_{clip} = (f + n)/(f - n)*Z - (2fn/(f-n)),  并且取 w_{clip} = -Z
    #     # 这样 Z=znear => z_{ndc}=-1, Z=zfar => z_{ndc}=+1. 
    #     # 这会变成类似 "DX 右手系" 投影，但仍保持 -1~+1 范围。
    #     #
    #     # 为简洁，这里直接给出一个较常用的做法：保持OpenGL式右手坐标，但把相机设置为-Z可见。
    #     # 如果你确实想 +Z 可见，需要先对点做 z->-z 或者在 viewMatrix 里旋转。
    #     # 
    #     # ------ 如果你想真正的 +Z 可见 + [-1,1] clip: ------
    #     # 下面这块就给一个可用的:

    #     P = np.zeros((4, 4), dtype=np.float32)
    #     P[0, 0] = 2.0 * znear / (r - l)
    #     P[1, 1] = 2.0 * znear / (t - b)
    #     P[0, 2] = (r + l) / (r - l)
    #     P[1, 2] = (t + b) / (t - b)
    #     # “正Z向前”并且 z_{ndc} in [-1,1]:
    #     P[2, 2] = -(zfar + znear) / (zfar - znear)
    #     P[2, 3] = -(2.0 * znear * zfar) / (zfar - znear)
    #     P[3, 2] = -1.0

    #     # 这样构造的矩阵，其实仍是“OpenGL默认的 -Z 朝里”风格，但我们给z做了负值
    #     # => 一般在OpenGL中物体要放在 z<0 才能看见。
    #     # 如果要把点 (X, Y, Z>0) 直接投影，就要么 z-> -Z, 要么自定义个变体。
    #     #
    #     # 下面这段也可以改成 left-handed 版本或修改 z 符号以符合 +Z 向前习惯，
    #     # 这方面可以根据你的坐标系再做细调。
        
    #     return P

    # def test_projection(pts):
    #     """演示如何使用上述函数，并测试投影结果与 K 矩阵近似一致。"""
    #     fx = 521.27
    #     fy = 521.27
    #     cx = 131.3553
    #     cy = 296.9030
    #     W, H = 288, 512
    #     znear, zfar = 0.01, 100.0

    #     # 构造Off-axis投影矩阵
    #     P = build_offaxis_projection_matrix(fx, fy, cx, cy, W, H, znear, zfar, x0_is_left=False, y0_is_top=True)

    #     print("Off-axis Projection Matrix:\n", P)

    #     # 下面给一个示例点(相机坐标系下), 假设Z>0(在相机前方).
    #     # 这个例子只是演示投影到NDC, 不考虑世界变换。
    #     import numpy as np
        
    #     p_3d = pts
    #     n = p_3d.shape[0]
    #     p_hom = np.hstack([p_3d, np.ones((n, 1), dtype=np.float32)])

    #     # (1) 通过Off-axis矩阵投影
    #     p_clip = (P @ p_hom.T).T
    #     # 透视除法
    #     p_ndc = p_clip[:, :3] / p_clip[:, 3:4]
        
    #     # 映射到像素
    #     # 注意：若我们没有对y做额外翻转，这里先“看一下”结果
    #     def ndc_to_pixel(u_ndc, size):
    #         # 直接将 [-1,1] -> [0,size], 不翻转
    #         return (u_ndc + 1.0) * 0.5 * size
        
    #     px = ndc_to_pixel(p_ndc[:,0], W)
    #     py = ndc_to_pixel(p_ndc[:,1], H)

    #     print("Projected pixel (without Y flip):")
    #     for i in range(n):
    #         print(f"3D {p_3d[i]} -> NDC=({p_ndc[i,0]:.3f},{p_ndc[i,1]:.3f}) -> Pixel=({px[i]:.2f},{py[i]:.2f})")

    #     # (2) 用 K 矩阵针孔模型投影 (忽略畸变)，看是否接近
    #     K = np.array([
    #         [fx, 0,   cx],
    #         [0,  fy,  cy],
    #         [0,   0,   1]
    #     ], dtype=np.float32)
        
    #     # p_cam: (X, Y, Z>0)
    #     p_pinhole = (K @ p_3d.T).T
    #     p_pinhole[:, 0] /= p_pinhole[:, 2]
    #     p_pinhole[:, 1] /= p_pinhole[:, 2]

    #     print("\nPinhole K-matrix projection:")
    #     for i in range(n):
    #         print(f"3D {p_3d[i]} -> Pixel=({p_pinhole[i,0]:.2f},{p_pinhole[i,1]:.2f})")

    #     print("\n你会发现数值可能较接近，但上下会有反转或一定偏差。"
    #         "可以对 y 像素做 H-1-flip 或者改用 left-handed 矩阵/旋转 Z 等进一步对齐。")


    #     test_projection(pts=pts.cpu().detach().numpy())

    
    # test
    # n = pts.cpu().detach().numpy().shape[0]
    # p_hom = np.hstack([pts.cpu().detach().numpy(), np.ones((n, 1))])
    # p_cam = (viewpoint_camera.world_view_transform.cpu().numpy().T @ p_hom.T).T
    # b = (viewpoint_camera.projection_matrix.T.cpu().numpy() @ p_cam.T).T

    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None

    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            if separate_sh:
                dc, shs = pc.get_features_dc, pc.get_features_rest
            else:
                shs = pc.get_features
    else:
        colors_precomp = override_color

    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    if separate_sh:
        
        # all_same_device = check_same_device(means3D, means2D, dc, shs, colors_precomp, opacity)
        # print(f"所有张量是否在同一设备上: {all_same_device}")
        
        rendered_image, radii, depth_image = rasterizer(
            means3D = means3D,
            means2D = means2D,
            dc = dc,
            shs = shs,
            colors_precomp = colors_precomp,
            opacities = opacity,
            scales = scales,
            rotations = rotations,
            cov3D_precomp = cov3D_precomp)
    else:
        rendered_image, radii, depth_image = rasterizer(
            means3D = means3D,
            means2D = means2D,
            shs = shs,
            colors_precomp = colors_precomp,
            opacities = opacity,
            scales = scales,
            rotations = rotations,
            cov3D_precomp = cov3D_precomp)
        
    # Apply exposure to rendered image (training only)
    if use_trained_exp:
        exposure = pc.get_exposure_from_name(viewpoint_camera.image_name)
        rendered_image = torch.matmul(rendered_image.permute(1, 2, 0), exposure[:3, :3]).permute(2, 0, 1) + exposure[:3, 3,   None, None]

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    rendered_image = rendered_image.clamp(0, 1)
    out = {
        "render": rendered_image,
        "viewspace_points": screenspace_points,
        "visibility_filter" : (radii > 0).nonzero(),
        "radii": radii,
        "depth" : depth_image
        }
    
    return out
