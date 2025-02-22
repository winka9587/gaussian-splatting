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
import numpy as np
from typing import NamedTuple

class BasicPointCloud(NamedTuple):
    points : np.array
    colors : np.array
    normals : np.array

def geom_transform_points(points, transf_matrix):
    P, _ = points.shape
    ones = torch.ones(P, 1, dtype=points.dtype, device=points.device)
    points_hom = torch.cat([points, ones], dim=1)
    points_out = torch.matmul(points_hom, transf_matrix.unsqueeze(0))

    denom = points_out[..., 3:] + 0.0000001
    return (points_out[..., :3] / denom).squeeze(dim=0)

def getWorld2View(R, t):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0
    return np.float32(Rt)

def getWorld2View2(R, t, translate=np.array([.0, .0, .0]), scale=1.0):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()  # c2w -> w2c
    Rt[:3, 3] = t  # w2c
    Rt[3, 3] = 1.0

    C2W = np.linalg.inv(Rt)  # get c2w
    cam_center = C2W[:3, 3]
    cam_center = (cam_center + translate) * scale
    C2W[:3, 3] = cam_center
    Rt = np.linalg.inv(C2W)
    return np.float32(Rt)

def getProjectionMatrix(znear, zfar, fovX, fovY, cx=None, cy=None, W=None, H=None):
    tanHalfFovY = math.tan((fovY / 2))
    tanHalfFovX = math.tan((fovX / 2))

    top = tanHalfFovY * znear
    bottom = -top
    right = tanHalfFovX * znear
    left = -right

    P = torch.zeros(4, 4)

    z_sign = 1.0

    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    
    if cx is not None and cy is not None and W is not None and H is not None:
        P[0, 2] = - (2 * cx / W - 1)  # 归一化 cx
        P[1, 2] = - (2 * cy / H - 1)  # 归一化 cy
    return P

def fov2focal(fov, pixels):
    return pixels / (2 * math.tan(fov / 2))

def focal2fov(focal, pixels):
    return 2*math.atan(pixels/(2*focal))

def build_offaxis_projection_matrix_p6(
    fx, fy, cx, cy, W, H,
    znear, zfar
):
    """
    构造与“P6”类似的右手系 +Z 朝前透视投影矩阵，以逼近针孔相机内参 (fx, fy, cx, cy)。
    
    参数:
    -------
    fx, fy : float
        相机焦距(像素)，对应内参 K[0,0], K[1,1].
    cx, cy : float
        主点(光心)在图像坐标中的像素位置，对应 K[0,2], K[1,2].
    W, H : int
        图像宽、高(像素)。
    znear, zfar : float
        近裁面、远裁面(单位与相机坐标系一致)，需 z>0 才可见。

    返回:
    -------
    P : (4,4) np.ndarray (float32)
        类似P6的投影矩阵(右手系、+Z可见)，clip-space 范围通常是 x,y,z ∈ [-1,1] 。
        
    用法:
    -------
    1) 点从相机坐标 (x,y,z,1) 乘以该矩阵得到 clip-space (x',y',z',w').
    2) 做透视除法 (x'/w', y'/w', z'/w') => NDC in [-1,1].
    3) NDC -> 像素时，如果图像(0,0)在左上角，需对 y 做翻转: 
         x_pix = (x_ndc + 1)*W/2
         y_pix = H - 1 - (y_ndc + 1)*H/2
       或者在Shader等其他环节处理。
    """

    # -------------------------
    # 1) x,y方向的缩放系数
    #    在传统针孔模型下，如果我们期望:
    #      x_ndc = (fx * x) / z  变换到 [-1,1] 则还需要归一化到 [-1,1] 宽度 => 结论是:
    #      s_x = 2*fx / W,   s_y = 2*fy / H
    #    这样当 x=±(W/2)/fx * z 时，可对应 x_ndc=±1.
    # -------------------------
    sx = 2.0 * fx / W
    sy = 2.0 * fy / H  # 注意: 此处是正号 => y 向上; 若希望 y 向下，可加负号(看具体需求)

    # -------------------------
    # 2) 偏移项 p_x, p_y
    #    理想上, 我们想让当 x=0 => x_ndc =  - (2*cx/W - 1),
    #    但因为本例把 “偏移” 合并进第三行以实现 off-axis,
    #    就需要先算出 “NDC层” 里对 x,y 的移位(常见公式: - (2*(cx/W) -1)).
    #    这里的 p_x, p_y 会出现在 Z_clip 行, 影响最终 (x_ndc, y_ndc) 间接加入主点.
    # -------------------------
    dx = (2.0 * cx / W) - 1.0   # 通常 "2*cx/W -1" 
    dy = (2.0 * cy / H) - 1.0

    # P6 里看到的偏移往往是 -dx, +dy(或相反)——具体因行列分配而定
    px = dx
    py = dy
    # 你可以根据实际渲染是否倒置, 再调这两个符号. 
    # 例如, 若发现图像水平方向正好反了, 可以改 px = +dx; 
    # 若垂直反了, 可以改 py = -dy 等.

    # -------------------------
    # 3) 近/远裁面项
    #    在典型 OpenGL(right-handed) 中, z_clip = A*z + B, w_clip = -z,  (面向 -Z).
    #    这里我们想 +Z 向前, 并把 z ∈ [znear, zfar] => z_ndc ∈ [-1,1].
    #    取: w_clip = z,  则 z_clip = alpha*z + beta. 
    #    令 alpha = (zfar+znear)/(zfar-znear), beta = -(2*znear*zfar)/(zfar-znear) 
    #    不同库/引擎可能选用不同的符号, 也可选 z_clip = alpha*z + 1, 具体看你要怎样映射.
    # 
    #    观测到 P6 里对第三行写了 ( ... , 1.0002, 1 ), 似乎把 beta=+1, alpha≈1.0002, 
    #    这说明 z_ndc = (alpha*z + 1)/z => alpha + 1/z, 
    #    不完全是传统 frustum 的做法，但在某些场景下也可以工作。
    #    如果你就是想完全复刻 P6 那种写法, 可以这么做:
    # -------------------------

    alpha = (zfar + znear) / (zfar - znear)  # ~ 1.0002 (当 near=0.01, far=100)
    # 而把 beta 固定成 +1, 这样第三行就跟P6一致:
    beta  = 1.0

    # -------------------------
    # 4) 组装矩阵 (形似 P6)
    #
    #   P6 = [
    #     [ sx, 0,   0,    0 ],
    #     [ 0,  sy,  0,    0 ],
    #     [ px, py, alpha, beta ],
    #     [ 0,  0,   1,    0 ]
    #   ]
    #
    #   => x_clip = sx * x
    #      y_clip = sy * y
    #      z_clip = px*x + py*y + alpha*z + beta
    #      w_clip = z
    #
    #   之后 x_ndc = (sx*x)/z, y_ndc=(sy*y)/z, z_ndc=(px*x+py*y+alpha*z+beta)/z.
    #   如此可满足:  (1) +Z>0  (2) off-axis 偏移  (3) near/far 区间映射 [-1,1].
    #
    #   注：之所以第三行包含对 x,y,z,1 的线性组合，是因为我们把“offset/near/far”一并塞到 z_clip 行。
    # -------------------------

    P = np.array([
        [ sx,   0.0,   0.0,     0.0 ],
        [ 0.0,  sy,    0.0,     0.0 ],
        [ px,   py,    alpha,   beta],
        [ 0.0,  0.0,   1.0,     0.0 ]
    ], dtype=np.float32)

    return P