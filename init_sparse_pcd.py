"""
    使用单帧depth图像生成稀疏点云
    1.读取深度图, 生成xyz
    2.读取对应RGB图, 生成color
    3.生成zeros的法向量
    4.创建BasicPointCloud对象
    5.保存为point3D.bin文件
"""
from submodules.vision_tool.utils import load_depth, backproject
import cv2
import numpy as np
from scene.gaussian_model import BasicPointCloud
from scene.dataset_readers import storePly
from scene.colmap_loader import read_points3D_binary, write_points3D_binary
import collections

def init_frame_pcd(depth, rgb, intrinsics, mask=None):
    """
        input:
            depth: (w, h)
            rgb: (w, h, 3)
            intrinsics: [fx, fy, cx, cy]
        return:
            BasicPointCloud
    """
    positions, choose = backproject(depth, intrinsics, mask)
    colors = rgb.reshape(-1, rgb.shape[-1])[choose]
    normals = np.zeros_like(positions)
    print("positions.shape: ", positions.shape)
    print("colors.shape: ", colors.shape)
    print("normals.shape: ", normals.shape)
    # pcd = BasicPointCloud(points=positions, colors=colors, normals=normals)
    
    Point3D = collections.namedtuple(
        "Point3D", ["id", "xyz", "rgb", "error", "image_ids", "point2D_idxs"]
    )
    pt_n = positions.shape[0]
    # error = np.zeros((positions.shape[0], 1), dtype=np.float32)
    errors = np.zeros(pt_n)  # 所有点的误差为0
    # 创建一个空的或者填充0的image_ids和point2D_idxs
    image_ids = np.zeros(0, dtype=np.uint32)
    point2D_idxs = np.zeros(0, dtype=np.uint32)

    # 创建Point3D对象
    pcd = {
        i: Point3D(i, positions[i], colors[i], errors[i], image_ids, point2D_idxs) for i in range(pt_n)
    }
    return pcd, positions, colors, normals

if __name__ == "__main__":
    # settings
    rgb_path = "/data4/cxx/dataset/desk_3/color/00001.png"
    depth_path = "/data4/cxx/dataset/desk_3/depth/00001.png"
    mask = None
    bin_save_path = "/data4/cxx/dataset/desk_3/gs_input/train_single/sparse/0/points3D.bin"
    ply_save_path = "/data4/cxx/dataset/desk_3/gs_input/train_single/sparse/0/points3D.ply"
    K = [390.029, 390.029, 320.854, 241.826]  # fx, fy, cx, cy
    
    depth = load_depth(depth_path)
    depth = depth.astype(np.uint16)
    rgb = cv2.imread(rgb_path)[:, :, :3]
    rgb = rgb[:, :, ::-1]
    
    # test read_points3D_binary
    bin_path = "/data4/cxx/dataset/desk_3/gs_input/train/sparse/0/points3D.bin"
    xyz, rgb, _ = read_points3D_binary(bin_path)
    
    # generate pcd
    pcd, positions, colors, normals = init_frame_pcd(depth, rgb, K, mask)
    
    # generate .bin file
    write_points3D_binary(pcd, bin_save_path)
    try:
        print("save pointcloud to {}".format(bin_save_path))
    except:
        print("save pointcloud to {} failed!".format(bin_save_path))
    
    # generate .ply file
    try:
        storePly(ply_save_path, positions, colors)
        print("save pointcloud to {}".format(ply_save_path))
    except:
        print("save pointcloud to {} failed!".format(ply_save_path))