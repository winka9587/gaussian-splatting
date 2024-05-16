"""
    测试colmap的二进制文件读取
"""
from scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, \
        read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary, read_points3D_text, \
        write_intrinsics_binary, write_extrinsics_binary, write_points3D_binary
import os
from scene.dataset_readers import sceneLoadTypeCallbacks

def load_files():
    source_path = "/data4/cxx/dataset/gs/ismar/"
    images = "images"
    eval = False
    scene_info = sceneLoadTypeCallbacks["Colmap"](source_path, images, eval)
    print(scene_info)




if __name__ == "__main__":
    load_files()