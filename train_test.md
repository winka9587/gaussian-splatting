
# gaussian splatting代码测试+解读

测试输入:

    训练图像194
    初始点云5.4k 
测试结果(粗略的计时, 因为这其中包含了加载和保存文件的时间):
    
    7k-iter 5min
    30k-iter 30min

测试训练结果

~~~bash 
./SIBR_viewers/install/bin/SIBR_gaussianViewer_app -m /home/lab/gs/output/bicycle/ -path /home/lab/gs/360_v2/bicycle
~~~

~~~bash
(gs) lab@lab-PC:~/workspace/cxx/gaussian-splatting$ python train.py -s /home/lab/gs/360_v2/bicycle -m /home/lab/gs/output/bicycle
Optimizing /home/lab/gs/output/bicycle
Output folder: /home/lab/gs/output/bicycle [06/12 19:00:54]
Tensorboard not available: not logging progress [06/12 19:00:54]
Reading camera 194/194 [06/12 19:00:54]
Converting point3d.bin to .ply, will happen only the first time you open the scene. [06/12 19:00:54]
Loading Training Cameras [06/12 19:00:54]
[ INFO ] Encountered quite large input images (>1.6K pixels width), rescaling to 1.6K.
 If this is not desired, please explicitly specify '--resolution/-r' as 1 [06/12 19:00:54]
Loading Test Cameras [06/12 19:01:55]
Number of points at initialisation :  54275 [06/12 19:01:55]
Training progress:  23%|██████████████████████████████████████████████▏                                                                                                                                                       | 7000/30000 [03:52<18:27, 20.76it/s, Loss=0.0939351]
[ITER 7000] Evaluating train: L1 0.03885948732495308 PSNR 23.782460403442386 [06/12 19:05:48]

[ITER 7000] Saving Gaussians [06/12 19:05:48]
Training progress: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 30000/30000 [30:02<00:00, 16.65it/s, Loss=0.0589145]

[ITER 30000] Evaluating train: L1 0.026784038916230202 PSNR 27.015393829345705 [06/12 19:31:58]

[ITER 30000] Saving Gaussians [06/12 19:31:58]

Training complete. [06/12 19:32:19]
~~~

/scene/colmap_loader.cpp 中有函数write_points3D_binary用于将一组三维点云(point3D)写入.bin文件中

~~~bash
def write_points3D_binary(points3D, path_to_model_file):
    """
    see: src/colmap/scene/reconstruction.cc
        void Reconstruction::ReadPoints3DBinary(const std::string& path)
        void Reconstruction::WritePoints3DBinary(const std::string& path)
    """
    with open(path_to_model_file, "wb") as fid:
        write_next_bytes(fid, len(points3D), "Q")
        for _, pt in points3D.items():
            write_next_bytes(fid, pt.id, "Q")
            write_next_bytes(fid, pt.xyz.tolist(), "ddd")
            write_next_bytes(fid, pt.rgb.tolist(), "BBB")
            write_next_bytes(fid, pt.error, "d")
            track_length = pt.image_ids.shape[0]
            write_next_bytes(fid, track_length, "Q")
            for image_id, point2D_id in zip(pt.image_ids, pt.point2D_idxs):
                write_next_bytes(fid, [image_id, point2D_id], "ii")， 
~~~

另一个函数用来读取bin文件：

~~~bash
def read_points3D_binary(path_to_model_file):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::ReadPoints3DBinary(const std::string& path)
        void Reconstruction::WritePoints3DBinary(const std::string& path)
    """


    with open(path_to_model_file, "rb") as fid:
        num_points = read_next_bytes(fid, 8, "Q")[0]

        xyzs = np.empty((num_points, 3))
        rgbs = np.empty((num_points, 3))
        errors = np.empty((num_points, 1))

        for p_id in range(num_points):
            binary_point_line_properties = read_next_bytes(
                fid, num_bytes=43, format_char_sequence="QdddBBBd")
            xyz = np.array(binary_point_line_properties[1:4])
            rgb = np.array(binary_point_line_properties[4:7])
            error = np.array(binary_point_line_properties[7])
            track_length = read_next_bytes(
                fid, num_bytes=8, format_char_sequence="Q")[0]  # "image_ids", 
            track_elems = read_next_bytes(
                fid, num_bytes=8*track_length,
                format_char_sequence="ii"*track_length)  # "point2D_idxs"
            xyzs[p_id] = xyz
            rgbs[p_id] = rgb
            errors[p_id] = error
    return xyzs, rgbs, errors 
~~~

观察这两个函数可以发现，在读取.bin文件后,仅使用了其中的xyz,rgb,error这三个属性。所以如果自定义三维点云来替换colmap的生成结果, 需要有：
    1.(n,3)的点云坐标xyz，
    2.(n,3)的颜色rgbs，
    3.(n,1)的errors，其中errors的值全部为0. 
创建一组points3D，填充其属性，使其可以使用write_points3D_binary写入为一个bin文件，同时还能够被read_points3D_binary读取。