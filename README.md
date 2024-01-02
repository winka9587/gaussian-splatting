# 3D Gaussian Splatting for Real-Time Radiance Field Rendering
<a href="README_origin.md"><img src="assets/teaser.png"> </a>

# Train & Test on custom dataset

    1214 images(RGB, 640×480) as input of colmap

<center>
<video src="./assets/desk_3.mp4" controls style="width: 40%;"></video>
<p>input images</p>

<video src="./assets/desk_3_3dgs.mp4" controls style="width: 40%;"></video>
<p>results of 3D gaussian</p>
</center>

# init from single depth image

init_sparse_pcd.py 读取rgb和depth图像, 生成points3D.bin文件, 用于取代colmap的生成结果, 进行后续的训练。
