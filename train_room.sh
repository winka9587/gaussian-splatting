conda activate gaussian_splatting
export PATH=/usr/local/cuda-11.6/bin:$PATH
nvcc -V
cd /data4/cxx/workplace/gs/gaussian-splatting
python train.py -s /data4/cxx/dataset/gs/room/ -m /data4/cxx/dataset/gs/room/output/
