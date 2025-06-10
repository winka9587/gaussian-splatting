conda activate 3dgs
export PATH=/usr/local/cuda-11.6/bin:$PATH
nvcc -V
cd /data3/cxx/workspace/match/gs_submodules/gaussian_splatting
python train.py -s /data3/cxx/workspace/match/data/fineRGBA -m /data3/cxx/workspace/match/data/fineRGBA/output/
