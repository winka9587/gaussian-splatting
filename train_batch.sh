conda activate gs
export PATH=/usr/local/cuda-11.6/bin:$PATH
nvcc -V
/home/lab/workspace/cxx/gaussian-splatting
python train.py -s /home/lab/gs/360_v2/bonsai -m /home/lab/gs/output/bonsai
python train.py -s /home/lab/gs/360_v2/counter -m /home/lab/gs/output/counter
python train.py -s /home/lab/gs/360_v2/drjohnson -m /home/lab/gs/output/drjohnson
python train.py -s /home/lab/gs/360_v2/flowers -m /home/lab/gs/output/flowers
python train.py -s /home/lab/gs/360_v2/garden -m /home/lab/gs/output/garden
python train.py -s /home/lab/gs/360_v2/kitchen -m /home/lab/gs/output/kitchen
python train.py -s /home/lab/gs/360_v2/playroom -m /home/lab/gs/output/playroom
python train.py -s /home/lab/gs/360_v2/room -m /home/lab/gs/output/room
python train.py -s /home/lab/gs/360_v2/stump -m /home/lab/gs/output/stump
python train.py -s /home/lab/gs/360_v2/train -m /home/lab/gs/output/train
python train.py -s /home/lab/gs/360_v2/treehill -m /home/lab/gs/output/treehill
python train.py -s /home/lab/gs/tandt_db/tandt/train -m /home/lab/gs/output/train
python train.py -s /home/lab/gs/tandt_db/tandt/truck -m /home/lab/gs/output/truck
