CUDA_VISIBLE_DEVICES=$1 python imagenet.py -a resnet50 -b $2 --dist-url 'tcp://127.0.0.1:6789' --dist-backend 'nccl' --multiprocessing-distributed --world-size 1 --rank 0 /home/dataset/imagenet
