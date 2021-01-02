CUDA_VISIBLE_DEVICES=$1 python imagenet.py -a $2 --pretrained --dist-url 'tcp://127.0.0.1:'$3 --dist-backend 'nccl' --multiprocessing-distributed --world-size 1 --rank 0 $4
