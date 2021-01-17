CUDA_VISIBLE_DEVICES=$1 python imagenet.py -a $2 --lr $3 $6 --dist-url 'tcp://127.0.0.1:'$4 --dist-backend 'nccl' --multiprocessing-distributed --world-size 1 --rank 0 $5

# bash ddp.sh 4,5,6,7 resnet18 0.1 6790 /home/dataset/imagenet