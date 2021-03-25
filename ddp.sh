CUDA_VISIBLE_DEVICES=$1 python imagenet_remapping.py -a $2 --lr $3 $6 --dist-url 'tcp://127.0.0.1:'$4 --dist-backend 'nccl' --multiprocessing-distributed --world-size 1 --rank 0 $5

# bash ddp.sh 3,4,5,7 mobilenet15 0.2 6719 /mnt/cephfs/dataset/imagenet '--ratio 0.4'
