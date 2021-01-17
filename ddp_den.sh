CUDA_VISIBLE_DEVICES=$1 python imagenet_den.py -a $2 --lr $3 --n_classes $4 --ratio $5 --pruning_amount $6 $9 --dist-url 'tcp://127.0.0.1:'$7 --dist-backend 'nccl' --multiprocessing-distributed --world-size 1 --rank 0 $8

# bash ddp_den.sh 4,5,6,7 w2 0.01 100 1.0 0.1 6789 /home/dataset/imagenet --pretrained --evaluate