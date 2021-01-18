CUDA_VISIBLE_DEVICES=$1 python imagenet.py -a $3 --lr 0.1 --n_classes $5 --ratio $6 \
    --output $9 --resume --dist-url 'tcp://127.0.0.1:'$2 --dist-backend 'nccl' --multiprocessing-distributed \
    --world-size 1 --rank 0 $8
CUDA_VISIBLE_DEVICES=$1 python imagenet_den.py -a $3 --lr $4 --n_classes $5 --ratio $6 \
    --pruning_amount $7 --output $9 --pretrained $9/checkpoint/$3-$5-$6_final_train.pth.tar --evaluate --resume \
    --dist-url 'tcp://127.0.0.1:'$2 --dist-backend 'nccl' \
    --multiprocessing-distributed --world-size 1 --rank 0 $8

# bash train_pruning_finetune.sh 4,5,6,7 6789 mobilenet 0.1 100 0.1 0.6 /home/dataset/imagenet exp