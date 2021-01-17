CUDA_VISIBLE_DEVICES=$1 python imagenet.py -a $2 --lr 0.1 --n_classes $4 --ratio $5 --epochs 2 \
    --dist-url 'tcp://127.0.0.1:'$6 --dist-backend 'nccl' --multiprocessing-distributed \
    --world-size 1 --rank 0 $8
CUDA_VISIBLE_DEVICES=$1 python imagenet_den.py -a $2 --lr $3 --n_classes $4 --ratio $5 --epochs 2 \
    --pruning_amount $7 --pretrained --evaluate --resume checkpoint/$2-$4-$5_model_best_train.pth.tar --dist-url 'tcp://127.0.0.1:'$6 --dist-backend 'nccl' \
    --multiprocessing-distributed --world-size 1 --rank 0 $8

# bash train_pruning_finetune.sh 4,5,6,7 mobilenet 0.1 100 1.0 6789 0.7 /home/dataset/imagenet