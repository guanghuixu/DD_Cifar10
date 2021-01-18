# Train CIFAR10 with PyTorch

I'm playing with [PyTorch](http://pytorch.org/) on the CIFAR10 dataset.

## Prerequisites
- Python 3.6+
- PyTorch 1.0+

## Training
```
# Start training with: 
python main.py

# You can manually resume the training with: 
python main.py --resume --lr=0.01
```
## Training with ratio
```
CIFAR10: bash run_model.sh 4 main_ratio.py ResNet10
CIFAR100: bash run_model.sh 4 main_ratio_cifar100.py ResNet10 
```

## ImageNet visulization with hist
```
mkdir results  # the save dir
CUDA_VISIBLE_DEVICES=4 python imagenet.py -a resnet50 --pretrained /mnt/dataset/imagenet
```

## ImageNet: sample_class_ratio
```
CUDA_VISIBLE_DEVICES=4 python imagenet.py -a resnet50 -b 64 --pretrained /mnt/dataset/imagenet
# see imagenet.py Line 259
# see image_folder.py Line 110
```

## Train ImageNet within mobilenet
```
bash ddp.sh 0,1,2,3 256
```

## Train mobilenet in depth & width level
```
bash ddp.sh 0,1,2,3,4,5,6,7 d4 6789 /home/dataset/imagenet
```

## Train & Val dataset
```
# see imagenet_trainval.py Lines 245,263
# see image_folder Lines 146
```

## Train ResNet18
```
# see ddp.sh
bash ddp.sh 4,5,6,7 resnet18 0.1 6790 /home/dataset/imagenet
```

## Training from pruning model
```
CUDA_VISIBLE_DEVICES=$1 python imagenet_den.py -a $2 --lr $3 --n_classes $4 --ratio $5 --pruning_amount $6 $9 --dist-url 'tcp://127.0.0.1:'$7 --dist-backend 'nccl' --multiprocessing-distributed --world-size 1 --rank 0 $8

1. GPUs: 4,5,6,7
2. Arch: mobilenet15
3. lr: 0.1
4. number of classes: 100 (imagenet)
5. sample ratio of dataset: 1.0
6. pruning_amount: 0.7, each layer pruning 70% units
7. port: 6789 or 6890
8. imagenet dir: /home/dataset/imagenet
9. other params: "--pretrained --evaluate --resume checkpoint/mobilenet15-100-1.0_model_best.pth.tar"

bash ddp_den.sh 4,5,6,7 mobilenet15 0.1 100 1.0 0.1 6789 /home/dataset/imagenet "--pretrained --evaluate --resume checkpoint/mobilenet15-100-1.0_model_best.pth.tar"
```

## Training & Pruning & Finetune
```
bash train_pruning_finetune.sh 4,5,6,7 6789 mobilenet 0.1 100 0.1 0.6 /home/dataset/imagenet
```

## exp about ratio, keep n_classes=100
pruning_amount =  1 - (new_model / 1.5)
| ratio             | pruning_amount    | new_model        |
| ----------------- | ----------------- | -----------------|
| 0.1               |    0.43           |  0.85            |
| 0.2               |    0.40           |  0.90            |
| 0.3               |    0.36           |  0.95            |
| 0.4               |    0.33           |  1.00            |
| 0.5               |    0.30           |  1.05            |
| 0.6               |    0.26           |  1.10            |
| 0.7               |    0.23           |  1.15            |
| 0.8               |    0.20           |  1.20            |
| 0.9               |    0.16           |  1.25            |
| 1.0               |    0.13           |  1.30            |

## exp about n_classes, keep ratio=1.0
pruning_amount =  1 - (new_model / 1.5)
| n_classes         | pruning_amount    | new_model        |
| ----------------- | ----------------- | -----------------|
| 20                |    0.40           |  0.90            |
| 40                |    0.33           |  1.00            |
| 80                |    0.20           |  1.20            |
| 100               |    0.13           |  1.30            |
| 200               |    0.10           |  1.35            |
| 1000              |    0              |  1.50            |


