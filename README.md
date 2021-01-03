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

## Accuracy
| Model             | Acc.        |
| ----------------- | ----------- |
| [VGG16](https://arxiv.org/abs/1409.1556)              | 92.64%      |
| [ResNet18](https://arxiv.org/abs/1512.03385)          | 93.02%      |
| [ResNet50](https://arxiv.org/abs/1512.03385)          | 93.62%      |
| [ResNet101](https://arxiv.org/abs/1512.03385)         | 93.75%      |
| [RegNetX_200MF](https://arxiv.org/abs/2003.13678)     | 94.24%      |
| [RegNetY_400MF](https://arxiv.org/abs/2003.13678)     | 94.29%      |
| [MobileNetV2](https://arxiv.org/abs/1801.04381)       | 94.43%      |
| [ResNeXt29(32x4d)](https://arxiv.org/abs/1611.05431)  | 94.73%      |
| [ResNeXt29(2x64d)](https://arxiv.org/abs/1611.05431)  | 94.82%      |
| [SimpleDLA](https://arxiv.org/abs/1707.064)           | 94.89%      |
| [DenseNet121](https://arxiv.org/abs/1608.06993)       | 95.04%      |
| [PreActResNet18](https://arxiv.org/abs/1603.05027)    | 95.11%      |
| [DPN92](https://arxiv.org/abs/1707.01629)             | 95.16%      |
| [DLA](https://arxiv.org/abs/1707.064)                 | 95.47%      |

