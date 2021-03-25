'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse

import models
from utils import progress_bar

import random
import numpy as np
from cifar import CIFAR10, CIFAR100
from models.my_mobilenet.derived_imagenet_net import ImageNetModel
from models.my_mobilenet.param_remap import remap_for_paramadapt
from models.my_mobilenet import configs
from tensorboardX import SummaryWriter

def seed_torch(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed) 
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) 
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    
seed_torch(2021)


parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--model_name', default='mobilenet', type=str, help='model name')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--ratio', default=0.5, type=float, help='sample ratio')
parser.add_argument('--remapping', '-r', action='store_true',
                    help='remapping from pretrained checkpoint')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch
writer_dir = 'runs/{}-{}-{}'.format(args.model_name, args.remapping, args.ratio)
writer = SummaryWriter(writer_dir)
global_training_steps = 0

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])


# trainset = torchvision.datasets.CIFAR10(
#     root='./data', train=True, download=True, transform=transform_train)
trainset = CIFAR100(
    root='./data', train=True, download=True, transform=transform_train,
    ratio=args.ratio)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=512, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR100(
    root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=100, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

# Model
print('==> Building model..')
# net = getattr(models, args.model_name)(num_classes=100)
net = ImageNetModel(
        net_config=getattr(configs, '{}_config'.format(args.model_name)), 
        num_classes=100)

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/ckpt.pth', map_location='cpu')
    net.load_state_dict(checkpoint['net'])
    # best_acc = checkpoint['acc']
    # start_epoch = checkpoint['epoch']

if args.remapping:
    state_dict = remap_for_paramadapt(
        load_path='checkpoint/mobilenet_ckpt-0.5-CIFAR100.pth', 
        model_dict=net.state_dict(), 
        seed_num_layers=[1, 1, 2, 3, 4, 3, 3, 1, 1])
    net.load_state_dict(state_dict)
    print('success remap_for_paramadapt')

net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr,
                      momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)


# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    global global_training_steps
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        acc = 100.*correct/total
        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss/(batch_idx+1), acc, correct, total))
        writer.add_scalar('training iter loss', loss.item(), global_training_steps)
        writer.add_scalar('training iter acc', acc, global_training_steps)
        global_training_steps += 1
    acc = 100.*correct/total
    writer.add_scalar('train acc', acc, epoch)

def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
                         
    # Save checkpoint.
    acc = 100.*correct/total
    writer.add_scalar('test acc', acc, epoch)
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        if args.remapping:
            torch.save(state, './checkpoint/{}_ckpt-{}-CIFAR100_remapping.pth'.format(args.model_name, args.ratio))
        else:
            torch.save(state, './checkpoint/{}_ckpt-{}-CIFAR100.pth'.format(args.model_name, args.ratio))
        best_acc = acc


for epoch in range(start_epoch, start_epoch+200):
    train(epoch)
    test(epoch)
    scheduler.step()