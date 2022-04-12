import argparse

import os
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from model import Net
from util import *

from torchvision import transforms, datasets

## Parser 생성
parser = argparse.ArgumentParser(description="Train the Net",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("--lr", default=1e-3, type=float, dest="lr")
parser.add_argument("--batch_size", default=64, type=int, dest="batch_size")
parser.add_argument("--num_epoch", default=10, type=int, dest="num_epoch")

parser.add_argument("--ckpt_dir", default='./checkpoint', type=str, dest="ckpt_dir")
parser.add_argument("--log_dir", default='./log', type=str, dest="log_dir")

parser.add_argument("--mode", default="train", type=str, dest="mode")
parser.add_argument("--train_continue", default="off", type=str, dest="train_continue")

args = parser.parse_args()

## parameters setting for training
lr = args.lr
batch_size = args.batch_size
num_epoch = args.num_epoch

ckpt_dir = args.ckpt_dir  # 학습된 네트워크가 저장이 될 디렉토리
log_dir = args.log_dir   # 텐서보드의 로그가 저장이 될 디렉토리

mode = args.mode
train_continue = args.train_continue

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')

print("learning rate: %.4e" % lr)
print("batch size: %d" % batch_size)
print("number of epoch: %d" % num_epoch)
print("ckpt dir: %s" % ckpt_dir)
print("log dir: %s" % log_dir)
print("mode: %s" % mode)

## MNIST 데이터 불러오기
if mode == "train":
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=(0.5,), std=(0.5,))])

    dataset = datasets.MNIST(download=True, root='./', train=True, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    num_data = len(loader.dataset)
    num_batch = np.ceil(num_data / batch_size)
else:
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=(0.5,), std=(0.5,))])

    dataset = datasets.MNIST(download=True, root='./', train=True, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    num_data = len(loader.dataset)
    num_batch = np.ceil(num_data / batch_size)


## 네트워크 설정 및 필요한 손실함수 구하기
net = Net().to(device)
params = net.parameters()

fn_loss = nn.CrossEntropyLoss().to(device)
fn_pred = lambda output: torch.softmax(output, dim=1)
fn_acc = lambda pred, label: ((pred.max(dim=1)[1] == label).type(torch.float)).mean()

optim = torch.optim.Adam(params, lr=lr)

writer = SummaryWriter(log_dir=log_dir)

## 트레이닝 시작
# TRAIN MODE
if mode == "train":
    if train_continue == "on":
        net, optim = load(ckpt_dir=ckpt_dir, net=net, optim=optim)

    for epoch in range(1, num_epoch + 1):    # 네트워크 학습을 위한 for문 구현
        net.train()     # network에게 train() 단계라고 알려주기

        loss_arr = []
        acc_arr = []

        for batch, (input, label) in enumerate(loader, 1):
            input = input.to(device)
            label = label.to(device)

            output = net(input)
            pred = fn_pred(output)

            optim.zero_grad()

            loss = fn_loss(output, label)
            acc = fn_acc(pred, label)

            loss.backward()

            optim.step()

            loss_arr += [loss.item()]
            acc_arr += [acc.item()]

            print('TRAIN: EPOCH %04d/%04d | BATCH %04d/%04d | LOSS: %.4f | ACC %.4f' %
            (epoch, num_epoch, batch, num_batch, np.mean(loss_arr), np.mean(acc_arr)))

        writer.add_scalar('loss', np.mean(loss_arr), epoch)
        writer.add_scalar('acc', np.mean(acc_arr), epoch)

        save(ckpt_dir=ckpt_dir, net=net, optim=optim, epoch=epoch)

    writer.close()

## TEST MODE
else:
    net, optim = load(ckpt_dir=ckpt_dir, net=net, optim=optim)

    with torch.no_grad():
        net.eval()

        loss_arr = []
        acc_arr = []

        for batch, (input, label) in enumerate(loader, 1):
            input = input.to(device)
            label = label.to(device)

            output = net(input)
            pred = fn_pred(output)

            loss = fn_loss(output, label)
            acc = fn_acc(pred, label)

            loss_arr += [loss.item()]
            acc_arr += [acc.item()]

            print('TEST: BATCH %04d/%04d | LOSS: %.4f | ACC %.4f' %
                  (batch, num_batch, np.mean(loss_arr), np.mean(acc_arr)))

    print("AVERAGE TEST: ACCURACY %.4f | LOSS %.4f" %
          (np.mean(acc_arr), np.mean(loss_arr)))













