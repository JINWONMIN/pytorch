import os
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from torchvision import transforms, datasets

## Network building
class Net(nn.Module):   # torch의 Module 클래스를 Net 클래스에 상속
    def __init__(self):
        super(Net, self).__init__()

        # Initialization of Layers for Network Construction
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=10, kernel_size=5, stride=1, padding=0, bias=True)
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv2d(in_channels=10, out_channels=20, kernel_size=5, stride=1, padding=0, bias=True)
        self.drop2 = nn.Dropout2d(p=0.5)
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        self.relu2 = nn.ReLU()

        self.fc1 = nn.Linear(in_features=320, out_features=50, bias=True)
        self.relu1_fc1 = nn.ReLU()
        self.drop1_fc1 = nn.Dropout2d(p=0.5)

        self.fc2 = nn.Linear(in_features=50, out_features=10, bias=True)

    # 초기화된 레이어를 연결하여 네트워크 구축
    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.drop2(x)
        x = self.pool2(x)
        x = self.relu2(x)

        x = x.view(-1, 320)

        x = self.fc1(x)
        x = self.relu1_fc1(x)
        x = self.drop1_fc1(x)

        x = self.fc2(x)

        return x
