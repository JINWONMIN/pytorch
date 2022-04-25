import os
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

from util import *

## 데이터 로더 구현
class Dataset(torch.utils.data.Dataset):    # Dataset 클래스에 torch.utils.data.Dataset 클래스를 상속
    def __init__(self, data_dir, transform=None, task=None, opts=None):   # 할당 받을 인자 선언 (첫 선언)
        self.data_dir = data_dir
        self.transform = transform
        self.task = task
        self.opts = opts


        # prefixed word를 이용해 prefixed 되어 있는 input / label data를 나눠서 불러오기
        lst_data = os.listdir(self.data_dir) # os.listdir 메소드를 이용해서 data_dir에 있는 모든 파일을 불러온다.
        lst_data = [f for f in lst_data if f.endswith('jpg') | f.endswith('png')]  # 확장자가 jpg, png인 파일만 불러온다.

        lst_data.sort()

        self.lst_data = lst_data  # lst_data 라는 instance 선언

    def __len__(self):
        return len(self.lst_data)

    def __getitem__(self, index):   # index를 인자로 받아서 index에 해당하는 파일을 로드해서 리턴하는 형태로 정의
        # label = np.load(os.path.join(self.data_dir, self.lst_label[index])) # NumPy 형식으로 저장되어 있어 np.load로 불러온다.
        # input = np.load(os.path.join(self.data_dir, self.lst_input[index]))

        img = plt.imread(os.path.join(self.data_dir, self.lst_data[index]))
        sz = img.shape

        # 이미지가 세로가 긴 이미지, 가로가 긴 이미지 이렇게 뒤죽박죽 섞여 있어 이것을 하나의 기준으로 정렬하기 위해 아래 코드 작성
        if sz[0] > sz[1]:
            img = img.transpose((1, 0, 2))  # 항상 가로로 긴 이미지로 transpose 될 수 있도록 if문 작성

        # 이미지를 normalization 해주는 경우는 data type이 uint8인 경우에만 해줘서 data type이 uint8인 경우에만 정규화 해주는 코드 작성
        if img.dtype == np.uint8:
            img = img / 255.0

        if img.ndim == 2:     # input 값은 최소 3차원 으로 넣어야 해서 2차원인 경우
            img = img[:, :, np.newaxis]     # 마지막 axis 임의로 생성

        label = img     # 정규화 된 이미지는 label로 정의한다.

        if self.task == "denoising":
            input = add_noise(img, type=self.opts[0], opts=self.opts[1])
        elif self.task == "inpainting":
            input = add_sampling(img, type=self.opts[0], opts=self.opts[1])
        elif self.task == "super_resolution":
            input = add_blur(img, type=self.opts[0], opts=self.opts[1])

        data = {'input': input, 'label': label} # 생성된 label 과 input을 딕셔너리 형태로 저장

        if self.transform:  # transform function을 data loader의 argument로 넣어주고,
            data = self.transform(data)     # transform 함수가 정의 되어 있다면, transform 함수를 통과한 data를 리턴

        return data


## Data transform 구현하기
class ToTensor(object): # (NumPy -> Tensor)
    def __call__(self, data):
        label, input = data['label'], data['input']

        label = label.transpose((2, 0, 1)).astype(np.float32) # [Image의 numpy 차원 = (Y, X, CH)] -> [Image의 pytorch tensor 차원 = (CH, Y, X)]
        input = input.transpose((2, 0, 1)).astype(np.float32)

        data = {'label': torch.from_numpy(label), 'input': torch.from_numpy(input)}  # from_numpy 함수: numpy를 tensor로 넘겨줌.

        return data



'''
segmentation 같은 경우는 input 이미지에 대해서만 normalization을 해 주었지만,
regression | restoration을 수행하는 경우에는 label 이미지도 normalization을 해주어야 한다.
'''
class Normalization(object):
    def __init__(self, mean=0.5, std=0.5):
        self.mean = mean
        self.std = std

    def __call__(self, data):
        label, input = data['label'], data['input']

        label = (label - self.mean) / self.std
        input = (input - self.mean) / self.std
        # 여기서 label data는 0 or 1 클래스로 되어 있어서 정규화 X

        data = {'label': label, 'input': input}

        return data

class RandomFlip(object):
    def __call__(self, data):
        label, input = data['label'], data['input']

        if np.random.rand() > 0.5:
            label = np.fliplr(label)    # 항상 input 과 label을 동시에 해줘야한다.
            input = np.fliplr(input)

        if np.random.rand() > 0.5:
            label = np.flipud(label)
            input = np.flipud(input)

        data = {'label': label, 'input': input}

        return data

class RandomCrop(object):
    def __init__(self, shape):
        self.shape = shape  # crop할 shape을 인자로 받는다.

    def __call__(self, data):   # data가 들어오면 label과 input에도 crop 적용
        input, label = data['input'], data['label']

        h, w = input.shape[:2]
        new_h, new_w = self.shape

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        id_y = np.arange(top, top + new_h, 1)[:, np.newaxis]
        id_x = np.arange(left, left + new_w, 1)

        input = input[id_y, id_x]
        label = label[id_y, id_x]

        data = {'input': input, 'label': label}

        return data