import pandas as pd
from PIL import Image
from torch.utils import data
from skimage import io
import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision import transforms
import json
import matplotlib


def getData(mode):
    if mode == 'train':
        img_names = None
        labels = None
        with open('train.json') as f:
            data = json.load(f)
            img_names = list(data.keys())
            labels = list(data.values())
        return img_names, labels
        
    elif mode == 'test' or mode == 'new_test':
        labels = None
        file_name = 'test.json' if mode == 'test' else 'new_test.json'
        with open(file_name) as f:
            labels = json.load(f)
        return [], labels    

    # elif mode == "valid":
    #     df = pd.read_csv('valid.csv')
    #     path = df['Path'].tolist()
    #     label = df['label'].tolist()
    #     return path, label

    # else:
    #     df = pd.read_csv('resnet_18_test.csv')
    #     path = df['Path'].tolist()
    #     return path, []


class iclevrDataset(data.Dataset):
    def __init__(self, root, mode, transform=None):
        self.root = root
        self.img_names, self.labels = getData(mode=mode)
        self.transform = transform
        self.mode = mode

        self.label_map = None
        with open('objects.json') as f:
            self.label_map = json.load(f)
        print("> Found %d images..." % (len(self.labels)))

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, index):
        if self.mode == 'train':
            path = self.root + self.img_names[index]
            label = self.labels[index]
            img = io.imread(path) # np.array rgba
            img = img[:, :, :-1]
            if self.transform:
                img = self.transform(img)
                # print(img.size()) # tensor(3, 450, 450)
                # img = img.float() / 255.0

            # one_hot_version----------------
            one_hot_label = torch.zeros(24)
            for item in label:
                one_hot_label[self.label_map[item]] = 1.0
            
            # one_hot_label = torch.zeros(3)
            # one_hot_label += 24
            # i = 0
            # for item in label:
            #     one_hot_label[i] = self.label_map[item]
            #     i += 1

            return img, one_hot_label.int()
        elif self.mode == 'test' or self.mode == 'new_test':
            # one_hot_version----------------
            label = self.labels[index]
            one_hot_label = torch.zeros(24)
            for item in label:
                one_hot_label[self.label_map[item]] = 1.0

            # one_hot_label = torch.zeros(3)
            # one_hot_label += 24
            # i = 0
            # for item in label:
            #     one_hot_label[i] = self.label_map[item]
            #     i += 1
            return one_hot_label.int()
        
        
        # if self.mode != 'test':
        #     label = self.label[index]
        #     return img, label
        # else:
        #     return img

