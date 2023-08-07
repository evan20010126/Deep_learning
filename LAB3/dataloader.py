import pandas as pd
from PIL import Image
from torch.utils import data
import imageio
import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision import transforms


def getData(mode):
    if mode == 'train':
        df = pd.read_csv('train.csv')
        path = df['Path'].tolist()
        label = df['label'].tolist()
        # count = len(label)/2
        # try:
        #     for i in range(len(label)):
        #         if label[i] == 1 and count > 0:
        #             del path[i]
        #             del label[i]
        #             count -= 1
        # except:
        #     pass
        return path, label

    elif mode == "valid":
        df = pd.read_csv('valid.csv')
        path = df['Path'].tolist()
        label = df['label'].tolist()
        return path, label

    else:
        df = pd.read_csv('resnet_18_test.csv')
        path = df['Path'].tolist()
        return path, []


class LeukemiaLoader(data.Dataset):
    def __init__(self, root, mode, transform=None):
        """
        Args:
            mode : Indicate procedure status(training or testing)

            self.img_name (string list): String list that store all image names.
            self.label (int or float list): Numerical list that store all ground truth label values.
        """
        self.root = root
        self.img_name, self.label = getData(mode)  # both are list
        self.mode = mode
        self.transform = transform
        print("> Found %d images..." % (len(self.img_name)))

    def __len__(self):
        """'return the size of dataset"""
        return len(self.img_name)

    def __getitem__(self, index):
        """something you should implement here"""

        """
           step1. Get the image path from 'self.img_name' and load it.
                  hint : path = root + self.img_name[index] + '.jpeg'
           
           step2. Get the ground truth label from self.label
                     
           step3. Transform the .jpeg rgb images during the training phase, such as resizing, random flipping, 
                  rotation, cropping, normalization etc. But at the beginning, I suggest you follow the hints. 
                       
                  In the testing phase, if you have a normalization process during the training phase, you only need 
                  to normalize the data. 
                  
                  hints : Convert the pixel value to [0, 1]
                          Transpose the image shape from [H, W, C] to [C, H, W]
                         
            step4. Return processed image and label
        """
        path = self.root + self.img_name[index]
        img = imageio.imread(path)
        # print(img.shape) # ndarray(450, 450, 3)

        if self.transform:
            img = self.transform(img)
            # print(img.size()) # tensor(3, 450, 450)
            img = img.float() / 255.0

        if self.mode != 'test':
            label = self.label[index]
            return img, label
        else:
            return img

# training_loader = LeukemiaLoader("new_dataset","train")
# img, label = training_loader.__getitem__(0)
# print(img.shape)

# plt.imshow(img)
# plt.show()
