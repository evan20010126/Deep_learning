import pandas as pd
from PIL import Image
from torch.utils import data
import imageio
import matplotlib.pyplot as plt
import numpy as np
import torch

def getData(mode):
    if mode == 'train':
        df = pd.read_csv('train.csv')
        path = df['Path'].tolist()
        label = df['label'].tolist()
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
    def __init__(self, root, mode):
        """
        Args:
            mode : Indicate procedure status(training or testing)

            self.img_name (string list): String list that store all image names.
            self.label (int or float list): Numerical list that store all ground truth label values.
        """
        self.root = root
        self.img_name, self.label = getData(mode) # both are list
        self.mode = mode
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
        path = self.root +  self.img_name[index]
        img = imageio.imread(path)
        
        label = self.label[index]

        img = img.astype(float) / 255 
        img = img.reshape((img.shape[2], img.shape[0], img.shape[1]))
        
        return torch.from_numpy(img), label

# training_loader = LeukemiaLoader("new_dataset","train")
# img, label = training_loader.__getitem__(0)
# print(img.shape)

# plt.imshow(img)
# plt.show()
