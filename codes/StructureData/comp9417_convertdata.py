# -*- coding: utf-8 -*-
"""COMP9417_convertData.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1Zl0z2SoqyqVBkG0UEMC_JdFnMxK2JUJs

# Split & Convert Dataset

In this notebook, we split the whole dataset into training set, validation set, and testing set in the ratio of 8 : 1 : 1. Then we converted the unstructured image dataset into structured csv files.
"""

import os
import pandas as pd
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split

DIR_DATA = "drive/MyDrive/COMP9417/data/PreprocessedData/train"
DIR_CSV = "drive/MyDrive/COMP9417/data/"

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

dataset = ImageFolder(root=DIR_DATA, transform=transform)

"""Split the whole dataset"""

train_size = int(0.8 * len(dataset))
val_size = int(0.1 * len(dataset))
test_size = len(dataset) - train_size - val_size

train_data, val_data, test_data = random_split(dataset, [train_size, val_size, test_size])

"""Convert the dataset into sturctured format and save to local machine"""

def convert_data(data):
  indexes = []
  classes = []
  paths = []
  for idx, (image, label) in enumerate(data):
    indexes.append(idx)
    classes.append(dataset.classes[label])
    paths.append(dataset.imgs[data.indices[idx]][0])
  result = pd.DataFrame({'index': indexes, 'class': classes, 'path': paths})
  return result

df_train = convert_data(train_data)
df_val = convert_data(val_data)
df_test = convert_data(test_data)

df_train.to_csv(DIR_CSV+'train_data.csv', index=False)
df_val.to_csv(DIR_CSV+'val_data.csv', index=False)
df_test.to_csv(DIR_CSV+'test_data.csv', index=False)