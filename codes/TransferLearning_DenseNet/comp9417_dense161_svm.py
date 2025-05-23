# -*- coding: utf-8 -*-
"""COMP9417_dense161_svm.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1y1cMgOq9Dm61gLzVtACF0uyzJmGlFbQR

# Rice leaf classification


## Introduction
This code uses a CNN model to extract features then dose classification by SVM. The result will be compared with other methods to choose the best one. <br>

# 1. Import libraries
"""

import torch
import torch.nn as nn
import torchvision
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor, Resize, Compose
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.nn.functional as F
import numpy as np
from torchvision.utils import make_grid
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import os
from torch.utils.data import Dataset
from torchvision import transforms
import pandas as pd
from PIL import Image
from glob import glob
import time
import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn.model_selection import train_test_split
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import Dataset
import torchvision
import torch.optim as optim
from torchvision import datasets, transforms
from sklearn.svm import SVC
from torch.optim import Adam
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score,classification_report

from google.colab import drive
drive.mount('/content/drive')

"""# 2. Load data ##"""

class CustomImageDataset(Dataset):
  def __init__(self, csv_file, transform=None):
    self.data_frame = pd.read_csv(csv_file)
    self.transform = transform
    self.classes = sorted(self.data_frame['class'].unique())
    self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}

  def __len__(self):
    return len(self.data_frame)

  def __getitem__(self, idx):
    if torch.is_tensor(idx):
      idx = idx.tolist()

    img_path = self.data_frame.loc[idx, "path"]
    image = Image.open(img_path).convert("RGB")

    label = self.data_frame.loc[idx, "class"]
    label_idx = self.class_to_idx[label]
    if self.transform:
      image = self.transform(image)

    return image, label_idx

from torchvision import transforms
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(), # convert a PIL Image or numpy.ndarray to a PyTorch tensor
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

"""### Data source description
This dataset is a collection of multiple data sets found online. The size of this dataset is 7 GB due to the high resolution of images. The number of images in the dataset is 3355, including 3 types of rice diseases and the healthy leaf.

### Data source link
Download the dataset through the link below, unzip and save the dataset in the current directory https://www.kaggle.com/datasets/shayanriyaz/riceleafs <br>
Revise the below path so images can be loaded.
"""

# just for test
brown_spot = glob("../content/drive/MyDrive/riceleaf/RiceLeafs/validation/BrownSpot/*")
healthy = glob("../content/drive/MyDrive/riceleaf/RiceLeafs/validation/Healthy/*")
hispa = glob("../content/drive/MyDrive/riceleaf/RiceLeafs/validation/Hispa/*")
leaf_blast = glob("../content/drive/MyDrive/riceleaf/RiceLeafs/validation/LeafBlast/*")

print("Number of brown spot:",len(brown_spot))
print("Number of healthy:",len(healthy))
print("Number of hispa: ",len(hispa))
print("Number of leaf blast:",len(leaf_blast))
print('Total number:',len(brown_spot)+len(healthy)+len(hispa)+len(leaf_blast))

label_map = {0:"brown_spot",
             1:"healthy",
             2:"hispa",
             3:"leaf_blast"
            }

"""### Data preprocessing
Resize<br>
Normalization<br>
Standarlization<br>
"""

# Reference https://www.kaggle.com/code/mehmetlaudatekman/rice-leaf-pytorch-transfer-learning
class LeafDataset(Dataset):

    def __init__(self,paths):

        self.x = []
        self.y = []                         # This converts pil image to torch tensor.
        self.transform = transforms.Compose([transforms.ToTensor(),
                                             # We have to normalize data to use in torchvision models.
                                             transforms.Normalize(mean=[0.485, 0.456, 0.406],                                     std=[0.229, 0.224, 0.225])
                                            ])

        start = time.time()
        for label,class_paths in enumerate(paths):
            for sample_path in class_paths:
                img = Image.open(sample_path).resize((224,224))
                self.x.append(self.transform(img))
                self.y.append(label)
        end = time.time()
        process_time = round(end-start,2)
        print("Dataset has loaded, that took {} seconds".format(process_time))


    def __getitem__(self,index):
        return self.x[index],self.y[index]

    def __len__(self):
        return len(self.x)

dataset = LeafDataset((brown_spot,healthy,hispa,leaf_blast))

# Splitting indices into train and test sets.
train,test = train_test_split(list(range(len(dataset))))
print(len(train))
print(len(test))

train_sampler = SubsetRandomSampler(train)
test_sampler = SubsetRandomSampler(test)
print(len(train_sampler))
print(len(test_sampler))

BATCH_SIZE = 1
train_loader = DataLoader(dataset,batch_size=BATCH_SIZE,sampler=train_sampler)
test_loader = DataLoader(dataset,batch_size=BATCH_SIZE,sampler=test_sampler)

"""# 3. Load a pre-trained model """

# Load a pre-trained model
d161 = torchvision.models.densenet161(pretrained=True)

# Remove the last layer of the model

feature_extractor = nn.Sequential(*list(d161.children())[:-1])
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model.to(device)

"""### Extract features by CNN:"""

# Load the dataset and extract features
# Reference: https://discuss.pytorch.org/t/how-to-extract-features-of-an-image-from-a-trained-model/119/105?page=6
criterion = nn.CrossEntropyLoss()
optimizer = Adam(d161.parameters(), lr=0.0001)

X_train = []
y_train = []
X_test = []
y_test = []

for input, label in train_loader:

    with torch.no_grad():
        feature_vector = feature_extractor(input)

    X_train.append(feature_vector.numpy().flatten())

    y_train.append(label)

X_train = np.asarray(X_train, dtype=np.float32)
y_train = np.asarray(y_train, dtype=np.float32)

# same operation for test set
for input, label in test_loader:

    with torch.no_grad():
        feature_vector = feature_extractor(input)

    X_test.append(feature_vector.numpy().flatten())

    y_test.append(label)

X_test = np.asarray(X_test, dtype=np.float32)
y_test = np.asarray(y_test, dtype=np.float32)

"""### Classify by SVM:"""

clf2 = SVC(kernel='linear', C=1, decision_function_shape='ovr')
clf2.fit(X_train, y_train)

# Evaluate the SVM classifier on the test set
y_pred = clf2.predict(X_test)

val_acc = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

print(f"Validation Accuracy: {val_acc:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 score: {f1:.4f}")

"""## Findings:
- Based on the above attempt, the CNN + SVM did not get as good effect as other CNN models.
"""
