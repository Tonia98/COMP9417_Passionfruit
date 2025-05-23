# -*- coding: utf-8 -*-
"""COMP9417_dense201_processed.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1j_9Q9tFDIkEAa8GmJeSwUsy9fGXFFDdL

# Rice leaf classification


## Introduction
This code aims to test the chosen pretrained model with original images and preprocessed images. <br>

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
import matplotlib.pyplot as plt
import os
from torch.utils.data import Dataset
import pandas as pd
from PIL import Image

from google.colab import drive
drive.mount('/content/drive')

"""# 2. Load data ##
We have gathered all images in folder "train".<br>
Shuffled the images in each folder randomly.<br>
Split the images with ratio 8:1:1 as new training, validation and testing sets.<br>
Recorded the images' names in train_data.csv, val_data.csv and test_data.csv.
"""

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

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(), # convert a PIL Image or numpy.ndarray to a PyTorch tensor
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

import pandas as pd
DIR_CSV = "drive/MyDrive/"
train_set = CustomImageDataset(csv_file=DIR_CSV+"train_data1.csv", transform=transform)
val_set = CustomImageDataset(csv_file=DIR_CSV+"val_data1.csv", transform=transform)
test_set = CustomImageDataset(csv_file=DIR_CSV+"test_data1.csv", transform=transform)

from PIL import Image
train_set[301]

test_set[4]

DIR_MODEL = "drive/MyDrive/" # path to save model

len(train_set) # all data from csv
train_data = train_set
val_data = val_set
test_data = test_set

len(train_data)

train_dataloader = DataLoader(train_set, batch_size=16, shuffle=True, num_workers=4)
val_dataloader = DataLoader(val_set, batch_size=16, shuffle=False, num_workers=4)
test_dataloader = DataLoader(test_set, batch_size=16, shuffle=False, num_workers=4)

#@title


mean_train = np.zeros(3)
std_train = np.zeros(3)
mean_val = np.zeros(3)
std_val = np.zeros(3)

for images, _ in train_dataloader:
  mean_train += images.mean(axis=(0, 2, 3)).numpy()
  std_train += images.std(axis=(0, 2, 3)).numpy()

mean_train /= len(train_dataloader)
std_train /= len(train_dataloader)

for images, _ in val_dataloader:
  mean_val += images.mean(axis=(0, 2, 3)).numpy()
  std_val += images.std(axis=(0, 2, 3)).numpy()

mean_val /= len(val_dataloader)
std_val /= len(val_dataloader)

def denormalize(images, means, stds):
  means = torch.tensor(means).reshape(1, 3, 1, 1)
  stds = torch.tensor(stds).reshape(1, 3, 1, 1)
  return images * stds + means

def show_batch(dl):
  for images, labels in dl:
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.set_xticks([]); ax.set_yticks([])
    denorm_images = denormalize(images, *(mean_train, std_train))
    ax.imshow(make_grid(denorm_images[:64], nrow=8).permute(1, 2, 0).clamp(0,1))
    break

"""### Show some images
The below images have backgroud removed to help extract features more accurately in further process.
"""

show_batch(train_dataloader)

#@title
# def conv_block(in_channels, out_channels, pool=False):
#   layers = [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1), 
#           nn.BatchNorm2d(out_channels), 
#           nn.ReLU(inplace=True)]
#   if pool: layers.append(nn.MaxPool2d(2))
#   return nn.Sequential(*layers)

# class ResNet9(nn.Module):
#   def __init__(self, in_channels, num_classes):
#     super().__init__()
        
#     self.conv1 = conv_block(in_channels, 64)
#     self.conv2 = conv_block(64, 128, pool=True)
#     self.res1 = nn.Sequential(conv_block(128, 128), conv_block(128, 128))
        
#     self.conv3 = conv_block(128, 256, pool=True)
#     self.conv4 = conv_block(256, 512, pool=True)
#     self.res2 = nn.Sequential(conv_block(512, 512), conv_block(512, 512))
        
#     self.classifier = nn.Sequential(nn.AdaptiveMaxPool2d(1), 
#                     nn.Flatten(), 
#                     nn.Dropout(0.2),
#                     nn.Linear(512, num_classes))
        
#   def forward(self, xb):
#     out = self.conv1(xb)
#     out = self.conv2(out)
#     out = self.res1(out) + out
#     out = self.conv3(out)
#     out = self.conv4(out)
#     out = self.res2(out) + out
#     out = self.classifier(out)
#     return out

# model = ResNet9(3, 4)

d201 = torchvision.models.densenet201(pretrained=True)

# This will return how many features we'll have after flattening.
# num_features = d201.classifier.in_features
num_features = d201.classifier.in_features
# num_features

num_features

# We did not get the last layer (prediction layer) 
# we'll add our prediction layer.
layers = list(d201.classifier.children())[:-1]
# layers = list(r101.fc.children())[:-1]
# layers.append(nn.Linear(num_features,4))

# d201.classifier = nn.Sequential(*layers)

# model = d201

'''
new_layers = nn.Sequential(
                nn.Linear(num_features, 512),
                nn.ReLU(),
                nn.Dropout(0.5),  # Dropout layer with dropout probability of 0.2
                nn.Linear(512, 4)
                #nn.Softmax(dim=1) # Create a softmax module that applies softmax along the second dimension (i.e., dimension 1)
                
             )

new_layers = nn.Sequential(
                nn.Flatten(),  # Flatten the output of the last convolutional layer
                
                nn.Linear(num_features, 1024),
                # nn.Linear(1920 * 7 * 7, 1024),  # Add a new linear layer with input size of 1920 * 7 * 7 and output size of 1024
                nn.BatchNorm1d(1024),
                nn.ReLU(),
                nn.Dropout(0.5),  # Add a dropout layer to prevent overfitting

                nn.Linear(1024, 512),
                nn.BatchNorm1d(512),
                nn.ReLU(),
                nn.Dropout(0.5),  # Add a dropout layer to prevent overfitting

                nn.Linear(512, 4)  # Add the final linear layer with output size of num_classes
            )'''

new_layers = nn.Linear(num_features,4)            

layers.append(new_layers)

d201.classifier = nn.Sequential(*layers)

model = d201

#@title
print(model)

# If train the entire model, set the requires_grad attribute of all parameters to True
for param in d201.parameters():
    param.requires_grad = True

"""# 4. Train the model ##"""

from torch.optim import Adam
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score,classification_report

criterion = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=0.0001)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
# for drawing graph
acc_list, val_loss_list, precision_list, recall_list, f1_list = [],[],[],[],[]

train_acc_list, train_loss_list, train_precision_list, train_recall_list, train_f1_list = [],[],[],[],[]

# Training loop
num_epochs = 30
for epoch in range(num_epochs):
  model.train()
  train_loss = 0
  train_predictions, train_targets = [], []
  for inputs, labels in train_dataloader:
    inputs, labels = inputs.to(device), labels.to(device)
    optimizer.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()
        
    train_loss += loss.item()
    _, preds = torch.max(outputs, 1)
    train_predictions.extend(preds.cpu().numpy())
    train_targets.extend(labels.cpu().numpy())

  model.eval()
  with torch.no_grad():
    val_loss = 0
    predictions, targets = [], []
    for inputs, labels in val_dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        #print(loss.item())
        val_loss += loss.item()

        _, preds = torch.max(outputs, 1)
        predictions.extend(preds.cpu().numpy())
        targets.extend(labels.cpu().numpy())



        # Calculate metrics
    # print('t',targets[:15])
    # print('p',predictions[:15])

    
    # print(classification_report(targets, predictions))
    
    # print(len(targets))

    train_acc = accuracy_score(train_targets, train_predictions)
    train_loss /= len(train_dataloader)

    val_acc = accuracy_score(targets, predictions)
    precision = precision_score(targets, predictions, average='weighted')
    recall = recall_score(targets, predictions, average='weighted')
    f1 = f1_score(targets, predictions, average='weighted')
    val_loss /= len(val_dataloader)

    # record data of train set
    train_acc_list.append(train_acc)
    train_loss_list.append(train_loss)
    train_precision_list.append(precision_score(train_targets, train_predictions, average='weighted'))
    train_recall_list.append(recall_score(train_targets, train_predictions, average='weighted'))
    train_f1_list.append(f1_score(train_targets, train_predictions, average='weighted'))

    # record data of validation set
    acc_list.append(val_acc)
    val_loss_list.append(val_loss)
    precision_list.append(precision)
    recall_list.append(recall)
    f1_list.append(f1)

    print(f"Epoch {epoch+1}/{num_epochs}")
    print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.4f}")
    print(f"Validation Accuracy: {val_acc:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 score: {f1:.4f}")
    #print(classification_report(targets, predictions))
  if epoch % 10 == 0:
    m_name = 'model_res101_add_layers.pth'
    # torch.save(model.state_dict(), DIR_MODEL + test_name)
    torch.save(model, DIR_MODEL + m_name)

# load trained model for testing
import torchvision.models as models

model_file = DIR_MODEL + 'model_res101_add_layers.pth'
testmodel = torch.load(model_file)
testmodel.eval()

def test_model(test_dataloader, model):
    with torch.no_grad():
      predictions, targets = [], []
      for inputs, labels in test_dataloader:
          inputs, labels = inputs.to(device), labels.to(device)
          outputs = testmodel(inputs)
          _, preds = torch.max(outputs, 1)
          predictions.extend(preds.cpu().numpy())
          targets.extend(labels.cpu().numpy())

          # Calculate metrics
          # print('t',targets[:15])
          # print('p',predictions[:15])
          # print(classification_report(targets, predictions))
          # print(len(targets))
          #train_acc = accuracy_score(train_targets, train_predictions)
          #train_loss /= len(train_dataloader)

      acc = accuracy_score(targets, predictions)
      precision = precision_score(targets, predictions, average='weighted')
      recall = recall_score(targets, predictions, average='weighted')
      f1 = f1_score(targets, predictions, average='weighted')

      #print(f"Epoch {epoch+1}/{num_epochs}")
      #print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.4f}")
      print(f"Test Accuracy: {acc:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 score: {f1:.4f}")
      print(classification_report(targets, predictions))

test_model(test_dataloader, model)

from matplotlib.layout_engine import TightLayoutEngine
def plot(list1, list2, label1, label2, title, yl):
  epo = range(len(list1))
  plt.plot(epo, list1, 'r', label= label1)
  plt.plot(epo, list2, 'b', label= label2)
  plt.title(title)
  plt.legend(loc=0)
  plt.xlabel('epoches')
  plt.ylim(0,yl)
  plt.figure()
  plt.show()

plot(train_acc_list, acc_list, 'Accuracy of Training data', 'Accuracy of Validation data', 'Training vs validation accuracy',1.1 )

#@title
plot(train_loss_list, val_loss_list, 'Loss of Training data', 'Loss of Validation data', 'Training vs validation loss',1 )

plot(train_recall_list, recall_list, 'Recall of Training data', 'Recall of Validation data', 'Training vs validation recall',1.1 )

plot(train_f1_list, f1_list, 'F1 of Training data', 'F1 of Validation data', 'Training vs validation F1',1.1 )

plot(train_precision_list, precision_list, 'Precision of Training data', 'Precision of Validation data', 'Training vs validation Precision',1.1 )

# just for test

brown_spot = glob("../content/drive/MyDrive/train/BrownSpot/*")
healthy = glob("../content/drive/MyDrive/train/Health/*")
hispa = glob("../content/drive/MyDrive/train/Hispa/*")
leaf_blast = glob("../content/drive/MyDrive/train/LeafBlast/*")

print("Number of brown spot:",len(brown_spot))
print("Number of health:",len(healthy))
print("Number of hispa: ",len(hispa))
print("Number of leaf blast:",len(leaf_blast))

"""## Findings:
- We can observe the accuracy of using pre-processed images has increased 7.03% compared to the original images, which means the pre-processing of this dataset was effective. The accuracy 0.8601 has beaten other codes at Kaggle on this dataset.
"""

