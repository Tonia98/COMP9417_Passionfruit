import torch
from torch.utils.data import DataLoader
import numpy as np
import operator
import torch
import torchvision.datasets as da
import torchvision.transforms as transforms
import torchvision
import os
import cv2
from PIL import Image
from PIL import ImageFilter
from PIL import ImageEnhance
from sklearn.neighbors import KNeighborsClassifier

DIR_G_TRAIN = "drive/MyDrive/COMP9417/RiceLeafs_G/train/"
DIR_G_VALID = "drive/MyDrive/COMP9417/RiceLeafs_G/validation/"
classes1 = os.listdir(DIR_G_TRAIN)
classes2 = os.listdir(DIR_G_VALID)

# get train and validation dataset
classes = {'BrownSpot':0,'LeafBlast':1,'Healthy':2,'Hispa':3}
x_train = []
y_train = []
x_val = []
y_val = []
for index1 in range(len(classes1)):
  leaf = os.listdir(DIR_G_TRAIN + classes1[index1])
  for index2 in range(len(leaf)):
    im = cv2.imread(DIR_G_TRAIN + classes1[index1] + '/' + leaf[index2])
    gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    gray = gray.reshape(1,-1)
    gray = gray[0]
    x_train.append(gray)
    y_train.append(classes[classes1[index1]])

for index1 in range(len(classes2)):
  leaf = os.listdir(DIR_G_VALID + classes2[index1])
  for index2 in range(len(leaf)):
    im = cv2.imread(DIR_G_VALID + classes2[index1] + '/' + leaf[index2])
    gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    gray = gray.reshape(1,-1)
    gray = gray[0]
    x_val.append(gray)
    y_val.append(classes[classes2[index1]])

# centralized data to [-1,1], and mean = 0, standard variance = 1
def centralized(data):
  central_data = []
  for item in range(len(data)):
    image = data[item].astype(float)
    tensor_image = image/255
    mean_image = np.mean(tensor_image)
    std_image = tensor_image.std()
    central_image = (tensor_image-mean_image)/std_image
    central_data.append(central_image)
  return central_data

# convert array to list
def array_to_list(data):
  list_data = []
  for item in range(len(data)):
    image = list(data[item])
    temp = []
    for i in range(len(image)):
      num = np.float(image[i])
      if np.isnan(num) == True:
        temp.append(np.float(0))
      else:
        temp.append(num)
    list_data.append(temp)
  return list_data

x_train = centralized(x_train)
x_train = array_to_list(x_train)
x_val = centralized(x_val)
x_val = array_to_list(x_val)

# Re-partition the dataset into train and validation
num_shuffle = list(range(3355))
np.random.shuffle(num_shuffle)
xx = []
xx.extend(x_train)
xx.extend(x_val)
yy = []
yy.extend(y_train)
yy.extend(y_val)
X_train = []
Y_train = []
X_val = []
Y_val = []
print(num_shuffle)
for i in range(3355):
  if i < 2684:
    X_train.append(xx[num_shuffle[i]])
    Y_train.append(yy[num_shuffle[i]])
  else:
    X_val.append(xx[num_shuffle[i]])
    Y_val.append(yy[num_shuffle[i]])

# sklearn Knn (Euclidean distance)
from sklearn.neighbors import KNeighborsClassifier
for k in range(2,26):
  neigh = KNeighborsClassifier(n_neighbors=k)
  neigh.fit(X_train, Y_train)
  y_pred = []
  for i in range(len(X_val)):
    y_pred.append(neigh.predict([X_val[i]]))
  num_correct = 0
  for i in range(len(y_val)):
    if y_pred[i] == Y_val[i]:
      num_correct += 1
  print(num_correct/len(Y_val))

# get the best performing of train validation division after testing
finalx_train = X_train
finalx_val = X_val
finaly_train =Y_train
finaly_val = Y_val

# sklearn K (Euclidean distance)
neigh = KNeighborsClassifier(n_neighbors=25)
neigh.fit(finalx_train, finaly_train)
y_pred = []
for i in range(len(finalx_val)):
  y_pred.append(neigh.predict([finalx_val[i]]))

num_correct = 0
Tp0,Tp1,Tp2,Tp3 = 0,0,0,0
Fp0,Fp1,Fp2,Fp3 = 0,0,0,0
Fn0,Fn1,Fn2,Fn3 = 0,0,0,0
for i in range(len(y_val)):
  if y_pred[i] == Y_val[i]:
    num_correct += 1
    if y_pred[i] == 0:
      Tp0 += 1
    if y_pred[i] == 1:
      Tp1 += 1
    if y_pred[i] == 2:
      Tp2 += 1
    if y_pred[i] == 3:
      Tp3 += 1
  else:
    if y_pred[i] == 0:
      Fp0 += 1
    if y_pred[i] == 1:
      Fp1 += 1
    if y_pred[i] == 2:
      Fp2 += 1
    if y_pred[i] == 3:
      Fp3 += 1
    if Y_val[i] == 0:
      Fn0 += 1
    if Y_val[i] == 1:
      Fn1 += 1
    if Y_val[i] == 2:
      Fn2 += 1
    if Y_val[i] == 3:
      Fn3 += 1
print(f"Accuracy is {num_correct/len(Y_val)}")
precision = (Tp0/(Tp0+Fp0)+Tp1/(Tp1+Fp1)+Tp2/(Tp2+Fp2)+Tp3/(Tp3+Fp3))/4
recall = (Tp0/(Tp0+Fn0)+Tp1/(Tp1+Fn1)+Tp2/(Tp2+Fn2)+Tp3/(Tp3+Fn3))/4
print(f"Precision is {precision}")
print(f"Recall is {recall}")
print(f"F1 is {2*precision*recall/(precision+recall)}")
print(Tp0/(Tp0+Fp0),Tp1/(Tp1+Fp1),Tp2/(Tp2+Fp2),Tp3/(Tp3+Fp3))
print(Tp0/(Tp0+Fn0),Tp1/(Tp1+Fn1),Tp2/(Tp2+Fn2),Tp3/(Tp3+Fn3))

# sklearn Knn (Manhattan distance)
neigh = KNeighborsClassifier(n_neighbors=25, metric="manhattan")
neigh.fit(finalx_train, finaly_train)
y_pred = []
for i in range(len(finalx_val)):
  y_pred.append(neigh.predict([finalx_val[i]]))

num_correct = 0
Tp0,Tp1,Tp2,Tp3 = 0,0,0,0
Fp0,Fp1,Fp2,Fp3 = 0,0,0,0
Fn0,Fn1,Fn2,Fn3 = 0,0,0,0
for i in range(len(y_val)):
  if y_pred[i] == Y_val[i]:
    num_correct += 1
    if y_pred[i] == 0:
      Tp0 += 1
    if y_pred[i] == 1:
      Tp1 += 1
    if y_pred[i] == 2:
      Tp2 += 1
    if y_pred[i] == 3:
      Tp3 += 1
  else:
    if y_pred[i] == 0:
      Fp0 += 1
    if y_pred[i] == 1:
      Fp1 += 1
    if y_pred[i] == 2:
      Fp2 += 1
    if y_pred[i] == 3:
      Fp3 += 1
    if Y_val[i] == 0:
      Fn0 += 1
    if Y_val[i] == 1:
      Fn1 += 1
    if Y_val[i] == 2:
      Fn2 += 1
    if Y_val[i] == 3:
      Fn3 += 1
print(f"Accuracy is {num_correct/len(Y_val)}")
precision = (Tp0/(Tp0+Fp0)+Tp1/(Tp1+Fp1)+Tp2/(Tp2+Fp2)+Tp3/(Tp3+Fp3))/4
recall = (Tp0/(Tp0+Fn0)+Tp1/(Tp1+Fn1)+Tp2/(Tp2+Fn2)+Tp3/(Tp3+Fn3))/4
print(f"Precision is {precision}")
print(f"Recall is {recall}")
print(f"F1 is {2*precision*recall/(precision+recall)}")
print(Tp0/(Tp0+Fp0),Tp1/(Tp1+Fp1),Tp2/(Tp2+Fp2),Tp3/(Tp3+Fp3))
print(Tp0/(Tp0+Fn0),Tp1/(Tp1+Fn1),Tp2/(Tp2+Fn2),Tp3/(Tp3+Fn3))