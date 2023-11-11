# images(224*224*3) with removed background and enhanced contrast
DIR_M_TRAIN = "drive/MyDrive/COMP9417/RiceLeafs_M/train/"
DIR_M_VALID = "drive/MyDrive/COMP9417/RiceLeafs_M/validation/"

# image(56*56*3) which focus on the lesion area
DIR_S_TRAIN = "drive/MyDrive/COMP9417/RiceLeafs_S/train/"
DIR_S_VALID = "drive/MyDrive/COMP9417/RiceLeafs_S/validation/"

# gray image(56*56*1)
DIR_G_TRAIN = "drive/MyDrive/COMP9417/RiceLeafs_G/train/"
DIR_G_VALID = "drive/MyDrive/COMP9417/RiceLeafs_G/validation/"

import os
import random
from PIL import Image
from PIL import ImageFilter
from PIL import ImageEnhance
import cv2
from google.colab.patches import cv2_imshow
import numpy as np
from matplotlib import pyplot as plt
import csv

classes2 = os.listdir(DIR_M_TRAIN)
classes3 = os.listdir(DIR_M_VALID)
classes4 = os.listdir(DIR_S_TRAIN)
classes5 = os.listdir(DIR_S_VALID)

# find dark blobs location
def find_dark_point_position(img):
    temp = np.zeros(img.shape, np.uint8)
    for i in range(224):
      for j in range(224):
        if img[i][j][1]<142:
          temp[i][j] = img[i][j]
    temp =np.resize(temp,(224,224,3))

    res2 = temp
    # detect spots
    im = res2#cv2.cvtColor(res2, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(im, 100, 255,cv2.THRESH_BINARY)[1]

    params = cv2.SimpleBlobDetector_Params()
    params.filterByColor = True
    # params.blobColor = 0  # 0 only detect black blob
    params.blobColor = 255  # 255 only detect white blob
    params.filterByArea = True
    params.minArea = 5
    #params.maxArea = 10000
    params.filterByCircularity = True # Roundness control
    params.minCircularity = 0.01
    params.filterByInertia = True  # Smaller and more elliptical
    params.minInertiaRatio = 0.01
    params.filterByConvexity =True
    params.minConvexity = 0.01
    detector = cv2.SimpleBlobDetector_create(params)
    keypoints = detector.detect(thresh)

    # find real lesion region
    keypoint_p = cv2.KeyPoint_convert(keypoints)
    real_key = []
    for k in range(len(keypoints)):
      red_flag = 0
      size = int(keypoints[k].size/2)
      for i in range(int(keypoint_p[k][0]-size), int(keypoint_p[k][0]+size+1)):
        if red_flag == 1:
          break
        for j in range(int(keypoint_p[k][1]-size), int(keypoint_p[k][1]+size+1)):
          if 0<i<224 and 0<j<224:
            if thresh[j][i][1] < 35 and thresh[j][i][2] > 200:
              red_flag = 1
              break
      if red_flag == 1:
        real_key.append(keypoints[k])
    if len(keypoints)==1 and len(real_key)==0:
      real_key = keypoints

    final_keypoint_p = []
    max_size = 0
    if len(real_key)>=1:
      final_key = [real_key[0]]
      for i in range(len(real_key)):
        if real_key[i].size > max_size:
          final_key = [real_key[i]]
          max_size = real_key[i].size
      final_keypoint_p = cv2.KeyPoint_convert(final_key)
    return final_keypoint_p

# find light blobs
def find_light_point_position(img):
  temp = np.zeros(img.shape, np.uint8)
  for i in range(224):
    for j in range(224):
      if img[i][j][1]>170: #and img[i][j][2]>200:
        temp[i][j] = img[i][j]
  temp =np.resize(temp,(224,224,3))
  gray = cv2.cvtColor(temp, cv2.COLOR_BGR2GRAY)

  params = cv2.SimpleBlobDetector_Params()
  params.filterByColor = True
  #params.blobColor = 0  # 0 only detect black blob
  params.blobColor = 255  # 255 only detect white blob
  params.filterByArea = True
  params.minArea = 5
  #params.maxArea = 10000
  params.filterByCircularity = True # Roundness control
  params.minCircularity = 0.01
  params.filterByInertia = True  # Smaller and more elliptical
  params.minInertiaRatio = 0.01
  params.filterByConvexity =True
  params.minConvexity = 0.01
  detector = cv2.SimpleBlobDetector_create(params)
  keypoints = detector.detect(gray)

  final_keypoint_p = []
  max_size = 0
  if len(keypoints)>=1:
    final_key = [keypoints[0]]
    for i in range(len(keypoints)):
      if keypoints[i].size > max_size:
        final_key = [keypoints[i]]
        max_size = keypoints[i].size
    final_keypoint_p = cv2.KeyPoint_convert(final_key)
  return final_keypoint_p

# find all possible lesion region
# return their position
all_keypoint_p = []
for index1 in range(len(classes3)):
  leaf = os.listdir(DIR_M_VALID + classes3[index1])
  for index2 in range(len(leaf)):
    im = cv2.imread(DIR_M_VALID + classes3[index1] + '/' + leaf[index2])
    img = cv2.GaussianBlur(im, (5, 5), 0)
    keypoint_p = []
    keypoint_p = find_dark_point_position(img)
    if keypoint_p==[]:
      keypoint_p = find_light_point_position(img)
    if keypoint_p==[]:
      keypoint_p = np.array([[112, 112]])
    all_keypoint_p.append(keypoint_p)

# create dataset that image(56*56*3) focus on the lesion area
index_key = 0

for index1 in range(len(classes3)):
    leaf = os.listdir(DIR_M_VALID + classes3[index1])
    for index2 in range(len(leaf)):
        im = cv2.imread(DIR_M_VALID + classes3[index1] + '/' + leaf[index2])
        s_im = np.zeros((56, 56, 3), dtype=np.uint8)
        x = int(all_keypoint_p[index_key][0][0])
        y = int(all_keypoint_p[index_key][0][1])
        if x - 28 < 0:
            x = 28
        if x + 28 > 223:
            x = 195
        if y - 28 < 0:
            y = 28
        if y + 28 > 223:
            y = 195
        s_i = 28 - x
        s_j = 28 - y
        for i in range(x - 28, x + 28):
            for j in range(y - 28, y + 28):
                s_im[s_j + j][s_i + i] = im[j][i]

        index_key += 1
        cv2.imwrite(DIR_S_VALID + classes3[index1] + '/' + leaf[index2], s_im)