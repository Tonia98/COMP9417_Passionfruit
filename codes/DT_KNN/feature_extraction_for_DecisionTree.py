DIR_TRAIN = "drive/MyDrive/COMP9417/RiceLeafs/train/"
DIR_VALID = "drive/MyDrive/COMP9417/RiceLeafs/validation/"

DIR_M_TRAIN = "drive/MyDrive/COMP9417/RiceLeafs_M/train/"
DIR_M_VALID = "drive/MyDrive/COMP9417/RiceLeafs_M/validation/"

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

# original graphs
classes = os.listdir(DIR_TRAIN)
classes1 = os.listdir(DIR_VALID)
# modified graphs (removed background and enhanced contrast)
classes2 = os.listdir(DIR_M_TRAIN)
classes3 = os.listdir(DIR_M_VALID)

# deal with the original graphs (removed background and enhanced contrast)
# and save in DIR_M_TRAIN and DIR_M_VALID
for index1 in range(len(classes)):
  leaf = os.listdir(DIR_TRAIN + classes[index1])
  for index2 in range(len(leaf)):
    im = cv2.imread(DIR_TRAIN + classes[index1] + '/' + leaf[index2])
    # crop image
    res1 = cv2.resize(im,(224,224))

    # remove mottled white(and grey shadow) background
    hsv = cv2.cvtColor(dst1, code=cv2.COLOR_BGR2HSV)
    lower_white = (0,0,46)
    upper_white = (180,60,255)
    lower_blue = (100,43,46)  # 白底背景太多蓝色干扰了
    upper_blue = (124,255,255)
    lower_lightblue = (78,43,46)
    upper_lightblue = (99,255,255)
    mask1 = cv2.inRange(hsv, lower_white, upper_white)
    mask2 = cv2.inRange(hsv, lower_blue, upper_blue)
    mask3 = cv2.inRange(hsv, lower_lightblue, upper_lightblue)
    mask = cv2.add(mask1,mask2)
    mask = cv2.add(mask3,mask)

    # remove impurity shadow in the background (small color block)
    contours,_ = cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    small_contours = []
    for contour in contours:
      area = cv2.contourArea(contour)
      if area <= 400:  # smaller than 400 pixels color block
        small_contours.append(contour)
      else:
        continue
    cv2.fillPoly(mask, small_contours, (255,255,255))

    # reverse mask
    for i in range(len(mask)):
      for j in range(len(mask[0])):
        mask[i][j] = 255 - mask[i][j]

    # remove impurity shadow in leaf (small color block)
    contours,_ = cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    small_contours = []
    for contour in contours:
      area = cv2.contourArea(contour)
      if area <= 200:  # smaller than 200 pixels color block
        small_contours.append(contour)
      else:
        continue
    cv2.fillPoly(mask, small_contours, (255,255,255))

    # make the background of original image to black
    res2 = cv2.bitwise_and(res1,res1,mask=mask)

    # improve image contrast
    all_pixel = 0
    count_pixel = 0  # don't calculate black background influence
    for i in range(len(res2)):
      for j in range(len(res2[0])):
        for k in range(len(res2[0][0])):
          if res2[i][j][k] != 0:
            all_pixel += res2[i][j][k]
            count_pixel += 1
    bri_mean2 = all_pixel / count_pixel # image mean

    # Contrast coefficient, increase or decrease in units of 0.5
    contrast_value = 2
    img_b = contrast_value * (res2-bri_mean2) + bri_mean2
    img_b = np.clip(img_b,0,255).astype(np.uint8)
    cv2.imwrite(DIR_M_TRAIN + classes[index1] + '/' + leaf[index2], img_b)

Dark_points0 = []
Dark_points_size = []

# Detect dark spots
for index1 in range(len(classes3)):
    leaf = os.listdir(DIR_M_VALID + classes3[index1])
    for index2 in range(len(leaf)):
        im = cv2.imread(DIR_M_VALID + classes3[index1] + '/' + leaf[index2])

        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        img = cv2.GaussianBlur(im, (5, 5), 0)

        # Simple division of dark lesion locations by color threshold
        temp = np.zeros(img.shape, np.uint8)
        for i in range(224):
            for j in range(224):
                if img[i][j][1] < 142:
                    temp[i][j] = img[i][j]
        temp = np.resize(temp, (224, 224, 3))

        # Considering the mean value of the leaf color
        # and the difference of the lesion area, divide the lesion area
        r_param = []
        g_param = []
        b_param = []
        pixel_n = 0
        E_ij = np.zeros(gray.shape)
        for i in range(224):
            for j in range(224):
                if np.sum(img[i][j]) != 0:
                    pixel_n += 1
                    b_param.append(int(img[i][j][0]))
                    g_param.append(int(img[i][j][1]))
                    r_param.append(int(img[i][j][2]))
                    E_ij[i][j] = (int(img[i][j][0]) ** 2 + (int(img[i][j][1]) * 4) ** 2 + int(img[i][j][2]) ** 2) / (
                                int(img[i][j][0]) + int(img[i][j][1]) + int(img[i][j][2]))
        r_sq = 0
        g_sq = 0
        b_sq = 0
        for i in range(len(r_param)):
            r_sq += r_param[i] ** 2
            g_sq += (g_param[i] * 4) ** 2  # g is more important
            b_sq += b_param[i] ** 2
        E_f = (r_sq + g_sq + b_sq) / (np.sum(r_param) + np.sum(g_param) + np.sum(b_param))
        Fermi_img = np.zeros(gray.shape, np.uint8)
        for i in range(224):
            for j in range(224):
                if np.sum(img[i][j]) == 0:
                    continue
                elif E_ij[i][j] < E_f * 0.9:
                    Fermi_img[i][j] += 255
        Fermi_img = np.resize(Fermi_img, (224, 224))

        res2 = cv2.bitwise_and(temp, temp, mask=Fermi_img)

        # detect spots
        im = res2  # cv2.cvtColor(res2, cv2.COLOR_BGR2GRAY)
        thresh = cv2.threshold(im, 100, 255, cv2.THRESH_BINARY)[1]

        params = cv2.SimpleBlobDetector_Params()
        params.filterByColor = True
        # params.blobColor = 0  # 0 only detect black blob
        params.blobColor = 255  # 255 only detect white blob
        params.filterByArea = True
        params.minArea = 5
        # params.maxArea = 10000
        params.filterByCircularity = True  # Roundness control
        params.minCircularity = 0.01
        params.filterByInertia = True  # Smaller and more elliptical
        params.minInertiaRatio = 0.01
        params.filterByConvexity = True
        params.minConvexity = 0.01
        detector = cv2.SimpleBlobDetector_create(params)
        keypoints = detector.detect(thresh)
        with_keypoints = cv2.drawKeypoints(im, keypoints, np.array([]),
                                           (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

        # Find real lesions and remove false spots
        keypoint_p = cv2.KeyPoint_convert(keypoints)
        real_key = []
        for k in range(len(keypoints)):
            red_flag = 0
            size = int(keypoints[k].size / 2)
            for i in range(int(keypoint_p[k][0] - size), int(keypoint_p[k][0] + size + 1)):
                if red_flag == 1:
                    break
                for j in range(int(keypoint_p[k][1] - size), int(keypoint_p[k][1] + size + 1)):
                    if 0 < i < 224 and 0 < j < 224:
                        if thresh[j][i][1] < 35 and thresh[j][i][2] > 200:
                            red_flag = 1
                            break
            if red_flag == 1:
                real_key.append(keypoints[k])
        if len(keypoints) == 1 and len(real_key) == 0:
            real_key = keypoints
        keypoint_p = cv2.KeyPoint_convert(real_key)

        # find dark blob and dark-outside-white-inside blob
        dark_point = []
        dwhite_point = []
        gray = cv2.cvtColor(thresh, cv2.COLOR_BGR2GRAY)
        for i in range(len(keypoint_p)):
            red_flag = 0
            for j in range(10):
                if 10 < gray[int(keypoint_p[i][0])][int(keypoint_p[i][1] - 10 + j)] < 100 or \
                        10 < gray[int(keypoint_p[i][0] - 10 + j)][int(keypoint_p[i][1])] < 100:
                    red_flag = 1
            if red_flag == 0:
                dark_point.append(int(real_key[i].size))
            else:
                dwhite_point.append(int(real_key[i].size))

        if len(dwhite_point) != 0:
            Dark_points0.append('dark-white')
            Dark_points_size.append(max(dwhite_point))
        elif len(dark_point) != 0:
            Dark_points0.append('dark')
            Dark_points_size.append(max(dark_point))
        else:
            Dark_points0.append('none')
            Dark_points_size.append(0)

Light_points = []

# find light blob
for index1 in range(len(classes3)):
  leaf = os.listdir(DIR_M_VALID + classes3[index1])
  for index2 in range(len(leaf)):
    im = cv2.imread(DIR_M_VALID + classes3[index1] + '/' + leaf[index2])
    img = cv2.GaussianBlur(im, (5, 5), 0)
    temp = np.zeros(img.shape, np.uint8)
    for i in range(224):
      for j in range(224):
        if img[i][j][1]>170: #and img[i][j][2]>200:
          temp[i][j] = img[i][j]
    temp =np.resize(temp,(224,224,3))
    gray = cv2.cvtColor(temp, cv2.COLOR_BGR2GRAY)

    # remove too small white blobs
    contours,_ = cv2.findContours(gray,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    small_contours = []
    for contour in contours:
      area = cv2.contourArea(contour)
      if area <= 50:  # smaller than 50 pixels color block
        small_contours.append(contour)
      else:
        continue
    cv2.fillPoly(gray, small_contours, (0,0))

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

    if len(keypoints) > 0:
      Light_points.append('white-round')
    elif len(contours) > 0:
      Light_points.append('white-line')
    else:
      Light_points.append('none')

Mean_r = []
Mean_g = []
Mean_b = []

# mean color of leaf (rgb)
for index1 in range(len(classes3)):
  leaf = os.listdir(DIR_M_VALID + classes3[index1])
  for index2 in range(len(leaf)):
    im = cv2.imread(DIR_M_VALID + classes3[index1] + '/' + leaf[index2])
    mean_r = []
    mean_g = []
    mean_b = []
    for i in range(len(im)):
      for j in range(len(im[0])):
        mean_r.append(im[i][j][2])
        mean_g.append(im[i][j][1])
        mean_b.append(im[i][j][0])
    Mean_r.append(np.mean(mean_r))
    Mean_g.append(np.mean(mean_g))
    Mean_b.append(np.mean(mean_b))

# write all features in csv file
Light_points_d = {'white-round':0,'white-line':1,'none':2}
dark_point_d = {'dark-white':0,'dark':1,'none':2}
message = []
for index1 in range(len(classes2)):
  leaf = os.listdir(DIR_M_TRAIN + classes2[index1])
  for i in range(len(leaf)):
    message.append([classes2[index1]])
for i in range(len(message)):
  message[i].append(Light_points_d[Light_points[i][0]])
  message[i].append(Mean_r[i])
  message[i].append(Mean_g[i])
  message[i].append(Mean_b[i])
  message[i].append(dark_point_d[dark_points0[i][0]])
  message[i].append(Dark_points_size[i])

header = ['class','light point','mean_red','mean_green','mean_blue',
          'dark_point','point_size']

with open('leaf.csv', 'w', encoding='utf-8') as file_obj:
    writer = csv.writer(file_obj)
    writer.writerow(header)
    for p in message:
        writer.writerow(p)
