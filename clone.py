#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 20 16:54:58 2017

@author: jaerock
"""

import csv
import cv2
import numpy as np

lines = []
with open('./data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)
        
images = []
measurements = []
for line in lines:
    for i in range(3): # 0: center, 1: left, 2: right image
        image = cv2.imread(line[i]) 
        images.append(image)
 
    # steering angle based on the center image
    measurement = float(line[3]) 
    measurements.append(measurement)
    # add left and right steering angle to maintain the center of a road
    correction = 0.2 # param to tuen
    steering_left = measurement + correction
    steering_right = measurement - correction
    # the order of appending is important. 
    # must be center, left, and right
    measurements.append(steering_left)
    measurements.append(steering_right)
    
# augment data
augmented_images, augmented_measurements = [], []
for image, measurement in zip(images, measurements):
    augmented_images.append(image)
    augmented_measurements.append(measurement)
    augmented_images.append(cv2.flip(image,1))
    augmented_measurements.append(measurement*-1.0)
    
X_train = np.array(augmented_images)
y_train = np.array(augmented_measurements)

from keras.models import Sequential
from keras.layers import Flatten, Dense
from keras.layers import Lambda
from keras.layers import Convolution2D
from keras.layers import Cropping2D

image_size = (160, 320, 3)
crop_top_bottom = (50, 20)
crop_left_right = (0,0)
subsample_size = (2,2)

model = Sequential()
model.add(Lambda(lambda x: (x/ 255.0) - 0.5, input_shape=image_size))
model.add(Cropping2D(cropping=(crop_top_bottom, crop_left_right)))
model.add(Convolution2D(24,5,5,subsample=subsample_size,activation="relu"))
model.add(Convolution2D(36,5,5,subsample=subsample_size,activation="relu"))
model.add(Convolution2D(48,5,5,subsample=subsample_size,activation="relu"))
model.add(Convolution2D(64,3,3,activation="relu"))
model.add(Convolution2D(64,3,3,activation="relu"))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=3)

model.save('model_wo_gen.h5')