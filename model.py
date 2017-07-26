#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 20 16:54:58 2017
History:
    Use generators and change the location where the images are read.
@author: jaerock
"""

import csv
import cv2
import numpy as np

## read the csv file
lines = []
with open('./data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

## read image names and measurments from the lines
image_paths = []
measurements = []
for line in lines:
    for i in range(3): # 0: center, 1: left, 2: right image
        #image = cv2.imread(line[i]) 
        image_paths.append(line[i]) #image)
 
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
    
print('Total images: ', len(image_paths))
    
from keras.models import Sequential
from keras.layers import Flatten, Dense, Dropout
from keras.layers import Lambda
#from keras.layers.pooling import MaxPooling2D
from keras.layers import Convolution2D
from keras.layers import Cropping2D

# add generator
import sklearn
from sklearn.model_selection import train_test_split

samples = list(zip(image_paths, measurements))
train, valid = train_test_split(samples, test_size=0.2)

print('Train samples: ', len(train))
print('Valid samples: ', len(valid))

def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        samples = sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for image_path, measurement in batch_samples:
                image = cv2.imread(image_path)
                images.append(image)
                angles.append(float(measurement))
                
                # add the flipped image of the original
                images.append(cv2.flip(image,1))
                angles.append(measurement*-1.0)

            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)

train_generator = generator(train, batch_size=32)
valid_generator = generator(valid, batch_size=32)

image_size = (160, 320, 3)
crop_top_bottom = (50, 20)
crop_left_right = (0,0)
subsample_size = (2,2)
dropout_prob = 0.2

print('Input shape: ', image_size)

model = Sequential()
model.add(Lambda(lambda x: (x/ 255.0) - 0.5, input_shape=image_size))
model.add(Cropping2D(cropping=(crop_top_bottom, crop_left_right)))

model.add(Convolution2D(24,5,5,subsample=subsample_size,activation="relu"))
model.add(Convolution2D(36,5,5,subsample=subsample_size,activation="relu"))
model.add(Convolution2D(48,5,5,subsample=subsample_size,activation="relu"))
model.add(Convolution2D(64,3,3,activation="relu"))
model.add(Convolution2D(64,3,3,activation="relu"))
model.add(Flatten())
model.add(Dropout(dropout_prob))
model.add(Dense(1164))
model.add(Dropout(dropout_prob))
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')

# model filename
model_filename = 'model_with_gen.h5'

from keras.callbacks import ModelCheckpoint, EarlyStopping

# checkpoint
callbacks = []
checkpoint = ModelCheckpoint(model_filename, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
callbacks.append(checkpoint)

# early stopping
earlystop = EarlyStopping(monitor='val_loss', min_delta=0, patience=0, verbose=1, mode='min')
callbacks.append(earlystop)
        
history_object = model.fit_generator(train_generator, 
                                     samples_per_epoch=len(train),
                                     validation_data = valid_generator, 
                                     nb_val_samples=len(valid), 
                                     nb_epoch=5, verbose=1,
                                     callbacks=callbacks)
model.save(model_filename)

### print the keys contained in the history object
print(history_object.history.keys())

import matplotlib.pyplot as plt

### plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()
