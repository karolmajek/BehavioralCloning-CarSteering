#!/usr/bin/python3
from keras.layers import Input, Flatten, Dense, Activation, Convolution2D, MaxPooling2D, Dropout
from keras.models import Model, Sequential
from keras.optimizers import Adam
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import cv2
import json
import os

np.random.seed(123456)

def process_image(image):
    image=image[65:135:4,::4,0]
    # diff=image.max()-image.min()
    image=(image-image.min())/255.0-0.5
    # image=np.mean(image,axis=2)
    image=image.reshape(image.shape[0],image.shape[1],1)
    return image

def main(_):
    # define model
    kernel_size=[3,3]
    depth=[64,32,16,8]
    pool_size=(2,2)

    img=np.zeros(shape=(160,320,3))
    img=process_image(img)
    input_shape = img.shape
    print("img.shape:  ",img.shape)
    print("input shape:",input_shape)
    print("kernel_size:",kernel_size)
    print("depth:",depth)
    print("pool_size:",pool_size)

    model = Sequential()
    model.add(Convolution2D(depth[0], kernel_size[0], kernel_size[1],border_mode='valid',input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(Convolution2D(depth[1],kernel_size[0], kernel_size[1]))
    model.add(Activation('relu'))
    model.add(Convolution2D(depth[2],kernel_size[0], kernel_size[1]))
    model.add(Activation('relu'))
    model.add(Convolution2D(depth[3],kernel_size[0], kernel_size[1]))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=pool_size))
    model.add(Dropout(0.25))

    model.add(Flatten())

    model.add(Dense(16))
    model.add(Activation('relu'))
    model.add(Dense(16))
    model.add(Activation('relu'))
    model.add(Dense(16))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1))

    model.summary()
    json_string = model.to_json()
    parsed = json.loads(json_string)

    with open('model.json', 'w') as outfile:
        json.dump(json_string, outfile)
        print("Saved")

if __name__ == '__main__':
    tf.app.run()
