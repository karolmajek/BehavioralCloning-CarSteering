#!/usr/bin/python3
from sklearn.model_selection import train_test_split
from keras.optimizers import Adam
from keras.layers import Input, Flatten, Dense, Activation, Convolution2D, MaxPooling2D, Dropout
from keras.models import Model, Sequential, model_from_json
from tqdm import tqdm
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import cv2
import csv
import json
import os
import h5py

flags = tf.app.flags
FLAGS = flags.FLAGS

np.random.seed(123456)

# command line flags
flags.DEFINE_string('training_csv', 'recordings/track0_trial4/driving_log.csv', "")
flags.DEFINE_string('validation_csv', None, "")
flags.DEFINE_integer('epochs', 2, "The number of epochs.")
flags.DEFINE_integer('batch_size', 64, "The batch size.")
flags.DEFINE_integer('show', False, "Input data visualization")

image_labels=['Center image','Left image','Right image']
data_labels=['Steering Angle','Throttle','Break','Speed']

def load_data(training_csv, validation_csv=None):
    print('-'*60)
    X_train=[]
    y_train=[]
    with open(training_csv, 'r') as csvfile:
        data = list(csv.reader(csvfile, delimiter=','))
        pbar = tqdm(range(len(data)), desc='Loading Train', unit='images')
        for i in pbar:
            l=data[i]
            image0=cv2.imread(l[0].lstrip())
            image1=cv2.imread(l[1].lstrip())
            image2=cv2.imread(l[2].lstrip())
            image0=process_image(image0)
            image1=process_image(image1)
            image2=process_image(image2)
            if FLAGS.show:
                image=np.concatenate((image1,image0,image2),axis=1)
                cv2.imshow('cameras',image)
                cv2.waitKey(1)
            # Lets work with center image only

            X_train.append(image0)
            X_train.append(image1)
            X_train.append(image2)
            output=float(l[3])
            y_train.append(output)
            y_train.append(output)
            y_train.append(output)

    X_train=np.array(X_train)
    y_train=np.array(y_train)

    if validation_csv == None:
        print("Number of samples before split into train/val:",X_train.shape[0])

        X_train, X_val, y_train, y_val = train_test_split(X_train,
            y_train,
            test_size=0.2,
            random_state=122333)
    else:
        X_val=[]
        y_val=[]
        with open(validation_csv, 'r') as csvfile:
            data = list(csv.reader(csvfile, delimiter=','))
            pbar = tqdm(range(len(data)), desc='Loading Val', unit='images')
            for i in pbar:
                l=data[i]
                image0=cv2.imread(l[0].lstrip())
                image1=cv2.imread(l[1].lstrip())
                image2=cv2.imread(l[2].lstrip())
                image0=process_image(image0)
                image1=process_image(image1)
                image2=process_image(image2)
                if FLAGS.show:
                    image=np.concatenate((image1,image0,image2),axis=1)
                    cv2.imshow('cameras',image)
                    cv2.waitKey(1)
                # Lets work with center image only
                X_val.append(image0)
                X_val.append(image1)
                X_val.append(image2)
                output=float(l[3])
                y_val.append(output)
                y_val.append(output)
                y_val.append(output)
    X_val=np.array(X_val)
    y_val=np.array(y_val)
    X_train = X_train.astype('float32')
    X_val = X_val.astype('float32')
    print(X_val.shape)
    print(y_val.shape)
    print(X_train.shape)
    print(y_train.shape)
    return X_train, y_train, X_val, y_val

def getModel():
    with open('model.json', 'r') as f:
        model = model_from_json(json.load(f))
    model.compile("adam", "mse", metrics=['accuracy'])
    print("Imported model")
    return model

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
        print("Saved model.json")


    # load bottleneck data
    X_train, y_train, X_val, y_val = load_data(FLAGS.training_csv, FLAGS.validation_csv)
    print(X_train.shape, y_train.shape)
    print(X_val.shape, y_val.shape)

    # the histogram of the data
    # n, bins, patches = plt.hist(y_val, 20, normed=1, facecolor='green', alpha=0.75)
    # plt.show()

    # define model
    kernel_size=[3,3]
    depth=[16,8,4,2]
    pool_size=(2,2)
    input_shape = X_train.shape[1:]
    print("input shape:",input_shape)
    print("kernel_size:",kernel_size)
    print("depth:",depth)
    print("pool_size:",pool_size)

    model = getModel()

    # train model
    model.fit(X_train, y_train, nb_epoch=FLAGS.epochs, batch_size=FLAGS.batch_size, validation_data=(X_val, y_val), shuffle=True)

    # save weights
    model.save_weights('./model.h5')
    print("Saved model.h5")

if __name__ == '__main__':
    tf.app.run()
