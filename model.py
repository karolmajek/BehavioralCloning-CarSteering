#!/usr/bin/python3
from sklearn.model_selection import train_test_split
from keras.optimizers import Adam
from keras.layers import Input, Flatten, Dense, Activation, Convolution2D, MaxPooling2D, Dropout
from keras.models import Model, Sequential, model_from_json
from keras.utils.visualize_util import plot
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

#set random seed for reproducibility
random_seed=123456
np.random.seed(random_seed)

# command line flags
flags.DEFINE_string('training_csv', 'recordings/track0_trial3/driving_log.csv', "")
flags.DEFINE_string('validation_csv', None, "")
flags.DEFINE_integer('epochs', 10, "The number of epochs.")
flags.DEFINE_integer('batch_size', 400, "The batch size.")
flags.DEFINE_integer('show', False, "Input data visualization")

#Labels for CSV columns - not used, just for info
image_labels=['Center image','Left image','Right image']
data_labels=['Steering Angle','Throttle','Break','Speed']

def process_image(image):
    '''
    Function that process each image during training and while driving.
    It reduces shape from 320x160x3 to 18x160x1.
    It also normalize values to zero mean.
    '''
    image=image[60:130:4,::2,:]
    image=np.mean(image,axis=2)
    image=(image-image.min())/255.0-0.5
    image=image.reshape(image.shape[0],image.shape[1],1)
    return image

def load_data(training_csv, validation_csv=None, val_size=0.2):
    '''
    Function loads the data training and validation data.
    If validation_csv is not specified it splits data from training_csv
     into training and validation using val_size parameter.
    Otherwise it loads validation data from validation_csv.
    Function returns X_train, y_train, X_val, y_val
    '''
    print('-'*60)
    X_train=[]
    y_train=[]

    first_image=True

    with open(training_csv, 'r') as csvfile:
        data = list(csv.reader(csvfile, delimiter=','))
        pbar = tqdm(range(len(data)), desc='Loading Train', unit='images')
        for i in pbar:
            l=data[i]
            image0=cv2.imread(l[0].lstrip())
            image1=cv2.imread(l[1].lstrip())
            image2=cv2.imread(l[2].lstrip())
            if first_image:
                cv2.imwrite('images/raw.jpg',image0)
            image0=process_image(image0)
            image1=process_image(image1)
            image2=process_image(image2)
            if first_image:
                cv2.imwrite('images/processed.jpg',(image0+0.5)*255.0)
                first_image=False
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
            test_size=val_size,
            random_state=random_seed)
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
    print('Shape of training input:',X_train.shape)
    print('Shape of training output:',y_train.shape)
    print('Shape of validation input:',X_val.shape)
    print('Shape of validation output:',y_val.shape)
    return X_train, y_train, X_val, y_val

def getModel(model_file='model.json'):
    '''
    Function loads model from model.json file.
    '''
    with open(model_file, 'r') as f:
        model = model_from_json(json.load(f))
    model.compile("adam", "mse", metrics=['accuracy'])
    print("Model imported")
    return model

def createModel(input_shape):
    '''
    Function creates a model of a the network
    '''
    kernel_size=[3,3]
    depth=[64,32,16,8]
    pool_size=(2,2)
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
    model.add(Dropout(0.3))

    model.add(Flatten())

    model.add(Dense(16))
    model.add(Activation('relu'))
    model.add(Dense(16))
    model.add(Activation('relu'))
    model.add(Dense(16))
    model.add(Activation('relu'))
    model.add(Dropout(0.3))
    model.add(Dense(1))
    return model

def saveModel(model,fname='model.json'):
    '''
    Function saves the model to file
    '''
    json_string = model.to_json()
    with open(fname, 'w') as outfile:
        json.dump(json_string, outfile)
        print("Saved model to:",fname)

def main(_):
    #get the input shape after processing
    img=np.zeros(shape=(160,320,3))
    img=process_image(img)
    input_shape = img.shape

    print("Image shape before processing:",img.shape)
    print("Image shape after processing:",input_shape)

    #create the model
    model =createModel(input_shape)

    #print summary
    model.summary()

    #save the model
    saveModel(model)

    #plot the model for README.md
    plot(model, to_file='images/model.png',show_shapes=True)

    # load input data
    X_train, y_train, X_val, y_val = load_data(FLAGS.training_csv, FLAGS.validation_csv)

    # the histogram of the steering data - train and val
    n, bins, patches = plt.hist(y_train, 40, facecolor='green', alpha=0.75,label=['Training data'])
    n, bins, patches = plt.hist(y_val, 40, facecolor='red', alpha=0.75, stacked=False,label=['Validation data'])
    plt.xlabel('Steering value')
    plt.ylabel('Number of images')
    plt.title('Histogram of steering values')
    plt.legend()
    plt.savefig('images/histogram.png', bbox_inches='tight')


    #Load model from file
    model = getModel()

    # train model
    model.fit(X_train, y_train, nb_epoch=FLAGS.epochs, batch_size=FLAGS.batch_size, validation_data=(X_val, y_val), shuffle=True)

    # save weights
    model.save_weights('./model.h5')
    print("Saved model.h5")

if __name__ == '__main__':
    tf.app.run()
