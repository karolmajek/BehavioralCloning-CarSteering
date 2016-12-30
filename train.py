#!/usr/bin/python3
from keras.layers import Input, Flatten, Dense
from keras.models import Model
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import tensorflow as tf
import numpy as np
import cv2
import csv

flags = tf.app.flags
FLAGS = flags.FLAGS

# command line flags
flags.DEFINE_string('training_dir', 'recordings/track0_trail1', "Directory with training data from simulator")
flags.DEFINE_string('validation_dir', None, "Directory with validation data from simulator")
flags.DEFINE_integer('epochs', 50, "The number of epochs.")
flags.DEFINE_integer('batch_size', 256, "The batch size.")
flags.DEFINE_integer('show_input', False, "Input data visualization")

image_labels=['Center image','Left image','Right image']
data_labels=['Steering Angle','Throttle','Break','Speed']

def normalize(data):
    return 0.8*(data-data.min())/(data.max()-data.min())+0.1

def load_data(training_dir, validation_dir=None):
    X_train=[]
    y_train=[]
    with open(training_dir+'/driving_log.csv', 'r') as csvfile:
        data = list(csv.reader(csvfile, delimiter=','))
        pbar = tqdm(range(len(data)), desc='Loading Train', unit='images')
        for i in pbar:
            l=data[i]
            image0=cv2.imread(l[0].lstrip())
            if FLAGS.show_input:
                image1=cv2.imread(l[1].lstrip())
                image2=cv2.imread(l[2].lstrip())
                image=np.concatenate((image1,image0,image2),axis=1)
                cv2.imshow('cameras',image)
                cv2.waitKey(1)
            # Lets work with center image only
            X_train.append(normalize(image0))
            output=float(l[3])
            y_train.append(output)
    X_train=np.array(X_train)
    y_train=np.array(y_train)

    if validation_dir == None:
        print("Number of samples before split into train/val:",X_train.shape[0])
        
        X_train, X_val, y_train, y_val = train_test_split(X_train,
            y_train,
            test_size=0.2,
            random_state=122333)
    else:
        X_val=[]
        y_val=[]
        with open(validation_dir+'/driving_log.csv', 'r') as csvfile:
            data = list(csv.reader(csvfile, delimiter=','))
            pbar = tqdm(range(len(data)), desc='Loading Val', unit='images')
            for i in pbar:
                l=data[i]
                image0=cv2.imread(l[0].lstrip())
                if FLAGS.show_input:
                    image1=cv2.imread(l[1].lstrip())
                    image2=cv2.imread(l[2].lstrip())
                    image=np.concatenate((image1,image0,image2),axis=1)
                    cv2.imshow('cameras',image)
                    cv2.waitKey(1)
                # Lets work with center image only
                X_val.append(normalize(image0))
                output=float(l[3])
                y_val.append(output)
        X_val=np.array(X_val)
        y_val=np.array(y_val)

    return X_train, y_train, X_val, y_val

def main(_):
    # load bottleneck data
    X_train, y_train, X_val, y_val = load_data(FLAGS.training_dir, FLAGS.validation_dir)
    print(X_train.shape, y_train.shape)
    print(X_val.shape, y_val.shape)
    return

    nb_classes = len(np.unique(y_train))

    # define model
    input_shape = X_train.shape[1:]
    inp = Input(shape=input_shape)
    x = Flatten()(inp)
    x = Dense(nb_classes, activation='softmax')(x)
    model = Model(inp, x)
    model.summary()
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # train model
    model.fit(X_train, y_train, nb_epoch=FLAGS.epochs, batch_size=FLAGS.batch_size, validation_data=(X_val, y_val), shuffle=True)

    json_string = model.to_json()
    print(json_string)


if __name__ == '__main__':
    tf.app.run()
