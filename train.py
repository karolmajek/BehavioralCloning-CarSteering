import pickle
import tensorflow as tf
# TODO: import Keras layers you need here
import numpy as np
from keras.layers import Input, Flatten, Dense
from keras.models import Model
import csv
import cv2

flags = tf.app.flags
FLAGS = flags.FLAGS

# command line flags
flags.DEFINE_string('training_dir', 'recordings/track0_trail0', "Directory with training data from simulator")
flags.DEFINE_string('validation_dir', 'recordings/track0_trail0', "Directory with validation data from simulator")
flags.DEFINE_integer('epochs', 50, "The number of epochs.")
flags.DEFINE_integer('batch_size', 256, "The batch size.")


def load_data(training_dir, validation_dir=None):
    if validation_dir == None:
        validation_dir=training_dir

    with open(training_dir+'/driving_log.csv', 'r') as csvfile:
        data = csv.reader(csvfile, delimiter=',')
        for l in data:
            image0=cv2.imread(l[0].lstrip())
            image1=cv2.imread(l[1].lstrip())
            image2=cv2.imread(l[2].lstrip())
            image=np.concatenate((image1,image0,image2),axis=1)
            cv2.imshow('cameras',image)
            cv2.waitKey(1)
            print(l[3:])
    return #X_train, y_train, X_val, y_val

def main(_):
    # load bottleneck data
    # X_train, y_train, X_val, y_val = load_data(FLAGS.training_dir, FLAGS.validation_dir)
    load_data(FLAGS.training_dir, FLAGS.validation_dir)
    return
    print(X_train.shape, y_train.shape)
    print(X_val.shape, y_val.shape)

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


# parses flags and calls the `main` function above
if __name__ == '__main__':
    tf.app.run()
