# Copyright (c) 2020 Seamus Lankford
# Licensed under MIT License

import warnings
warnings.filterwarnings('ignore')

# do not delete the following imports. Importing all on keras.datasets enables autonas to be generic
# A neural architecture for any keras dataset can be devised simply by specifying the name of the data and
# the number of colour channels. Two examples are illustrated below.
import cv2
from keras.datasets import *

from keras.utils import to_categorical
from matplotlib import pyplot
from pandas import np


def pre_processor(ui):
    # Generic pre processor which can handle both grayscale and RGB datasets

    # load train and test datasets, reshape if necessary and one hot encode labels
    def process_dataset():

        # The eval function takes a string as argument and evaluates this string as a Python expression.
        # The result of an expression is an object.
        str_to_dataset = eval(ui.dataset)

        (trainX, trainY), (testX, testY) = str_to_dataset.load_data()  # load dataset

        if ui.mode == 'load':   # when in loader mode, display a sample of the images.

            # summarise loaded dataset
            print('Train: X=%s, y=%s' % (trainX.shape, trainY.shape))
            print('Test: X=%s, y=%s' % (testX.shape, testY.shape))

            for i in range(9):              # plot first few images
                # define subplot
                pyplot.subplot(330 + 1 + i)
                pyplot.imshow(trainX[i])
            pyplot.show()   # display the figure

        if ui.mode == 'pretrain' or ui.mode == 'hybrid':

            if ui.n_channel == 1:
                # Following is done for pre-trained only, convert images to 32 x 32 x 3 (RGB)
                trainX = [cv2.cvtColor(cv2.resize(i, (32, 32)), cv2.COLOR_GRAY2BGR) for i in trainX]
                trainX = np.concatenate([arr[np.newaxis] for arr in trainX]).astype('float32')

                testX = [cv2.cvtColor(cv2.resize(i, (32, 32)), cv2.COLOR_GRAY2BGR) for i in testX]
                testX = np.concatenate([arr[np.newaxis] for arr in testX]).astype('float32')

        if ui.mode == 'train' or ui.mode == 'load'\
                or ui.mode == 'hybrid':
            # check if images are grayscale. If not, then assume images are RGB
            if ui.n_channel == 1:
                # grayscale images => reshape dataset to have just one channel
                # convert dataset to shape of (28, 28, 1)
                trainX = trainX.reshape((trainX.shape[0], 28, 28, 1))
                testX = testX.reshape((testX.shape[0], 28, 28, 1))

        if ui.mode != 'hybrid':
            # one hot encode target values.
            # hybrid looks after to to_categorical in main hybrid function
            trainY = to_categorical(trainY)
            testY = to_categorical(testY)

        return trainX, trainY, testX, testY

    # convert to floats and normalise
    def process_pixels(train, test):
        # integers -> floats
        train_n = train.astype('float32')
        test_n = test.astype('float32')

        # normalize to range 0-1
        train_n = train_n / 255.0
        test_n = test_n / 255.0

        return train_n, test_n  # return normalized images

    if ui.mode == 'ensemble':
        # ensemble mode involves passing output from keras models to sci-kit linear classifiers
        if ui.n_channel == 1:
            print("1 channel land !")
            (trainX, trainY), (testX, testY) = fashion_mnist.load_data()  # load the training and testing data.

            if ui.optimize == 'hybrid':     # convert to 32 x 32 x 3 for pre trained ensemble networks
                trainX = [cv2.cvtColor(cv2.resize(i, (32, 32)), cv2.COLOR_GRAY2BGR) for i in trainX]
                trainX = np.concatenate([arr[np.newaxis] for arr in trainX]).astype('float32')

                testX = [cv2.cvtColor(cv2.resize(i, (32, 32)), cv2.COLOR_GRAY2BGR) for i in testX]
                testX = np.concatenate([arr[np.newaxis] for arr in testX]).astype('float32')

            else:
                trainX = trainX.reshape((trainX.shape[0], 28, 28, 1))  # reshape dataset to have a single channel
                testX = testX.reshape((testX.shape[0], 28, 28, 1))

        if ui.n_channel == 3:
            print("3 channel land !")
            (trainX, trainY), (testX, testY) = cifar10.load_data()  # load the training and testing data.
            trainX = trainX.reshape((trainX.shape[0], 32, 32, 3))  # reshape dataset to have a single channel
            testX = testX.reshape((testX.shape[0], 32, 32, 3))

    else:       # has been tested on autokeras

        trainX, trainY, testX, testY = process_dataset()            # load dataset
        trainX, testX = process_pixels(trainX, testX)               # prepare pixel data

    return trainX, trainY, testX, testY
