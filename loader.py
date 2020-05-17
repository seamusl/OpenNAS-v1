# Copyright (c) 2020 Seamus Lankford
# Licensed under MIT License

# LOADER command line calls (samples)

# CIFAR
# python autonas.py -mode load -optimize aco -d cifar10 -n_channel 3 --cf 5 --m_name aco1-cifar.h5
# python autonas.py -mode load -optimize pso -d cifar10 -n_channel 3 --cf 5 --m_name pso1-cifar
# python autonas.py -mode load -optimize mobilenet -d cifar10 -n_channel 3 --cf 5 --m_name mobilenet-cifar.h5
# python autonas.py -mode load -optimize resnet50 -d cifar10 -n_channel 3 --cf 5 --m_name resnet50-cifar.h5

# FASHION
# python autonas.py -mode load -optimize aco -d fashion_mnist -n_channel 1 --cf 5 --m_name aco1-fashion.h5
# python autonas.py -mode load -optimize pso -d fashion_mnist -n_channel 1 --cf 5 --m_name pso2-fashion.h5
# python autonas.py -mode load -optimize VGG16 -d fashion_mnist -n_channel 3 --cf 5 --m_name vgg16-fashion.h5
# python autonas.py -mode load -optimize resnet50 -d fashion_mnist -n_channel 3 --cf 5 --m_name resnet50-fashion.h5

# ---- AUTOKERAS LOADING---- #
# To load previously built models for fashion_mnist (grayscale):
# python autonas.py -mode load -optimize autokeras -d fashion_mnist -n_channel 1 --m_name fashion_mnist-model_number_0

import warnings
warnings.simplefilter(action='ignore')

import joblib
import keras

from keras.datasets import fashion_mnist
from tensorflow.python.keras.datasets import cifar10
from keras.utils import to_categorical

from sklearn.model_selection import KFold
from keras.models import model_from_yaml

# need the following for graphing model structures
from tensorflow.keras.utils import plot_model
import tensorflow as tf

from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import time


def load_autonas(ui):

    def create_model():     # Function to create model, required for KerasClassifier

        if ui.optimize == 'pso':        # don't specify the file extension when loading model
            f_name = ui.m_name
            f_yaml = f_name + '.yaml'
            f_weights = f_name + '.h5'

            yaml_file = open(f_yaml, 'r')
            loaded_model_yaml = yaml_file.read()
            yaml_file.close()

            loaded_model = model_from_yaml(loaded_model_yaml)
            loaded_model.load_weights(f_weights)

            opt = 'Adagrad'
            loaded_model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
            loaded_model.save(ui.m_name)  # save the model (overwrite weights file with full arch)

        opt = 'Adagrad'
        loaded_model = tf.keras.models.load_model(ui.m_name)  # LOAD autonas base model
        loaded_model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

        return loaded_model

    print("[INFO] pre-processing", ui.dataset, "...")

    if ui.n_channel == 1:
        (trainX, trainY), (testX, testY) = fashion_mnist.load_data()
        # grayscale images => reshape dataset to have just one channel
        # convert dataset to shape of (28, 28, 1)
        trainX = trainX.reshape((trainX.shape[0], 28, 28, 1))
        testX = testX.reshape((testX.shape[0], 28, 28, 1))

    if ui.n_channel == 3:       # note all pre-train networks are triple channel
        (trainX, trainY), (testX, testY) = cifar10.load_data()

    trainY = to_categorical(trainY)
    testY = to_categorical(testY)

    timestr = time.strftime("%m%d-%H%M")
    # graph structure and store graph of architecture
    filename = "plots/" + ui.optimize + "-" + str(ui.my_seed) + "-" + timestr + '.png'

    print("Applying Kfold now ...")
    # seed is specified on the command line at run time for reproducibility
    # create model using KerasClassifier wrapper for a Keras classification neural net
    # scikit-learn library functions can then be accessed

    model = KerasClassifier(build_fn=create_model, epochs=ui.epochs, batch_size=10)

    # evaluate using cross validation. number of folds is specified on the command line at run time
    kfold = KFold(n_splits=ui.cf, shuffle=True, random_state=ui.my_seed)

    results = cross_val_score(model, testX, testY, cv=kfold, verbose=0)
    print("Mean results", "%.2f" % results.mean())
    print("Standard deviation", "%.4f" % results.std())
    print("All results", results)

    loaded_model = tf.keras.models.load_model(ui.m_name)
    print("Loaded model from disk ...", loaded_model.summary())

    plot_model(loaded_model, to_file=filename)

    store_fig = "plots/" + ui.optimize + "-" + timestr + ".png"

    fig = plt.figure()
    fig.suptitle('Compare folds')
    plt.xlabel(ui.m_name)
    plt.boxplot(results)        # box plot algorithm comparison
    plt.savefig(store_fig)
    plt.show()

    if ui.optimize == 'pretrain' or ui.optimize == 'ensemble':
        # no structure with classifier applied to pretrain or model of ensembler outputs
        # => just load model disk and evaluate

        filename = ui.m_name
        loaded_model = joblib.load(filename)
        result = loaded_model.score(testX, testY)
        print(result)

    return trainX, trainY, testX, testY
