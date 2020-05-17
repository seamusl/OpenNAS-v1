# Copyright (c) 2020 Seamus Lankford
# Licensed under MIT License

import warnings
warnings.simplefilter(action='ignore')

from keras.utils import to_categorical
from sklearn.metrics import accuracy_score

import tensorflow as tf
from tensorflow.keras.models import load_model

import numpy as np
import matplotlib.pyplot as plt

import joblib

from pre_processor import pre_processor
import config as cfg


def fit_model(ui):

    print("Compiling model...")  # initialize the optimizer and model
    opt = tf.keras.optimizers.SGD(lr=0.0001)

    model = load_model(ui.m_name)     # LOAD autonas base model
    print(model.summary())
    config = model.to_json()
    loaded_model = tf.keras.models.model_from_json(config)
    loaded_model.compile(loss="sparse_categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

    print("Training base learner...")  # train the network
    loaded_model.fit(trainX, trainY, validation_data=(testX, testY),
              batch_size=32, epochs=ui.epochs, verbose=2)

    return loaded_model


def load_models(num_models):            # load models from disk
    all_models = list()
    for i in range(num_models):
        filename = 'model_' + str(i + 1) + '.h5'            # filename for this ensemble
        model = tf.keras.models.load_model(filename)    # load model from the file

        all_models.append(model)        # add model to the list
        print('Loaded %s' % filename)
    return all_models


def stacked_dataset(members, inputX):
    stackX = None                                       # initially no layers in stack
    for model in members:
        y_pred = model.predict(inputX, verbose=0)       # make prediction
        if stackX is None:          # stack predictions into [rows, members, probabilities]
            stackX = y_pred         # add first layer to stack
        else:
            stackX = np.dstack((stackX, y_pred))  # add new layer to stack
    # flatten predictions to [rows, members x probabilities]
    stackX = stackX.reshape((stackX.shape[0], stackX.shape[1]*stackX.shape[2]))

    return stackX


def stacked_prediction(members, model, inputX):     # make a prediction with the stacked model
    stackedX = stacked_dataset(members, inputX)     # create dataset using ensemble
    y_pred = model.predict(stackedX)                # make predictions
    return y_pred


# Using ensemble outputs, create stacked training dataset for meta learner.
# feed outputs from ensemble base learners and fit a meta learner.

def fit_stacked_model(members, inputX, inputy, algorithm):

    # create meta learner data set using ensemble base learner predictions. The features of the new data set
    # returned, stackedX, are simply the predictions of each of the base learners for each instance. Therefore
    # more base learners => greater number of features in the new data set. stackedX and inputy (i.e. the
    # corresponding correct label outputs are used to fit a new model with the chosen classifier.

    stackedX = stacked_dataset(members, inputX)     # create data set from ensemble base learners
    # start of Phase B
    model = algorithm   # assign user defined model algorithm for training
    # fit using aggregate feature data from base learners and output labels
    model.fit(stackedX, inputy)
    return model


def ensemble_classifiers(base_learners, ui):

    testYc = to_categorical(testY)

    # evaluate standalone models on test set
    # performance of standalone models can be compared with ensemble performance
    for model in base_learners:
        _, acc = model.evaluate(testX, testYc, verbose=0)
        print('Model Accuracy: %.4f' % acc)

    results = []
    names = []

    for name, meta in cfg.meta_learners:         # evaluate multiple classifier models

        print("Training meta learner with ", meta, "...")  # train the meta learner wi
        # th its own data set
        # fit stacked model using the ensemble
        model = fit_stacked_model(base_learners, testX, testY, meta)

        # evaluate meta learner on test set
        y_pred = stacked_prediction(base_learners, model, testX)
        acc = accuracy_score(y_pred, testY)

        results.append(acc)
        names.append(name)

        print('Ensemble Meta learner Test Accuracy:   %.4f' % acc)

        # since we are using a sci-kit learn classifier (and not keras), use joblib library to store model
        filename = 'model_' + str(name) + '.sav'            # save the model to disk
        joblib.dump(model, filename)
        print('Saved %s' % filename)

    plt.figure(figsize=(9, 3))
    plt.subplot(132)
    plt.scatter(names, results)
    plt.suptitle('Algorithm Comparison')
    plt.savefig('ensemble_comparison.png')

    return


def create_base_learners(ui):
    # fit each base learner with same dataset and save models
    # weights of each model randomly initialised
    # => different base learner model saved with each iteration

    for i in range(ui.n_base):
        H = fit_model(ui)
        filename = 'model_' + str(i + 1) + '.h5'
        H.save(filename)  # save model
        print('Saved %s' % filename)

    return


def ensemble(ui):

    global trainX, trainY, testX, testY     # these globals only needed within ensemble module

    print("[INFO] pre-processing", ui.dataset, "...")
    trainX, trainY, testX, testY = pre_processor(ui)

    if ui.learners:
        create_base_learners(ui)

    # train meta-learner using predictions from base learners
    base_learners = load_models(ui.n_base)  # load all models

    print('Loaded %d models' % len(base_learners))  # check if all base learner models loaded

    ensemble_classifiers(base_learners, ui)

    return
