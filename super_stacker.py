# Copyright (c) 2020 Seamus Lankford
# Licensed under MIT License

from keras.datasets import fashion_mnist, cifar10

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from mlens.ensemble import SuperLearner
import numpy as np
from sklearn.ensemble import StackingClassifier

import config as cfg


def get_super_learner(X):		# create the super learner
    ensemble = SuperLearner(scorer=accuracy_score, folds=3,
                            shuffle=True, verbose=True, sample_size=len(X))
    ensemble.add(cfg.sl_models)         # add base models
    ensemble.add_meta(cfg.sl_meta)      # add meta model
    return ensemble


def stack_layers(ui):

    print("stacking layers ... ")
    # Initialize Stacking Classifiers

    def stack_one_layer():
        clf = StackingClassifier(estimators=cfg.layer1_learners,
                                 final_estimator=cfg.l1_fe, cv=ui.cf)
        return clf

    def stack_two_layer():
        layer_two = StackingClassifier(estimators=cfg.layer2_learners,
                                       final_estimator=cfg.l2_fe, cv=ui.cf)
        clf = StackingClassifier(estimators=cfg.layer1_learners, final_estimator=layer_two)
        return clf

    if ui.stack == 1:
        clf = stack_one_layer()
    if ui.stack == 2:
        clf = stack_two_layer()

    return clf


def super_stacker(ui):

    # load dataset
    if ui.dataset == 'fashion_mnist':
        (trainX, trainY), (testX, testY) = fashion_mnist.load_data()
    elif ui.dataset == 'cifar10':
        (trainX, trainY), (testX, testY) = cifar10.load_data()

    print('trainX shape:', trainX.shape)
    print('testX shape:', testX.shape)

    newX = np.concatenate((trainX, testX), axis=0)
    y = np.concatenate((trainY, testY), axis=0)

    # Normalize the images features
    newX = (newX / 255) - 0.5

    print("After stacking ...  X shape:", newX.shape)
    print('all labels', y.shape[0])
    print('y shape:', y.shape)

    if ui.dataset == 'fashion_mnist':
        X = newX.reshape((-1, 784))  # Flatten images
    elif ui.dataset == 'cifar10':
        y = y.ravel()
        print('y shape:', y.shape)
        X = newX.reshape((-1, 3072))  # Flatten

    print("After flattening ... X shape", X.shape)

    if ui.stack:

        clf = stack_layers(ui)

        X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=ui.my_seed)
        clf.fit(X_train, y_train)
        print(clf.score(X_test, y_test))

    else:

        X, X_val, y, y_val = train_test_split(X, y, test_size=0.50)
        print('Train', X.shape, y.shape, 'Test', X_val.shape, y_val.shape)
        ensemble = get_super_learner(X)  # create the super learner

        ensemble.fit(X, y)  # fit the super learner
        print(ensemble.data)  # summarize base learners

        yhat = ensemble.predict(X_val)  # make predictions on hold out set
        print('Super Learner: %.3f' % (accuracy_score(y_val, yhat) * 100))

    return
