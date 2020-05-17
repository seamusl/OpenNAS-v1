# Copyright (c) 2020 Seamus Lankford
# Licensed under MIT License

import warnings
warnings.filterwarnings('ignore')

import time
from datetime import datetime
import os
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from ruamel.yaml import YAML
import random
import cv2

import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)  # stops unwanted console errors

from sklearn import metrics, model_selection
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import train_test_split

from scipy.stats import uniform
from scipy.stats import norm

from psoCNN import psoCNN

import keras
from keras.datasets import *
from keras.layers.core import Dropout
from keras.layers.normalization import BatchNormalization
from keras.layers import Input, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import VGG16, VGG19, MobileNet, ResNet50, Xception, InceptionV3
from keras.applications.vgg16 import preprocess_input
from keras.models import Model
from keras.utils import np_utils

from ensembler import ensemble
from pre_processor import pre_processor
from super_stacker import super_stacker

import config as cfg

class Trainer:

    def __init__(self, ):

        pass

        return

    def set_baseModel(self, custom_model):

        # process_dataset() in ui.py transforms has already tranformed data to the following shape
        height, width, depth, classes = 32, 32, 3, 10

        # the pre-trained CNN model, user specified on the command line, sets the primary model architecture
        # keras function call must be modified in line with user choice
        # this loads up a model which has already been fitted

        if custom_model == 'VGG16':
            starter_model = VGG16(weights='imagenet', include_top=False)
        elif custom_model == 'VGG19':
            starter_model = VGG19(weights='imagenet', include_top=False)
        elif custom_model == 'MobileNet':
            starter_model = MobileNet(weights='imagenet', include_top=False)
        elif custom_model == 'ResNet50':
            starter_model = ResNet50(weights='imagenet', include_top=False)

        return starter_model

    def tune_secondary_model(self, feat_Train, feat_Val, trainY, testY, ui):
        # The hyper parameters of the secondary model (user specified on the command line) are tuned

        # construct the set of hyper parameters to tune
        # tune the hyper parameters via a randomized search
        # apply optimal parameter grid to the model and fit
        # cross fold validation used to reduce over fitting (with cv=5)

        print("Applying secondary ML algorithm ... ")
        print("Tuning hyper parameters via Random Search ...")

        if ui.tune == 'knn':

            print("Testing Random Search on KNN ... ")
            clf = KNeighborsClassifier()

            params = {"n_neighbors": np.arange(1, 31, 2), "metric": ["euclidean", "cityblock"]}
            model = RandomizedSearchCV(clf, params, random_state=random.randrange(10000000), cv=ui.cf)

        if ui.tune == 'mlp':

            print("Testing Random Search on MLP ... ")
            clf = MLPClassifier(max_iter=100)

            parameter_space = {
                    'hidden_layer_sizes': [(50, 50, 50), (50, 100, 50), (100,)],
                    'activation': ['tanh', 'relu'], 'solver': ['sgd', 'adam'],
                    'alpha': [1, 1], 'learning_rate': ['constant', 'adaptive'], }
            model = RandomizedSearchCV(clf, parameter_space, n_jobs=-1, verbose=2,
                                       random_state=random.randrange(10000000), cv=ui.cf)

        if ui.tune == 'rforest':

            print("Testing Random Search on Random Forest ... ")
            clf = RandomForestClassifier()

            n_estimators = [int(x) for x in np.linspace(start=200, stop=2000, num=20)] # num of trees in random forest
            max_features = ['auto', 'sqrt']  # number of features at every split

            max_depth = [int(x) for x in np.linspace(100, 500, num=11)]
            max_depth.append(None)
            random_grid = {'n_estimators': n_estimators, 'max_features': max_features, 'max_depth': max_depth}

            # Random search of parameters
            model = RandomizedSearchCV(clf, param_distributions=random_grid, n_iter=5, cv=ui.cf, verbose=2,
                                       random_state=random.randrange(10000000), n_jobs=-1)

        if ui.tune == 'svc':

            print("Testing Random Search on Support Vector Machine (SVC) ... ")
            clf = SVC()

            parameter_space = {'kernel': ['linear', 'poly', 'rbf'], 'C': norm(loc=0.5, scale=0.15)}
            model = RandomizedSearchCV(clf, parameter_space, n_jobs=-1, random_state=random.randrange(10000000), cv=ui.cf, verbose=2)

        if ui.tune == 'lr':

            print("Testing Random Search on Logistic Regression ... ")
            clf = LogisticRegression()                      # Create logistic regression

            penalty = ['l1', 'l2']          # Create regularization penalty space
            C = uniform(loc=0, scale=4)     # Use uniform distribution to create regularization hyperparameter
            parameter_space = dict(C=C, penalty=penalty)    # Create hyperparameter options
            model = RandomizedSearchCV(clf, parameter_space, random_state=random.randrange(10000000), n_iter=10, cv=ui.cf,
                                       verbose=2, n_jobs=-1)

        model.fit(feat_Train, trainY)

        va_pred = model.predict(feat_Val)
        val_acc = metrics.accuracy_score(testY, va_pred)
        print("Validation accuracy: ", val_acc)

        tr_pred = model.predict(feat_Train)
        tr_acc = metrics.accuracy_score(trainY, tr_pred)
        print("Training accuracy: ", tr_acc)
        print("")

        return

    def multiple_classifiers(self, feat_Val, testY, ui):

        models = cfg.meta           # evaluate each model in turn
        results = []
        names = []

        # prepare models smf prepare configuration

        for name, model in models:
            kfolds = model_selection.KFold(n_splits=ui.cf, random_state=random.randrange(10000000))
            cv_results = model_selection.cross_val_score(model, feat_Val, testY, cv=kfolds, scoring='accuracy')

            results.append(cv_results)
            names.append(name)
            msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
            print(msg)

        # boxplot algorithm comparison
        fig = plt.figure()
        fig.suptitle('Algorithm Comparison')
        ax = fig.add_subplot(111)
        plt.boxplot(results)
        ax.set_xticklabels(names)
        plt.savefig('models_comparison.png')

        return

    def pre_train(self, ui):
        # the user specified pre-trained network is used as a feature extractor
        # to create a new dataset before it is passed to the secondary classifiers

        print("[INFO] pre-processing", ui.dataset, "...")
        trainX, trainY, testX, testY = pre_processor(ui)

        baseModel = self.set_baseModel(ui.optimize)

        print("Pushing initial Training dataset though custom model chosen by user ...")
        featuresTrain = baseModel.predict(trainX)
        featuresTrain = featuresTrain.reshape(featuresTrain.shape[0], -1)
        print("New Training dataset now created for secondary ML algorithm.")

        print("Pushing initial Test dataset though custom model chosen by user ...")
        featuresVal = baseModel.predict(testX)
        featuresVal = featuresVal.reshape(featuresVal.shape[0], -1)
        print("New Test dataset now created for secondary ML algorithm.")

        return featuresTrain, featuresVal, trainY, testY

    def hybrid(self, ui):

        # the user specified pre-trained network is used as core model
        # additional layers are stacked on this core to create hybrid
        # all layers of hybrid (including inner layers of pre-trained model) are trained to maximize efficiency

        def preprocess_hybrid(x):
            X = np.expand_dims(x, axis=0)
            X = preprocess_input(X)
            return X[0]

        if ui.dataset == "cifar10":
            print("[INFO] pre-processing", ui.dataset, "...")
            (trainX, trainY), (testX, testY) = cifar10.load_data()

        if ui.dataset == "fashion_mnist":
            (trainX, trainY), (testX, testY) = fashion_mnist.load_data()

        if ui.n_channel == 1:
            # Following is done for pre-trained only, convert images to 32 x 32 x 3 (RGB)
            trainX = [cv2.cvtColor(cv2.resize(i, (32, 32)), cv2.COLOR_GRAY2BGR) for i in trainX]
            trainX = np.concatenate([arr[np.newaxis] for arr in trainX]).astype('float32')

            testX = [cv2.cvtColor(cv2.resize(i, (32, 32)), cv2.COLOR_GRAY2BGR) for i in testX]
            testX = np.concatenate([arr[np.newaxis] for arr in testX]).astype('float32')

        # Concatenate train and test images
        X = np.concatenate((trainX, testX))
        y = np.concatenate((trainY, testY))

        print(X.shape)  # (60000, 32, 32, 3)    # Check shape
        # Split data using user defined seed (need seed since val data being used)
        x_train, x_test, y_train, y_test \
            = train_test_split(X, y, test_size=10000, random_state=random.randrange(10000000))

        x_train, val_train, y_train, val_test = \
            train_test_split(x_train, y_train, test_size=0.2, random_state=random.randrange(10000000))

        print(x_train.shape, val_train.shape, x_test.shape, val_test.shape)

        num_classes = 10

        y_train = np_utils.to_categorical(y_train, num_classes)
        val_test = np_utils.to_categorical(val_test, num_classes)
        y_test = np_utils.to_categorical(y_test, num_classes)       # used for evaluation

        datagen = ImageDataGenerator(
            rotation_range=25, width_shift_range=0.25, height_shift_range=0.25,
            horizontal_flip=True, preprocessing_function=preprocess_hybrid)
        input = Input(shape=(32, 32, 3), name='image_input')

        baseline = self.set_baseModel(ui.optimize)
        output_baseline = baseline(input)             # Use the generated model

        # Add the fully-connected layers
        x = Flatten(name='flatten')(output_baseline)

        x = Dense(4096, activation='relu', name='fc1',
                  kernel_initializer=keras.initializers.glorot_normal(seed=random.randrange(10000000)))(x)

        x = BatchNormalization()(x)
        x = Dropout(.5, seed=random.randrange(10000000))(x)
        x = Dense(4096, activation='relu', name='fc2',
                  kernel_initializer=keras.initializers.glorot_normal(seed=random.randrange(10000000)))(x)

        x = BatchNormalization()(x)
        x = Dropout(.5, seed=random.randrange(10000000))(x)

        x = Dense(10, activation='softmax', name='predictions',
                  kernel_initializer=keras.initializers.glorot_normal(seed=random.randrange(10000000)))(x)

        my_model = Model(input=input, output=x)     # Create your own model

        Adam = keras.optimizers.Adam(lr=.0001)
        my_model.compile(optimizer=Adam, loss='categorical_crossentropy', metrics=['accuracy'])

        batch_size = 256
        history = my_model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size, seed=random.randrange(10000000)),
                               validation_data=datagen.flow(val_train, val_test, batch_size=batch_size,
                               seed=random.randrange(10000000)), verbose=2,
                               validation_steps=len(val_train) / 10, steps_per_epoch=len(x_train) / 10,
                               epochs=ui.epochs, shuffle=False)

        timestr = time.strftime("%m%d-%H%M")
        filename = "pretrained/" + ui.optimize + "-Hybrid-" + str(ui.my_seed) + "-" + timestr + ".h5"
        my_model.save(filename)

        evalgen = ImageDataGenerator(preprocessing_function=preprocess_hybrid)
        score = my_model.evaluate_generator(evalgen.flow(x_test, y_test, batch_size=256,
                                                         seed=random.randrange(10000000)), steps=len(x_test) / 10)
        print(my_model.metrics_names, score)

        timestr = time.strftime("%m%d-%H%M-")

        # summarize history for accuracy
        store_acc = "plots/" + ui.optimize + "-Hybrid-Accuracy-" + timestr + str(ui.my_seed)
        plt.plot(history.history['acc'])
        plt.plot(history.history['val_acc'])
        plt.title(ui.optimize + "-Accuracy-Hybrid-" + timestr + str(ui.my_seed))
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.savefig(store_acc)

        # summarize history for loss
        plt.clf()   # clear previous plot so that loss can now be plotted
        store_loss = "plots/" + ui.optimize + "-Hybrid-Loss-" + timestr + str(ui.my_seed)
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title(ui.optimize + "-Loss-Hybrid-" + timestr + str(ui.my_seed))
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.savefig(store_loss)

        return

    #  PSO training
    def pso_train(self, ui):

        # Custom changes to psoCNN:
        # within this function: cfg.epochs_pso set to 10 as per original paper for optimal performance
        # within particle.py module, an argument verbose=2 was added to .fit call in model_fit and model_fit_complete

        # function which uses pso to train on either the fashion_mnist or cifar10 datasets
        # one hot encoding, using to_categorical is carried out within pso library (pso_cnn.py)

        if ui.dataset == "fashion-mnist":
            dataset = "fashion-mnist"
        elif ui.dataset == "cifar10":
            dataset = "cifar10"

        # Algorithm parameters #
        number_runs = ui.pso_runs                      # 10 runs is often used as default
        number_iterations = ui.pso_iter                # 10 runs is often used as default
        population_size = ui.pso_pop                   # 20 runs is often used as default

        # Run the algorithm #
        results_path = "./results/" + dataset + "/"

        if not os.path.exists(results_path):
            os.makedirs(results_path)

        all_gBest_metrics = np.zeros((number_runs, 2))
        runs_time = []
        all_gbest_par = []
        best_gBest_acc = 0

        for i in range(number_runs):
            print("Run number: " + str(i))
            start_time = time.time()
            new_run = datetime.now()
            print(new_run)
            pso = psoCNN(dataset=dataset, n_iter=number_iterations, pop_size=population_size, \
                        batch_size=cfg.batch_size_pso, epochs=cfg.epochs_pso, min_layer=cfg.min_layer,
                        max_layer=cfg.max_layer, \
                        conv_prob=cfg.probability_convolution, pool_prob=cfg.probability_pooling, \
                        fc_prob=cfg.probability_fully_connected, max_conv_kernel=cfg.max_conv_kernel_size, \
                        max_out_ch=cfg.max_fully_connected_neurons, max_fc_neurons=cfg.max_fully_connected_neurons,
                        dropout_rate=cfg.dropout)

            pso.fit(Cg=cfg.Cg, dropout_rate=cfg.dropout)

            print(pso.gBest_acc)

            # Plot current gBest
            matplotlib.use('Agg')
            plt.plot(pso.gBest_acc)
            plt.xlabel("Iteration")
            plt.ylabel("gBest acc")
            plt.savefig(results_path + "gBest-iter-" + str(i) + ".png")
            plt.close()

            print('gBest architecture: ')
            print(pso.gBest)

            np.save(results_path + "gBest_inter_" + str(i) + "_acc_history.npy", pso.gBest_acc)
            np.save(results_path + "gBest_iter_" + str(i) + "_test_acc_history.npy", pso.gBest_test_acc)

            end_time = time.time()

            running_time = end_time - start_time

            runs_time.append(running_time)

            # Fully train the gBest model found
            n_parameters = pso.fit_gBest(batch_size=cfg.batch_size_full_training, epochs=cfg.epochs_full_training,
                                         dropout_rate=dropout)

            all_gbest_par.append(n_parameters)

            # Evaluate the fully trained gBest model
            gBest_metrics = pso.evaluate_gBest(batch_size=cfg.batch_size_full_training)

            if gBest_metrics[1] >= best_gBest_acc:
                best_gBest_acc = gBest_metrics[1]

                # Save best gBest model
                best_gBest_yaml = pso.gBest.model.to_yaml()

                with open(results_path + "best-gBest-model.yaml", "w") as yaml_file:
                    yaml_file.write(best_gBest_yaml)

                # Save best gBest model weights to HDF5 file
                pso.gBest.model.save_weights(results_path + "best-gBest-weights.h5")

            all_gBest_metrics[i, 0] = gBest_metrics[0]
            all_gBest_metrics[i, 1] = gBest_metrics[1]

            print("This run took: " + str(running_time) + " seconds.")

            # Compute mean accuracy of all runs
            all_gBest_mean_metrics = np.mean(all_gBest_metrics, axis=0)

            np.save(results_path + "/time_to_run.npy", runs_time)

            # Save all gBest metrics
            np.save(results_path + "/all_gBest_metrics.npy", all_gBest_metrics)

            # Save results in a text file
            output_str = "All gBest number of parameters: " + str(all_gbest_par) + "\n"
            output_str = output_str + "All gBest test accuracies: " + str(all_gBest_metrics[:, 1]) + "\n"
            output_str = output_str + "All running times: " + str(runs_time) + "\n"
            output_str = output_str + "Mean loss of all runs: " + str(all_gBest_mean_metrics[0]) + "\n"
            output_str = output_str + "Mean accuracy of all runs: " + str(all_gBest_mean_metrics[1]) + "\n"

            print(output_str)

            with open(results_path + "/final_results.txt", "w") as f:
                try:
                    print(output_str, file=f)
                except SyntaxError:
                    print >> f, output_str

        return

    # Ant Colony Optimization training
    def aco_train(self, ui):

        # function which uses aco to train on the inputted dataset
        global x_train, y_train, x_test, y_test, trained_topology

        # modify YAML file by taking a parameter
        if ui.dataset == 'cifar10':

            inp_fo = open("settings/cifar10.yaml").read()  # Read the Yaml File
            yaml = YAML()  # Load the yaml object
            # print(ui.m_depth, ui.n_ant, ui.epochs)

            code = yaml.load(inp_fo)  # Load content of YAML file to yaml object
            code['DeepSwarm']['max_depth'] = ui.m_depth  # Update Yaml file with new parameter in object
            code['DeepSwarm']['aco']['ant_count'] = ui.n_ant  # Update Yaml file with new parameter in object
            code['DeepSwarm']['backend']['epochs'] = ui.epochs  # Update Yaml file with new parameter in object

            inp_fo2 = open("settings/cifar10.yaml", "w")  # Open the file for Write
            yaml.dump(code, inp_fo2)  # Write to file with new parameter
            inp_fo2.close()  # close the file

        elif ui.dataset == 'fashion-mnist':

            inp_fo = open("settings/fashion-mnist.yaml").read()  # Read the Yaml File
            yaml = YAML()  # Load the yaml object
            # print(ui.m_depth, ui.n_ant, ui.epochs)

            code = yaml.load(inp_fo)  # Load content of YAML file to yaml object
            code['DeepSwarm']['max_depth'] = ui.m_depth  # Update Yaml file with new parameter in object
            code['DeepSwarm']['aco']['ant_count'] = ui.n_ant  # Update Yaml file with new parameter in object
            code['DeepSwarm']['backend']['epochs'] = ui.epochs  # Update Yaml file with new parameter in object

            inp_fo2 = open("settings/fashion-mnist.yaml", "w")  # Open the file for Write
            yaml.dump(code, inp_fo2)  # Write to file with new parameter
            inp_fo2.close()  # close the file

        # important to leave deepswarm import at this point otherwise
        # user customization won't update the yaml file
        from deepswarm.backends import Dataset, TFKerasBackend

        print("ACO Training of dataset ...")

        if ui.dataset == 'fashion-mnist':

            (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()    # Load Fashion MNIST dataset
            x_train, x_test = x_train / 255.0, x_test / 255.0                   # Normalize and reshape data
            x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
            x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

        elif ui.dataset == 'cifar10':

            (x_train, y_train), (x_test, y_test) = cifar10.load_data()          # Load CIFAR-10 dataset
            y_train = tf.keras.utils.to_categorical(y_train, 10)    # Convert class vectors to binary class matrices
            y_test = tf.keras.utils.to_categorical(y_test, 10)

        # import is at this point since _init_.py for deepswarm module is invoked once module is imported
        # the modified yaml settings (above) will not be loaded if deepswarm imported at start of prorgram.
        from deepswarm.deepswarm import DeepSwarm

        # Create dataset object, which controls all the data
        dataset = Dataset(
            training_examples=x_train, training_labels=y_train,
            testing_examples=x_test, testing_labels=y_test,
            validation_split=0.1)

        backend = TFKerasBackend(dataset=dataset)   # Create backend responsible for training & validating
        deepswarm = DeepSwarm(backend=backend)      # Create DeepSwarm object responsible for optimization
        topology = deepswarm.find_topology()        # Find the topology for a given dataset
        deepswarm.evaluate_topology(topology)       # Evaluate discovered topology

        if ui.dataset == 'cifar10':
            trained_topology = deepswarm.train_topology(topology, 50, augment={
                'rotation_range': 15,
                'width_shift_range': 0.1,
                'height_shift_range': 0.1,
                'horizontal_flip': True,
            })
        if ui.dataset == 'fashion-mnist':
            # Train topology on augmented data for additional 50 epochs
            trained_topology = deepswarm.train_topology(topology, 50, augment={
                'rotation_range': 15,
                'width_shift_range': 0.1,
                'height_shift_range': 0.1,
                'horizontal_flip': True,
            })

        deepswarm.evaluate_topology(trained_topology)   # Evaluate the final topology

        return

    def autokeras_train(self, ui):

        import autokeras as ak
        # simple pre-processing required of just scaling the data
        # autokeras makes a mess of reshaping input and one hot encoding label outputs by creating a custom layer
        # load the training and testing data, then scale it into the range [0, 1]

        print("[INFO] pre-processing", ui.dataset, "...")
        trainX, trainY, testX, testY = pre_processor(ui)

        for model in range(ui.ak_model):

            # train our Auto-Keras model
            print("[INFO] Training a new Autokeras model: ", model)

            # max_trials is max number of Keras Models to try.
            # search may finish before reaching the max_trials. Defaults to 100.
            # for exhaustive search set max_trials=50
            if ui.optimize == 'autokeras':

                print("Training with Autokeras ..")
                test_model = ak.ImageClassifier(max_trials=ui.ak_trial, seed=ui.my_seed)

            if ui.optimize == 'automodel':

                print("Training with Automodel ..")

                input_node = ak.ImageInput()
                output_node = ak.ImageBlock(
                    # Specify architecture subset to search
                    block_type=ui.ak_seek,
                    # Normalize the dataset
                    normalize=True,
                    # don't use data augmentation.
                    augment=False)(input_node)
                output_node = ak.ClassificationHead()(output_node)
                test_model = ak.AutoModel(inputs=input_node, outputs=output_node,
                                          max_trials=ui.ak_trial,  objective="val_accuracy", seed=ui.my_seed)

            # number of epochs to train each model during search, if unspecified, defaults to max of 1000.
            # Training stops if the validation loss stops improving for 10 epochs.
            # for exhaustive search set epochs=100
            test_model.fit(trainX, trainY, validation_split=0.3, epochs=ui.epochs, verbose=2)

            print("saving the model ..")
            best_model = test_model.export_model()
            best_model.summary()

            filename = ui.dataset + '-' + 'model_number_' + str(model) + '.h5'
            print("filename: ", filename)
            best_model.save(filename)                           # Save the model

        return


def train_autonas(ui):

    train_nas = Trainer()

    if ui.mode == 'train':

        if ui.optimize == 'pso':
            print("LOOKS GOOD - pso")
            train_nas.pso_train(ui)
        elif ui.optimize == 'aco':
            print("LOOKS GOOD - aco ")
            train_nas.aco_train(ui)
        elif ui.optimize == 'autokeras' or ui.optimize == 'automodel':
            train_nas.autokeras_train(ui)

    if ui.mode == 'pretrain':                                   # custom_model will be used within pre_train

        featuresTrain, featuresVal, trainY, testY = train_nas.pre_train(ui)

        if ui.second_model:
            print("Running multiple classifiers")
            train_nas.multiple_classifiers(featuresVal, testY, ui)

        if ui.tune:
            # tuning in this context means fine tuning of hyper parameters of chosen secondary model
            # tuning of inner layers of pre-trained models also to be implemented
            print("Tuning the secondary model")
            train_nas.tune_secondary_model(featuresTrain, featuresVal, trainY, testY, ui)

    if ui.mode == 'hybrid':                                   # custom_model will be used within pre_train
        print("LOOKS GOOD - hybrid")
        train_nas.hybrid(ui)

    if ui.mode == 'ensemble':
        print("LOOKS GOOD - ensemble ")
        ensemble(ui)

    if ui.mode == 'super_stacker':
        print("LOOKS GOOD - super_stacker ")
        super_stacker(ui)

    return
