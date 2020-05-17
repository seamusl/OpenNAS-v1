# Copyright (c) 2020 Seamus Lankford
# Licensed under MIT License

from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

#  *** START CONFIGURATION SETTINGS FOR PSO ***
epochs_pso = 1
epochs_full_training = 100

batch_size_pso = 32
batch_size_full_training = 32

max_conv_output_channels = 256
max_fully_connected_neurons = 300

min_layer = 3
max_layer = 20

# Probability of each layer type (should sum to 1)
probability_convolution = 0.6
probability_pooling = 0.3
probability_fully_connected = 0.1

max_conv_kernel_size = 7

Cg = 0.5
dropout = 0.5
#  *** END CONFIGURATION SETTINGS FOR PSO***

#  *** CONFIGURATION Ensemble Meta Learners ***
meta_learners = [('RF', RandomForestClassifier(n_estimators=100, max_depth=8, min_samples_leaf=4)),
                 ('SVC', SVC(kernel="linear", C=0.025)),
                 ('KNN', KNeighborsClassifier()),
                 ('MLPC', MLPClassifier(alpha=1, shuffle=False)),
                 ('LR', LogisticRegression())]

#  *** CONFIGURATION Super Learner ***
sl_models = [DecisionTreeClassifier(),
             KNeighborsClassifier(),
             ExtraTreesClassifier(n_estimators=100),
             RandomForestClassifier(n_estimators=100),
             MLPClassifier(
             hidden_layer_sizes=(100, 100), max_iter=400, alpha=1e-4,
             solver='adam', learning_rate_init=0.001, verbose=1, tol=1e-4)
             ]
sl_meta = RandomForestClassifier(n_estimators=100, max_depth=8, min_samples_leaf=4, n_jobs=-1)

#  *** CONFIGURATION FOR STACKING ***

layer1_learners = [('MLP1', MLPClassifier(
        hidden_layer_sizes=(100, 100), max_iter=100, alpha=1e-4,
        solver='sgd', learning_rate_init=0.2, momentum=0.9, tol=1e-4, random_state=1)),
    ('MLP2', MLPClassifier(
        hidden_layer_sizes=(100, 100), max_iter=100, alpha=1e-4,
        solver='adam', learning_rate_init=0.001, tol=1e-4, random_state=0)),
    ('rf_2', KNeighborsClassifier(n_neighbors=5))]

layer2_learners = [('dt_2', DecisionTreeClassifier()),
                   ('rf_2', RandomForestClassifier(n_estimators=50))]
l1_fe = RandomForestClassifier(n_estimators=100)
l2_fe = SVC()
