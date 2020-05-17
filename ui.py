# Copyright (c) 2020 Seamus Lankford
# Licensed under MIT License

# SUPER STACKING
# superlearner call
# python autonas.py -mode super_stacker --superlearner 1 --stack 0 -d fashion_mnist
# python autonas.py -mode super_stacker --superlearner 1 --stack 0 -d cifar10

# stacking  calls
# set layer estimators in config prior to running

# single layer on fashion_mnist ...
# python autonas.py -mode super_stacker --stack 1 -d fashion_mnist --cf 2

# double layers on cifar10 ...
# python autonas.py -mode super_stacker --stack 2 -d cifar10 --cf 2

# ENSEMBLE
# use existing base learners:
# when creating ensembles of pre-trained, must use -optimize hybrid (grayscale -> RGB in pre_processor)
# python autonas.py -mode ensemble -d fashion_mnist -n_channel 1 -optimize hybrid --learners 0 --n_base 4

# non hybrid fashion homogeneous ensembles:
# python autonas.py -mode ensemble -d fashion_mnist -n_channel 1 --learners 0 --n_base 4

# all cifar10 ensembles (homogeneous or heterogenous)
# python autonas.py -mode ensemble -d cifar10 -n_channel 3 --learners 0 --n_base 4

# create new base learners:
# python autonas.py -mode ensemble -d cifar10 -n_channel 3 --learners 1 --n_base 2 --epochs 5 --m_name xxx
# python autonas.py -mode ensemble -d fashion_mnist -n_channel 1 --learners 1 --n_base 2 --epochs 5 --m_name xxx


# ---- AUTOMODEL TRAINING ---- #
# To develop models for cifar10 (RGB):
# python autonas.py -mode train -optimize automodel -d cifar10 -n_channel 3 --ak_model 1 --ak_trial 3 --epoch 10 --ak_seek resnet

# ---- PRETRAIN MODE---- #
# python autonas.py -mode pretrain -optimize VGG16 -d fashion_mnist -n_channel 1 --second_model true --my_seed 5 --cf 5
# python autonas.py -mode pretrain -optimize VGG16 -d cifar10 -n_channel 3 --second_model true --my_seed 246810 --cf 3

# python autonas.py -mode pretrain_hybrid -optimize VGG16 -d cifar10 -n_channel 3 --my_seed 246810 --epochs 2
# python autonas.py -mode pretrain -optimize VGG16 -d cifar10 -n_channel 3 --tune mlp --my_seed 246810 --cf 3 --my

# PSO TESTS - the default (using epochs = 5):
# python autonas.py -mode train -optimize pso -d cifar10 --pso_runs 5 --pso_iter 20 --pso_pop 10 --my_seed 10
# python autonas.py -mode train -optimize pso -d cifar10 --pso_runs 5 --pso_iter 10 --pso_pop 20 --my_seed 10

# python autonas.py -mode load -optimize pso -d cifar10 -n_channel 3 --m_name cifar10-model_number_0

# ACO TESTS - the default:
# python autonas.py -mode train -optimize aco -d cifar10 --m_depth 20 --epochs 15 --n_ant 16 --my_seed 246810
# python autonas.py -mode train -optimize aco -d fashion_mnist --m_depth 20 --epochs 30 --n_ant 8 --my_seed 246810

# ---- AUTOKERAS TRAINING ---- 

# To develop models for cifar10 (RGB):
# python autonas.py -mode train -optimize autokeras -d cifar10 -n_channel 3 --ak_model 1 --ak_trial 3 --epoch 10 --my_seed 246810

# To develop models for fashion_mnist (grayscale):
# python autonas.py -mode train -optimize autokeras -d fashion_mnist -n_channel 1 --ak_model 1 --ak_trial 3 --epoch 10

import warnings
warnings.filterwarnings('ignore')

import argparse
import sys


class Choices:
    def __init__(self, mode, optimize, dataset, n_channel, second_model, tune, m_name, epochs,
                 my_seed, cf,
                 ak_trial, ak_model, ak_seek,
                 pso_runs, pso_iter, pso_pop,
                 m_depth, n_ant,
                 learners, n_base,
                 superlearner, stack):

        self.mode = mode
        self.optimize = optimize
        self.dataset = dataset
        self.n_channel = n_channel
        self.second_model = second_model
        self.tune = tune
        self.m_name = m_name
        self.epochs = epochs                # epochs used to train ak, pso, aco and ensemble models

        self.my_seed = my_seed
        self.cf = cf

        self.ak_trial = ak_trial
        self.ak_model = ak_model
        self.ak_seek = ak_seek

        self.pso_runs = pso_runs
        self.pso_iter = pso_iter
        self.pso_pop = pso_pop

        self.m_depth = m_depth
        self.n_ant = n_ant

        self.learners = learners
        self.n_base = n_base

        self.superlearner = superlearner
        self.stack = stack


def process_input():

    # Create the parser
    my_parser = argparse.ArgumentParser(prog='autonas',
                                        description='Using Neural Architecture Search, find the optimal architecture')
    # Add the arguments
    my_parser.add_argument('-mode', type=str,
                           help='options are: pretrain, train, load, ensemble')

    my_parser.add_argument('-optimize', type=str,
                           help='choose one of the following optimisation methods: '
                                'autokeras, automodel, aco, pso, VGG16, VGG19, ResNet50, Xception, InceptionV3')

    my_parser.add_argument('-d', type=str,
                           help='choose a  dataset e.g. fashion_mnist, cifar10 etc ..')

    my_parser.add_argument('-n_channel', type=int,
                           help='number of channels: 1 for grayscale, 3 for RGB')

    my_parser.add_argument('--m_name', type=str,
                           help='the path of the model to be loaded')

    my_parser.add_argument('--second_model', default=False,
                           type=lambda x: (str(x).lower() == 'true'),
                           help='True to run secondary models, otherwise False')

    my_parser.add_argument('--tune', type=str, default=False,
                           help='True for tuning secondary models otherwise, False')

    my_parser.add_argument('--epochs', type=int,
                           help='number of epochs to train on')

    my_parser.add_argument('--my_seed', type=int,
                           help='set the seed value for random number generator ')

    my_parser.add_argument('--cf', type=int,
                           help='specify the number of cross folds to use for validation')

    my_parser.add_argument('--ak_model', type=int,
                           help='number of models which autokeras will build ')

    my_parser.add_argument('--ak_trial', type=int,
                           help='number of trials which autokeras will run')

    my_parser.add_argument('--ak_seek', type=str,
                           help='the search space for automodel (note required for autokeras)')

    my_parser.add_argument('--pso_runs', type=int,
                           help='number of runs for pso search')

    my_parser.add_argument('--pso_iter', type=int,
                           help='number of iterations for pso search')

    my_parser.add_argument('--pso_pop', type=int,
                           help='population size for pso search')

    my_parser.add_argument('--m_depth', type=int,
                           help='depth of search for aco search')

    my_parser.add_argument('--n_ant', type=int,
                           help='number of ants to create for aco search')

    my_parser.add_argument('--learners', type=int,
                           help='create new learner models for the meta learner i.e. 1, otherwise 0')

    my_parser.add_argument('--n_base', type=int,
                           help='number of base learners to create for meta learner')

    my_parser.add_argument('--superlearner', type=int,
                           help='cross validated stacking with a super learner')

    my_parser.add_argument('--stacking', type=int,
                           help='specify number of levels to stack: 0, 1 or 2')

    if (len(sys.argv)) < 6:
        my_parser.print_help()
        sys.exit(0)

    # Execute the parse_args() method
    args = my_parser.parse_args()

    mode = args.mode                # train or load mode
    optimize = args.optimize        # autokeras, automodel, aco, pso, VGG16, VGG19, ResNet50, Xception, InceptionV3
    dataset = args.d
    n_channel = args.n_channel
    second_model = args.second_model
    tune = args.tune

    m_name = args.m_name
    epochs = args.epochs
    my_seed = args.my_seed
    cf = args.cf

    # initialise autokeras parameters
    ak_model = args.ak_model
    ak_trial = args.ak_trial
    ak_seek = args.ak_seek

    # initialise pso parameters
    pso_runs = args.pso_runs
    pso_iter = args.pso_iter
    pso_pop = args.pso_pop

    # initialise aco parameters
    m_depth = args.m_depth
    n_ant = args.n_ant

    # initialise ensemble parameters
    learners = args.learners
    n_base = args.n_base

    # initialise superlearner and stacking parameters
    superlearner = args.superlearner
    stacking = args.stacking

    ui = Choices(mode, optimize, dataset, n_channel, second_model, tune, m_name, epochs,
                 my_seed, cf,
                 ak_trial, ak_model, ak_seek,
                 pso_runs, pso_iter, pso_pop,
                 m_depth, n_ant,
                 learners, n_base,
                 superlearner, stacking
                 )
    
    return ui
