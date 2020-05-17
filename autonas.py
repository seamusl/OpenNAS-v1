# Copyright (c) 2020 Seamus Lankford
# Licensed under MIT License

# AUTONAS
#
# -- PREPROCESSOR (pre-processing of image dataset)
#
# ---- TRAINER   LOADER   PRETRAIN   HYBRID   ENSEMBLER   SUPER_STACKER
# one of 6 modes chosen on the command line from -mode switch
#
# ------ AUTOKERAS   AUTOMODEL   PSO   ACO   VGG16   VGG19   RESNET50
# one of 7 optimizations chosen on the command line from -optimize switch
#
# -------- NEW MODELS   EXISTING MODELS
# new models generated or existing models loaded using -m_name switch
# new models generated in TRAINER, PRETRAIN, HYBRID or ENSEMBLER modes
# existing models generated in LOADER or HYBRID modes

from __future__ import absolute_import

import warnings
warnings.filterwarnings('ignore')

import os
os.environ['PYTHONHASHSEED'] = '0'

# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'   # use CPU only
# following will lead to reproducible but slow performance
# os.environ['TF_DETERMINISTIC_OPS'] = '1'
# os.environ['HOROVOD_FUSION_THRESHOLD']='0'

# os.environ['TF_CUDNN_DETERMINISTIC'] = '1' # (Hasn't been tested)

# ui method has to be called at this stage so that user specified seed can be used for RNGs
from ui import process_input
ui = process_input()

# aco deepswarm library search uses python random seed, therefore it must be set
import random as rn
rn.seed(ui.my_seed)

# numpy random seed previously set for other optimizer functions
import numpy as np
np.random.seed(ui.my_seed)

import tensorflow as tf
# following will lead to reproducible but slow performance
# from tfdeterminism import patch
# patch()
tf.compat.v1.set_random_seed(ui.my_seed)

import sys
import time
import datetime
from datetime import datetime

from loader import load_autonas
from trainer import train_autonas


# log all program output
class Logger(object):
    def __init__(self):
        self.terminal = sys.stdout
        self.log = open("autonas.log", "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        # python 3 compatibility: this handles the flush command by doing nothing.
        pass


def main():

    sys.stdout = Logger()

    # log all user choices specified on the command line by printing the vars() of the object
    print(vars(ui))

    start = time.time()
    now = datetime.now()                    # current date and time
    print("+++ Running autonas +++", now)

    if ui.mode == 'load':
        load_autonas(ui)        # invoke autonas loader
    else:
        train_autonas(ui)

    runtime = time.time() - start
    print("+++ Completed autonas +++ in ", runtime, ' seconds')

    return


if __name__ == '__main__':
    main()
