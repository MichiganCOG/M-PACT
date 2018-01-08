import os

import numpy      as np
import tensorflow as tf


def make_dir(path):
    """ Create a directory if there isn't one already. """
    try:
        os.mkdir(path)
    except OSError:
        pass


def load_checkpoint(model, dataset, experiment_name):
    checkpoints_file = os.path.join('results', model, dataset, experiment_name, 'checkpoints', 'checkpoint')
    f = open(checkpoints_file, 'r')
    filename = f.readline()
    f.close()
    filename = filename.split(' ')[1].split('\"')[1]
    ckpt = np.load(os.path.dirname(os.path.join('results', model, dataset,  experiment_name, 'checkpoints',filename)))


def save_checkpoint():
    return 0
