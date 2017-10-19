"""
helper.py
"""
__author__ = "giorgio@ac.upc.edu"

import time
import os
import json
import sys
import argparse
from collections import OrderedDict
import numpy as np


def experiment_load():
    parser = argparse.ArgumentParser()
    parser.add_argument("CLUSTER")
    parser.add_argument("EXPERIMENT")
    args = parser.parse_args()

    # let experiment type (function) come from commandline arg
    with open(args.EXPERIMENT) as jconfig:
        DDPG_config = json.load(jconfig)

    DDPG_config['CLUSTER'] = args.CLUSTER
    DDPG_config['EXPERIMENT'] = args.EXPERIMENT.lower().split('.')[0]

    if DDPG_config['CLUSTER'] == 'local':
        import experiment.local
        runwrapper = experiment.local.runwrapper
        DDPG_config['EXPERIMENT'] = setup_exp(DDPG_config['EXPERIMENT'])

    return DDPG_config, runwrapper


def setup_exp(experiment=''):
    folder = 'runs/'
    os.makedirs(folder, exist_ok=True)

    folder += experiment + '/'
    os.makedirs(folder, exist_ok=True)

    return folder


def setup_run(DDPG_config):
    folder = DDPG_config['EXPERIMENT']
    epoch = 't%.6f/' % time.time()
    folder += epoch.replace('.', '')
    os.makedirs(folder, exist_ok=True)

    with open(folder + 'folder.ini', 'w') as ifile:
        ifile.write('[General]\n')
        ifile.write('**.folderName = "' + folder + '"\n')

    with open(folder + 'DDPG.json', 'w') as jconfig:
        json.dump(OrderedDict(sorted(DDPG_config.items(), key=lambda t: t[0])), jconfig, indent=4)

#     with open(folder + 'Routing.txt', 'w') as rfile:
#         rfile.write(DDPG_config['U_ROUTING'] + '\n')

    if DDPG_config['TRAFFIC'].startswith('STAT:'):
        with open(folder + 'Traffic.txt', 'w') as rfile:
            rfile.write(DDPG_config['TRAFFIC'].split('STAT:')[-1] + '\n')

    return folder


def setup_brute(DDPG_config):
    folder = 'runs/brute'
    epoch = 't%.6f/' % time.time()
    folder += epoch.replace('.', '')
    folder += '/'
    os.makedirs(folder, exist_ok=True)

    with open(folder + 'folder.ini', 'w') as ifile:
        ifile.write('[General]\n')
        ifile.write('**.folderName = "' + folder + '"\n')

#     with open(folder + 'Routing.txt', 'w') as rfile:
#         rfile.write(DDPG_config['U_ROUTING'] + '\n')

    if DDPG_config['TRAFFIC'].startswith('STAT:'):
        with open(folder + 'Traffic.txt', 'w') as rfile:
            rfile.write(DDPG_config['TRAFFIC'].split('STAT:')[-1] + '\n')

    with open(folder + 'DDPG.json', 'w') as jconfig:
        json.dump(OrderedDict(sorted(DDPG_config.items(), key=lambda t: t[0])), jconfig, indent=4)

    return folder


def parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("CLUSTER")
    parser.add_argument("EXPERIMENT")
    parser.add_argument("--RSEED", type=int, action="store", default=None)
    parser.add_argument("--PRINT", action="store_true")
    parser.add_argument("--ACTIVE_NODES", type=int, action="store", required=True)
    parser.add_argument("--MU", type=float, action="store", required=True)
    parser.add_argument("--THETA", type=float, action="store", required=True)
    parser.add_argument("--SIGMA", type=float, action="store", required=True)
    parser.add_argument("--BUFFER_SIZE", type=int, action="store", required=True)
    parser.add_argument("--BATCH_SIZE", type=int, action="store", required=True)
    parser.add_argument("--GAMMA", type=float, action="store", required=True)
    parser.add_argument("--TAU", type=float, action="store", required=True)
    parser.add_argument("--LRA", type=float, action="store", required=True)
    parser.add_argument("--LRC", type=float, action="store", required=True)
    parser.add_argument("--EXPLORE", type=float, action="store", required=True)
    parser.add_argument("--EPISODE_COUNT", type=int, action="store", required=True)
    parser.add_argument("--MAX_STEPS", type=int, action="store", required=True)
    parser.add_argument("--HACTI", action="store", required=True)
    parser.add_argument("--HIDDEN1_UNITS", type=int, action="store", required=True)
    parser.add_argument("--HIDDEN2_UNITS", type=int, action="store", required=True)
    parser.add_argument("--TRAFFIC", action="store", required=True)
    parser.add_argument("--STATUM", action="store", required=True)
    parser.add_argument("--PRAEMIUM", action="store", required=True)
    parser.add_argument("--ACTUM", action="store", required=True)
    parser.add_argument("--MAX_DELTA", type=float, action="store", default=None)
    parser.add_argument("--BN", action="store", default=None)
    parser.add_argument("--U_ROUTING", action="store", default=None)
    parser.add_argument("--ROUTING", action="store", required=True)
    parser.add_argument("--ENV", action="store", required=True)

    args = parser.parse_args()

    DDPG_config = vars(args)

    return DDPG_config


def pretty(f):
    try:
        float(f)
        return str.format('{0:.3f}', f).rstrip('0').rstrip('.')
    except:
        return str(f)


def scale(array):
    mean = array.mean()
    std = array.std()
    if std == 0:
        std = 1
    return np.asarray((array - mean)/std)


def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)


def selu(x):
    from keras.activations import elu
    """Scaled Exponential Linear Unit. (Klambauer et al., 2017)

    # Arguments
        x: A tensor or variable to compute the activation function for.

    # References
        - [Self-Normalizing Neural Networks](https://arxiv.org/abs/1706.02515)
    """
    alpha = 1.6732632423543772848170429916717
    scale = 1.0507009873554804934193349852946
    return scale * elu(x, alpha)
