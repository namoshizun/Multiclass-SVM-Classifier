from __future__ import print_function
from __future__ import division
import itertools
import pandas as pd
import numpy as np
import os, time, shutil

##################
# MISC Utilities #
##################
def one_vs_one_pairs(lst):
    primitive_pairs =  list(itertools.combinations(lst, 2))
    return [([p[0]], [p[1]]) for p in primitive_pairs]


def one_vs_rest_pairs(lst):
    if len(lst) <= 2:
        return [([lst[0]], [lst[1]])]
    if type(lst) is not list:
        lst = list(lst)

    return [([e], lst[:i] + lst[i + 1:]) for i, e in enumerate(lst)]


def chunkify(lst, n):
    return [lst[i::n] for i in range(n)]


def setup_tmp(f):
    def handler(*args):
        tmp_dump = './tmp'
        if not os.path.exists(tmp_dump):
            os.mkdir(tmp_dump)
        ret = f(*args)
        if os.path.exists(tmp_dump):
            shutil.rmtree(tmp_dump, ignore_errors=True)
        return ret
    return handler


def timing(f):
    def timmer(*args):
        start = time.time()
        ret = f(*args)
        finish = time.time()
        print('%s function took %0.3f s' % (f.__name__, (finish-start)))
        return ret
    return timmer


####################
# Kernel Utilities #
####################
def kernel_function(name):
    return {
        'linear': linear,
        'poly': poly
    }.get(name, linear)


def linear(x, y):
    return np.dot(x, y)


def poly(degree=3, offset=1e0):
    def kernel(x, y):
        return (offset + np.dot(x, y)) ** degree
    return kernel

#################################
# Data pre-processing utilities #
#################################
def feature_selection(training_dataframe):
    # # remove unused features
    features = training_dataframe.columns[:-1]
    selected_features = read_selected_features('./selected_features.txt')  # TODO

    redundant_features = list(set(features) - set(selected_features))
    training_dataframe.drop(redundant_features, axis=1, inplace=True)


def read_selected_features(path):
    with open(path, 'r') as source:
        return source.read().splitlines()


def build_dataframe(source):
    # read source data, use the app name as index
    print('===== building data frame for {} ====='.format(os.path.basename(source)))

    data = pd.read_csv(source, index_col=0, header=None)
    data.index = data.index.rename('app')

    # create semantic columns
    tagged = ['t' + str(i) for i in range(len(data.columns))]
    data.columns = tagged

    return data


def make_training_data(training_path, labels_path):
    # read training and label data, use app name as index
    data = build_dataframe(training_path)

    # read labels and join them together
    labels = pd.read_csv(labels_path, index_col=0, header=None)
    labels.index = labels.index.rename('app')
    labels.columns = ['labels']
    assert(len(data) == len(labels))

    # nice and clean
    data = data.join(labels)
    data.set_index(['labels'], inplace=True)

    return data

