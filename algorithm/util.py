import itertools
import pandas as pd
import numpy as np
import os, time, shutil
from collections import Counter, defaultdict
from prettytable import PrettyTable
from functools import reduce

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


def folds_indexes(df, num_folds):
    """
    get the indexes of dataframe splitted into N folds
    """
    def chunkify(lst, n):
        return [lst[i::n] for i in range(n)]

    folds_idx = []
    chunks_idx = chunkify(list(range(len(df))), num_folds)
    tmp = set(list(range(num_folds)))

    for i in tmp:
        other_folds = list(tmp - set([i]))
        other_folds_pos = list(itertools.chain.from_iterable([chunks_idx[j] for j in other_folds]))
        folds_idx.append((other_folds_pos, chunks_idx[i]))

    return folds_idx


def accuracy_score(predictions, real_values):
    assert(len(predictions) == len(real_values))
    matches = reduce(lambda good, curr: good + int(curr[0] == curr[1]), zip(predictions, real_values), 0)
    return matches / len(predictions)


##############
# Decorators #
##############
def setup_tmp(f):
    """
    Manages the lifetime of 'tmp' folder.
    create at the beginning of function and remove when the function completes
    """
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
    """
    Time the function execution
    """
    def timmer(*args, **kwargs):
        start = time.time()
        ret = f(*args, **kwargs)
        finish = time.time()
        print('%s function took %0.3f s' % (f.__name__, (finish-start)))
        return ret
    return timmer


####################
# Kernel Utilities #
####################
def kernel_function(name, degree=3, offset=1e0):
    return {
        'linear': linear,
        'poly': poly,
        'rbf': rbf
    }.get(name, linear)


def linear(x, y):
    return np.dot(x, y)


def poly(x, y):
    # default degree = 3., default offset = 1e0
    return (1e0 + np.dot(x, y)) ** 3


def rbf(x, y):
    # deprecated. no enough time to implement
    # default sigma = 5.0
    return np.exp(-np.linalg.norm(x-y)**2 / (2 * (5.0 ** 2)))


#################################
# Data pre-processing utilities #
#################################
def build_dataframe(source):
    """
    Read source csv and build a pandas dataframe
    """
    # read source data, use the app name as index
    print('===== building data frame for {} ====='.format(os.path.basename(source)))

    data = pd.read_csv(source, index_col=0, header=None)
    data.index = data.index.rename('app')

    # create semantic columns
    tagged = ['t' + str(i) for i in range(len(data.columns))]
    data.columns = tagged

    return data


def make_training_data(training_path, labels_path):
    """
    Read training and labels source, use labels as dataframe index and
    columns are tagged using prefix 't'.
    """
    # read training and label data, use app name as index
    data = build_dataframe(training_path)
    print('===== making training data =====')

    # read labels and join them together
    labels = pd.read_csv(labels_path, index_col=0, header=None)
    labels.index = labels.index.rename('app')
    labels.columns = ['labels']
    assert(len(data) == len(labels))

    # nice and clean
    data = data.join(labels)
    data.set_index(['labels'], inplace=True)

    return data


def feature_selection(dataframe, features_ig, min_ig=0.0107036):
    """
    Compute information gain and remove redundant features whose info gain values are below min_ig.
    Reference: [A survey of text classification algorithms](www.time.mk/trajkovski/thesis/text-class.pdf)
    """
    print('===== selecting usefull features [usecache={}] ====='.format(use_cache))
    features = dataframe.columns
    selected_features = [ft for ft, ig in features_ig.items() if ig >= min_ig]
    redundant_features = list(set(features) - set(selected_features))
    dataframe.drop(redundant_features, axis=1, inplace=True)


@timing
def compute_info_gains(training_dataframe, save=False):
    """
     Compute info gain value for each of the training data feature (13627 unique words)
    """
    def entropy(vals):
        if len(vals) == 0:
            return 0.0
        # preprocessing - convert nominal to numeral so the efficient bincount can be used
        cnt = Counter(vals)
        for i, k in enumerate(cnt.keys()):
            cnt[k] = i
        trans = np.vectorize(lambda kls: cnt[kls])(vals)
        # count occurrences
        x = np.atleast_2d(trans)
        nrows, ncols = x.shape
        nbins = x.max() + 1
        counts = np.vstack((np.bincount(row, minlength=nbins) for row in x))
        p = counts / float(ncols)
        # compute Shannon entropy in bits
        return -np.sum(p * np.log2(p), axis=1)[0]

    features = training_dataframe.columns
    kls_entropy = entropy(training_dataframe.index.values)
    N = len(training_dataframe)
    word_ig = {}

    print('*** computing info gain ***')
    for word in features:
        selector = training_dataframe[word] > 0
        w_docs =  [i for i, ok in selector.iteritems() if ok]
        nw_docs =  [i for i, ok in selector.iteritems() if not ok]
        F_w = selector.sum() / N
        word_ig[word] = kls_entropy - F_w * entropy(w_docs) - (1 - F_w) * entropy(nw_docs)

    if save:
        save_info_gain(word_ig)
    return word_ig


def read_info_gain(path='./information_gain_result.txt'):
    """
    Read pre-computed info gain values for training_data.csv
    """
    import re
    with open(path, 'r') as source:
        features_ig = {}
        regex=re.compile(r'^|\s+\w+\s+|\s+\d+\.\d+\s+|$')
        lines = source.read().splitlines()[3:]
        for ln in lines:
            parts = regex.findall(ln)
            if len(parts) != 4: continue
            features_ig[parts[1].strip()] = float(parts[2].strip())

        return features_ig


def save_info_gain(result, path='./information_gain_result.txt'):
    """
    Save info gain values for training_data.csv to a local file
    """
    pt = PrettyTable()
    pt.add_column('feature name', list(result.keys()))
    pt.add_column('information gain', list(result.values()))
    pt.reversesort = True
    pt.sortby = 'information gain'

    with open(path, 'w+') as thefile:
        thefile.write(pt.get_string())