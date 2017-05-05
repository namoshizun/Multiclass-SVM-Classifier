import itertools
import pandas as pd
import numpy as np
import os, time, shutil
from collections import Counter, defaultdict
from numba import jit
from prettytable import PrettyTable


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


def feature_selection(training_dataframe, min_ig=0.0107036, use_cache=False):
    """
    Compute information gain to remove redundant features.
    Reference: [A survey of text classification algorithms](www.time.mk/trajkovski/thesis/text-class.pdf)
    """
    print('===== selecting usefull features [usecache={}] ====='.format(use_cache))
    features = training_dataframe.columns
    if use_cache:
        features_ig = read_info_gain()
    else:
        features_ig = compute_info_gains(training_dataframe, save=True)

    selected_features = [ft for ft, ig in features_ig.items() if ig >= min_ig]
    redundant_features = list(set(features) - set(selected_features))
    training_dataframe.drop(redundant_features, axis=1, inplace=True)


def read_info_gain(path='./information_gain_result.txt'):
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
    from operator import itemgetter
    pt = PrettyTable()
    pt.add_column('feature name', list(result.keys()))
    pt.add_column('information gain', list(result.values()))
    pt.reversesort = True
    pt.sortby = 'information gain'

    with open(path, 'w+') as thefile:
        thefile.write(pt.get_string())


@timing
def compute_info_gains(training_dataframe, save=False):
    def entropy(vals):
        if len(vals) == 0:
            return 0.0
        # preprocessing - convert nonimal to numeral so the efficient bincount can be used
        cnt = Counter(vals)
        for i, k in enumerate(cnt.keys()):
            cnt[k] = i
        trans = np.vectorize(lambda kls: cnt[kls])(vals)
        # begin
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