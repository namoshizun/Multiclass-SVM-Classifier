from __future__ import print_function
from __future__ import division
from Queue import Empty
from collections import Counter
from util import timing, setup_tmp, chunkify
from svm import SVM
from evaluator import ClassifierEvaluator
import numpy as np
import itertools, os
import multiprocessing
import cPickle as pickle

num_cpus = multiprocessing.cpu_count()


def make_svm_unit(params):
    _id, pair, X, Y, config, mailbox = params['_id'], params['pair'], params['X'], params['Y'], params['config'], params['mailbox']
    unit = SVMUnit(pair, X, Y, config)
    fpath = os.path.join('./tmp', str(_id) + '.pkl')
    with open(fpath, 'wb') as f: pickle.dump(unit, f)
    mailbox.put(fpath)


class SVMUnit:
    def __init__(self, pair, X, Y, config):
        self.pair = pair
        self.class_lookup = {pair[0]: -1., pair[1]: 1.}
        self.prediction_lookup = {-1.: pair[0], 1.: pair[1]}
        self.svm = SVM(**config).fit(X, self.transform_y(Y))

    def transform_y(self, Y):
        """
        convert labels to (-1, 1)
        """
        vfunc = np.vectorize(lambda kls: self.class_lookup[kls])
        return vfunc(Y)

class Trainer:
    def __init__(self, data, config):
        self.data = data
        self.config = config
        self.svm_units = []
        self.evaluator = ClassifierEvaluator

    @setup_tmp
    @timing
    def train(self, data=None):
        """
        Train a multi-class SVM classifier by creating many One-vs-One SVM units.
        Parallelise the training of units for optimising the performance
        """
        def create_packets(pairs, mailbox):
            ret = []
            for idx, p in enumerate(pairs):
                subset = data.loc[list(p)]
                ret.append({
                    '_id': idx,
                    'pair': p,
                    'X': subset.values,
                    'Y': subset.index.values,
                    'config': self.config,
                    'mailbox': mailbox
                })
            return ret

        if data is None:
            data = self.data

        # get all one-vs-one pairs
        classes = np.unique(data.index.values)
        pairs = list(itertools.combinations_with_replacement(classes, 2))
        pairs = filter(lambda p: p[0] != p[1], pairs)

        # create worker processes
        num_workers = min(num_cpus, len(pairs))
        pool = multiprocessing.Pool(num_workers)
        pmanager = multiprocessing.Manager()
        mailbox = pmanager.Queue()

        # workers starting training many One-vs-One SVM classifiers
        pool.map(make_svm_unit, create_packets(pairs, mailbox))
        # each time a worker returns the pickle path to the trained SVM unit
        while True:
            try:
                fpath = mailbox.get(timeout=0.05)
                with open(fpath, 'rb') as f:
                    self.svm_units.append(pickle.load(f))
            except Empty:
                break

        pool.close()
        pool.join()
        print('done')

    @timing
    def cross_validate(self, data=None, num_folds=10):
        if data is None:
            data = self.data

        num_folds = min(len(data), num_folds)
        folds_idx = chunkify(list(range(len(data))), num_folds)
        folds_accuracy = []

        tmp = set(list(range(num_folds)))
        for i in tmp:
            print('Fold {}'.format(i))
            other_folds = list(tmp - set([i]))
            other_folds_pos = list(itertools.chain.from_iterable([folds_idx[j] for j in other_folds]))

            training_data = data.iloc[other_folds_pos]
            test_data = data.iloc[folds_idx[i]]

            self.train(training_data)
            accuracy = self.evaluator.accuracy_score(self.predict(test_data.values), test_data.index.values)
            self.svm_units = []
            
            print(accuracy)
            folds_accuracy.append(accuracy)

        print('Mean Accuracy : {}'.format(np.mean(folds_accuracy)))
        return self

    def predict(self, X):
        """
        cast projections over testing data using the major vote strategy
        """
        predictions = []
        for x in X:
            counter = Counter()
            for unit in self.svm_units:
                pred = unit.svm.predict([x])[0]
                vote = unit.prediction_lookup[pred]
                counter[vote] += 1
            predictions.append(counter.most_common(1)[0][0])

        return predictions
