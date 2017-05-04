import numpy as np
import itertools, queue, pickle, os
import multiprocessing
import util
from collections import Counter
from svm import SVM
from evaluator import ClassifierEvaluator
from util import timing, setup_tmp, chunkify, one_vs_one_pairs, one_vs_rest_pairs


num_cpus = multiprocessing.cpu_count()


def make_svm_unit(params):
    _id, pair, X, Y, config, mailbox = params['_id'], params['pair'], params['X'], params['Y'], params['config'], params['mailbox']
    unit = SVMUnit(pair, X, Y, config)
    fpath = os.path.join('./tmp', str(_id) + '.pkl')
    with open(fpath, 'wb') as f: pickle.dump(unit, f)
    mailbox.put(fpath)


class SVMUnit:
    def __init__(self, pair, X, Y, config):
        self.class_lookup = dict({kls: 1. for kls in pair[0]}, **{kls: -1. for kls in pair[1]})
        self.prediction_lookup = {1. : pair[0], -1. : pair[1] }
        self.strategy = config['strategy']
        self.svm = SVM(**config).fit(X, self.transform_y(Y))

    def lookup_prediction(self, pred):
        if self.strategy == 'one_vs_one':
            return self.prediction_lookup[pred][0]

        if self.strategy == 'one_vs_rest':
            return self.prediction_lookup[1.][0]

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
                subset = data.loc[p[0] + p[1]]
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
        pairs = getattr(util, self.config['strategy'] + '_pairs')(classes)

        # create worker processes
        num_workers = min(num_cpus, len(pairs))
        pool = multiprocessing.Pool(num_workers)
        pmanager = multiprocessing.Manager()
        mailbox = pmanager.Queue()

        # workers starting training many One-vs-N SVM classifiers
        pool.map(make_svm_unit, create_packets(pairs, mailbox))

        # each time a worker returns the pickle path to the trained SVM unit
        while True:
            try:
                fpath = mailbox.get(timeout=0.05)
                with open(fpath, 'rb') as f:
                    self.svm_units.append(pickle.load(f))
            except queue.Empty:
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

    def __one_vs_one_predict__(self, X):
        predictions = []
        for x in X:
            counter = Counter()
            for unit in self.svm_units:
                pred = unit.svm.predict([x])[0]
                vote = unit.lookup_prediction(pred)
                counter[vote] += 1
            predictions.append(counter.most_common(1)[0][0])

        return predictions

    def __one_vs_rest_predict__(self, X):
        predictions = []
        for x in X:
            margins = [unit.svm.predict([x])[0] for unit in self.svm_units]
            idx_max = np.argmax(margins)
            predictions.append(self.svm_units[idx_max].lookup_prediction(margins[idx_max]))
        return predictions


    def predict(self, X):
        """
        cast projections over testing data using the major vote strategy
        """
        return {
            'one_vs_one': self.__one_vs_one_predict__,
            'one_vs_rest': self.__one_vs_rest_predict__
        }.get(self.config['strategy'])(X)
