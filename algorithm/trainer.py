import numpy as np
import itertools, queue, pickle, os
import multiprocessing
import util
from collections import Counter
from svm import SVM
from confusion_matrix import ConfusionMatrix
from util import timing, setup_tmp, one_vs_one_pairs, one_vs_rest_pairs
from util import folds_indexes, accuracy_score


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

    def lookup_predictions(self, preds):
        if self.strategy == 'one_vs_one':
            vfunc = np.vectorize(lambda val: self.prediction_lookup[val][0])
            return vfunc(preds)

        if self.strategy == 'one_vs_rest':
            vfunc = np.vectorize(lambda val: self.prediction_lookup[1.][0])
            return vfunc(preds)

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
        if len(data) < num_folds:
            raise ValueError('Not enough data to make {} folds'.format(num_folds))

        folds = folds_indexes(data, num_folds)
        folds_accuracy = []
        cm = ConfusionMatrix(np.unique(data.index.values))

        for i, (train_idx, test_idx) in enumerate(folds):
            # find the indexes of other folds data in the dataframe
            print('Fold {}'.format(i))
            training_data = data.iloc[train_idx]
            test_data = data.iloc[test_idx]
            self.train(training_data)
            # predict and evaluate
            predictions = self.predict(test_data.values)
            real_vals = test_data.index.values
            accuracy = accuracy_score(predictions, real_vals)
            cm.update(predictions, real_vals)
            # clean up
            folds_accuracy.append(accuracy)
            self.svm_units = []
            print(accuracy)

        print('Mean Accuracy : {}'.format(np.mean(folds_accuracy)))
        print('***** Detailed Summary *****')
        print(cm.statistics())
        print('***** Confusion Matrix *****')
        print(cm)
        cm.save(folder='../experiments/')
        return self

    def __one_vs_one_predict__(self, X):
        unit_preds = []
        n = len(X)
        for unit in self.svm_units:
            pred = unit.lookup_predictions(unit.svm.predict(X))
            unit_preds.append(pred)
        unit_preds = np.array(unit_preds)
        return [Counter(unit_preds[:, i]).most_common(1)[0][0] for i in range(n)]

    def __one_vs_rest_predict__(self, X):
        n = len(X)
        unit_margins = np.array([unit.svm.predict(X) for unit in self.svm_units])
        idx_max = [np.argmax(unit_margins[:, i]) for i in range(n)]
        predictions = []

        for i in range(n):
            best = idx_max[i]
            predictions.append(str(self.svm_units[best].lookup_predictions(unit_margins[best, i])))
        
        return predictions

    def predict(self, X):
        """
        cast projections over testing data using the major vote strategy
        """
        return {
            'one_vs_one': self.__one_vs_one_predict__,
            'one_vs_rest': self.__one_vs_rest_predict__
        }.get(self.config['strategy'])(X)
