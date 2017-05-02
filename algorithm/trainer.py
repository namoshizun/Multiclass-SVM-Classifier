import numpy as np
import itertools, queue, pickle, os
import multiprocessing
from collections import Counter
from util import timing, setup_tmp, chunkify
from svm import SVM
from evaluator import ClassifierEvaluator
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

num_cpus = multiprocessing.cpu_count()


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


class SVMWorker(multiprocessing.Process):
    def __init__(self, taskQueue, reportSucc, reportFailed):
        super(SVMWorker, self).__init__()
        self.taskQueue = taskQueue
        self.reportSucc = reportSucc
        self.reportFailed = reportFailed

    def run(self):
        while True:
            try:
                _id, pair, X, Y, config = self.taskQueue.get(timeout=0.01)
                print('Training on pair {}'.format(pair))
                unit = SVMUnit(pair, X, Y, config)

                # due to the queue size limit, the svm units need to be dumped to a local folder
                fpath = os.path.join('./tmp', str(_id) + '.pkl')
                with open(fpath, 'wb') as f: pickle.dump(unit, f)
                self.reportSucc.put(fpath)
            except queue.Empty:
                break
            except Exception:
                self.reportFailed.put('Job {} failed'.format(_id))


class Trainer:
    def __init__(self, data, config):
        self.data = data
        self.config = config
        self.svm_units = []
        self.evaluator = ClassifierEvaluator

    def make_task_queues(self, data, pairs, num):
        """
        Create SVM task queues, each queue contains the training asset for a SVM worker
        """
        queues = [multiprocessing.Queue() for _ in range(num)]
        for idx, pair in enumerate(pairs):
            qidx = idx % num
            subset = data.loc[list(pair)]
            queues[qidx].put([idx, pair, subset.values, subset.index.values, self.config])
        return queues

    @setup_tmp
    @timing
    def train(self, data=None):
        """
        Train a multi-class SVM classifier by creating many One-vs-One SVM units.
        Parallelise the training of units for optimising the performance
        """
        if data is None:
            data = self.data

        # get all one-vs-one pairs
        classes = np.unique(data.index.values)
        pairs = list(itertools.combinations_with_replacement(classes, 2))
        pairs = list(filter(lambda p: p[0] != p[1], pairs))

        # create worker processes
        num_workers = min(num_cpus, len(pairs))
        reportSucc, reportFailed = multiprocessing.Queue(), multiprocessing.Queue()
        taskQueues = self.make_task_queues(data, pairs, num_workers)
        processes = [SVMWorker(taskQueues[i], reportSucc, reportFailed) for i in range(num_workers)]

        # workers starting training many One-vs-One SVM classifiers
        list(map(lambda p: p.start(), processes))
        list(map(lambda p: p.join(), processes))

        # each time a worker returns the pickle path to the trained SVM unit
        while True:
            try:
                fpath = reportSucc.get(timeout=0.05)
                with open(fpath, 'rb') as f:
                    self.svm_units.append(pickle.load(f))
            except queue.Empty:
                break

        # any bad news?
        while True:
            try:
                failure = reportFailed.get(timeout=0.05)
                print(failure)
            except queue.Empty:
                break
        print('done')

    # @timing
    # def split_validate(self, schedule=(0.9,)):
    #     for train_size in schedule:
    #         X, Y = self.data.ix[:, :-1].values, self.transform_y(self.data['labels'].values)
    #         X_train, X_test, Y_train, Y_test = train_test_split(X, Y,
    #                                                             train_size=train_size)
    #
    #         predictions = SVM(**self.config).fit(X_train, Y_train).predict(X_test)
    #         print('TrainingDataSize = {}, Accuracy: {}'
    #               .format(train_size, accuracy_score(Y_test, predictions)))
    #
    #     return self
    #

    @timing
    def cross_validate(self, data=None, num_folds=10):
        if data is None:
            data = self.data
        num_folds = min(len(data), num_folds)

        folds_idx = chunkify(list(range(len(data))), num_folds)

        tmp = set(list(range(num_folds)))
        for i in tmp:
            print('Fold {}'.format(i))
            other_folds = list(tmp - set([i]))
            other_folds_pos = list(itertools.chain.from_iterable([folds_idx[j] for j in other_folds]))

            training_data = data.iloc[other_folds_pos]
            test_data = data.iloc[folds_idx[i]]

            self.train(training_data)
            accuracy = accuracy_score(self.predict(test_data.values), test_data.index.values)
            self.svm_units = []

            print(accuracy)

        # TODO
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
