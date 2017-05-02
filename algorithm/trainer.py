import numpy as np
import itertools, queue, pickle, os
import multiprocessing
from collections import Counter
from util import timing, setup_tmp
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
                unit = SVMUnit(pair, X, Y, config)

                # due to the queue size limit, the svm units need to be dumped to a local folder
                fpath = os.path.join('./tmp', str(_id) + '.pkl')
                with open(fpath, 'wb') as f: pickle.dump(unit, f)
                self.reportSucc.put(fpath)
            except queue.Empty:
                break


class Trainer:
    def __init__(self, data, config):
        self.idx_lookup = {}
        self.data = data
        self.config = config

        self.svm_units = []
        self.evaluator = ClassifierEvaluator

    @setup_tmp
    @timing
    def train(self, data=None):
        """
        Train a multi-class SVM classifier by creating many One-vs-One SVM units.
        Parallelise the training of units for maximum speed
        """
        data = data or self.data

        classes = np.unique(data.index.values)
        pairs = list(itertools.combinations_with_replacement(classes, 2))
        pairs = list(filter(lambda p: p[0] != p[1], pairs))
        taskQueue, reportSucc, reportFailed = multiprocessing.Queue(), multiprocessing.Queue(), multiprocessing.Queue()
        processes = [SVMWorker(taskQueue, reportSucc, reportFailed) for i in range(min(num_cpus, len(pairs)))]

        # Create SVM tasks
        for idx, pair in enumerate(pairs):
            subset = data.loc[list(pair)]
            taskQueue.put([idx, pair, subset.values, subset.index.values, self.config])

        # Workers starting training many One-vs-One SVM classifiers
        list(map(lambda p: p.start(), processes))
        list(map(lambda p: p.join(), processes))

        # Each time a worker returns a SVM unit that represents a one-vs-one SVM classifier
        while True:
            try:
                fpath = reportSucc.get(timeout=0.05)
                with open(fpath, 'rb') as f:
                    self.svm_units.append(pickle.load(f))
            except queue.Empty:
                break

        # Any bad news?
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
    # def cross_validate(self, fold=10):
    #     # TODO
    #     return self

    def predict(self, X):
        predictions = []
        for x in X:
            counter = Counter()
            for unit in self.svm_units:
                pred = unit.svm.predict([x])[0]
                vote = unit.prediction_lookup[pred]
                counter[vote] += 1
            predictions.append(counter.most_common(1)[0][0])

        print(predictions)
        return self
