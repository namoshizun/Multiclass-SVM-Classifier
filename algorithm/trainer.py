import numpy as np
import itertools
import multiprocessing
from util import timing
from svm import SVM
from evaluator import ClassifierEvaluator
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

num_cpus = multiprocessing.cpu_count()


class SVMUnit:
    def __init__(self, id, X, Y, config):
        self.id = id
        self.svm = SVM(**config).fit(X, self.transform_y(Y))
        self.class_lookup = None

    def transform_y(self, Y):
        pass


class Trainer:
    def __init__(self, data, config):
        self.idx_lookup = {}
        self.data = data
        self.config = config

        self.svm_units = []
        self.evaluator = ClassifierEvaluator

    def transform_y(self, Y):
        """
        convert nominal labels to float digitals for the purpose of matrix computation
        :param Y: nominal labels
        :return: digitalised Y labels
        """
        classes = np.unique(Y)
        cls_lookup = {
            'Education': 1.,
            'Media and Video': -1.,
        }

        vfunc = np.vectorize(lambda cls: cls_lookup[cls])
        return vfunc(Y)

    @timing
    def train(self):
        X, Y = self.data.ix[:, :-1].values, self.transform_y(self.data['labels'].values)
        self.classifier.fit(X, Y)
        return self

    def _train(self):
        """
        Train a multi-class SVM classifier by creating many One-vs-One SVM units.
        Parallelise the training of units for maximum speed
        """
        def add_svm_unit():
            pass

        X, Y = self.data.ix[:, :-1].values, self.data['labels'].values
        classes = np.unique(Y)
        pairs = list(itertools.combinations_with_replacement(classes, 2))
        pairs = list(filter(lambda p: p[0] != p[1], pairs))

        pool = multiprocessing.Pool(min(len(pairs), num_cpus))
        pool.map(add_svm_unit, [{

        } for p in pairs])


    @timing
    def split_validate(self, schedule=(0.9,)):
        for train_size in schedule:
            X, Y = self.data.ix[:, :-1].values, self.transform_y(self.data['labels'].values)
            X_train, X_test, Y_train, Y_test = train_test_split(X, Y,
                                                                train_size=train_size)

            predictions = SVM(**self.config).fit(X_train, Y_train).predict(X_test)
            print('TrainingDataSize = {}, Accuracy: {}'
                  .format(train_size, accuracy_score(Y_test, predictions)))

        return self

    def cross_validate(self, fold=10):
        # TODO
        return self

    def predict(self, X):
        print(self.classifier.predict(X))
        return self
