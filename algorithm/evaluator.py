from __future__ import print_function
from __future__ import division
from functools import reduce
from prettytable import PrettyTable


class ClassifierEvaluator:
    @staticmethod
    def accuracy_score(predictions, real_values):
        assert(len(predictions) == len(real_values))
        matches = reduce(lambda good, curr: good + int(curr[0] == curr[1]), zip(predictions, real_values), 0)
        return matches / len(predictions)

    @staticmethod
    def build_confusion_matrix(predictions, real_values):
        return None

    @staticmethod
    def build_report(predictions, real_values):
        return None
