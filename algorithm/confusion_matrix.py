from __future__ import print_function
from __future__ import division
import pandas as pd
import numpy as np
import os, time
from prettytable import PrettyTable

class ConfusionMatrix:
    def __init__(self, klasses):
        self.cm = pd.DataFrame(0, index=klasses, columns=klasses)
        self.stats = None


    def update(self, predictions, real_vals):
        for pred, actual in zip(predictions, real_vals):
            self.cm.loc[actual, pred] += 1

    def statistics(self, rebuild=False):
        if self.stats and not rebuild:
            return self.stats.get_string()

        columns = ['TP_Rate', 'FP_Rate', 'Precision', 'Recall', 'F-measure', 'Class']
        cm, pt = self.cm, PrettyTable()
        pt.field_names = columns

        for actual, predictions in cm.iterrows():
            N = cm.loc[actual].sum()  # number of the klss
            M = cm[actual].sum()     # number instance classifed as the klass

            tp = cm.loc[actual, actual]
            fp = (M - cm.loc[actual, actual])
            fn = N - tp
            tn = M - fp
            precision = tp / (tp + fp)
            recall = tp / (tp + fn)
            f_measure = 2 * fp / (2 * fp + fn + fp)

            pt.add_row([tp / N, fp / M, precision, recall, f_measure, actual])

        self.stats = pt
        return self.stats.get_string()


    def save(self, folder):
        # Save confusion matrix
        self.cm.to_csv(os.path.join(folder, 'confusion_matrix.txt'))
        # Save detailed stats summary
        with open(os.path.join(folder, 'accuracy_measurements.txt'), 'w+') as outfile:
            outfile.write(self.statistics())


    def __repr__(self):
        return self.cm.__str__()
