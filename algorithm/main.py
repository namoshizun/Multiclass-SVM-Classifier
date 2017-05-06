import argparse
import pandas as pd
from util import build_dataframe, make_training_data, feature_selection
from trainer import Trainer


def read_dev_data():
    """
    Reads pre-built dataframe (dev-data.csv) or the subset (subset_5kls.csv) of original training_data.csv.
    Used for development only
    """
    source = '../input-dev/subset_5kls.csv'
    # source = '../input-dev/dev-data.csv'
    return pd.read_csv(source, index_col=0), None


def read_full_data():
    """
    Read from the original assignment data and perform data pre-processing
    """
    training_data = '../input/training_data.csv'
    training_labels = '../input/training_labels.csv'
    test_data = '../input/test_data.csv'

    training_data = make_training_data(training_data, training_labels)
    test = build_dataframe(test_data)
    feature_selection(training_data, use_cache=True)

    return training_data, test


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SVM Classifier')
    parser.add_argument('kernel', nargs='?', type=str, default='linear',
                        help='The kernel function to use')
    parser.add_argument('strategy', nargs='?', type=str, default='one_vs_one',
                        help='The strategy to implement a multiclass SVM. Choose "one_vs_one" or "one_vs_rest"')
    parser.add_argument('C', nargs='?', type=float, default=1.0,
                        help='The regularization parameter that trades off margin size and training error')
    parser.add_argument('min_lagmult', nargs='?', type=float, default=1e-5,
                        help='The support vector\'s minimum Lagrange multipliers value')
    parser.add_argument('cross_validate', nargs='?', type=bool, default=False,
                        help='Whether or not to cross validate SVM')
    config = vars(parser.parse_args())

    training_data, test = read_full_data()
    trainer = Trainer(training_data, config)

    if config['cross_validate']:
        trainer.cross_validate()
    else:
        trainer.train()
        predictions = trainer.predict(test.values)
        # TODO
