import argparse
import pandas as pd
from util import build_dataframe, make_training_data, feature_selection
from trainer import Trainer

def read_small_data():
    source = '../input-dev/subset_2kls.csv'
    return pd.read_csv(source, index_col=None), None


def read_full_data():
    training_data = '../input/training_data.csv'
    training_labels = '../input/training_labels.csv'
    test_data = '../input/test_data.csv'

    training_data = make_training_data(training_data, training_labels)
    feature_selection(training_data)

    test = build_dataframe(test_data)

    return training_data, test

def read_mock_data():
    training_data = '../input-dev/mock_training.csv'
    training_labels = '../input-dev/mock_training_labels.csv'
    test_data = '../input-dev/mock_test.csv'
    test_labels = '../input-dev/mock_test_label.csv'
    training_data = make_training_data(training_data, training_labels)
    test = build_dataframe(test_data)

    return training_data, test

if __name__ == '__main__':
    # training_data = '../input/training_data.csv'
    # training_labels = '../input/training_labels.csv'
    # test_data = '../input/test_data.csv'

    # DATA SOURCE

    # RECEIVE CONFIG
    parser = argparse.ArgumentParser(description='SVM Classifier')
    parser.add_argument('kernel', nargs='?', type=str, default='linear', help='The kernel function to use')
    parser.add_argument('C', nargs='?', type=float, default=1.0, help='The penalty constant C value')
    parser.add_argument('min_lagmult', nargs='?', type=float, default=1e-5,
                        help='The support vector\'s minimum lagrange multipliers value')
    config = vars(parser.parse_args())
    # training_data, test = read_mock_data()
    training_data, test = read_small_data()

    # HAVE FUN!
    trainer = Trainer(training_data, config)
    # trainer.train().predict(test.values)

    trainer.split_validate()
    # trainer.cross_validate()

