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


def read_full_data(evaluate_features=False):
    """
    Read from the original assignment data and perform data pre-processing
    """
    training_data = '../input/training_data.csv'
    training_labels = '../input/training_labels.csv'
    test_data = '../input/test_data.csv'

    training_data = make_training_data(training_data, training_labels)
    test_data = build_dataframe(test_data)
    feature_selection(training_data, use_cache=not evaluate_features)
    feature_selection(test_data, use_cache=not evaluate_features)

    return training_data, test_data


def save_predictions(predictions, test_data):
    df = pd.DataFrame(predictions, index=test_data.index)
    df.to_csv('../output/predicted_labels.csv', header=None)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SVM Classifier')
    parser.add_argument('kernel', nargs='?', type=str, default='poly', choices=['linear', 'poly'],
                        help='The kernel function to use')
    parser.add_argument('strategy', nargs='?', type=str, default='one_vs_one', choices=['one_vs_one', 'one_vs_rest'],
                        help='The strategy to implement a multiclass SVM. Choose "one_vs_one" or "one_vs_rest"')
    parser.add_argument('C', nargs='?', type=float, default=1.0,
                        help='The regularization parameter that trades off margin size and training error')
    parser.add_argument('min_lagmult', nargs='?', type=float, default=1e-5,
                        help='The support vector\'s minimum Lagrange multipliers value')
    parser.add_argument('cross_validate', nargs='?', type=bool, default=False,
                        help='Whether or not to cross validate SVM')
    parser.add_argument('evaluate_features', nargs='?', type=bool, default=False,
                        help='Will read the cache of feature evaluation results if set to False')
    parser.add_argument('mode', nargs='?', type=str, default='prod', choices=['dev', 'prod'],
                        help='Reads dev data in ../input-dev/ if set to dev mode, otherwise looks for datasets in ../input/')
    config = vars(parser.parse_args())
    svm_params = {k: config[k] for k in ('kernel', 'strategy', 'C', 'min_lagmult')}


    if config['mode'] == 'dev':
        training_data, test_data = read_dev_data()
    elif config['mode'] == 'prod':
        training_data, test_data = read_full_data(config['evaluate_features'])


    trainer = Trainer(training_data, svm_params)


    if config['cross_validate']:
        trainer.cross_validate()
    else:
        print('===== training SVM units =====')
        trainer.train()
        print('===== predicting test data =====')
        predictions = trainer.predict(test_data.values)
        save_predictions(predictions, test_data)
