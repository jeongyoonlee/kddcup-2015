#!/usr/bin/env python

from __future__ import division
from datetime import datetime
from sklearn.cross_validation import StratifiedKFold
from sklearn.datasets import dump_svmlight_file
from sklearn.metrics import roc_auc_score as AUC

import argparse
import logging
import numpy as np
import os
import subprocess
import time

from kaggler.data_io import load_data


def train_predict(train_file, test_file, predict_valid_file, predict_test_file,
                  n_iter=100, dim=4, lrate=.1, n_fold=5):

    dir_feature = os.path.dirname(train_file)
    dir_val = os.path.dirname(predict_valid_file)

    feature_name = os.path.basename(train_file)[:-8]
    algo_name = 'libfm_{}_{}_{}'.format(n_iter, dim, lrate)
    model_name = '{}_{}'.format(algo_name, feature_name)

    logging.basicConfig(format='%(asctime)s   %(levelname)s   %(message)s',
                        level=logging.DEBUG, filename='{}.log'.format(model_name))

    logging.info('Loading training data')
    X, y = load_data(train_file)
    n_tst = sum(1 for line in open(test_file))

    cv = StratifiedKFold(y, n_folds=n_fold, shuffle=True, random_state=2015)

    p_val = np.zeros_like(y)
    for i, (i_trn, i_val) in enumerate(cv, 1):
        logging.info('Training model #{}...'.format(i))
        valid_train_file = os.path.join(dir_feature, '{}.trn{}.sps'.format(feature_name, i))
        valid_test_file = os.path.join(dir_feature, '{}.val{}.sps'.format(feature_name, i))
        valid_predict_file = os.path.join(dir_val, '{}.val{}.yht'.format(model_name, i))

        # if there is no CV training or validation file, then generate them
        # first.
        if (not os.path.isfile(valid_train_file) or not os.path.isfile(valid_test_file)):
            dump_svmlight_file(X[i_trn], y[i_trn], valid_train_file,
                               zero_based=False)
            dump_svmlight_file(X[i_val], y[i_val], valid_test_file,
                               zero_based=False)

        subprocess.call(["libFM",
                         "-task", "c",
                         '-dim', '1,1,{}'.format(dim),
                         '-init_stdev', str(lrate),
                         '-iter', str(n_iter),
                         '-train', valid_train_file,
                         '-test', valid_test_file,
                         '-out', valid_predict_file])

        p_val[i_val] = np.loadtxt(valid_predict_file)
        os.remove(valid_predict_file)

    logging.info('AUC = {:.6f}'.format(AUC(y, p_val)))
    np.savetxt(predict_valid_file, p_val, fmt='%.6f')

    logging.info('Retraining with 100% data...')
    subprocess.call(["libFM",
                     "-task", "c",
                     '-dim', '1,1,{}'.format(dim),
                     '-init_stdev', str(lrate),
                     '-iter', str(n_iter),
                     '-train', train_file,
                     '-test', test_file,
                     '-out', predict_test_file])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-file', required=True, dest='train_file')
    parser.add_argument('--test-file', required=True, dest='test_file')
    parser.add_argument('--predict-valid-file', required=True,
                        dest='predict_valid_file')
    parser.add_argument('--predict-test-file', required=True,
                        dest='predict_test_file')
    parser.add_argument('--n-iter', type=int, dest='n_iter')
    parser.add_argument('--dim', type=int, dest='dim')
    parser.add_argument('--lrate', type=float, dest='lrate')

    args = parser.parse_args()

    start = time.time()
    train_predict(train_file=args.train_file,
                  test_file=args.test_file,
                  predict_valid_file=args.predict_valid_file,
                  predict_test_file=args.predict_test_file,
                  n_iter=args.n_iter,
                  dim=args.dim,
                  lrate=args.lrate)
    logging.info('finished ({:.2f} min elasped)'.format((time.time() - start) /
                                                        60))
