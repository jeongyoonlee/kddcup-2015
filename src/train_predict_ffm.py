#!/usr/bin/env python

from __future__ import division
from datetime import datetime
from sklearn.cross_validation import StratifiedKFold
from sklearn.datasets import dump_svmlight_file, load_svmlight_file
from sklearn.metrics import roc_auc_score as AUC

import argparse
import logging
import numpy as np
import os
import subprocess
import time

from svm_to_ffm import svm_to_ffm


def train_predict(train_file, test_file, train_svm_file,
                  predict_valid_file, predict_test_file, model_file,
                  n_iter=100, dim=4, lrate=.1, n_fold=5):

    dir_feature = os.path.dirname(train_file)
    dir_model = os.path.dirname(model_file)
    dir_val = os.path.dirname(predict_valid_file)

    feature_name = os.path.basename(train_file)[:-8]
    algo_name = 'ffm_{}_{}_{}'.format(n_iter, dim, lrate)
    model_name = '{}_{}'.format(algo_name, feature_name)
    logging.basicConfig(format='%(asctime)s   %(levelname)s   %(message)s',
                        level=logging.DEBUG, filename='{}.log'.format(model_name))

    logging.info('Loading training data')
    X, y = load_svmlight_file(train_svm_file)

    cv = StratifiedKFold(y, n_folds=n_fold, shuffle=True, random_state=2015)

    p = np.zeros_like(y)
    for i, (i_trn, i_val) in enumerate(cv, 1):
        logging.info('Training model #{}...'.format(i))
        valid_train_file = os.path.join(dir_feature, '{}.trn{}.ffm'.format(feature_name, i))
        valid_test_file = os.path.join(dir_feature, '{}.val{}.ffm'.format(feature_name, i))
        valid_model_file = os.path.join(dir_model, '{}.trn{}.mdl'.format(model_name, i))
        valid_predict_file = os.path.join(dir_val, '{}.val{}.yht'.format(model_name, i))

        # if there is no CV training or validation file, then generate them
        # first.
        if (not os.path.isfile(valid_train_file) or not os.path.isfile(valid_test_file)):
            # generate libsvm files
            valid_train_svm_file = os.path.join(dir_feature, '{}.trn{}.sps'.format(feature_name, i))
            valid_test_svm_file = os.path.join(dir_feature, '{}.val{}.sps'.format(feature_name, i))

            if not os.path.isfile(valid_train_svm_file):
                dump_svmlight_file(X[i_trn], y[i_trn], valid_train_svm_file, zero_based=False)

            if not os.path.isfile(valid_test_svm_file):
                dump_svmlight_file(X[i_val], y[i_val], valid_test_svm_file, zero_based=False)

            # then convert libsvm files into libffm formats
            svm_to_ffm(valid_train_svm_file, valid_train_file, feature_name)
            svm_to_ffm(valid_test_svm_file, valid_test_file, feature_name)

        subprocess.call(["ffm-train",
                         '-k', '{}'.format(dim),
                         '-r', str(lrate),
                         '-t', str(n_iter),
                         '-p', valid_test_file,
                         valid_train_file,
                         valid_model_file])

        subprocess.call(["ffm-predict",
                         valid_test_file,
                         valid_model_file,
                         valid_predict_file])

        p[i_val] = np.loadtxt(valid_predict_file)

    logging.info('AUC = {:.4f}'.format(AUC(y, p)))
    np.savetxt(predict_valid_file, p, fmt='%.6f')

    logging.info('Retraining with 100% data...')
    subprocess.call(["ffm-train",
                     '-k', '{}'.format(dim),
                     '-r', str(lrate),
                     '-t', str(n_iter),
                     '-p', test_file,
                     train_file,
                     model_file])

    subprocess.call(["ffm-predict",
                     test_file,
                     model_file,
                     predict_test_file])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-file', required=True, dest='train_file')
    parser.add_argument('--test-file', required=True, dest='test_file')
    parser.add_argument('--train-svm-file', required=True, dest='train_svm_file')
    parser.add_argument('--model-file', required=True, dest='model_file')
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
                  train_svm_file=args.train_svm_file,
                  predict_valid_file=args.predict_valid_file,
                  predict_test_file=args.predict_test_file,
                  model_file=args.model_file,
                  n_iter=args.n_iter,
                  dim=args.dim,
                  lrate=args.lrate)
    logging.info('finished ({:.2f} min elasped)'.format((time.time() - start) /
                                                        60))
