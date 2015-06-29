#!/usr/bin/env python

from __future__ import division
from sklearn.cross_validation import StratifiedKFold
from sklearn.metrics import roc_auc_score as AUC

import argparse
import logging
import numpy as np
import os
import time

from kaggler.data_io import load_data

import xgboost as xgb


def train_predict(train_file, test_file, predict_valid_file, predict_test_file,
                  n_est=100, lrate=.1, l1=.0, l2=.0, n_fold=5):

    dir_feature = os.path.dirname(train_file)
    dir_val = os.path.dirname(predict_valid_file)

    feature_name = os.path.basename(train_file)[:-4]
    algo_name = 'xgl_{}_{}_{}_{}'.format(n_est, lrate, l1, l2)
    model_name = '{}_{}'.format(algo_name, feature_name)
    logging.basicConfig(format='%(asctime)s   %(levelname)s   %(message)s',
                        level=logging.DEBUG,
                        filename='{}.log'.format(model_name))

    param = {'eta': lrate,
             'objective': 'binary:logistic',
             'colsample_bytree': .7,
             'subsample': .5,
             'eval_metric': 'auc',
             'seed': 2015,
             'booster': 'gblinear',
             'alpha': l1,
             'lambda': l2}

    logging.info('Loading training and test data...')
    X, y = load_data(train_file)
    X_tst, _ = load_data(test_file)

    cv = StratifiedKFold(y, n_folds=n_fold, shuffle=True, random_state=2015)

    p_val = np.zeros_like(y)
    for i, (i_trn, i_val) in enumerate(cv, 1):
        logging.info('Training model #{}...'.format(i))
        dtrain = xgb.DMatrix(X[i_trn], label=y[i_trn])
        dvalid = xgb.DMatrix(X[i_val], label=y[i_val])
        watchlist = [(dvalid, 'eval'), (dtrain, 'train')]

        clf = xgb.train(param, dtrain, n_est, watchlist)

        p_val[i_val] = clf.predict(dvalid)
        logging.info('AUC TRN = {:.6f}'.format(AUC(y[i_trn], clf.predict(dtrain))))
        logging.info('AUC VAL = {:.6f}'.format(AUC(y[i_val], p_val[i_val])))

    logging.info('AUC = {:.6f}'.format(AUC(y, p_val)))

    logging.info('Retraining with 100% data...')
    dtrain = xgb.DMatrix(X, label=y)
    dtest = xgb.DMatrix(test_file)
    watchlist = [(dtrain, 'train')]

    clf = xgb.train(param, dtrain, n_est, watchlist)
    p_tst = clf.predict(dtest)

    logging.info('Saving predictions...')
    np.savetxt(predict_valid_file, p_val, fmt='%.6f')
    np.savetxt(predict_test_file, p_tst, fmt='%.6f')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-file', required=True, dest='train_file')
    parser.add_argument('--test-file', required=True, dest='test_file')
    parser.add_argument('--predict-valid-file', required=True,
                        dest='predict_valid_file')
    parser.add_argument('--predict-test-file', required=True,
                        dest='predict_test_file')
    parser.add_argument('--n-est', type=int, dest='n_est')
    parser.add_argument('--l1', type=float, dest='l1')
    parser.add_argument('--l2', type=float, dest='l2')
    parser.add_argument('--lrate', type=float, dest='lrate')

    args = parser.parse_args()

    start = time.time()
    train_predict(train_file=args.train_file,
                  test_file=args.test_file,
                  predict_valid_file=args.predict_valid_file,
                  predict_test_file=args.predict_test_file,
                  n_est=args.n_est,
                  l1=args.l1,
                  l2=args.l2,
                  lrate=args.lrate)
    logging.info('finished ({:.2f} min elasped)'.format((time.time() - start) /
                                                        60))
