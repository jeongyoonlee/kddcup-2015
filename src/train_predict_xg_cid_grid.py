#!/usr/bin/env python

from __future__ import division
from sklearn.cross_validation import StratifiedKFold
from sklearn.grid_search import GridSearchCV

import argparse
import logging
import numpy as np
import os
import time

from kaggler.data_io import load_data

import xgboost as xgb


def train_predict(train_file, test_file, predict_test_file,
                  cid_train_file, cid_test_file, n_fold=5):

    feature_name = os.path.basename(train_file)[:-4]
    logging.basicConfig(format='%(asctime)s   %(levelname)s   %(message)s',
                        level=logging.DEBUG,
                        filename='xg_cid_grid_{}.log'.format(feature_name))

    logging.info('Loading course IDs for training and test data')
    cid_trn = np.loadtxt(cid_train_file, dtype=int)
    cid_tst = np.loadtxt(cid_test_file, dtype=int)
    
    logging.info('Loading training and test data...')
    X, y = load_data(train_file)
    X_tst, y_tst = load_data(test_file)

    xg = xgb.XGBClassifier(subsample=0.5, colsample_bytree=0.8, nthread=6)
    param = {'learning_rate': [0.005, .01, .02], 'max_depth': [4, 6, 8],
             'n_estimators': [200, 400, 600]}

    p_tst = np.zeros_like(y_tst)
    for j in range(39):
        idx_trn = np.where(cid_trn == j)[0]
        idx_tst = np.where(cid_tst == j)[0]

        cv = StratifiedKFold(y[idx_trn], n_folds=n_fold, shuffle=True,
                             random_state=2015)
        clf = GridSearchCV(xg, param, scoring='roc_auc', verbose=1, cv=cv)
        clf.fit(X[idx_trn], y[idx_trn])

        logging.info('CID #{}: {:.4f} {}'.format(j, clf.best_score_,
                                                 clf.best_params_))

        logging.info('Retraining with 100% data...')
        clf.best_estimator_.fit(X[idx_trn], y[idx_trn])
        p_tst[idx_tst] = clf.best_estimator_.predict_proba(X_tst[idx_tst])[:, 1]

    logging.info('Saving test predictions...')
    np.savetxt(predict_test_file, p_tst, fmt='%.6f')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-file', required=True, dest='train_file')
    parser.add_argument('--test-file', required=True, dest='test_file')
    parser.add_argument('--cid-train-file', required=True,
                        dest='cid_train_file')
    parser.add_argument('--cid-test-file', required=True,
                        dest='cid_test_file')
    parser.add_argument('--predict-test-file', required=True,
                        dest='predict_test_file')

    args = parser.parse_args()

    start = time.time()
    train_predict(train_file=args.train_file,
                  test_file=args.test_file,
                  cid_train_file=args.cid_train_file,
                  cid_test_file=args.cid_test_file,
                  predict_test_file=args.predict_test_file)
    logging.info('finished ({:.2f} min elasped)'.format((time.time() - start) /
                                                        60))
