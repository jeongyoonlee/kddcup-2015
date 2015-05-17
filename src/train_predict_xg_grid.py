#!/usr/bin/env python

from __future__ import division
from sklearn.cross_validation import StratifiedKFold
from sklearn.datasets import load_svmlight_file
from sklearn.grid_search import GridSearchCV

import argparse
import logging
import numpy as np
import os
import time

import xgboost as xgb


def train_predict(train_file, test_file, predict_valid_file, predict_test_file,
                  n_fold=5):

    feature_name = os.path.basename(train_file)[:-4]
    logging.basicConfig(format='%(asctime)s   %(levelname)s   %(message)s',
                        level=logging.DEBUG,
                        filename='xg_grid_{}.log'.format(feature_name))

    logging.info('Loading training and test data...')
    X, y = load_svmlight_file(train_file)
    X_tst, _ = load_svmlight_file(test_file)

    xg = xgb.XGBClassifier(subsample=0.5, colsample_bytree=0.8, nthread=6)
    param = {'learning_rate': [.01, .05, .1], 'max_depth': [4, 8, 12],
             'n_estimators': [100, 200, 400]}
    cv = StratifiedKFold(y, n_folds=n_fold, shuffle=True, random_state=2015)
    clf = GridSearchCV(xg, param, scoring='roc_auc', verbose=1, cv=cv)

    logging.info('Cross validation for grid search...')
    clf.fit(X, y)
    p = clf.predict_proba(X)[:, 1]

    logging.info('best model = {}'.format(clf.best_estimator_))
    logging.info('best score = {:.4f}'.format(clf.best_score_))

    logging.info('Retraining with 100% data...')
    clf.best_estimator_.fit(X, y)
    p_tst = clf.best_estimator_.predict_proba(X_tst)[:, 1]

    logging.info('Saving predictions...')
    np.savetxt(predict_valid_file, p, fmt='%.6f')
    np.savetxt(predict_test_file, p_tst, fmt='%.6f')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-file', required=True, dest='train_file')
    parser.add_argument('--test-file', required=True, dest='test_file')
    parser.add_argument('--predict-valid-file', required=True,
                        dest='predict_valid_file')
    parser.add_argument('--predict-test-file', required=True,
                        dest='predict_test_file')

    args = parser.parse_args()

    start = time.time()
    train_predict(train_file=args.train_file,
                  test_file=args.test_file,
                  predict_valid_file=args.predict_valid_file,
                  predict_test_file=args.predict_test_file)
    logging.info('finished ({:.2f} min elasped)'.format((time.time() - start) /
                                                        60))
