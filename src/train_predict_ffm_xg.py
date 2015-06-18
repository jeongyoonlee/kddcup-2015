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
import xgboost as xgb

from svm_to_ffm import svm_to_ffm


def np_to_ffm(X, y, ffm_feature_file):
    with open(ffm_feature_file, 'w') as f:
        for i in range(X.shape[0]):
            features = ['{}:{}:1'.format(j, X[i, j])
                        for j in range(X.shape[1])]
            f.write('{} {}\n'.format(y[i], ' '.join(features)))


def train_predict(train_file, test_file,
                  predict_valid_file, predict_test_file, model_file,
                  n_iter=100, dim=4, lrate=.1, n_tree=30, depth=4, eta=0.05,
                  n_fold=5):

    # XGB parameters and configuration
    param = {'max_depth': depth,
             'eta': eta,
             'objective': 'binary:logistic',
             'colsample_bytree': .8,
             'subsample': .5,
             'eval_metric': 'auc',
             'seed': 2015}

    dir_feature = os.path.dirname(train_file)
    dir_model = os.path.dirname(model_file)
    dir_val = os.path.dirname(predict_valid_file)

    feature_name = os.path.basename(train_file)[:-8]
    xg_feature_name = 'xg_{}_{}_{}_{}'.format(n_tree, depth, eta, feature_name)
    algo_name = 'ffm_{}_{}_{}'.format(n_iter, dim, lrate)
    model_name = '{}_{}'.format(algo_name, feature_name)

    logging.basicConfig(format='%(asctime)s   %(levelname)s   %(message)s',
                        level=logging.DEBUG,
                        filename='{}.log'.format(model_name))

    logging.info('Loading training data')
    X, y = load_svmlight_file(train_file)

    cv = StratifiedKFold(y, n_folds=n_fold, shuffle=True, random_state=2015)

    p = np.zeros_like(y)
    for i, (i_trn, i_val) in enumerate(cv, 1):
        logging.info('Training model #{}...'.format(i))
        valid_train_file = os.path.join(dir_feature, '{}.trn{}.ffm'.format(xg_feature_name, i))
        valid_test_file = os.path.join(dir_feature, '{}.val{}.ffm'.format(xg_feature_name, i))
        valid_model_file = os.path.join(dir_model, '{}.trn{}.mdl'.format(model_name, i))
        valid_predict_file = os.path.join(dir_val, '{}.val{}.yht'.format(model_name, i))
        valid_train_svm_file = os.path.join(dir_feature, '{}.trn{}.sps'.format(feature_name, i))
        valid_test_svm_file = os.path.join(dir_feature, '{}.val{}.sps'.format(feature_name, i))

        # if there are no libsvm files, generate them first.
        if (not os.path.isfile(valid_train_file)) or \
           (not os.path.isfile(valid_test_file)):

            # if there are no libsvm files, generate them first.
            if not os.path.isfile(valid_train_svm_file):
                dump_svmlight_file(X[i_trn], y[i_trn], valid_train_svm_file,
                                   zero_based=False)

            if not os.path.isfile(valid_test_svm_file):
                dump_svmlight_file(X[i_val], y[i_val], valid_test_svm_file,
                                   zero_based=False)

            # generate XGB features
            dtrain = xgb.DMatrix(valid_train_svm_file)
            dtest = xgb.DMatrix(valid_test_svm_file)
            watchlist = [(dtest, 'eval'), (dtrain, 'train')]

            logging.info('Generating XGB features')
            xg = xgb.train(param, dtrain, n_tree, watchlist)
            xg_trn_feature = xg.predict(dtrain, pred_leaf=True)
            xg_tst_feature = xg.predict(dtest, pred_leaf=True)

            # save XGB features as the libffm format
            np_to_ffm(xg_trn_feature, dtrain.get_label(), valid_train_file)
            np_to_ffm(xg_tst_feature, dtest.get_label(), valid_test_file)

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

    ffm_train_file = os.path.join(dir_feature, '{}.trn.ffm'.format(xg_feature_name))
    ffm_test_file = os.path.join(dir_feature, '{}.tst.ffm'.format(xg_feature_name))
    if (not os.path.isfile(ffm_train_file)) or \
       (not os.path.isfile(ffm_test_file)):

        # generate XGB features
        dtrain = xgb.DMatrix(train_file)
        dtest = xgb.DMatrix(test_file)
        watchlist = [(dtrain, 'train')]

        logging.info('Generating XGB features')
        xg = xgb.train(param, dtrain, n_tree, watchlist)
        xg_trn_feature = xg.predict(dtrain, pred_leaf=True)
        xg_tst_feature = xg.predict(dtest, pred_leaf=True)

        # save XGB features as the libffm format
        np_to_ffm(xg_trn_feature, dtrain.get_label(), ffm_train_file)
        np_to_ffm(xg_tst_feature, dtest.get_label(), ffm_test_file)

    logging.info('Retraining with 100% data...')
    subprocess.call(["ffm-train",
                     '-k', '{}'.format(dim),
                     '-r', str(lrate),
                     '-t', str(n_iter),
                     '-p', ffm_test_file,
                     ffm_train_file,
                     model_file])

    subprocess.call(["ffm-predict",
                     ffm_test_file,
                     model_file,
                     predict_test_file])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-file', required=True, dest='train_file')
    parser.add_argument('--test-file', required=True, dest='test_file')
    parser.add_argument('--model-file', required=True, dest='model_file')
    parser.add_argument('--predict-valid-file', required=True,
                        dest='predict_valid_file')
    parser.add_argument('--predict-test-file', required=True,
                        dest='predict_test_file')
    parser.add_argument('--n-iter', type=int, dest='n_iter')
    parser.add_argument('--dim', type=int, dest='dim')
    parser.add_argument('--lrate', type=float, dest='lrate')
    parser.add_argument('--eta', type=float, dest='eta')
    parser.add_argument('--n-tree', type=int, dest='n_tree')
    parser.add_argument('--depth', type=int, dest='depth')

    args = parser.parse_args()

    start = time.time()
    train_predict(train_file=args.train_file,
                  test_file=args.test_file,
                  predict_valid_file=args.predict_valid_file,
                  predict_test_file=args.predict_test_file,
                  model_file=args.model_file,
                  n_iter=args.n_iter,
                  dim=args.dim,
                  n_tree=args.n_tree,
                  depth=args.depth,
                  lrate=args.lrate,
                  eta=args.eta)
    logging.info('finished ({:.2f} min elasped)'.format((time.time() - start) /
                                                        60))
