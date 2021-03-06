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
from kaggler.online_model import NN


def train_predict(train_file, test_file, predict_valid_file, predict_test_file,
                  n_iter=100, hidden=4, lrate=.1, n_fold=5):

    feature_name = os.path.basename(train_file)[:-4]
    logging.basicConfig(format='%(asctime)s   %(levelname)s   %(message)s',
                        level=logging.DEBUG,
                        filename='nn_{}_{}_{}_{}.log'.format(n_iter,
                                                             hidden,
                                                             lrate,
                                                             feature_name))

    _, y_val = load_data(train_file)

    cv = StratifiedKFold(y_val, n_folds=n_fold, shuffle=True, random_state=2015)

    p_val = np.zeros_like(y_val)
    auc = 0.
    for i_cv, (i_trn, i_val) in enumerate(cv, 1):
        logging.info('Training model #{}...'.format(i_cv))
        clf = NN(n=100000, h=hidden, a=lrate, seed=2015)

        logging.info('Epoch\tTrain\tValid')
        logging.info('=========================')
        for i_iter in range(n_iter):
            cnt_trn = 0
            for i, (x, y) in enumerate(clf.read_sparse(train_file)):
                if i in i_val:
                    p_val[i] = clf.predict(x)
                else:
                    p = clf.predict(x)
                    clf.update(x, p - y)
                    cnt_trn += 1

            auc_val = AUC(y_val[i_val], p_val[i_val])

            if (i_iter == 0) or ((i_iter + 1) % int(n_iter / 10) == 0) or (i_iter == n_iter - 1):
                logging.info('#{:4d}\t______\t{:.6f}'.format(i_iter + 1,
                                                             auc_val))

        auc += auc_val

    logging.info('AUC = {:.6f}'.format(auc / n_fold))

    logging.info('Retraining with 100% data...')
    clf = NN(n=100000, h=hidden, a=lrate, seed=2015)
    for i_iter in range(n_iter):
        for x, y in clf.read_sparse(train_file):
            p = clf.predict(x)
            clf.update(x, p - y)

        if (i_iter == 0) or ((i_iter + 1) % int(n_iter / 10) == 0) or (i_iter == n_iter - 1):
            logging.info('#{:4d}'.format(i_iter + 1))

    _, y_tst = load_data(test_file)
    p_tst = np.zeros_like(y_tst)
    for i, (x, _) in enumerate(clf.read_sparse(test_file)):
        p_tst[i] = clf.predict(x)

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
    parser.add_argument('--n-iter', type=int, dest='n_iter')
    parser.add_argument('--hidden', type=int, dest='hidden')
    parser.add_argument('--lrate', type=float, dest='lrate')

    args = parser.parse_args()

    start = time.time()
    train_predict(train_file=args.train_file,
                  test_file=args.test_file,
                  predict_valid_file=args.predict_valid_file,
                  predict_test_file=args.predict_test_file,
                  n_iter=args.n_iter,
                  hidden=args.hidden,
                  lrate=args.lrate)
    logging.info('finished ({:.2f} min elasped)'.format((time.time() - start) /
                                                        60))
