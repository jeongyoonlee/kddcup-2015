#!/usr/bin/env python

from sklearn.cross_validation import StratifiedKFold
from sklearn.metrics import roc_auc_score as AUC
from sklearn.linear_model import LogisticRegression as LR
from sklearn.preprocessing import StandardScaler

import argparse
import logging
import numpy as np
import os
import time

from kaggler.data_io import load_data


def train_predict_lr_fs(train_file, test_file, predict_valid_file,
                        predict_test_file, C, n_fold=5):

    feature_name = os.path.basename(train_file)[:-8]
    algo_name = 'lr_fs_{}'.format(C)
    model_name = '{}_{}'.format(algo_name, feature_name)
    logging.basicConfig(format='%(asctime)s   %(levelname)s   %(message)s',
                        level=logging.DEBUG,
                        filename='{}.log'.format(model_name))

    logging.info("Loading training and test data...")
    X_trn, y_trn = load_data(train_file)
    X_tst, _ = load_data(test_file)

    logging.info('Normalizing data')
    scaler = StandardScaler()
    X_trn = scaler.fit_transform(X_trn)
    X_tst = scaler.transform(X_tst)
 
    cv = StratifiedKFold(y_trn, n_folds=n_fold, shuffle=True, random_state=2015)

    selected_features = []
    features_to_test = [x for x in range(X_trn.shape[1])
                        if x not in selected_features]

    auc_cv_old = .5
    is_improving = True
    while is_improving:
        auc_cvs = []
        for feature in features_to_test:
            logging.info('{}'.format(selected_features + [feature]))
            X = X_trn[:, selected_features + [feature]]

            p_val = np.zeros_like(y_trn)
            for i, (i_trn, i_val) in enumerate(cv, start=1):
                clf = LR(C=C, class_weight='auto', random_state=2014)
                clf.fit(X[i_trn], y_trn[i_trn])
                p_val[i_val] = clf.predict_proba(X[i_val])[:, 1]

            auc_cv = AUC(y_trn, p_val)
            logging.info('AUC CV: {:.6f}'.format(auc_cv))
            auc_cvs.append(auc_cv)

        auc_cv_new = max(auc_cvs)
        if auc_cv_new > auc_cv_old:
            auc_cv_old = auc_cv_new
            feature = features_to_test.pop(auc_cvs.index(auc_cv_new))
            selected_features.append(feature)
            logging.info('selected features: {}'.format(selected_features))
        else:
            is_improving = False
            logging.info('final selected features: {}'.format(selected_features))

    logging.info('saving selected features as a file')
    with open('{}_selected.txt'.format(model_name), 'w') as f:
        f.write('{}\n'.format(selected_features))

    X = X_trn[:, selected_features]
    logging.debug('feature matrix: {}x{}'.format(X.shape[0], X.shape[1]))

    p_val = np.zeros_like(y_trn)
    for i, (i_trn, i_val) in enumerate(cv, start=1):
        logging.info('Training CV #{}'.format(i))
        clf = LR(C=C, class_weight='auto', random_state=2015)
        clf.fit(X[i_trn], y_trn[i_trn])
        p_val[i_val] = clf.predict_proba(X[i_val])[:, 1]

    auc_cv = AUC(y_trn, p_val)
    logging.info('AUC CV: {:.6f}'.format(auc_cv))
    logging.info("Writing test predictions to file")
    np.savetxt(predict_valid_file, p_val, fmt='%.6f', delimiter=',')

    logging.info('Retraining with 100% data...')
    clf.fit(X, y_trn)
    p_tst = clf.predict_proba(X_tst[:, selected_features])[:, 1]
    np.savetxt(predict_test_file, p_tst, fmt='%.6f', delimiter=',')

    
if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-file', '-t', required=True, dest='train')
    parser.add_argument('--test-file', '-v', required=True, dest='test')
    parser.add_argument('--predict-valid-file', '-p', required=True,
                        dest='predict_train')
    parser.add_argument('--predict-test-file', '-q', required=True,
                        dest='predict_test')
    parser.add_argument('--C', '-C', required=True, type=float, dest='C')

    args = parser.parse_args()

    start = time.time()
    train_predict_lr_fs(train_file=args.train,
                        test_file=args.test,
                        predict_valid_file=args.predict_train,
                        predict_test_file=args.predict_test,
                        C=args.C)

    logging.info('Finished ({:.2f} sec elasped).'.format(time.time() - start))
