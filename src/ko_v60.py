# -*- coding: utf-8 -*-
"""
# Result

$ python ko_v60.py
Fold1: 0.90500418
Fold2: 0.90752129
Fold3: 0.90467691
Fold4: 0.90556196
Fold5: 0.90198943
CV Result: 0.90495076
"""
import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score


# Output files
VAL_FILE_FMT = "ko_v60.valCV.{0:.8f}.csv"
TEST_FILE_FMT = "ko_v60.testCV.{0:.8f}.csv"

# Given data
CV_FILE = "cv_id.txt"
TRUTH_TRAIN_FILE = "truth_train.csv"

# SK feature: DropBox/feature_sk_v52.7z
SK_FEATURE_TRAIN = "trn_sk_v52.csv"
SK_FEATURE_TEST = "tst_sk_v52.csv"

# RW feature: DropBox/rw_data
RW_FEATURE_TRAIN = "xgb_train_rw"
RW_FEATURE_TEST = "xgb_test_rw"


def load_X():
    y_train = np.array(
        pd.read_csv(TRUTH_TRAIN_FILE, names=['enrollment_id', 'y'])['y']).ravel()

    Xsk_train = np.array(pd.read_csv(SK_FEATURE_TRAIN))
    Xsk_test = np.array(pd.read_csv(SK_FEATURE_TEST))

    Xrw_train = np.array(pd.read_csv(RW_FEATURE_TRAIN))
    Xrw_test = np.array(pd.read_csv(RW_FEATURE_TEST))

    X_train = np.hstack([Xsk_train, Xrw_train])
    X_test = np.hstack([Xsk_test, Xrw_test])

    return X_train, X_test, y_train


def model():
    params = {
        'silent': 1,
        'nthread': 8,
        'seed': 899,
        'objective': 'binary:logistic',
        'n_estimators': 720,
        'max_depth': 9,
        'min_child_weight': 1.5900763988832267,
        'subsample': 0.9014464225893877,
        'gamma': 5.814857437011067,
        'colsample_bytree': 0.5704419449928451,
        'learning_rate': 0.04,
    }
    return xgb.XGBClassifier(**params)


def cv():
    X, _, y = load_X()
    cvset = pd.read_csv(CV_FILE, names=['fold_id'])
    fold_list = sorted(cvset['fold_id'].unique())

    scores = []
    pred_list = []
    for fold_id in fold_list:
        idx_train = (np.array(cvset) != fold_id).ravel()
        idx_test = (np.array(cvset) == fold_id).ravel()
        idx_list = np.arange(len(cvset))[idx_test]
        X_train, y_train = X[idx_train, :], y[idx_train]
        X_test, y_test = X[idx_test, :], y[idx_test]

        clf = model()
        clf.fit(X_train, y_train)
        y_pred = clf.predict_proba(X_test)
        score = roc_auc_score(y_test, y_pred[:, 1])
        scores.append(score)
        print("Fold{0}: {1:.8f}".format(fold_id, score))
        pred_list.append(pd.DataFrame({
            'pred': y_pred[:, 1],
            'idx': idx_list,
        }))

    cv_score = np.mean(scores)
    print("CV Result: {0:.8f}".format(cv_score))
    pd.concat(pred_list).sort('idx')['pred'].to_csv(
        VAL_FILE_FMT.format(cv_score), index=False, header=None)
    return cv_score


def solve(cv_score):
    X_train, X_test, y_train = load_X()

    clf = model()
    clf.fit(X_train, y_train)
    y_pred = clf.predict_proba(X_test)[:, 1]
    pd.DataFrame({'pred': y_pred}).to_csv(TEST_FILE_FMT.format(cv_score),
        index=False, header=None)


if __name__ == '__main__':
    cv_score = cv()
    solve(cv_score)
