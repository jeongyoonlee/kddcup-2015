# -*- coding: utf-8 -*-
"""
# Model description

* sk features (base)
* rw features (selected)
* jeong features (selected)

# Results

$ python ko_v61.py
0.90523011

"""
import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.datasets import load_svmlight_file


# Output files
VAL_FILE_FMT = "ko_v61.valCV.{0:.8f}.csv"
TEST_FILE_FMT = "ko_v61.testCV.{0:.8f}.csv"

# Given data
CV_FILE = "cv_id.txt"
TRUTH_TRAIN_FILE = "truth_train.csv"

# Features
JEONG9_TRAIN_FILE = "feature9.trn.sps"
JEONG9_TEST_FILE = "feature9.tst.sps"
SK_TRAIN_FILE = "trn_sk_v52.csv"
SK_TEST_FILE = "tst_sk_v52.csv"
RW_TRAIN_FILE = "xgb_train_rw"
RW_TEST_FILE = "xgb_test_rw"


def load_X():
    y_train = np.array(
        pd.read_csv(TRUTH_TRAIN_FILE, names=['enrollment_id', 'y'])['y']).ravel()

    rw_feat = [
        'max_absent_days',
        'min_days_from_first_visit_to_next_course_begin',
        'min_days_from_10days_after_last_visit_to_next_course_begin',
        'min_days_from_last_visit_to_next_course_end',
        'min_days_from_next_course_end_to_last_visit',
        'min_days_from_10days_after_current_course_end_to_next_course_begin',
        'min_days_from_10days_after_current_course_end_to_next_course_end',
        'min_days_from_course_end_to_next_visit',
        'time_span',
        'days_from_lastact_to_course_end',
        'active_days_from_last_visit_to_course_end',
        'active_days_in_10days_from_course_end',
        'course_drop_rate',
        'course_num_enrolled',
        'obj_problem_visited_ratio',
        'obj_chapter_not_visited',
        'obj_chapter_visited_ratio',
        'obj_video_not_visited',
        'obj_video_visited_ratio',
        'avg_hour_per_day',
        'last_month',
    ]
    j9_feature_names = [
        'obj_10_days_after_last_date',
    ]

    # Load sk features
    Xsk_train = np.array(pd.read_csv(SK_TRAIN_FILE))
    Xsk_test = np.array(pd.read_csv(SK_TEST_FILE))

    # Load (selected) rw features
    Xrw_train = np.array(pd.read_csv(RW_TRAIN_FILE, usecols=rw_feat))
    Xrw_test = np.array(pd.read_csv(RW_TEST_FILE, usecols=rw_feat))

    # Load Jeong's feature9
    Xj9_train, Xj9_test = load_jeong()
    Xj9_train = np.hstack([
        Xj9_train[fname].todense() for fname in j9_feature_names])
    Xj9_test = np.hstack([
        Xj9_test[fname].todense() for fname in j9_feature_names])

    # Horizontal stack
    X_train = np.hstack([Xsk_train, Xrw_train, Xj9_train])
    X_test = np.hstack([Xsk_test, Xrw_test, Xj9_test])

    return X_train, X_test, y_train


def model(up_params={}):
    params = {
        'nthread': 8,
        'seed': 899,
        'objective': 'binary:logistic',
        'n_estimators': 600,
        'max_depth': 9,
        'min_child_weight': 1.5900763988832267,
        'subsample': 0.9014464225893877,
        'gamma': 5.814857437011067,
        'colsample_bytree': 0.5704419449928451,
        'learning_rate': 0.04,
    }
    params.update(up_params)
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


def load_jeong():
    Xj9_train, y_train = load_svmlight_file(JEONG9_TRAIN_FILE)
    Xj9_test, y_test = load_svmlight_file(JEONG9_TEST_FILE)

    feature_info = [
        ('username', 20113),
        ('course_id', 39),
        ('source_event', 10),
        ('object', 3554),
        ('count', 1),
        ('category', 6),
        ('n_children', 7),
        ('obj_days_before_last_date', 10),
        ('days_before_last_date', 30),
        ('weeks_before_last_date', 4),
        ('last_month', 7),
        ('days_after_obj_date', 7),
        ('obj_10_days_after_last_date', 2),
    ]

    Xj9_train_dict = {}
    Xj9_test_dict = {}
    for feature_name, feature_size in feature_info:
        Xj9_train_subset, Xj9_train = (
            Xj9_train[:, :feature_size],
            Xj9_train[:, feature_size:])
        Xj9_train_dict[feature_name] = Xj9_train_subset

        Xj9_test_subset, Xj9_test = (
            Xj9_test[:, :feature_size],
            Xj9_test[:, feature_size:])
        Xj9_test_dict[feature_name] = Xj9_test_subset
    return Xj9_train_dict, Xj9_test_dict


if __name__ == '__main__':
    cv_score = cv()
    solve(cv_score)
