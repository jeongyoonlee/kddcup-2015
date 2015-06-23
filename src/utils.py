# -*- coding: utf-8 -*-
import os
import logging as l
import datetime
import subprocess

import pandas as pd
import numpy as np
from ume.externals.xgboost import XGBoost as XGB
from sklearn.ensemble import GradientBoostingClassifier as GBC
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.base import BaseEstimator
from sklearn.ensemble import BaggingClassifier as BC
from sklearn.preprocessing import LabelEncoder as LE
from sklearn.preprocessing import OneHotEncoder as OHE
from sklearn.datasets import dump_svmlight_file
from sklearn.metrics import roc_auc_score


ENROLL_TEST = "data/input/enrollment_test.csv"
ENROLL_TRAIN = "data/input/enrollment_train.csv"
LOG_TEST = "data/input/log_test.csv"
LOG_TRAIN = "data/input/log_train.csv"
TRUTH_TRAIN = "data/input/truth_train.csv"
OBJECT = "data/input/object.csv"
SAMPLE_SUBMISSION = "data/input/sampleSubmission.csv"


def reshape(series, col_sz=1):
    elem_sz = len(series)
    row_sz = elem_sz / col_sz
    assert row_sz * col_sz == elem_sz
    return np.array(series).reshape((row_sz, col_sz))


def agg_by(df, merge_on=None, columns=None, agg_func=None):
    feat = []
    group_cols = [merge_on] + columns

    for idx, row in df.groupby(group_cols):
        key_value = zip(group_cols, idx)
        ret_dict = agg_func(row)
        ret_dict.update(key_value)
        feat.append(ret_dict)

    feat = pd.DataFrame(feat)
    val_cols = [col for col in feat.columns if col in group_cols]

    feat_wide = feat.pivot_table(
        values=val_cols,
        index=merge_on,
        columns=columns).reset_index()
    col_sz = len(feat_wide.columns) - 1
    feat_wide.columns = [merge_on] + list(range(col_sz))
    return feat_wide


def toordinal(s, fmt='%Y-%m-%dT%H:%M:%S'):
    """
    Day level

    TODO: is it useful to use more fine grained datetime for generating features?

    >>> datetime.datetime(2015, 6, 6, 15, 56, 40).toordinal()
    735755
    >>> datetime.datetime(2015, 6, 6, 20, 56, 40).toordinal()
    735755
    """
    dt = datetime.datetime.strptime(s, fmt)
    return dt.toordinal()


def to_seconds_v2(s2, fmt='%Y-%m-%dT%H:%M:%S'):
    dt1 = datetime.datetime.strptime('1990-01-01T00:00:00', fmt)
    dt2 = datetime.datetime.strptime(s2, fmt)
    return (dt2 - dt1).total_seconds()


def to_seconds(s2, fmt='%Y-%m-%dT%H:%M:%S'):
    dt1 = datetime.datetime.strptime('2013-10-01T00:00:00', fmt)
    dt2 = datetime.datetime.strptime(s2, fmt)
    return (dt2 - dt1).total_seconds()


def to_hours(s2, fmt='%Y-%m-%dT%H:%M:%S'):
    dt1 = datetime.datetime.strptime('2013-10-01T00:00:00', fmt)
    dt2 = datetime.datetime.strptime(s2, fmt)
    return (dt2 - dt1).total_seconds() / 60.0 / 60.0


def timediff_seconds(s1, s2, fmt='%Y-%m-%dT%H:%M:%S'):
    dt1 = datetime.datetime.strptime(s1, fmt)
    dt2 = datetime.datetime.strptime(s2, fmt)
    return (dt2 - dt1).total_seconds() / 60.0 / 60.0


def onehot_encode(series, sparse=False):
    arr = LE().fit_transform(series)
    X = OHE(sparse=sparse).fit_transform(arr.reshape((len(arr), 1)))
    return X


def load_truth():
    truth_df = pd.read_csv(TRUTH_TRAIN, names=['enrollment_id', 'target'])
    return truth_df


def load_enroll():
    enroll_train_df = pd.read_csv(ENROLL_TRAIN)
    enroll_test_df = pd.read_csv(ENROLL_TEST)
    df = pd.concat([enroll_train_df, enroll_test_df])
    return df


def load_log(**args):
    log_train_df = pd.read_csv(LOG_TRAIN)
    log_test_df = pd.read_csv(LOG_TEST)
    df = pd.concat([log_train_df, log_test_df])

    # Merge with enrollment (username)
    enr_df = load_enroll()
    df = df.merge(enr_df, how='left', on='enrollment_id')

    return df


def load_obj():
    obj_df = pd.read_csv(OBJECT, na_values=["", "null"]).rename(
        columns={'module_id': 'object'})
    obj_df['n_children'] = obj_df['children'].fillna("").apply(
        lambda x: len(x.split(" ")))
    obj_df['start_flag'] = obj_df['start'].fillna("").apply(
        lambda x: len(x) == 19)

    return obj_df


def load_log_with_obj_attrib():
    df = load_log()
    df_obj = load_obj()
    df = df.merge(df_obj[[
        'object', 'category', 'n_children', 'start', 'start_flag']],
            how='left', on='object')
    return df


def to_evaluation_period(s, days=10, offset=1):
    base_date = datetime.datetime.strptime(s, '%Y-%m-%dT%H:%M:%S')
    next_day = base_date + datetime.timedelta(days=offset)
    next_day_bgn = datetime.datetime(
        base_date.year, base_date.month, base_date.day)
    next_day_end = (next_day_bgn + datetime.timedelta(days=days) -
            datetime.timedelta(seconds=1))
    return {
        'begin': next_day_bgn.strftime('%Y-%m-%dT%H:%M:%S'),
        'end': next_day_end.strftime('%Y-%m-%dT%H:%M:%S'),
    }


def before_evaluation_period(s):
    base_date = datetime.datetime.strptime(s, '%Y-%m-%dT%H:%M:%S')
    next_day = base_date - datetime.timedelta(days=9)
    next_day_bgn = datetime.datetime(
        base_date.year, base_date.month, base_date.day)
    next_day_end = (next_day_bgn + datetime.timedelta(days=10) -
            datetime.timedelta(seconds=1))
    return {
        'begin': next_day_bgn.strftime('%Y-%m-%dT%H:%M:%S'),
        'end': next_day_end.strftime('%Y-%m-%dT%H:%M:%S'),
    }


def drop_header(fp):
    pd.read_csv(fp).to_csv(fp + ".drop_head.csv", index=False, header=False)


LIBFM_BIN = "onetime/libFM"
LIBSVM_TRAIN_VAL = "data/input/sparse_feature_set_val_train.libsvm"
LIBSVM_TEST_VAL = "data/input/sparse_feature_set_val_test.libsvm"
LIBFM_VAL_OUTPUT = "data/working/libfm.tmp"


class FactorizationMachines(BaseEstimator):
    def __init__(self, task="c", use_bias=1, use_1way_interactions=1,
            dim_2way_interactions=2, seed=999, iteration=10,
            method="mcmc", init_stdev=0.001, regular="r2", verbosity=0,
            learn_rate=0.01):
        self.task = task
        self.use_bias = str(use_bias)
        self.use_1way_interactions = str(use_1way_interactions)
        self.dim_2way_interactions = str(dim_2way_interactions)
        self.method = method
        self.init_stdev = str(init_stdev)
        self.seed = str(seed)
        self.iteration = str(iteration)
        self.regular = regular
        self.verbosity = str(verbosity)
        self.learn_rate = str(learn_rate)

    def _dump_dataset(self, X, y, filepath):
        with open(filepath, 'w') as f:
            dump_svmlight_file(X, y, f)

    def fit(self, X_train, y_train):
        self._dump_dataset(X_train, y_train, LIBSVM_TRAIN_VAL)

    def predict_proba(self, X_test):
        y_test = np.zeros(X_test.shape[0])
        self._dump_dataset(X_test, y_test, LIBSVM_TEST_VAL)

        libfm_v5 = [
            LIBFM_BIN,
            "-task", self.task,
            "-train", LIBSVM_TRAIN_VAL,
            "-test", LIBSVM_TEST_VAL,
            "-dim", "'{},{},{}'".format(
                self.use_bias,
                self.use_1way_interactions,
                self.dim_2way_interactions),
            "-seed", self.seed,
            "-iter", self.iteration,
            "-method", self.method,
            "-init_stdev", self.init_stdev,
            "-regular", self.regular,
            "-verbosity", self.verbosity,
            "-learn_rate", self.learn_rate,
            "-out", LIBFM_VAL_OUTPUT
        ]
        subprocess.call(libfm_v5)
        y_pred = np.array(pd.read_csv(LIBFM_VAL_OUTPUT, names=['y_pred']).y_pred)
        return y_pred


class Mixer(BaseEstimator):
    def __init__(self, **params):
        self.X_train = None
        self.y_train = None

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def predict_proba(self, X_test):
        pred = np.zeros((X_test.shape[0], 2))

        l.info("XGB")
        clf = XGB(silent=2, nthread=4, seed=999,
            objective='binary:logistic', eval_metric='auc',
            num_round=422, max_depth=7, min_child_weight=4.305,
            subsample=0.9, gamma=1.8519, eta=0.0583779)
        clf.fit(self.X_train, self.y_train)
        clf_pred = clf.predict_proba(X_test)
        pred = pred + clf_pred * 0.9

        l.info("RFC")
        clf = RFC(n_jobs=6, n_estimators=200, criterion="entropy",
                random_state=999)
        clf.fit(self.X_train, self.y_train)
        clf_pred = clf.predict_proba(X_test)
        pred = pred + clf_pred * 0.1

        return pred / 2.0


class XGBoostAverageModel2(BaseEstimator):
    def __init__(self, **params):
        self.n_models = params.get('n_models', 100)
        params.pop('n_models')

        self.params = params
        self.X_train = None
        self.y_train = None
        self.params['eta'] = (
            np.random.randint(10) - 5
        ) * 0.01

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def predict_proba(self, X_test):
        pred = np.zeros((X_test.shape[0], 2))
        for i in range(self.n_models):
            p = dict(self.params)
            p['num_round'] += i
            p['seed'] += i
            clf = XGB(**p)
            l.info("Runninr: {0} ({1} of {2})".format(
                str(clf), i, self.n_models))
            clf.fit(self.X_train, self.y_train)
            clf_pred = clf.predict_proba(X_test)
            pred = pred + clf_pred

        return pred / float(self.n_models)


class XGBoostAverageModel(BaseEstimator):
    def __init__(self, **params):
        self.n_models = params.get('n_models', 100)
        params.pop('n_models')

        self.params = params
        self.X_train = None
        self.y_train = None

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def predict_proba(self, X_test):
        pred = np.zeros((X_test.shape[0], 2))
        for i in range(self.n_models):
            p = dict(self.params)
            p['num_round'] += i
            p['seed'] += i
            clf = XGB(**p)
            l.info("Runninr: {0} ({1} of {2})".format(
                str(clf), i, self.n_models))
            clf.fit(self.X_train, self.y_train)
            clf_pred = clf.predict_proba(X_test)
            pred = pred + clf_pred

        return pred / float(self.n_models)


class XGBoostRankAverageModel(BaseEstimator):
    def __init__(self, **params):
        self.n_models = params.get('n_models', 100)
        params.pop('n_models')

        self.params = params
        self.X_train = None
        self.y_train = None

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def predict_proba(self, X_test):
        pred = np.zeros((X_test.shape[0], 2))
        for i in range(self.n_models):
            p = dict(self.params)
            p['num_round'] += i
            p['seed'] += i
            clf = XGB(**p)
            l.info("Run: {0} ({1} of {2})".format(
                str(clf), i, self.n_models))
            clf.fit(self.X_train, self.y_train)
            clf_pred = clf.predict_proba(X_test)

            df_pred = pd.DataFrame({'Id': list(range(clf_pred.shape[0]))})
            df_pred['Proba_0'] = 1.0 - clf_pred
            df_pred['Proba_1'] = clf_pred
            df_pred = df_pred.sort('Proba_1')
            df_pred['Rank_0'] = 1.0 - np.arange(0, len(df_pred)) / (1.0 * len(df_pred) - 1)
            df_pred['Rank_1'] = np.arange(0, len(df_pred)) / (1.0 * len(df_pred) - 1)
            df_pred = df_pred.sort('Id')

            pred = pred + np.array(df_pred[['Rank_0', 'Rank_1']])

        return pred / float(self.n_models)
