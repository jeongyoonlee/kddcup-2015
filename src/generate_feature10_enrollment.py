#!/usr/bin/env python

from __future__ import division
from sklearn.datasets import dump_svmlight_file

import argparse
import logging
import numpy as np
import pandas as pd
import time

from kaggler.util import encode_categorical_features


logging.basicConfig(format='%(asctime)s   %(levelname)s   %(message)s',
                    level=logging.DEBUG)


def generate_feature(train_file, test_file, object_file,
                     user_feature_file, course_feature_file,
                     train_feature_file, test_feature_file):

    logging.info('loading input data')
    trn = pd.read_csv(train_file)
    tst = pd.read_csv(test_file)
    obj = pd.read_csv(object_file, header=None)
    obj.columns = ['course_id', 'object', 'category', 'children', 'start']

    n_trn = trn.shape[0]

    trn.time = pd.to_datetime(trn.time)
    tst.time = pd.to_datetime(tst.time)

    df = pd.concat([trn, tst], axis=0)
    df['count'] = 1

    # get last dates of courses
    last_date = df[['course_id', 'time']].groupby('course_id', as_index=False).max()
    last_date.columns = ['course_id', 'last_date']

    # extract object information
    obj.children.fillna('', inplace=True)
    obj['n_children'] = obj.children.apply(lambda x: int(np.log2(1 + len(x.split()))))
    obj.start.replace('null', '2013-10-27 00:00:00', inplace=True)
    obj.start = pd.to_datetime(obj.start)
    obj = pd.merge(obj, last_date, on='course_id', how='left')

    obj['obj_days_before_last_date'] = (obj.last_date - obj.start).apply(lambda x: pd.Timedelta(x).days)
    obj.ix[obj.obj_days_before_last_date > 30, 'obj_days_before_last_date'] = 32
    obj.ix[obj.obj_days_before_last_date < -10, 'obj_days_before_last_date'] = -16

    # merge log data with last coursedate and object information
    df = pd.merge(df, last_date, on='course_id', how='left')
    df = pd.merge(df, obj[['object', 'category', 'n_children', 'obj_days_before_last_date']],
                  on='object', how='left')

    df['days_before_last_date'] = (df.last_date - df.time).apply(lambda x: pd.Timedelta(x).days)
    df['weeks_before_last_date'] = df.days_before_last_date // 7
    df.ix[df.weeks_before_last_date == 4, 'weeks_before_last_date'] = 3
    df['last_month'] = df.last_date.apply(lambda x: x.month)

    df['days_after_obj_date'] = df.obj_days_before_last_date - df.days_before_last_date
    df.ix[df.days_after_obj_date < 0, 'days_after_obj_date'] = -1

    df.days_after_obj_date = df.days_after_obj_date.apply(
        lambda x: np.sign(x) * int(np.log2(1 + np.abs(x)))
                  if ~pd.isnull(x) else x
    )

    df['obj_10_days_after_last_date'] = df.obj_days_before_last_date.apply(
        lambda x: 1 if x < 0 and x >= -10 else 0
    )
    df.obj_days_before_last_date = df.obj_days_before_last_date.apply(
        lambda x: np.sign(x) * int(np.log2(1 + np.abs(x)))
                  if ~pd.isnull(x) else x
    )

    eid = df.drop(['time', 'last_date'], axis=1)
    eid.set_index('enrollment_id', inplace=True)

    X = encode_categorical_features(eid, n=n_trn, min_obs=10, nan_as_var=True)
    X = X.tocsr()

    dump_svmlight_file(X[:n_trn], trn.enrollment_id.values, train_feature_file,
                       zero_based=False)
    dump_svmlight_file(X[n_trn:], tst.enrollment_id.values, test_feature_file,
                       zero_based=False)

    cid = df.drop(['enrollment_id', 'time', 'last_date'], axis=1)
    cid.set_index('course_id', inplace=True)

    X = encode_categorical_features(cid, n=n_trn, min_obs=10, nan_as_var=True)
    X = X.tocsr()
    with open(course_feature_file, 'w') as f:
        for i in range(X.shape[0]):
            x = X[i].toarray().flatten()
            idx = np.where(x != 0)[0]
            features = ' '.join(['{}:{}'.format(j + 1, x[j]) for j in idx])
            f.write('{} {}\n'.format(df.index.values[i], features))

    uid = df.drop(['enrollment_id', 'time', 'last_date'], axis=1)
    uid.set_index('username', inplace=True)

    X = encode_categorical_features(uid, n=n_trn, min_obs=10, nan_as_var=True)
    X = X.tocsr()
    with open(user_feature_file, 'w') as f:
        for i in range(X.shape[0]):
            x = X[i].toarray().flatten()
            idx = np.where(x != 0)[0]
            features = ' '.join(['{}:{}'.format(j + 1, x[j]) for j in idx])
            f.write('{} {}\n'.format(df.index.values[i], features))

 
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-file', required=True, dest='train_file')
    parser.add_argument('--test-file', required=True, dest='test_file')
    parser.add_argument('--object-file', required=True, dest='object_file')
    parser.add_argument('--user-feature-file', required=True,
                        dest='user_feature_file')
    parser.add_argument('--course-feature-file', required=True,
                        dest='course_feature_file')
    parser.add_argument('--train-feature-file', required=True,
                        dest='train_feature_file')
    parser.add_argument('--test-feature-file', required=True,
                        dest='test_feature_file')

    args = parser.parse_args()
    start = time.time()
    generate_feature(args.train_file,
                     args.test_file,
                     args.object_file,
                     args.user_feature_file,
                     args.course_feature_file,
                     args.train_feature_file,
                     args.test_feature_file)
    logging.info('finished ({:.2f} sec elapsed)'.format(time.time() - start))

