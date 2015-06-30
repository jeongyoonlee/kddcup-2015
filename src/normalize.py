#!/usr/bin/env python

from sklearn.preprocessing import StandardScaler
from sklearn.datasets import dump_svmlight_file

import argparse

from kaggler.data_io import load_data


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-file', required=True, dest='train_file')
    parser.add_argument('--test-file', required=True, dest='test_file')
    parser.add_argument('--normalized-train-file', required=True,
                        dest='normalized_train_file')
    parser.add_argument('--normalized-test-file', required=True,
                        dest='normalized_test_file')

    args = parser.parse_args()
    X_trn, y_trn = load_data(args.train_file)
    X_tst, y_tst = load_data(args.test_file)

    scaler = StandardScaler(with_mean=False)
    X_trn = scaler.fit_transform(X_trn)
    X_tst = scaler.transform(X_tst)

    dump_svmlight_file(X_trn, y_trn, args.normalized_train_file,
                       zero_based=False)
    dump_svmlight_file(X_tst, y_tst, args.normalized_test_file,
                       zero_based=False)
