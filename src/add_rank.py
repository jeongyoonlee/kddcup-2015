#!/usr/bin/env python

from __future__ import division
from scipy.stats import rankdata
from sklearn.datasets import dump_svmlight_file

import argparse
import numpy as np

from kaggler.data_io import load_data


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-file', required=True, dest='input_file')
    parser.add_argument('--output-file', required=True, dest='output_file')

    args = parser.parse_args()

    X, y = load_data(args.input_file)
    n_obs = X.shape[0]
    n_feature = X.shape[1]
    X_rank = np.zeros((n_obs, n_feature * 2))

    X_rank[:, :n_feature] = X
    for i in range(n_obs):
        X_rank[i, n_feature:] = rankdata(X[i])


    dump_svmlight_file(X_rank, y, args.output_file, zero_based=False)
