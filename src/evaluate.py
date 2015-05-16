#!/usr/bin/env python

from __future__ import division
from sklearn.metrics import roc_auc_score as AUC

import argparse
import numpy as np
import os


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--target-file', '-t', required=True,
                        dest='target_file')
    parser.add_argument('--predict-file', '-p', required=True,
                        dest='predict_file')
    args = parser.parse_args()

    p = np.loadtxt(args.predict_file, delimiter=',')
    y = np.loadtxt(args.target_file, delimiter=',')[:, 1]

    model_name = os.path.basename(args.predict_file)[:-8]

    print('{}\t{:.6f}'.format(model_name, AUC(y, p)))
