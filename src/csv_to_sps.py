#!/usr/bin/env python

import argparse
from sklearn.datasets import dump_svmlight_file
from kaggler.data_io import load_data


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--csv-file', required=True, dest='csv_file')
    parser.add_argument('--sps-file', required=True, dest='sps_file')

    args = parser.parse_args()

    X, y = load_data(args.csv_file)
    dump_svmlight_file(X, y, args.sps_file, zero_based=False)
