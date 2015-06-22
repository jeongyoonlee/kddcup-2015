#!/usr/bin/env python

import argparse


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--csv-file', required=True, dest='csv_file')
    parser.add_argument('--ffm-file', required=True, dest='ffm_file')

    args = parser.parse_args()

    with open(args.csv_file) as fin, open(args.ffm_file, 'w') as fout:
        # skip header of the input CSV file.
        fin.readline()
        for row in fin:
            row = row.strip().split(',')
            newrow = ['{}:{}:1'.format(i, x.split('_')[1]) for i, x in enumerate(row, 15)]

            fout.write('{}\n'.format(' '.join(newrow)))
