#!/usr/bin/env python

import argparse
import logging
import numpy as np
import time


logging.basicConfig(format='%(asctime)s   %(levelname)s   %(message)s',
                    level=logging.DEBUG)

FEATURE_PER_FIELD = {'feature3': [20113, 39, 2, 7, 3554, 30, 4]}

def svm_to_ffm(svm_file, ffm_file, feature_name):

    feature_per_field = FEATURE_PER_FIELD[feature_name]
    max_idx_per_field = np.cumsum(feature_per_field)
    n_field = len(feature_per_field)

    with open(ffm_file, 'w') as fout, open(svm_file) as fin:

        for i, row in enumerate(fin, 1):
            if i % 100000 == 0:
                logging.info('processing {} line(s)'.format(i))

            # split a row into the ID and features
            row = row.strip().split(' ')
            new_row = [row[0]]
            features = row[1:]

            i_field = 0
            prev_max_idx = 0
            for feature in features:
                idx, value = feature.split(':')
                idx = int(idx)
                while idx > max_idx_per_field[i_field]:
                    prev_max_idx = max_idx_per_field[i_field]
                    i_field += 1

                # this point, idx <= max_idx_per_field[i_field]
                new_row.append('{}:{}:{}'.format(i_field + 1,
                                                 idx - prev_max_idx,
                                                 value))

            fout.write('{}\n'.format(' '.join(new_row)))

        logging.info('{} lines processed'.format(i))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--svm-file', '-s', required=True, dest='svm_file')
    parser.add_argument('--ffm-file', '-f', required=True, dest='ffm_file')

    args = parser.parse_args()

    start = time.time()
    svm_to_ffm(args.svm_file, args.ffm_file, 'feature3')
    logging.info('finished ({:.2f} sec elapsed)'.format(time.time() - start))
