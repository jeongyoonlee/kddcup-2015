#!/usr/bin/env python

import argparse
import logging
import numpy as np
import time


logging.basicConfig(format='%(asctime)s   %(levelname)s   %(message)s',
                    level=logging.DEBUG)

FEATURE_FIELD = {'feature3': [20113, 39, 2, 7, 3554, 30, 4],
                 'feature4': [55907, 39, 2, 7, 5268, 30, 4, 7],
                 'feature5': [55907, 39, 2, 7, 5268, 6, 7, 13, 30, 4, 7, 2],
                 'feature6': [20113, 39, 2, 7, 3554, 6, 7, 10, 30, 4, 7, 2],
                 'feature7': [20113, 39, 2, 7, 3554, 6, 7, 10, 30, 4, 7, 12, 2],
                 'feature8': [1, 39, 10, 3554, 6, 7, 10, 30, 4, 7, 12, 2],
                 'feature_tam': [1, 1, 33, 33, 33, 33, 33, 33, 32, 33, 33,
                                 33, 33, 33, 33, 33, 32, 33, 33, 33, 33, 33,
                                 33, 33, 32, 33, 12, 12, 12, 12, 12, 12, 12,
                                 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12,
                                 12, 12, 12, 11, 11, 11, 11, 11, 11, 11, 11,
                                 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11,
                                 11, 11, 11, 11, 11, 33, 33, 33, 33, 33, 33,
                                 6, 6, 6, 4, 1, 1, 3, 3, 1, 4, 4, 4],
                 'feature9': [20113, 39, 10, 3554, 6, 7, 10, 30, 4, 7, 12, 2,
                              86, 86],
                 'feature9_esb30': [20113, 39, 10, 3554, 6, 7, 10, 30, 4, 7,
                                    12, 2, 86, 86, 30],
                 'esb30_course': [30, 39]}


def svm_to_ffm(svm_file, ffm_file, feature_name):

    feature_per_field = FEATURE_FIELD[feature_name]
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
    parser.add_argument('--feature-name', '-n', required=True,
                        dest='feature_name')

    args = parser.parse_args()

    start = time.time()
    svm_to_ffm(args.svm_file, args.ffm_file, args.feature_name)
    logging.info('finished ({:.2f} sec elapsed)'.format(time.time() - start))
