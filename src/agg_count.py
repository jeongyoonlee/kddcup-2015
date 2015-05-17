#!/usr/bin/env python

import argparse
import logging
import numpy as np
import time


logging.basicConfig(format='%(asctime)s   %(levelname)s   %(message)s',
                    level=logging.DEBUG)


def _write_target_features(f, feature_dict, label):
    row = []
    for k in sorted(feature_dict.keys()):
        row.append('{}:{}'.format(k, feature_dict[k]))

    f.write('{} {}\n'.format(label, ' '.join(row)))


def agg_count(in_file, out_file, label_file=None):

    if label_file:
        logging.info('loading labels from {}'.format(label_file))
        labels = np.loadtxt(label_file, delimiter=',')[:, 1]

    with open(out_file, 'w') as fout, open(in_file) as fin:

        i_out = 1
        feature_dict = {}
        for i_in, row in enumerate(fin, 1):
            if i_in % 1000000 == 0:
                logging.info('processing {} line(s)'.format(i_in))

            # split a row into the ID and features
            row = row.strip().split(' ')
            _id = row[0]
            features = row[1:]

            # if this is a row with a new ID, write features for the
            # previous ID to the file, and reset feature dictionary.
            if (i_in > 1) and (_id != prev_id):
                _write_target_features(fout, feature_dict, prev_label)
                i_out += 1
                feature_dict = {}

            # add features to the feature dictionary.
            # features are formatted in "index:value"
            for feature in features:
                idx, value = feature.split(':')
                try:
                    feature_dict[int(idx)] += int(value)
                except KeyError:
                    feature_dict[int(idx)] = int(value)

            prev_id = _id
            if label_file:
                prev_label = int(labels[i_out - 1])
            else:
                prev_label = 0

        # at the end of processing, write the last row to the file.
        _write_target_features(fout, feature_dict, prev_label)

        logging.info('{} lines were written'.format(i_out))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--in-file', '-i', required=True, dest='in_file')
    parser.add_argument('--out-file', '-o', required=True, dest='out_file')
    parser.add_argument('--label-file', '-l', default=None, dest='label_file')

    args = parser.parse_args()

    start = time.time()
    agg_count(args.in_file, args.out_file, args.label_file)
    logging.info('finished ({:.2f} sec elapsed)'.format(time.time() - start))
