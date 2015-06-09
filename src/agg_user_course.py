#!/usr/bin/env python

import argparse
import logging
import numpy as np
import time


logging.basicConfig(format='%(asctime)s   %(levelname)s   %(message)s',
                    level=logging.DEBUG)


def _write_target_features(f, feature_dict, _id):
    row = []
    for k in sorted(feature_dict.keys()):
        row.append('{}:{}'.format(k, np.log2(1 + feature_dict[k])))

    f.write('{} {}\n'.format(_id, ' '.join(row)))


def agg_count(in_file, out_file):

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
                _write_target_features(fout, feature_dict, prev_id)
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

        # at the end of processing, write the last row to the file.
        _write_target_features(fout, feature_dict, prev_id)

        logging.info('{} lines were written'.format(i_out))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--in-file', '-i', required=True, dest='in_file')
    parser.add_argument('--out-file', '-o', required=True, dest='out_file')

    args = parser.parse_args()

    start = time.time()
    agg_count(args.in_file, args.out_file)
    logging.info('finished ({:.2f} sec elapsed)'.format(time.time() - start))
