#!/usr/bin/env python

import argparse
import numpy as np


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--ffm-file', required=True, dest='ffm_file')
    parser.add_argument('--cv-file', required=True, dest='cv_file')

    args = parser.parse_args()

    cv_id = np.loadtxt(args.cv_file, dtype='int')

    trn1 = args.ffm_file[:-4] + '1.ffm'
    trn2 = args.ffm_file[:-4] + '2.ffm'
    trn3 = args.ffm_file[:-4] + '3.ffm'
    trn4 = args.ffm_file[:-4] + '4.ffm'
    trn5 = args.ffm_file[:-4] + '5.ffm'

    val1 = args.ffm_file[:-7] + 'val1.ffm'
    val2 = args.ffm_file[:-7] + 'val2.ffm'
    val3 = args.ffm_file[:-7] + 'val3.ffm'
    val4 = args.ffm_file[:-7] + 'val4.ffm'
    val5 = args.ffm_file[:-7] + 'val5.ffm'

    with open(args.ffm_file) as fin, \
         open(trn1, 'w') as ft1, \
         open(trn2, 'w') as ft2, \
         open(trn3, 'w') as ft3, \
         open(trn4, 'w') as ft4, \
         open(trn5, 'w') as ft5, \
         open(val1, 'w') as fv1, \
         open(val2, 'w') as fv2, \
         open(val3, 'w') as fv3, \
         open(val4, 'w') as fv4, \
         open(val5, 'w') as fv5:

        for i, row in enumerate(fin):
            if cv_id[i] == 1:
                fv1.write(row)
                ft2.write(row)
                ft3.write(row)
                ft4.write(row)
                ft5.write(row)
            if cv_id[i] == 2:
                ft1.write(row)
                fv2.write(row)
                ft3.write(row)
                ft4.write(row)
                ft5.write(row)
            if cv_id[i] == 3:
                ft1.write(row)
                ft2.write(row)
                fv3.write(row)
                ft4.write(row)
                ft5.write(row)
            if cv_id[i] == 4:
                ft1.write(row)
                ft2.write(row)
                ft3.write(row)
                fv4.write(row)
                ft5.write(row)
            if cv_id[i] == 5:
                ft1.write(row)
                ft2.write(row)
                ft3.write(row)
                ft4.write(row)
                fv5.write(row)
