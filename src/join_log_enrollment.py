#!/usr/bin/env python

import argparse
import pandas as pd


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--log-file', required=True, dest='log_file')
    parser.add_argument('--enrollment-file', required=True,
                        dest='enrollment_file')
    parser.add_argument('--out-file', required=True, dest='out_file')

    args = parser.parse_args()

    log = pd.read_csv(args.log_file)
    enrollment = pd.read_csv(args.enrollment_file)

    df = pd.merge(log, enrollment, on='enrollment_id', how='left')
    df[['enrollment_id', 'username', 'course_id', 'time', 'source', 'event',
        'object']].to_csv(args.out_file, index=False)
