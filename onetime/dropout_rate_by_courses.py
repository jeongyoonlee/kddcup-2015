# -*- coding: utf-8 -*-
import logging as l
import sys
sys.path.append('..')
import os
import datetime

import numpy as np
import pandas as pd
import brewer2mpl
import matplotlib.pyplot as plt
import seaborn as sns


# Output file
OUTPUT_PATH = "course_cd_dropout_rate.png"

# List of file path
ENROLL_TEST = "enrollment_test.csv"
ENROLL_TRAIN = "enrollment_train.csv"
TRUTH_TRAIN = "truth_train.csv"

REF_COURSE_CODE = "ref_course_code.csv"


def load_truth():
    truth_df = pd.read_csv(TRUTH_TRAIN, names=['enrollment_id', 'target'])
    return truth_df


def load_enroll():
    enroll_train_df = pd.read_csv(ENROLL_TRAIN)
    enroll_test_df = pd.read_csv(ENROLL_TEST)
    df = pd.concat([enroll_train_df, enroll_test_df])
    return df


def course_dropout_rate():
    df_tru = load_truth()
    df_enr = load_enroll()
    df_enr = df_enr.merge(df_tru, on='enrollment_id', how='left')
    df_enr = df_enr.merge(pd.read_csv(REF_COURSE_CODE), on='course_id', how='left')
    df_enr.dropna(subset=['target'], inplace=True)

    sns.set_style("whitegrid", {
        "xtick.major.size": 0,
        "ytick.major.size": 0,
    })
    sns.barplot(x='course_cd', y='target', data=df_enr, color="salmon", order=list(range(39)))
    plt.tight_layout()
    plt.savefig(OUTPUT_PATH)
    plt.close()


if __name__ == '__main__':
    l.basicConfig(format='%(asctime)s %(message)s', level=l.INFO)
    course_dropout_rate()
