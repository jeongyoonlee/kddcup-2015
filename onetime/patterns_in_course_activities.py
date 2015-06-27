# -*- coding: utf-8 -*-
"""
Visualize course activitiy

Warning: this script requires 2.5G diskspace for preprocessing step
"""
import os
import datetime
import pickle
import sys

import pandas as pd
from sklearn.preprocessing import LabelEncoder as LE
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

sns.set(style='ticks')


# Parameter
TARGET_COURSE_CD = 6

# Output filepath
OUTPUT_PATH = "course_{course_cd:d}.png".format(course_cd=TARGET_COURSE_CD)

# Input files
OBJECT = "object.csv"
ENR_TEST = "enrollment_test.csv"
ENR_TRAIN = "enrollment_train.csv"
LOG_TEST = "log_test.csv"
LOG_TRAIN = "log_train.csv"
TRUTH = "truth_train.csv"

REF_COURSE_CODE = "ref_course_code.csv"
TEMPFILE = "logfile.pkl"


def load_truth():
    truth_df = pd.read_csv(TRUTH, names=['enrollment_id', 'target'])
    return truth_df


def load_enroll():
    enroll_train_df = pd.read_csv(ENR_TRAIN)
    enroll_test_df = pd.read_csv(ENR_TEST)
    df = pd.concat([enroll_train_df, enroll_test_df])
    return df


def load_log(**args):
    log_train_df = pd.read_csv(LOG_TRAIN)
    log_test_df = pd.read_csv(LOG_TEST)
    df = pd.concat([log_train_df, log_test_df])

    # Merge with enrollment (username)
    enr_df = load_enroll()
    df = df.merge(enr_df, how='left', on='enrollment_id')
    return df


def preprocessing():
    """ Preprocessing part """
    log_df = load_log()
    ref_course_df = pd.read_csv(REF_COURSE_CODE)
    log_df = log_df.merge(ref_course_df, how='left', on='course_id')
    with open(TEMPFILE, 'wb') as f:
        pickle.dump(log_df, f)


def parse_date(s):
    return datetime.datetime.strptime(s, "%Y-%m-%dT%H:%M:%S")


def _change_tick_fontsize(ax, size):
    for tl in ax.get_xticklabels():
        tl.set_fontsize(size)
    for tl in ax.get_yticklabels():
        tl.set_fontsize(size)

def show_course_activity(course_cd=9):
    with open(TEMPFILE, 'rb') as f:
        log_df = pickle.load(f)

    # Load logs
    enr_df = load_enroll()
    ref_course_df = pd.read_csv(REF_COURSE_CODE)
    truth_df = load_truth().fillna(-1)

    enr_df = enr_df.merge(truth_df, how='left', on='enrollment_id')
    enr_df = enr_df.merge(ref_course_df, how='left', on='course_id')
    enr_df = enr_df[enr_df['course_cd'] == course_cd]
    log_df = log_df[log_df['course_cd'] == course_cd]

    obj_df = pd.read_csv(OBJECT, usecols=['module_id', 'category']).rename(columns={'module_id': 'object'})
    log_df = log_df.merge(obj_df, how='left', on='object')

    # Encode object ids
    log_df = log_df.sort('time')
    obj_time = log_df.groupby('object').head(1).reset_index()[['object', 'time']]
    obj_encoder = LE()
    obj_time_ls = obj_encoder.fit(obj_time['object'])
    uniq_obj = len(log_df['object'].unique())
    uniq_obj_names = sorted(obj_df['category'].unique())

    true_enr_id = enr_df[enr_df['target'] == 1].head(50)
    false_enr_id = enr_df[enr_df['target'] == 0].head(50)

    f, ax_list = plt.subplots(50, 2, figsize=(10, 13), sharex=True)

    # For top50 dropout enrollment
    for i, (idx, row) in enumerate(true_enr_id.iterrows()):
        enr_id = row['enrollment_id']
        df = log_df[log_df['enrollment_id'] == enr_id]
        ax = ax_list[i, 0]

        sns.set_palette('husl')
        for category_name in uniq_obj_names:
            selected_df = df[df['category'] == category_name]
            ax.plot(selected_df['time'].map(parse_date),
                    obj_encoder.transform(selected_df['object']),
                    '.')

        ax.set_ylim((0, uniq_obj))
        _change_tick_fontsize(ax, 8)

        dateFmt = mpl.dates.DateFormatter('%Y-%m-%d')
        ax.xaxis.set_major_formatter(dateFmt)
        daysLoc = mpl.dates.DayLocator()
        hoursLoc = mpl.dates.HourLocator(interval=6)
        ax.xaxis.set_major_locator(daysLoc)
        ax.xaxis.set_minor_locator(hoursLoc)
        for ticklabel in ax.xaxis.get_ticklabels():
            ticklabel.set_rotation(80)

    # For top50 continue enrollment
    for i, (idx, row) in enumerate(false_enr_id.iterrows()):
        enr_id = row['enrollment_id']
        df = log_df[log_df['enrollment_id'] == enr_id]
        ax = ax_list[i, 1]

        sns.set_palette('husl')
        for category_name in uniq_obj_names:
            selected_df = df[df['category'] == category_name]
            ax.plot(selected_df['time'].map(parse_date),
                    obj_encoder.transform(selected_df['object']),
                    '.')

        ax.set_ylim((0, uniq_obj))
        _change_tick_fontsize(ax, 8)

        dateFmt = mpl.dates.DateFormatter('%Y-%m-%d')
        ax.xaxis.set_major_formatter(dateFmt)
        daysLoc = mpl.dates.DayLocator()
        hoursLoc = mpl.dates.HourLocator(interval=6)
        ax.xaxis.set_major_locator(daysLoc)
        ax.xaxis.set_minor_locator(hoursLoc)
        for ticklabel in ax.xaxis.get_ticklabels():
            ticklabel.set_rotation(80)

    plt.tight_layout()
    plt.subplots_adjust(top=0.962, hspace=0.09)
    plt.savefig(OUTPUT_PATH)


if __name__ == '__main__':
    #preprocessing()
    show_course_activity(course_cd=TARGET_COURSE_CD)
