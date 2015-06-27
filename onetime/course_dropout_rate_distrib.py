# -*- coding: utf-8 -*-
import os
import cPickle as pickle

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder as LE
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl


sns.set(font="Hiragino Kaku Gothic ProN", font_scale=0.8)

ENROLL_TEST = "data/input/enrollment_test.csv"
ENROLL_TRAIN = "data/input/enrollment_train.csv"
LOG_TRAIN = "data/input/log_train.csv"
TRUTH_TRAIN = "data/input/truth_train.csv"


def _show_all_fonts():
    font_paths = mpl.font_manager.findSystemFonts()
    font_objects = mpl.font_manager.createFontList(font_paths)
    font_names = [f.name for f in font_objects]
    print(font_names)


def _load_dataset():
    df_train = pd.read_csv(ENROLL_TRAIN)
    df_train['dataset'] = 'train'
    df_test = pd.read_csv(ENROLL_TEST)
    df_test['dataset'] = 'test'
    df = pd.concat([df_train, df_test])
    truth_df = pd.read_csv(TRUTH_TRAIN, names=[
        'enrollment_id', 'target',
    ]).replace(1.0, 'dropout').replace(0.0, 'continue')
    df = df.merge(truth_df, how='left', on='enrollment_id')
    df['target'] = df['target'].fillna('testcase')
    df['course_num'] = LE().fit_transform(df['course_id'])
    return df


def course_id_distribution():
    df = _load_dataset()
    sns.countplot(x="course_num", hue="target", data=df)
    plt.savefig("data/output/plot/confirm_train_test_course_id_distribution.png")


def check_course_num():
    df = _load_dataset()
    df = df[df['course_num'] == 1]
    # enrollment_id is an unique value for an username and course_id
    assert len(df['enrollment_id'].unique()) == len(df['username'].unique())


def course_num_1_detail(course_id = '3VkHkmOtom3jM2wCu94xgzzu1d6Dn7or'):
    df = pd.read_csv(LOG_TRAIN)
    df = df[df['course_id'] == course_id]

    truth_df = pd.read_csv(TRUTH_TRAIN, names=['enrollment_id', 'target'])
    df = df.merge(truth_df, how='left', on='enrollment_id')

    log_by_user = df.groupby('enrollment_id').agg({
        'target': 'sum',
        'username': 'count',
    }).reset_index().rename(columns={
        'username': 'num_logs',
    })

    transform_target = lambda x: 'continue' if x == 0 else 'dropout'
    log_by_user['target'] = log_by_user['target'].apply(transform_target)

    bins = np.linspace(0, 1000, 41)
    g = sns.FacetGrid(log_by_user, col="target")
    g.map(plt.hist, "num_logs", color="steelblue", bins=bins, lw=0)
    plt.savefig("data/output/plot/course_user_log_plot.png")


def _load_pickle():
    working_single_course_log = "data/working/single_course_log.pickle"
    if os.path.exists(working_single_course_log):
        with open(working_single_course_log, 'rb') as f:
            df = pickle.load(f)
    else:
        df = pd.read_csv(LOG_TRAIN)
        df = df[df['course_id'] == course_id]
        truth_df = pd.read_csv(TRUTH_TRAIN, names=['enrollment_id', 'target'])
        df = df.merge(truth_df, how='left', on='enrollment_id')

        with open(working_single_course_log, 'wb') as f:
            pickle.dump(df, f)
    return df


def course_num_1_single_record(course_id = '3VkHkmOtom3jM2wCu94xgzzu1d6Dn7or'):
    """
    course_num: 1
    """
    # Log by users
    log_by_user = df.groupby('enrollment_id').agg({
        'target': 'sum',
        'username': 'count',
    }).reset_index().rename(columns={
        'username': 'num_logs',
    })
    log_by_user = log_by_user[log_by_user['num_logs'] == 1]
    log_activity_user = log_by_user[['enrollment_id', 'num_logs']]
    log_activity_user = log_activity_user.merge(df, how='outer', on='enrollment_id')
    df = log_activity_user[log_activity_user['num_logs'] == 1]


if __name__ == '__main__':
    course_id_distribution()
    course_num_1_detail()
    course_num_1_single_record()
