# -*- coding: utf-8 -*-
import logging as l
import os
import datetime

from scipy.stats.stats import pearsonr
import numpy as np
import pandas as pd
import brewer2mpl
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression as LR

sns.set(style='ticks')


# Input files
REF_COURSE_CODE = "ref_course_code.csv"
ENROLL_TEST = "enrollment_test.csv"
ENROLL_TRAIN = "enrollment_train.csv"
LOG_TEST = "log_test.csv"
LOG_TRAIN = "log_train.csv"
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


def load_log(**args):
    log_train_df = pd.read_csv(LOG_TRAIN)
    log_test_df = pd.read_csv(LOG_TEST)
    df = pd.concat([log_train_df, log_test_df])

    # Merge with enrollment (username)
    enr_df = load_enroll()
    df = df.merge(enr_df, how='left', on='enrollment_id')

    return df


def course_cd_by_last_time(X, course_cd=8, bins=80):
    df_tru = load_truth()
    df_tru['is_train'] = 1
    df_enr = load_enroll()
    df_enr['last_access'] = X.ravel()

    df_enr = df_enr.merge(df_tru, on='enrollment_id', how='left')
    df_enr = df_enr.merge(pd.read_csv(REF_COURSE_CODE), on='course_id', how='left')
    df_enr['target_orig'] = df_enr['target']
    df_enr['target'].fillna("testcase", inplace=True)
    df_enr['target'].replace(1.0, "dropout", inplace=True)
    df_enr['target'].replace(0.0, "continue", inplace=True)
    df_enr = df_enr[df_enr['target'] != 'testcase']

    # Compute correlation and p-value
    corr, pval = pearsonr(df_enr['last_access'], df_enr['target_orig'])
    corr_text = "corr={:.6f}, pval={:.6f}".format(corr, pval)

    f, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True)
    mpl_colors = brewer2mpl.get_map('Set1', 'qualitative', 5).mpl_colors

    f.suptitle(corr_text, fontsize=8)
    ax1.set_title("course_cd = {}, target = dropout".format(course_cd))
    ax1.set_ylabel("# enrollment")
    ax1.set_xlabel("last access (last_access - base_time)")
    ax1.hist(np.array(df_enr[
        (df_enr['course_cd'] == course_cd) &
        (df_enr['target'] == 'dropout')
        ]['last_access']), bins=bins, color=mpl_colors[0])

    ax2.set_title("course_cd = {}, target = continue".format(course_cd))
    ax2.set_ylabel("# enrollment")
    ax2.set_xlabel("last access (last_access - base_time)")
    ax2.hist(np.array(df_enr[
        (df_enr['course_cd'] == course_cd) &
        (df_enr['target'] == 'continue')
        ]['last_access']), bins=bins, color=mpl_colors[1])

    # fitting result
    clf = LR()
    clf.fit(
        reshape(df_enr['last_access']),
        reshape(df_enr['target_orig']))
    ax3.set_title("fitting result")
    ax3.plot(np.arange(800),
             clf.predict_proba(reshape(np.arange(800)))[:, 0])

    plt.tight_layout()
    plt.savefig("course_cd_by_last_access_working.png")
    plt.close()


def reshape(series, col_sz=1):
    elem_sz = len(series)
    row_sz = elem_sz / col_sz
    assert row_sz * col_sz == elem_sz
    return np.array(series).reshape((row_sz, col_sz))


def to_hours(s2, fmt='%Y-%m-%dT%H:%M:%S'):
    dt1 = datetime.datetime.strptime('2013-10-01T00:00:00', fmt)
    dt2 = datetime.datetime.strptime(s2, fmt)
    return (dt2 - dt1).total_seconds() / 60.0 / 60.0


def gen_time_by_enrollment_fine():
    enr_df = load_enroll()
    df = load_log()
    dx = df.groupby('course_id').agg({'time': 'min'}).reset_index()

    course_min_time = {}
    for idx, row in dx.iterrows():
        course_min_time[row['course_id']] = to_hours(row['time'])

    feat = []
    df = df.sort('time')
    for idx, row in df.groupby('enrollment_id'):
        times = sorted(row['time'].tolist())
        course_id = row['course_id'].tolist()[0]
        first_time = to_hours(times[0])
        last_time = to_hours(times[-1])
        min_time = course_min_time[course_id]
        feat.append({
            'enrollment_id': idx,
            'first_time': first_time - min_time,
            'last_time': last_time - min_time,
        })

    feat = pd.DataFrame(feat)
    enr_df = enr_df.merge(feat, how='left', on='enrollment_id')
    enr_df['first_time'] = enr_df['first_time'].fillna(-1)
    enr_df['last_time'] = enr_df['last_time'].fillna(-1)

    return {
        'first': reshape(enr_df['first_time']),
        'last': reshape(enr_df['last_time']),
    }


if __name__ == '__main__':
    l.basicConfig(format='%(asctime)s %(message)s', level=l.INFO)
    X = gen_time_by_enrollment_fine()['last']
    course_cd_by_last_time(X, course_cd=11)
