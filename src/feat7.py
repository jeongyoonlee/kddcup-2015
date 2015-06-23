# -*- coding: utf-8 -*-
"""
* 1 day feature after evaluation period
"""
import datetime
import logging as l
from collections import Counter
import cPickle as pickle

from sklearn.preprocessing import StandardScaler as SS
import pandas as pd
import numpy as np

import utils


def gen_uniq_object_in_judgement_time():
    enr_df = utils.load_enroll()

    df = utils.load_log()
    df_by_course = df.groupby('course_id').agg({'time': 'max'}).reset_index()
    course_evaluation_period = {
        row['course_id']: utils.to_evaluation_period(row['time'], days=1)
        for idx, row in df_by_course.iterrows()
    }
    course_list = course_evaluation_period.keys()

    course_df = {
        course_id: df[
            (df['time'] >= course_evaluation_period[course_id]['begin']) &
            (df['time'] <= course_evaluation_period[course_id]['end'])
        ]
        for course_id in course_list
    }

    feat = []
    df = df.sort('time')
    sz = len(df)
    for i, (idx, df_part) in enumerate(df.groupby(['username', 'course_id'])):
        if i % 100 == 0:
            l.info("{0} of 200k".format(i))
        username = idx[0]
        course_id = idx[1]
        n_activities = len(course_df[course_id][
            (course_df[course_id]['username'] == username)
        ]['object'].unique())

        feat.append({
            'username': idx[0],
            'course_id': idx[1],
            'n_activities': n_activities,
        })

    feat = pd.DataFrame(feat)
    enr_df = enr_df.merge(feat, how='left', on=['username', 'course_id'])
    enr_df.fillna(-1, inplace=True)

    return {'X': utils.reshape(enr_df['n_activities'])}


def gen_proobjuniq_object_in_judgement_time():
    enr_df = utils.load_enroll()

    df = utils.load_log()
    df = df[df['event'] == 'problem']

    df_by_course = df.groupby('course_id').agg({'time': 'max'}).reset_index()
    course_evaluation_period = {
        row['course_id']: utils.to_evaluation_period(row['time'], days=1)
        for idx, row in df_by_course.iterrows()
    }
    course_list = course_evaluation_period.keys()

    course_df = {
        course_id: df[
            (df['time'] >= course_evaluation_period[course_id]['begin']) &
            (df['time'] <= course_evaluation_period[course_id]['end'])
        ]
        for course_id in course_list
    }

    feat = []
    df = df.sort('time')
    sz = len(df)
    for i, (idx, df_part) in enumerate(df.groupby(['username', 'course_id'])):
        if i % 100 == 0:
            l.info("{0} of 200k".format(i))
        username = idx[0]
        course_id = idx[1]
        n_activities = len(course_df[course_id][
            (course_df[course_id]['username'] == username)
        ]['object'].unique())

        feat.append({
            'username': idx[0],
            'course_id': idx[1],
            'n_activities': n_activities,
        })

    feat = pd.DataFrame(feat)
    enr_df = enr_df.merge(feat, how='left', on=['username', 'course_id'])
    enr_df.fillna(-1, inplace=True)

    return {'X': utils.reshape(enr_df['n_activities'])}


def gen_uniq_course_in_judgement_time():
    enr_df = utils.load_enroll()

    df = utils.load_log()
    df_by_course = df.groupby('course_id').agg({'time': 'max'}).reset_index()
    course_evaluation_period = {
        row['course_id']: utils.to_evaluation_period(row['time'], days=1)
        for idx, row in df_by_course.iterrows()
    }
    course_list = course_evaluation_period.keys()

    course_df = {
        course_id: df[
            (df['time'] >= course_evaluation_period[course_id]['begin']) &
            (df['time'] <= course_evaluation_period[course_id]['end'])
        ]
        for course_id in course_list
    }

    feat = []
    df = df.sort('time')
    sz = len(df)
    for i, (idx, df_part) in enumerate(df.groupby(['username', 'course_id'])):
        if i % 100 == 0:
            l.info("{0} of 200k".format(i))
        username = idx[0]
        course_id = idx[1]
        n_activities = len(course_df[course_id][
            (course_df[course_id]['username'] == username)
        ]['course_id'].unique())

        feat.append({
            'username': idx[0],
            'course_id': idx[1],
            'n_activities': n_activities,
        })

    feat = pd.DataFrame(feat)
    enr_df = enr_df.merge(feat, how='left', on=['username', 'course_id'])
    enr_df.fillna(-1, inplace=True)

    return {'X': utils.reshape(enr_df['n_activities'])}


def gen_loglen_in_judgement_time():
    enr_df = utils.load_enroll()

    df = utils.load_log()
    df_by_course = df.groupby('course_id').agg({'time': 'max'}).reset_index()
    course_evaluation_period = {
        row['course_id']: utils.to_evaluation_period(row['time'], days=1)
        for idx, row in df_by_course.iterrows()
    }
    course_list = course_evaluation_period.keys()

    course_df = {
        course_id: df[
            (df['time'] >= course_evaluation_period[course_id]['begin']) &
            (df['time'] <= course_evaluation_period[course_id]['end'])
        ]
        for course_id in course_list
    }

    feat = []
    df = df.sort('time')
    sz = len(df)
    for i, (idx, df_part) in enumerate(df.groupby(['username', 'course_id'])):
        if i % 100 == 0:
            l.info("{0} of 200k".format(i))
        username = idx[0]
        course_id = idx[1]
        n_activities = len(course_df[course_id][
            (course_df[course_id]['username'] == username)
        ])
        print(n_activities)

        feat.append({
            'username': idx[0],
            'course_id': idx[1],
            'n_activities': n_activities,
        })

    feat = pd.DataFrame(feat)
    enr_df = enr_df.merge(feat, how='left', on=['username', 'course_id'])
    enr_df.fillna(-1, inplace=True)

    return {'X': utils.reshape(enr_df['n_activities'])}


def gen_range_size_in_judgement_time():
    enr_df = utils.load_enroll()

    df = utils.load_log()
    df_by_course = df.groupby('course_id').agg({'time': 'max'}).reset_index()
    course_evaluation_period = {
        row['course_id']: utils.to_evaluation_period(row['time'], days=1)
        for idx, row in df_by_course.iterrows()
    }
    course_list = course_evaluation_period.keys()

    course_df = {
        course_id: df[
            (df['time'] >= course_evaluation_period[course_id]['begin']) &
            (df['time'] <= course_evaluation_period[course_id]['end'])
        ]
        for course_id in course_list
    }

    feat = []
    df = df.sort('time')
    sz = len(df)
    for i, (idx, df_part) in enumerate(df.groupby(['username', 'course_id'])):
        if i % 100 == 0:
            l.info("{0} of 200k".format(i))
        username = idx[0]
        course_id = idx[1]
        d = course_df[course_id][
            (course_df[course_id]['username'] == username)
        ]
        diff = -1 if len(d) == 0 else (
            utils.timediff_seconds(
                d['time'].min(),
                d['time'].max()
            )
        )

        feat.append({
            'username': idx[0],
            'course_id': idx[1],
            'n_activities': diff,
        })

    feat = pd.DataFrame(feat)
    enr_df = enr_df.merge(feat, how='left', on=['username', 'course_id'])
    enr_df.fillna(-1, inplace=True)

    return {'X': utils.reshape(enr_df['n_activities'])}


def gen_first_last_in_judgement_time():
    enr_df = utils.load_enroll()

    df = utils.load_log()
    df_by_course = df.groupby('course_id').agg({'time': 'max'}).reset_index()
    course_evaluation_period = {
        row['course_id']: utils.to_evaluation_period(row['time'], days=1)
        for idx, row in df_by_course.iterrows()
    }
    course_list = course_evaluation_period.keys()

    course_df = {
        course_id: df[
            (df['time'] >= course_evaluation_period[course_id]['begin']) &
            (df['time'] <= course_evaluation_period[course_id]['end'])
        ]
        for course_id in course_list
    }

    feat = []
    df = df.sort('time')
    sz = len(df)
    for i, (idx, df_part) in enumerate(df.groupby(['username', 'course_id'])):
        if i % 100 == 0:
            l.info("{0} of 200k".format(i))
        username = idx[0]
        course_id = idx[1]
        d = course_df[course_id][
            (course_df[course_id]['username'] == username)
        ]
        first_time = -1 if len(d) == 0 else utils.to_seconds(d['time'].min())
        last_time = -1 if len(d) == 0 else utils.to_seconds(d['time'].max())

        feat.append({
            'username': idx[0],
            'course_id': idx[1],
            'last_time': last_time,
            'first_time': first_time,
        })

    feat = pd.DataFrame(feat)
    enr_df = enr_df.merge(feat, how='left', on=['username', 'course_id'])
    enr_df.fillna(-1, inplace=True)

    return {
        'first_time': utils.reshape(enr_df['first_time']),
        'last_time': utils.reshape(enr_df['last_time']),
    }


def gen_prob_first_last_in_judgement_time():
    enr_df = utils.load_enroll()

    df = utils.load_log()
    df = df[df['event'] == 'problem']

    df_by_course = df.groupby('course_id').agg({'time': 'max'}).reset_index()
    course_evaluation_period = {
        row['course_id']: utils.to_evaluation_period(row['time'], days=1)
        for idx, row in df_by_course.iterrows()
    }
    course_list = course_evaluation_period.keys()

    course_df = {
        course_id: df[
            (df['time'] >= course_evaluation_period[course_id]['begin']) &
            (df['time'] <= course_evaluation_period[course_id]['end'])
        ]
        for course_id in course_list
    }

    feat = []
    df = df.sort('time')
    sz = len(df)
    for i, (idx, df_part) in enumerate(df.groupby(['username', 'course_id'])):
        if i % 100 == 0:
            l.info("{0} of 200k".format(i))
        username = idx[0]
        course_id = idx[1]
        d = course_df[course_id][
            (course_df[course_id]['username'] == username)
        ]
        first_time = -1 if len(d) == 0 else utils.to_seconds(d['time'].min())
        last_time = -1 if len(d) == 0 else utils.to_seconds(d['time'].max())

        feat.append({
            'username': idx[0],
            'course_id': idx[1],
            'last_time': last_time,
            'first_time': first_time,
        })

    feat = pd.DataFrame(feat)
    enr_df = enr_df.merge(feat, how='left', on=['username', 'course_id'])
    enr_df.fillna(-1, inplace=True)

    return {
        'first_time': utils.reshape(enr_df['first_time']),
        'last_time': utils.reshape(enr_df['last_time']),
    }


def gen_active_hours_in_judgement_time():
    enr_df = utils.load_enroll()

    df = utils.load_log()
    df_by_course = df.groupby('course_id').agg({'time': 'max'}).reset_index()
    course_evaluation_period = {
        row['course_id']: utils.to_evaluation_period(row['time'], days=1)
        for idx, row in df_by_course.iterrows()
    }

    # Preparing extracted logs for each courses
    course_df = {
        course_id: df[
            (df['time'] >= course_evaluation_period[course_id]['begin']) &
            (df['time'] <= course_evaluation_period[course_id]['end'])
        ]
        for course_id in course_evaluation_period.keys()
    }

    feat = []
    df = df.sort('time')
    sz = len(df)
    for i, (idx, df_part) in enumerate(df.groupby(['username', 'course_id'])):
        if i % 100 == 0:
            l.info("{0} of 200k".format(i))
        username = idx[0]
        course_id = idx[1]
        d = course_df[course_id][
            (course_df[course_id]['username'] == username)
        ]
        uniq_hour = len(d['time'].apply(lambda x:
            datetime.datetime.strptime(x, '%Y-%m-%dT%H:%M:%S').strftime(
                '%Y%m%d%H')).unique())

        feat.append({
            'username': idx[0],
            'course_id': idx[1],
            'uniq_hour': uniq_hour,
        })

    feat = pd.DataFrame(feat)
    enr_df = enr_df.merge(feat, how='left', on=['username', 'course_id'])
    enr_df.fillna(-1, inplace=True)

    return {'X': utils.reshape(enr_df['uniq_hour'])}
