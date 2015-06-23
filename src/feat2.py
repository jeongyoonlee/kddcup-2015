# -*- coding: utf-8 -*-
import datetime
import logging as l
from collections import Counter
import cPickle as pickle

import pandas as pd
import numpy as np

import utils


def gen_course_user_active_hours():
    enr_df = utils.load_enroll()
    df = utils.load_log()

    feat = []
    df = df.sort('time')
    for idx, row in df.groupby(['username', 'course_id']):
        uniq_hour = len(row['time'].apply(lambda x:
            datetime.datetime.strptime(x, '%Y-%m-%dT%H:%M:%S').strftime(
                '%Y%m%d%H')).unique())

        feat.append({
            'username': idx[0],
            'course_id': idx[1],
            'uniq_hour': uniq_hour,
        })

    feat = pd.DataFrame(feat)
    featp = feat.pivot_table(values='uniq_hour', index='username',
            columns='course_id').reset_index()
    featp.columns = ['username'] + list(range(39))

    enr_df = enr_df.merge(featp, how='left', on='username')
    enr_df.fillna(0, inplace=True)

    return {
        'X': np.array(enr_df[list(range(39))]),
    }


def gen_course_user_last_time_fine():
    enr_df = utils.load_enroll()
    df = utils.load_log()
    min_date = utils.to_hours(df['time'].min())

    feat = []
    df = df.sort('time')
    for idx, row in df.groupby(['username', 'course_id']):
        times = sorted(row['time'].tolist())
        last_time = utils.to_hours(times[-1])
        feat.append({
            'username': idx[0],
            'course_id': idx[1],
            'last_time': last_time - min_date,
        })

    feat = pd.DataFrame(feat)
    featp = feat.pivot_table(values='last_time', index='username',
            columns='course_id').reset_index()
    featp.columns = ['username'] + list(range(39))

    enr_df = enr_df.merge(featp, how='left', on='username')
    enr_df.fillna(-1, inplace=True)

    return {
        'X': np.array(enr_df[list(range(39))]),
    }


def gen_course_user_prob_last_time_fine():
    enr_df = utils.load_enroll()
    df = utils.load_log()
    min_date = utils.to_hours(df['time'].min())

    feat = []
    df = df.sort('time')
    df = df[df['event'] == 'problem']
    for idx, row in df.groupby(['username', 'course_id']):
        times = sorted(row['time'].tolist())
        last_time = utils.to_hours(times[-1])
        feat.append({
            'username': idx[0],
            'course_id': idx[1],
            'last_time': last_time - min_date,
        })

    feat = pd.DataFrame(feat)
    featp = feat.pivot_table(values='last_time', index='username',
            columns='course_id').reset_index()
    featp.columns = ['username'] + list(range(39))

    enr_df = enr_df.merge(featp, how='left', on='username')
    enr_df.fillna(-1, inplace=True)

    return {
        'X': np.array(enr_df[list(range(39))]),
    }


def gen_time_by_enrollment_fine():
    # same as "time_feat.gen_first_time.npz" in initial_analysis
    enr_df = utils.load_enroll()
    df = utils.load_log()
    dx = df.groupby('course_id').agg({'time': 'min'}).reset_index()
    course_min_time = {}
    for idx, row in dx.iterrows():
        course_min_time[row['course_id']] = utils.to_hours(row['time'])

    feat = []
    df = df.sort('time')
    for idx, row in df.groupby('enrollment_id'):
        times = sorted(row['time'].tolist())
        course_id = row['course_id'].tolist()[0]
        first_time = utils.to_hours(times[0])
        last_time = utils.to_hours(times[-1])
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
        'first': utils.reshape(enr_df['first_time']),
        'last': utils.reshape(enr_df['last_time']),
    }


def gen_prob_time_by_enrollment_fine():
    # same as "time_feat.gen_first_time.npz" in initial_analysis
    enr_df = utils.load_enroll()

    df = utils.load_log()
    dx = df.groupby('course_id').agg({'time': 'min'}).reset_index()
    course_min_time = {}
    for idx, row in dx.iterrows():
        course_min_time[row['course_id']] = utils.to_seconds(row['time'])

    feat = []
    df = df.sort('time')
    df = df[df['event'] == 'problem']
    for idx, row in df.groupby('enrollment_id'):
        times = sorted(row['time'].tolist())
        course_id = row['course_id'].tolist()[0]
        first_time = utils.to_seconds(times[0])
        last_time = utils.to_seconds(times[-1])
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
        'first': utils.reshape(enr_df['first_time']),
        'last': utils.reshape(enr_df['last_time']),
    }


def gen_course_user_first_time_fine():
    enr_df = utils.load_enroll()
    df = utils.load_log()
    min_date = utils.to_hours(df['time'].min())

    feat = []
    df = df.sort('time')
    for idx, row in df.groupby(['username', 'course_id']):
        times = sorted(row['time'].tolist())
        first_time = utils.to_hours(times[0])
        feat.append({
            'username': idx[0],
            'course_id': idx[1],
            'first_time': first_time - min_date,
        })

    feat = pd.DataFrame(feat)
    featp = feat.pivot_table(values='first_time', index='username',
            columns='course_id').reset_index()
    featp.columns = ['username'] + list(range(39))

    enr_df = enr_df.merge(featp, how='left', on='username')
    enr_df.fillna(-1, inplace=True)

    return {'X': np.array(enr_df[list(range(39))])}


#def gen_course_user_first_time_fine():
def gen_user_event_last_time():
    enr_df = utils.load_enroll()
    df = utils.load_log()
    min_date = utils.to_seconds(df['time'].min())
    df['course_id_x_event'] = df['course_id'] + 'x' + df['event']

    feat = []
    df = df.sort('time')
    for idx, row in df.groupby(['username', 'course_id_x_event']):
        times = sorted(row['time'].tolist())
        last_time = utils.to_seconds(times[-1])
        feat.append({
            'username': idx[0],
            'course_id_x_event': idx[1],
            'last_time': last_time - min_date,
        })

    feat = pd.DataFrame(feat)
    featp = feat.pivot_table(values='last_time', index='username',
            columns='course_id_x_event').reset_index()
    colsz = len(featp.columns) - 1
    featp.columns = ['username'] + list(range(colsz))

    enr_df = enr_df.merge(featp, how='left', on='username')
    enr_df.fillna(-1, inplace=True)

    return {'X': np.array(enr_df[list(range(colsz))])}


def gen_time_by_username_fine():
    # same as "time_feat.gen_time_by_username.npz" in initial_analysis
    enr_df = utils.load_enroll()
    df = utils.load_log()
    min_date = utils.to_hours(df['time'].min())

    feat = []
    df = df.sort('time')
    for idx, row in df.groupby('username'):
        times = sorted(row['time'].tolist())
        first_time = utils.to_hours(times[0])
        last_time = utils.to_hours(times[-1])
        feat.append({
            'username': idx,
            'first_time': first_time - min_date,
            'last_time': last_time - min_date,
        })

    feat = pd.DataFrame(feat)
    enr_df = enr_df.merge(feat, how='left', on='username')
    enr_df['first_time'] = enr_df['first_time'].fillna(-1)
    enr_df['last_time'] = enr_df['last_time'].fillna(-1)

    return {
        'first': utils.reshape(enr_df['first_time']),
        'last': utils.reshape(enr_df['last_time']),
    }


def gen_prob_time_by_username_fine():
    # same as "time_feat.gen_time_by_username.npz" in initial_analysis
    enr_df = utils.load_enroll()
    df = utils.load_log()
    min_date = utils.to_seconds(df['time'].min())
    df = df[df['event'] == 'problem']

    feat = []
    df = df.sort('time')
    for idx, row in df.groupby('username'):
        times = sorted(row['time'].tolist())
        first_time = utils.to_seconds(times[0])
        last_time = utils.to_seconds(times[-1])
        feat.append({
            'username': idx,
            'first_time': first_time - min_date,
            'last_time': last_time - min_date,
        })

    feat = pd.DataFrame(feat)
    enr_df = enr_df.merge(feat, how='left', on='username')
    enr_df['first_time'] = enr_df['first_time'].fillna(-1)
    enr_df['last_time'] = enr_df['last_time'].fillna(-1)

    return {
        'first': utils.reshape(enr_df['first_time']),
        'last': utils.reshape(enr_df['last_time']),
    }
