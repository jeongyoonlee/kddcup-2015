# -*- coding: utf-8 -*-
import datetime
import logging as l
from collections import Counter
import cPickle as pickle

from sklearn.preprocessing import StandardScaler as SS
import pandas as pd
import numpy as np

import utils


def gen_base():
    df = utils.load_enroll()
    train_sz = len(pd.read_csv(utils.ENROLL_TRAIN))
    truth_df = pd.read_csv(utils.TRUTH_TRAIN, names=['enrollment_id', 'target'])

    df = df.merge(truth_df, how='left', on='enrollment_id')
    assert train_sz == 120542
    assert len(df) == 200904
    return {
        'y': utils.reshape(df['target'])[:train_sz],
        'id_train': utils.reshape(df['enrollment_id'])[:train_sz],
        'id_test': utils.reshape(df['enrollment_id'])[train_sz:],
    }


def gen_course_cat():
    df = utils.load_enroll()
    X = utils.onehot_encode(df['course_id'])
    return {'X': X}


def gen_user_cat():
    df = utils.load_enroll()
    X = utils.onehot_encode(df['username'])
    return {'X': X}


def gen_course_user_source_last_time():
    enr_df = utils.load_enroll()
    df = utils.load_log()
    min_date = utils.toordinal(df['time'].min())

    feat = []
    df = df.sort('time')
    for idx, row in df.groupby(['username', 'source', 'course_id']):
        times = sorted(row['time'].tolist())
        first_time = utils.toordinal(times[0])
        last_time = utils.toordinal(times[-1])
        feat.append({
            'username': idx[0],
            'source': idx[1],
            'course_id': idx[2],
            'last_time': last_time - min_date,
        })

    feat = pd.DataFrame(feat)
    featp = feat.pivot_table(values='last_time', index='username',
            columns=['course_id', 'source']).reset_index()
    col_sz = len(featp.columns) - 1
    featp.columns = ['username'] + list(range(col_sz))

    enr_df = enr_df.merge(featp, how='left', on='username')
    enr_df.fillna(-1, inplace=True)

    return {
        'X': np.array(enr_df[list(range(col_sz))]),
    }


def gen_course_user_source_first_time():
    enr_df = utils.load_enroll()
    df = utils.load_log()
    min_date = utils.toordinal(df['time'].min())
    df.sort('time')

    feat = []
    df = df.sort('time')
    for idx, row in df.groupby(['username', 'source', 'course_id']):
        times = sorted(row['time'].tolist())
        first_time = utils.toordinal(times[0])
        last_time = utils.toordinal(times[-1])
        feat.append({
            'username': idx[0],
            'source': idx[1],
            'course_id': idx[2],
            'first_time': first_time - min_date,
        })

    feat = pd.DataFrame(feat)
    featp = feat.pivot_table(values='first_time', index='username',
            columns=['course_id', 'source']).reset_index()
    col_sz = len(featp.columns) - 1
    featp.columns = ['username'] + list(range(col_sz))

    enr_df = enr_df.merge(featp, how='left', on='username')
    enr_df.fillna(-1, inplace=True)

    return {
        'X': np.array(enr_df[list(range(col_sz))]),
    }


def gen_course_user_event_last_time():
    enr_df = utils.load_enroll()
    df = utils.load_log()
    min_date = utils.toordinal(df['time'].min())

    feat = []
    df = df.sort('time')
    for idx, row in df.groupby(['username', 'event', 'course_id']):
        times = sorted(row['time'].tolist())
        first_time = utils.toordinal(times[0])
        last_time = utils.toordinal(times[-1])
        feat.append({
            'username': idx[0],
            'event': idx[1],
            'course_id': idx[2],
            'last_time': last_time - min_date,
        })

    feat = pd.DataFrame(feat)
    featp = feat.pivot_table(values='last_time', index='username',
            columns=['course_id', 'event']).reset_index()
    col_sz = len(featp.columns) - 1
    featp.columns = ['username'] + list(range(col_sz))

    enr_df = enr_df.merge(featp, how='left', on='username')
    enr_df.fillna(-1, inplace=True)

    return {
        'X': np.array(enr_df[list(range(col_sz))]),
    }


def gen_course_user_last_time():
    enr_df = utils.load_enroll()
    df = utils.load_log()
    min_date = utils.toordinal(df['time'].min())

    feat = []
    df = df.sort('time')
    for idx, row in df.groupby(['username', 'course_id']):
        times = sorted(row['time'].tolist())
        first_time = utils.toordinal(times[0])
        last_time = utils.toordinal(times[-1])
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


def gen_course_user_last_time_with_category():
    enr_df = utils.load_enroll()
    df = utils.load_log_with_obj_attrib()
    min_date = utils.toordinal(df['time'].min())
    df['course_x_cat'] = df['category'] + df['course_id']

    feat = []
    df = df.sort('time')
    for idx, row in df.groupby(['username', 'course_x_cat']):
        times = sorted(row['time'].tolist())
        first_time = utils.toordinal(times[0])
        last_time = utils.toordinal(times[-1])
        feat.append({
            'username': idx[0],
            'course_x_cat': idx[1],
            'last_time': last_time - min_date,
        })

    feat = pd.DataFrame(feat)
    featp = feat.pivot_table(values='last_time', index='username',
            columns='course_x_cat').reset_index()
    col_sz = len(featp.columns) - 1
    featp.columns = ['username'] + list(range(col_sz))

    enr_df = enr_df.merge(featp, how='left', on='username')
    enr_df.fillna(-1, inplace=True)

    return {
        'X': np.array(enr_df[list(range(col_sz))]),
    }


def gen_event_last_time():
    enr_df = utils.load_enroll()
    df = utils.load_log_with_obj_attrib()
    min_date = utils.toordinal(df['time'].min())

    feat = []
    df = df.sort('time')
    for idx, row in df.groupby(['username', 'event']):
        times = sorted(row['time'].tolist())
        first_time = utils.toordinal(times[0])
        last_time = utils.toordinal(times[-1])
        feat.append({
            'username': idx[0],
            'event': idx[1],
            'last_time': last_time - min_date,
        })

    feat = pd.DataFrame(feat)
    featp = feat.pivot_table(values='last_time', index='username',
            columns='event').reset_index()
    col_sz = len(featp.columns) - 1
    featp.columns = ['username'] + list(range(col_sz))

    enr_df = enr_df.merge(featp, how='left', on='username')
    enr_df.fillna(-1, inplace=True)

    return {
        'X': np.array(enr_df[list(range(col_sz))]),
    }


def gen_season_user_last_time():
    enr_df = utils.load_enroll()
    df = utils.load_log()
    df = pd.merge(df, pd.read_csv("data/input/ref_course_code.csv"),
            how='left', on='course_id')
    min_date = utils.toordinal(df['time'].min())

    feat = []
    df = df.sort('time')
    for idx, row in df.groupby(['username', 'season']):
        times = sorted(row['time'].tolist())
        first_time = utils.toordinal(times[0])
        last_time = utils.toordinal(times[-1])
        feat.append({
            'username': idx[0],
            'season': idx[1],
            'last_time': last_time - min_date,
        })

    feat = pd.DataFrame(feat)
    featp = feat.pivot_table(values='last_time', index='username',
            columns='season').reset_index()
    featp.columns = ['username'] + list(range(2))

    enr_df = enr_df.merge(featp, how='left', on='username')
    enr_df.fillna(-1, inplace=True)

    return {
        'X': np.array(enr_df[list(range(2))]),
    }


def gen_course_user_first_time():
    enr_df = utils.load_enroll()
    df = utils.load_log()
    min_date = utils.toordinal(df['time'].min())

    feat = []
    df = df.sort('time')
    for idx, row in df.groupby(['username', 'course_id']):
        times = sorted(row['time'].tolist())
        first_time = utils.toordinal(times[0])
        last_time = utils.toordinal(times[-1])
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


def gen_user_source_loglen():
    enr_df = utils.load_enroll()
    df = utils.load_log()

    feat = []
    for idx, row in df.groupby(['username', 'source']):
        sz = len(row['object'])
        feat.append({
            'username': idx[0],
            'source': idx[1],
            'count': sz,
        })
    feat = pd.DataFrame(feat)
    featp = feat.pivot_table(values='count', index='username',
            columns='source').reset_index()
    colsz = len(featp.columns) - 1
    featp.columns = ['username'] + list(range(colsz))

    enr_df = enr_df.merge(featp, how='left', on='username')
    enr_df.fillna(-1, inplace=True)

    return {'X': np.array(enr_df[list(range(colsz))])}


def gen_user_course_source_loglen():
    enr_df = utils.load_enroll()
    df = utils.load_log()

    feat = []
    for idx, row in df.groupby(['username', 'course_id', 'source']):
        sz = len(row['object'])
        feat.append({
            'username': idx[0],
            'course_id': idx[1],
            'source': idx[2],
            'count': sz,
        })
    feat = pd.DataFrame(feat)
    featp = feat.pivot_table(values='count', index='username',
            columns=['course_id', 'source']).reset_index()
    colsz = len(featp.columns) - 1
    featp.columns = ['username'] + list(range(colsz))

    enr_df = enr_df.merge(featp, how='left', on='username')
    enr_df.fillna(-1, inplace=True)

    return {'X': np.array(enr_df[list(range(colsz))])}


def gen_enroll_source_event_loglen():
    enr_df = utils.load_enroll()
    df = utils.load_log()

    feat = []
    for idx, row in df.groupby(['enrollment_id', 'event', 'source']):
        sz = len(row['object'])
        feat.append({
            'enrollment_id': idx[0],
            'source': idx[1],
            'event': idx[2],
            'count': sz,
        })
    feat = pd.DataFrame(feat)
    featp = feat.pivot_table(values='count', index='enrollment_id',
            columns=['event', 'source']).reset_index()
    colsz = len(featp.columns) - 1
    featp.columns = ['enrollment_id'] + list(range(colsz))

    enr_df = enr_df.merge(featp, how='left', on='enrollment_id')
    enr_df.fillna(-1, inplace=True)

    return {'X': np.array(enr_df[list(range(colsz))])}


def gen_user_event_loglen():
    enr_df = utils.load_enroll()
    df = utils.load_log()

    feat = []
    for idx, row in df.groupby(['username', 'event']):
        sz = len(row['object'])
        feat.append({
            'username': idx[0],
            'event': idx[1],
            'count': sz,
        })
    feat = pd.DataFrame(feat)
    featp = feat.pivot_table(values='count', index='username',
            columns='event').reset_index()
    colsz = len(featp.columns) - 1
    featp.columns = ['username'] + list(range(colsz))

    enr_df = enr_df.merge(featp, how='left', on='username')
    enr_df.fillna(-1, inplace=True)

    return {'X': np.array(enr_df[list(range(colsz))])}


def gen_course_user_loglen():
    enr_df = utils.load_enroll()
    df = utils.load_log()

    feat = []
    for idx, row in df.groupby(['username', 'course_id']):
        sz = len(row['object'])
        feat.append({
            'username': idx[0],
            'course_id': idx[1],
            'count': sz,
        })
    feat = pd.DataFrame(feat)
    featp = feat.pivot_table(values='count', index='username',
            columns='course_id').reset_index()
    featp.columns = ['username'] + list(range(39))

    enr_df = enr_df.merge(featp, how='left', on='username')
    enr_df.fillna(-1, inplace=True)

    return {'X': np.array(enr_df[list(range(39))])}


def gen_enroll_source_event_loguniq():
    enr_df = utils.load_enroll()
    df = utils.load_log()

    feat = []
    for idx, row in df.groupby(['enrollment_id', 'event', 'source']):
        unq = len(row['object'].unique())
        feat.append({
            'enrollment_id': idx[0],
            'source': idx[1],
            'event': idx[2],
            'count': unq,
        })
    feat = pd.DataFrame(feat)
    featp = feat.pivot_table(values='count', index='enrollment_id',
            columns=['event', 'source']).reset_index()
    colsz = len(featp.columns) - 1
    featp.columns = ['enrollment_id'] + list(range(colsz))

    enr_df = enr_df.merge(featp, how='left', on='enrollment_id')
    enr_df.fillna(-1, inplace=True)

    return {'X': np.array(enr_df[list(range(colsz))])}


def gen_user_event_loguniq():
    enr_df = utils.load_enroll()
    df = utils.load_log()

    feat = []
    for idx, row in df.groupby(['username', 'event']):
        unq = len(row['object'].unique())
        feat.append({
            'username': idx[0],
            'event': idx[1],
            'count': unq,
        })
    feat = pd.DataFrame(feat)
    featp = feat.pivot_table(values='count', index='username',
            columns='event').reset_index()
    colsz = len(featp.columns) - 1
    featp.columns = ['username'] + list(range(colsz))

    enr_df = enr_df.merge(featp, how='left', on='username')
    enr_df.fillna(-1, inplace=True)

    return {'X': np.array(enr_df[list(range(colsz))])}


def gen_user_source_loguniq():
    enr_df = utils.load_enroll()
    df = utils.load_log()

    feat = []
    for idx, row in df.groupby(['username', 'source']):
        unq = len(row['object'].unique())
        feat.append({
            'username': idx[0],
            'source': idx[1],
            'count': unq,
        })
    feat = pd.DataFrame(feat)
    featp = feat.pivot_table(values='count', index='username',
            columns='source').reset_index()
    colsz = len(featp.columns) - 1
    featp.columns = ['username'] + list(range(colsz))

    enr_df = enr_df.merge(featp, how='left', on='username')
    enr_df.fillna(-1, inplace=True)

    return {'X': np.array(enr_df[list(range(colsz))])}


def gen_user_category_loguniq():
    enr_df = utils.load_enroll()
    df = utils.load_log_with_obj_attrib()

    feat = []
    for idx, row in df.groupby(['username', 'category']):
        unq = len(row['object'].unique())
        feat.append({
            'username': idx[0],
            'category': idx[1],
            'count': unq,
        })
    feat = pd.DataFrame(feat)
    featp = feat.pivot_table(values='count', index='username',
            columns='category').reset_index()
    colsz = len(featp.columns) - 1
    featp.columns = ['username'] + list(range(colsz))

    enr_df = enr_df.merge(featp, how='left', on='username')
    enr_df.fillna(-1, inplace=True)

    return {'X': np.array(enr_df[list(range(colsz))])}


def gen_course_user_loguniq():
    enr_df = utils.load_enroll()
    df = utils.load_log()

    feat = []
    for idx, row in df.groupby(['username', 'course_id']):
        unq = len(row['object'].unique())
        feat.append({
            'username': idx[0],
            'course_id': idx[1],
            'count': unq,
        })
    feat = pd.DataFrame(feat)
    featp = feat.pivot_table(values='count', index='username',
            columns='course_id').reset_index()
    colsz = len(featp.columns) - 1
    featp.columns = ['username'] + list(range(colsz))

    enr_df = enr_df.merge(featp, how='left', on='username')
    enr_df.fillna(-1, inplace=True)

    return {'X': np.array(enr_df[list(range(colsz))])}


def gen_course_user_logproobjuniq():
    enr_df = utils.load_enroll()
    df = utils.load_log()

    feat = []
    for idx, row in df.groupby(['username', 'course_id']):
        ev = row[row['event'] == 'problem']
        feat.append({
            'username': idx[0],
            'course_id': idx[1],
            'count': len(ev['object'].unique()),
        })

    feat = pd.DataFrame(feat)
    featp = feat.pivot_table(values='count', index='username',
            columns='course_id').reset_index()
    colsz = len(featp.columns) - 1
    featp.columns = ['username'] + list(range(colsz))

    enr_df = enr_df.merge(featp, how='left', on='username')
    enr_df.fillna(0, inplace=True)

    return {'X': np.array(enr_df[list(range(colsz))])}


def gen_user_event_logtime():
    enr_df = utils.load_enroll()
    df = utils.load_log()

    feat = []
    for idx, row in df.groupby(['username', 'event']):
        uniq_day = len(row['time'].apply(lambda x:
            datetime.datetime.strptime(x, '%Y-%m-%dT%H:%M:%S').strftime(
                '%Y%m%d')).unique())
        feat.append({
            'username': idx[0],
            'event': idx[1],
            'count': uniq_day,
        })
    feat = pd.DataFrame(feat)
    featp = feat.pivot_table(values='count', index='username',
            columns='event').reset_index()
    colsz = len(featp.columns) - 1
    featp.columns = ['username'] + list(range(colsz))

    enr_df = enr_df.merge(featp, how='left', on='username')
    enr_df.fillna(-1, inplace=True)

    return {'X': np.array(enr_df[list(range(colsz))])}


def gen_user_source_logtime():
    enr_df = utils.load_enroll()
    df = utils.load_log()

    feat = []
    for idx, row in df.groupby(['username', 'source']):
        uniq_day = len(row['time'].apply(lambda x:
            datetime.datetime.strptime(x, '%Y-%m-%dT%H:%M:%S').strftime(
                '%Y%m%d')).unique())
        feat.append({
            'username': idx[0],
            'source': idx[1],
            'count': uniq_day,
        })
    feat = pd.DataFrame(feat)
    featp = feat.pivot_table(values='count', index='username',
            columns='source').reset_index()
    colsz = len(featp.columns) - 1
    featp.columns = ['username'] + list(range(colsz))

    enr_df = enr_df.merge(featp, how='left', on='username')
    enr_df.fillna(-1, inplace=True)

    return {'X': np.array(enr_df[list(range(colsz))])}


def gen_course_user_logtime():
    enr_df = utils.load_enroll()
    df = utils.load_log()

    feat = []
    for idx, row in df.groupby(['username', 'course_id']):
        uniq_day = len(row['time'].apply(lambda x:
            datetime.datetime.strptime(x, '%Y-%m-%dT%H:%M:%S').strftime(
                '%Y%m%d')).unique())
        feat.append({
            'username': idx[0],
            'course_id': idx[1],
            'count': uniq_day,
        })
    feat = pd.DataFrame(feat)
    featp = feat.pivot_table(values='count', index='username',
            columns='course_id').reset_index()
    featp.columns = ['username'] + list(range(39))

    enr_df = enr_df.merge(featp, how='left', on='username')
    enr_df.fillna(-1, inplace=True)

    return {'X': np.array(enr_df[list(range(39))])}


def gen_course_user_loghourtime():
    enr_df = utils.load_enroll()
    df = utils.load_log()

    feat = []
    for idx, row in df.groupby(['username', 'course_id']):
        uniq_day = len(row['time'].apply(lambda x:
            datetime.datetime.strptime(x, '%Y-%m-%dT%H:%M:%S').strftime(
                '%Y%m%d%H')).unique())
        feat.append({
            'username': idx[0],
            'course_id': idx[1],
            'count': uniq_day,
        })
    feat = pd.DataFrame(feat)
    featp = feat.pivot_table(values='count', index='username',
            columns='course_id').reset_index()
    featp.columns = ['username'] + list(range(39))

    enr_df = enr_df.merge(featp, how='left', on='username')
    enr_df.fillna(-1, inplace=True)

    return {'X': np.array(enr_df[list(range(39))])}


def gen_loglen():
    enr_df = utils.load_enroll()
    log_df = utils.load_log()
    log_count_df = log_df[['enrollment_id']].groupby('enrollment_id').agg({
        'enrollment_id': 'count'
    }).rename(columns={
        'enrollment_id': 'log_count'
    }).reset_index()

    enr_df = enr_df.merge(log_count_df, how='left', on='enrollment_id').fillna(0)
    return {'X': utils.reshape(enr_df['log_count'])}


def gen_enrollment_order():
    enr_df = utils.load_enroll()
    feat_raw = []
    for idx, enr_row in enr_df.groupby(['course_id']):
        enr_id_list = enr_row.sort('enrollment_id').enrollment_id.tolist()
        enr_order_list = np.arange(len(enr_id_list))
        feat_raw.append(pd.DataFrame({
            'enrollment_id': enr_id_list,
            'order': enr_order_list
        }))
    feat = pd.concat(feat_raw)
    enr_df = enr_df.merge(feat, how='left', on='enrollment_id')
    return {'X': utils.reshape(enr_df['order'])}


def gen_loguniq():
    # Compute number of uniq object by enrollment_id
    log_df = utils.load_log()
    arr = []
    for eid, part_df in log_df.groupby('enrollment_id'):
        part_d = {'enrollment_id': eid}
        part_d['uniq_object'] = len(part_df['object'].unique())
        arr.append(part_d)
    feat_df = pd.DataFrame(arr)

    # Merge with enrollment_id
    enr_df = utils.load_enroll()
    enr_df = enr_df.merge(feat_df, how='left', on='enrollment_id').fillna(0)

    return {'X': utils.reshape(enr_df['uniq_object'])}


def gen_uniq_source_event_obj():
    # Compute number of uniq object by enrollment_id
    log_df = utils.load_log()
    arr = []
    for eid, part_df in log_df.groupby('enrollment_id'):
        part_d = {'enrollment_id': eid}
        part_d['sz'] = len(part_df[['source', 'event',
            'object']].drop_duplicates())
        arr.append(part_d)
    feat_df = pd.DataFrame(arr)

    # Merge with enrollment_id
    enr_df = utils.load_enroll()
    enr_df = enr_df.merge(feat_df, how='left', on='enrollment_id').fillna(0)

    return {'X': utils.reshape(enr_df['sz'])}


def gen_logtime():
    df = utils.load_enroll()
    log_df = utils.load_log()

    arr = []
    for eid, part_df in log_df.groupby('enrollment_id'):
        part_d = {'enrollment_id': eid}
        part_d['uniq_days'] = len(part_df['time'].apply(lambda x:
            datetime.datetime.strptime(x, '%Y-%m-%dT%H:%M:%S').strftime(
                '%Y%m%d')).unique())
        arr.append(part_d)

    feat_df = pd.DataFrame(arr)
    df = df.merge(feat_df, how='left', on='enrollment_id').fillna(0)
    return {'X': utils.reshape(df['uniq_days'])}


def gen_video_loglen():
    df = utils.load_enroll()
    log_df = utils.load_log()
    log_df = log_df[log_df['event'] == 'video']

    arr = []
    for eid, part_df in log_df.groupby('enrollment_id'):
        part_d = {'enrollment_id': eid}
        part_d['sz'] = len(part_df)
        arr.append(part_d)

    feat_df = pd.DataFrame(arr)
    df = df.merge(feat_df, how='left', on='enrollment_id').fillna(0)
    return {'X': utils.reshape(df['sz'])}


def gen_prob_loglen():
    df = utils.load_enroll()
    log_df = utils.load_log()
    log_df = log_df[log_df['event'] == 'problem']

    arr = []
    for eid, part_df in log_df.groupby('enrollment_id'):
        part_d = {'enrollment_id': eid}
        part_d['sz'] = len(part_df)
        arr.append(part_d)

    feat_df = pd.DataFrame(arr)
    df = df.merge(feat_df, how='left', on='enrollment_id').fillna(0)
    return {'X': utils.reshape(df['sz'])}


def gen_nagi_loglen():
    df = utils.load_enroll()
    log_df = utils.load_log()
    log_df = log_df[log_df['event'] == 'nagivate']

    arr = []
    for eid, part_df in log_df.groupby('enrollment_id'):
        part_d = {'enrollment_id': eid}
        part_d['sz'] = len(part_df)
        arr.append(part_d)

    feat_df = pd.DataFrame(arr)
    df = df.merge(feat_df, how='left', on='enrollment_id').fillna(0)
    return {'X': utils.reshape(df['sz'])}


def gen_page_close_loglen():
    df = utils.load_enroll()
    log_df = utils.load_log()
    log_df = log_df[log_df['event'] == 'page_close']

    arr = []
    for eid, part_df in log_df.groupby('enrollment_id'):
        part_d = {'enrollment_id': eid}
        part_d['sz'] = len(part_df)
        arr.append(part_d)

    feat_df = pd.DataFrame(arr)
    df = df.merge(feat_df, how='left', on='enrollment_id').fillna(0)
    return {'X': utils.reshape(df['sz'])}


def gen_page_close_obj_topfreq():
    df = utils.load_enroll()
    log_df = utils.load_log()
    log_df = log_df[log_df['event'] == 'page_close']

    arr = []
    for eid, part_df in log_df.groupby('enrollment_id'):
        part_d = {'enrollment_id': eid}
        part_d['sz'] = part_df['object'].describe()['freq']
        arr.append(part_d)

    feat_df = pd.DataFrame(arr)
    df = df.merge(feat_df, how='left', on='enrollment_id').fillna(0)
    return {'X': utils.reshape(df['sz'])}


def gen_uniq_event_source():
    df = utils.load_enroll()
    log_df = utils.load_log()
    log_df['source_event'] = log_df['source'] + log_df['event']

    arr = []
    for eid, part_df in log_df.groupby('enrollment_id'):
        part_d = {'enrollment_id': eid}
        part_d['sz'] = len(part_df['source_event'].unique())
        arr.append(part_d)

    feat_df = pd.DataFrame(arr)
    df = df.merge(feat_df, how='left', on='enrollment_id').fillna(0)
    return {'X': utils.reshape(df['sz'])}


def gen_prob_logday():
    df = utils.load_enroll()
    log_df = utils.load_log()
    log_df = log_df[log_df['event'] == 'problem']

    arr = []
    for eid, part_df in log_df.groupby('enrollment_id'):
        part_d = {'enrollment_id': eid}
        part_d['uniq_days'] = len(part_df['time'].apply(lambda x:
            datetime.datetime.strptime(x, '%Y-%m-%dT%H:%M:%S').strftime(
                '%Y%m%d')).unique())
        arr.append(part_d)

    feat_df = pd.DataFrame(arr)
    df = df.merge(feat_df, how='left', on='enrollment_id').fillna(0)
    return {'X': utils.reshape(df['uniq_days'])}


def gen_prob_loghour():
    df = utils.load_enroll()
    log_df = utils.load_log()
    log_df = log_df[log_df['event'] == 'problem']

    arr = []
    for eid, part_df in log_df.groupby('enrollment_id'):
        part_d = {'enrollment_id': eid}
        part_d['uniq_days'] = len(part_df['time'].apply(lambda x:
            datetime.datetime.strptime(x, '%Y-%m-%dT%H:%M:%S').strftime(
                '%Y%m%d%H')).unique())
        arr.append(part_d)

    feat_df = pd.DataFrame(arr)
    df = df.merge(feat_df, how='left', on='enrollment_id').fillna(0)
    return {'X': utils.reshape(df['uniq_days'])}


def gen_logtime_th10():
    df = utils.load_enroll()
    log_df = utils.load_log()

    arr = []
    for eid, part_df in log_df.groupby('enrollment_id'):
        part_d = {'enrollment_id': eid}
        threshold = 10

        part = part_df.copy()
        part['count'] = 1
        part['time'] = part['time'].apply(lambda x:
            datetime.datetime.strptime(x, '%Y-%m-%dT%H:%M:%S').strftime(
                '%Y%m%d'))
        part_d['uniq_day_with_threshold'] = len(
            part.groupby('time').agg(np.sum).reset_index(
            ).query('count > {}'.format(threshold))
        )
        arr.append(part_d)

    feat_df = pd.DataFrame(arr)
    df = df.merge(feat_df, how='left', on='enrollment_id').fillna(0)
    return {'X': utils.reshape(df['uniq_day_with_threshold'])}


def gen_logtime_th20():
    df = utils.load_enroll()
    log_df = utils.load_log()

    arr = []
    for eid, part_df in log_df.groupby('enrollment_id'):
        part_d = {'enrollment_id': eid}
        threshold = 20

        part = part_df.copy()
        part['count'] = 1
        part['time'] = part['time'].apply(lambda x:
            datetime.datetime.strptime(x, '%Y-%m-%dT%H:%M:%S').strftime(
                '%Y%m%d'))
        part_d['uniq_day_with_threshold'] = len(
            part.groupby('time').agg(np.sum).reset_index(
            ).query('count > {}'.format(threshold))
        )
        arr.append(part_d)

    feat_df = pd.DataFrame(arr)
    df = df.merge(feat_df, how='left', on='enrollment_id').fillna(0)
    return {'X': utils.reshape(df['uniq_day_with_threshold'])}


def gen_logtime_th40():
    df = utils.load_enroll()
    log_df = utils.load_log()

    arr = []
    for eid, part_df in log_df.groupby('enrollment_id'):
        part_d = {'enrollment_id': eid}
        threshold = 40

        part = part_df.copy()
        part['count'] = 1
        part['time'] = part['time'].apply(lambda x:
            datetime.datetime.strptime(x, '%Y-%m-%dT%H:%M:%S').strftime(
                '%Y%m%d'))
        part_d['uniq_day_with_threshold'] = len(
            part.groupby('time').agg(np.sum).reset_index(
            ).query('count > {}'.format(threshold))
        )
        arr.append(part_d)

    feat_df = pd.DataFrame(arr)
    df = df.merge(feat_df, how='left', on='enrollment_id').fillna(0)
    return {'X': utils.reshape(df['uniq_day_with_threshold'])}


def gen_logtime_th80():
    df = utils.load_enroll()
    log_df = utils.load_log()

    arr = []
    for eid, part_df in log_df.groupby('enrollment_id'):
        part_d = {'enrollment_id': eid}
        threshold = 80

        part = part_df.copy()
        part['count'] = 1
        part['time'] = part['time'].apply(lambda x:
            datetime.datetime.strptime(x, '%Y-%m-%dT%H:%M:%S').strftime(
                '%Y%m%d'))
        part_d['uniq_day_with_threshold'] = len(
            part.groupby('time').agg(np.sum).reset_index(
            ).query('count > {}'.format(threshold))
        )
        arr.append(part_d)

    feat_df = pd.DataFrame(arr)
    df = df.merge(feat_df, how='left', on='enrollment_id').fillna(0)
    return {'X': utils.reshape(df['uniq_day_with_threshold'])}


def gen_loghourtime():
    df = utils.load_enroll()
    log_df = utils.load_log()
    log_sz = len(log_df.groupby('enrollment_id'))
    arr = []
    for i, (eid, part_df) in enumerate(log_df.groupby('enrollment_id')):
        if i % 1000 == 0:
            l.info("{0} of {1}".format(i, log_sz))

        part_d = {'enrollment_id': eid}
        part_d['uniq_hour'] = len(part_df['time'].apply(lambda x:
            datetime.datetime.strptime(x, '%Y-%m-%dT%H:%M:%S').strftime(
                '%Y%m%d%H')).unique())
        arr.append(part_d)

    feat_df = pd.DataFrame(arr)
    df = df.merge(feat_df, how='left', on='enrollment_id').fillna(0)
    return {'X': utils.reshape(df['uniq_hour'])}


def gen_logproobjuniq():
    df = utils.load_enroll()
    log_df = utils.load_log()
    arr = []
    log_sz = len(log_df.groupby('enrollment_id'))

    for i, (eid, part_df) in enumerate(log_df.groupby('enrollment_id')):
        if i % 1000 == 0:
            l.info("{0} of {1}".format(i, log_sz))

        ev = part_df[part_df['event'] == 'problem']
        part_d = {'enrollment_id': eid}
        part_d['evuniq'] = len(ev['object'].unique())
        arr.append(part_d)

    feat_df = pd.DataFrame(arr)
    df = df.merge(feat_df, how='left', on='enrollment_id').fillna(0)
    return {'X': utils.reshape(df['evuniq'])}


def gen_log_problem_obj_uniq():
    df = utils.load_enroll()
    log_df = utils.load_log()
    arr = []
    log_sz = len(log_df.groupby('enrollment_id'))

    for i, (eid, part_df) in enumerate(log_df.groupby('enrollment_id')):
        if i % 1000 == 0:
            l.info("{0} of {1}".format(i, log_sz))

        ev = part_df[part_df['event'] == 'problem']
        part_d = {'enrollment_id': eid}
        part_d['evuniq'] = len(ev['object'].unique())
        arr.append(part_d)

    feat_df = pd.DataFrame(arr)
    df = df.merge(feat_df, how='left', on='enrollment_id').fillna(0)
    return {'X': utils.reshape(df['evuniq'])}


def gen_log_problem_obj_uniq_ss():
    df = utils.load_enroll()
    log_df = utils.load_log()
    arr = []
    log_sz = len(log_df.groupby('enrollment_id'))

    for i, (eid, part_df) in enumerate(log_df.groupby('enrollment_id')):
        if i % 1000 == 0:
            l.info("{0} of {1}".format(i, log_sz))

        ev = part_df[part_df['event'] == 'problem']
        part_d = {'enrollment_id': eid}
        part_d['evuniq'] = len(ev['object'].unique())
        arr.append(part_d)

    feat_df = pd.DataFrame(arr)
    df = df.merge(feat_df, how='left', on='enrollment_id').fillna(0)
    return {'X': SS().fit_transform(utils.reshape(df['evuniq']))}


def gen_log_video_obj_uniq():
    df = utils.load_enroll()
    log_df = utils.load_log()
    arr = []
    log_sz = len(log_df.groupby('enrollment_id'))

    for i, (eid, part_df) in enumerate(log_df.groupby('enrollment_id')):
        if i % 1000 == 0:
            l.info("{0} of {1}".format(i, log_sz))

        ev = part_df[part_df['event'] == 'video']
        part_d = {'enrollment_id': eid}
        part_d['evuniq'] = len(ev['object'].unique())
        arr.append(part_d)

    feat_df = pd.DataFrame(arr)
    df = df.merge(feat_df, how='left', on='enrollment_id').fillna(0)
    return {'X': utils.reshape(df['evuniq'])}


def gen_log_seq_obj_uniq():
    df = utils.load_enroll()
    log_df = utils.load_log()
    arr = []
    log_sz = len(log_df.groupby('enrollment_id'))

    for i, (eid, part_df) in enumerate(log_df.groupby('enrollment_id')):
        if i % 1000 == 0:
            l.info("{0} of {1}".format(i, log_sz))

        ev = part_df[part_df['event'] == 'sequential']
        part_d = {'enrollment_id': eid}
        part_d['evuniq'] = len(ev['object'].unique())
        arr.append(part_d)

    feat_df = pd.DataFrame(arr)
    df = df.merge(feat_df, how='left', on='enrollment_id').fillna(0)
    return {'X': utils.reshape(df['evuniq'])}


def gen_log_cha_obj_uniq():
    df = utils.load_enroll()
    log_df = utils.load_log()
    arr = []
    log_sz = len(log_df.groupby('enrollment_id'))

    for i, (eid, part_df) in enumerate(log_df.groupby('enrollment_id')):
        if i % 1000 == 0:
            l.info("{0} of {1}".format(i, log_sz))

        ev = part_df[part_df['event'] == 'chapter']
        part_d = {'enrollment_id': eid}
        part_d['evuniq'] = len(ev['object'].unique())
        arr.append(part_d)

    feat_df = pd.DataFrame(arr)
    df = df.merge(feat_df, how='left', on='enrollment_id').fillna(0)
    return {'X': utils.reshape(df['evuniq'])}



def gen_time_by_enrollment():
    # same as "time_feat.gen_first_time.npz" in initial_analysis
    enr_df = utils.load_enroll()
    df = utils.load_log()
    dx = df.groupby('course_id').agg({'time': 'min'}).reset_index()
    course_min_time = {}
    for idx, row in dx.iterrows():
        course_min_time[row['course_id']] = utils.toordinal(row['time'])

    feat = []
    df = df.sort('time')
    for idx, row in df.groupby('enrollment_id'):
        times = sorted(row['time'].tolist())
        course_id = row['course_id'].tolist()[0]
        first_time = utils.toordinal(times[0])
        last_time = utils.toordinal(times[-1])
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


def gen_video_time_by_enrollment():
    # same as "time_feat.gen_first_time.npz" in initial_analysis
    enr_df = utils.load_enroll()
    df = utils.load_log()
    dx = df.groupby('course_id').agg({'time': 'min'}).reset_index()

    course_min_time = {}
    for idx, row in dx.iterrows():
        course_min_time[row['course_id']] = utils.toordinal(row['time'])

    feat = []
    df = df.sort('time')
    for idx, row in df.groupby('enrollment_id'):
        row_video = row[row['event'] == 'video']
        if len(row_video) == 0:
            continue

        times = sorted(row_video['time'].tolist())
        course_id = row_video['course_id'].tolist()[0]
        first_time = utils.toordinal(times[0])
        last_time = utils.toordinal(times[-1])
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


def gen_prob_time_by_enrollment():
    # same as "time_feat.gen_first_time.npz" in initial_analysis
    enr_df = utils.load_enroll()
    df = utils.load_log()
    dx = df.groupby('course_id').agg({'time': 'min'}).reset_index()

    course_min_time = {}
    for idx, row in dx.iterrows():
        course_min_time[row['course_id']] = utils.toordinal(row['time'])

    feat = []
    df = df.sort('time')
    for idx, row in df.groupby('enrollment_id'):
        row_prob = row[row['event'] == 'problem']
        if len(row_prob) == 0:
            continue

        times = sorted(row_prob['time'].tolist())
        course_id = row_prob['course_id'].tolist()[0]
        first_time = utils.toordinal(times[0])
        last_time = utils.toordinal(times[-1])
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


def gen_time_by_username():
    # same as "time_feat.gen_time_by_username.npz" in initial_analysis
    enr_df = utils.load_enroll()
    df = utils.load_log()
    min_date = utils.toordinal(df['time'].min())

    feat = []
    df = df.sort('time')
    for idx, row in df.groupby('username'):
        times = sorted(row['time'].tolist())
        first_time = utils.toordinal(times[0])
        last_time = utils.toordinal(times[-1])
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


def gen_user_loglen():
    df = utils.load_enroll()
    log_df = utils.load_log()
    arr = []
    for eid, part_df in log_df.groupby('username'):
        part_d = {'username': eid}
        part_d['evuniq'] = len(part_df['object'])
        arr.append(part_d)

    feat_df = pd.DataFrame(arr)
    df = df.merge(feat_df, how='left', on='username').fillna(0)
    return {'X': utils.reshape(df['evuniq'])}


def gen_user_loguniq():
    df = utils.load_enroll()
    log_df = utils.load_log()
    arr = []
    for eid, part_df in log_df.groupby('username'):
        part_d = {'username': eid}
        part_d['evuniq'] = len(part_df['object'].unique())
        arr.append(part_d)

    feat_df = pd.DataFrame(arr)
    df = df.merge(feat_df, how='left', on='username').fillna(0)
    return {'X': utils.reshape(df['evuniq'])}


def gen_user_logproobjuniq():
    df = utils.load_enroll()
    log_df = utils.load_log()
    arr = []
    for eid, part_df in log_df.groupby('username'):
        ev = part_df[part_df['event'] == 'problem']

        part_d = {'username': eid}
        part_d['evuniq'] = len(ev['object'].unique())
        arr.append(part_d)

    feat_df = pd.DataFrame(arr)
    df = df.merge(feat_df, how='left', on='username').fillna(0)
    return {'X': utils.reshape(df['evuniq'])}


def gen_userday():
    df = utils.load_enroll()
    log_df = utils.load_log()
    arr = []
    for eid, part_df in log_df.groupby('username'):
        part_d = {'username': eid}
        part_d['user_uniq_day'] = len(part_df['time'].apply(lambda x:
            datetime.datetime.strptime(x, '%Y-%m-%dT%H:%M:%S').strftime(
                '%Y%m%d')).unique())
        arr.append(part_d)

    feat_df = pd.DataFrame(arr)
    df = df.merge(feat_df, how='left', on='username').fillna(0)
    return {'X': utils.reshape(df['user_uniq_day'])}


def gen_userhour():
    df = utils.load_enroll()
    log_df = utils.load_log()
    arr = []
    for eid, part_df in log_df.groupby('username'):
        part_d = {'username': eid}
        part_d['user_uniq_hour'] = len(part_df['time'].apply(lambda x:
            datetime.datetime.strptime(x, '%Y-%m-%dT%H:%M:%S').strftime(
                '%Y%m%d%H')).unique())
        arr.append(part_d)

    feat_df = pd.DataFrame(arr)
    df = df.merge(feat_df, how='left', on='username').fillna(0)
    return {'X': utils.reshape(df['user_uniq_hour'])}


def gen_user_uniq_course():
    df = utils.load_enroll()
    log_df = utils.load_log()
    user_df = log_df[['username', 'course_id']].groupby('username').agg({
        'course_id': lambda x: len(x.unique())}).rename(columns={
        'course_id': 'course_uniq'}).reset_index()

    df = df.merge(user_df, how='left', on='username').fillna(0)
    return {'X': utils.reshape(df['course_uniq'])}


def gen_userdiscuss():
    df = utils.load_enroll()
    log_df = utils.load_log()
    arr = []
    log_sz = len(log_df.groupby('username'))

    for i, (eid, part_df) in enumerate(log_df.groupby('username')):
        if i % 1000 == 0:
            l.info("{0} of {1}".format(i, log_sz))

        part_d = {'username': eid}
        dis = part_df[part_df['event'] == 'discussion']
        part_d['user_discuss'] = len(dis)
        arr.append(part_d)

    feat_df = pd.DataFrame(arr)
    df = df.merge(feat_df, how='left', on='username').fillna(0)
    return {'X': utils.reshape(df['user_discuss'])}


def gen_enrollment_uniq_discuss():
    df = utils.load_enroll()
    log_df = utils.load_log()
    arr = []
    log_sz = len(log_df.groupby('enrollment_id'))

    for i, (eid, part_df) in enumerate(log_df.groupby('enrollment_id')):
        if i % 1000 == 0:
            l.info("{0} of {1}".format(i, log_sz))

        part_d = {'enrollment_id': eid}
        dis = part_df[part_df['event'] == 'discussion']
        part_d['enr_discuss'] = len(dis['object'].unique())
        arr.append(part_d)

    feat_df = pd.DataFrame(arr)
    df = df.merge(feat_df, how='left', on='enrollment_id').fillna(0)
    return {'X': utils.reshape(df['enr_discuss'])}


def gen_username_uniq_discuss():
    df = utils.load_enroll()
    log_df = utils.load_log()
    arr = []
    log_sz = len(log_df.groupby('username'))

    for i, (eid, part_df) in enumerate(log_df.groupby('username')):
        if i % 1000 == 0:
            l.info("{0} of {1}".format(i, log_sz))

        part_d = {'username': eid}
        dis = part_df[part_df['event'] == 'discussion']
        part_d['user_discuss'] = len(dis['object'].unique())
        arr.append(part_d)

    feat_df = pd.DataFrame(arr)
    df = df.merge(feat_df, how='left', on='username').fillna(0)
    return {'X': utils.reshape(df['user_discuss'])}


def gen_unresolved_problem():
    """
    Opened (browser,problem), but not submitted (server,problem).

    * # of unique browser,problem,object by enrollment_id
    * # of unique server,problem,object by enrollment_id
    * # of unique un resolved problem,object by enrollment_id
    """
    df = utils.load_enroll()
    log_df = utils.load_log()
    log_sz = len(log_df.groupby('enrollment_id'))

    feat = []
    for i, (eid, part_df) in enumerate(log_df.groupby('enrollment_id')):
        if i % 1000 == 0:
            l.info("{0} of {1}".format(i, log_sz))

        uniq_open_prob = len(part_df[
            (part_df['source'] == 'browser') &
            (part_df['event'] == 'problem')
        ]['object'].unique())

        uniq_serv_prob = len(part_df[
            (part_df['source'] == 'server') &
            (part_df['event'] == 'problem')
        ]['object'].unique())

        uniq_unresolved = uniq_open_prob - uniq_serv_prob

        part_d = {'enrollment_id': eid}
        part_d['uopen'] = uniq_open_prob
        part_d['userv'] = uniq_serv_prob
        part_d['unreslv'] = uniq_unresolved
        feat.append(part_d)

    feat_df = pd.DataFrame(feat)
    df = df.merge(feat_df, how='left', on='enrollment_id').fillna(-1)
    return {
        'uopen': utils.reshape(df['uopen']),
        'userv': utils.reshape(df['userv']),
        'unreslv': utils.reshape(df['unreslv']),
    }


def gen_multiple_server_access():
    """
    # of multiple server,access,xxxxxxx
    """
    df = utils.load_enroll()
    log_df = utils.load_log()
    log_sz = len(log_df.groupby('enrollment_id'))

    feat = []
    for i, (eid, part_df) in enumerate(log_df.groupby('enrollment_id')):
        if i % 1000 == 0:
            l.info("{0} of {1}".format(i, log_sz))

        object_count = Counter(part_df[
            (part_df['source'] == 'server') &
            (part_df['event'] == 'problem')
        ]['object'])
        len_multi_server = len([k for k, v in object_count.items() if v > 1])

        part_d = {'enrollment_id': eid}
        part_d['multi'] = len_multi_server
        feat.append(part_d)

    feat_df = pd.DataFrame(feat)
    df = df.merge(feat_df, how='left', on='enrollment_id').fillna(-1)
    return {'X': utils.reshape(df['multi'])}


def gen_last_category():
    df = utils.load_enroll()
    log_df = utils.load_log_with_obj_attrib()
    log_sz = len(log_df.groupby('enrollment_id'))
    log_df = log_df.groupby('enrollment_id').agg({'category': 'last'}).reset_index()
    df = df.merge(log_df, how='left', on='enrollment_id').fillna(-1)

    return {'X': utils.onehot_encode(df['category'])}


def gen_last_event():
    df = utils.load_enroll()
    log_df = utils.load_log()
    log_sz = len(log_df.groupby('enrollment_id'))
    log_df = log_df.groupby('enrollment_id').agg({'event': 'last'}).reset_index()
    df = df.merge(log_df, how='left', on='enrollment_id').fillna(-1)

    return {'X': utils.onehot_encode(df['event'])}


def gen_last_source():
    df = utils.load_enroll()
    log_df = utils.load_log()
    log_sz = len(log_df.groupby('enrollment_id'))
    log_df = log_df.groupby('enrollment_id').agg({'source': 'last'}).reset_index()
    df = df.merge(log_df, how='left', on='enrollment_id').fillna(-1)

    return {'X': utils.onehot_encode(df['source'])}


def gen_user_active_time_ratio():
    # dim: 24 by username
    df = utils.load_enroll()
    log_df = utils.load_log()
    arr = []
    log_sz = len(log_df.groupby('username'))

    for i, (eid, part_df) in enumerate(log_df.groupby('username')):
        if i % 1000 == 0:
            l.info("{0} of {1}".format(i, log_sz))

        base = {v: 0 for v in range(24)}
        hour_counter = Counter(part_df['time'].apply(lambda x:
            int(datetime.datetime.strptime(x, '%Y-%m-%dT%H:%M:%S').strftime(
                '%H'))))
        base.update(hour_counter)
        base.update({'username': eid})
        arr.append(base)

    feat_df = pd.DataFrame(arr)
    df = df.merge(feat_df, how='left', on='username').fillna(0)

    # row normalize
    X = np.array(df[list(range(24))])
    X = X.astype(np.float32)
    rowsum = X.sum(axis=1)
    X = X / rowsum[:, np.newaxis]
    nan_place = np.isnan(X)
    X[nan_place] = 0.0

    return {'X': X}
