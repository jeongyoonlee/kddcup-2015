# -*- coding: utf-8 -*-
import datetime
import logging as l
from collections import Counter
import cPickle as pickle

from sklearn.feature_extraction import DictVectorizer as DV
import pandas as pd
import numpy as np

import utils


def gen_onehot_course_by_enrollment():
    df = utils.load_enroll()
    X = utils.onehot_encode(df['course_id'], sparse=True)
    return {'X': X}


def gen_onehot_user_by_enrollment():
    df = utils.load_enroll()
    df['username'] = df['username'].apply(lambda x: x[:6])
    X = utils.onehot_encode(df['username'], sparse=True)
    return {'X': X}


#def gen_onehot_object_by_enrollment():
#    df = utils.load_enroll()
#    df['object'] = df['object'].apply(lambda x: x[:5])
#    x = utils.onehot_encode(df['object'], sparse=True)
#    return {'x': x}


if __name__ == '__main__':
    gen_onehot_course_by_username()
