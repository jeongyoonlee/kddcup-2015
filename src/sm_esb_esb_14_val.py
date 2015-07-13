# -*- coding: utf-8 -*-
"""
Created on Thu Jul 02 15:41:37 2015

@author: nguyentt
"""
import pandas as pd
import numpy as np
import statsmodels.api as sm

from sklearn.metrics import roc_auc_score

def train_and_predict(X, y, X_test):
    sub_file = "esb_esb14"
    cv = np.loadtxt("cv_id.txt")
    val_pred = np.zeros(X.shape[0])
    test_pred = 0

    for i in xrange(1,6):
        print i
        trainId = (cv!=i)
        valId = (cv==i)
        X_train = X[trainId,:]
        y_train = y[trainId]
        X_val = X[valId,:]
        y_val = y[valId]
        
        elf = sm.OLS(y_train, X_train).fit()
        
        val_pred[valId] = elf.predict(X_val)
        
        print "score: %f" % roc_auc_score(y_val, val_pred[valId])
    
    score = roc_auc_score(y, val_pred)
    np.savetxt("build/val/sm_%s_%f.val.yht" % (sub_file, score), val_pred)

    clf = sm.OLS(y, X).fit()
    test_pred = clf.predict(X_test)
    np.savetxt("build/tst/sm_%s_%f.tst.yht" % (sub_file, score), test_pred)


print "loading data..."

train = pd.read_csv("build/feature/esb.esb14.trn.csv", header=None)
y_train = np.asarray(train[0])
X_train = np.asarray(train[train.columns[1:]])

test = pd.read_csv("build/feature/esb.esb14.tst.csv", header=None)
X_test = np.asarray(test[test.columns[1:]])

X_tr = X_train.copy()
X_te = X_test.copy()

X_tr = sm.add_constant(X_tr)
X_te = sm.add_constant(X_te)
  
train_and_predict(X_tr, y_train, X_te)
