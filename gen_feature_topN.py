#!/usr/bin/env python
# -*- coding: GBK -*-
import time
from datetime import datetime
from datetime import timedelta
import pandas as pd
import pickle
import os
import math
import numpy as np
from dateutil.parser import parse
import re
import time
import datetime
import  lightgbm as lgb
import xgboost as xgb
data_path = u'/Users/Xiaobang/jupyter/blood_glucose/data/'

def get_gender(data_x):
    if data_x == u'男':
        return 1
    else :
        return 2

def get_top50_feature():
    param = {
        'boosting_type': 'gbdt',
        'objective': 'regression',
        'metric': 'l2',
        'num_leaves': 12,
        # 'min_data_in_leaf': 20,
        # 'min_sum_hessian_in_leaf': 1e-3,
        'learning_rate': 0.01,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 10,
        # 'reg_alpha': 0.1,
        # 'reg_lambda': 10,
        'verbose': 0,
        'is_unbalance': 'true',
        # 'max_bin': 300,
    }
    train_sorce = pd.read_csv(data_path + 'd_train_20180102.csv',encoding='GBK')
    ulimit = np.percentile(train_sorce[u'血糖'].values, 99.5)
    llimit = np.percentile(train_sorce[u'血糖'].values, 0.2)  # 线下1208 线上8375
    train_sorce = train_sorce[train_sorce[u'血糖'] < ulimit]
    train_sorce = train_sorce[train_sorce[u'血糖'] > llimit]
    ##########################
    train_sorce.to_csv(data_path + "train_sorce_preprocess.csv",index=False,encoding='GBK')
    train_sorce = pd.read_csv(data_path + "train_sorce_preprocess.csv",encoding='GBK')
    ##########################
    label_train = train_sorce[u'血糖']

    data_train = train_sorce.copy()
    data_train = data_train.drop([u'id',u'性别',u'体检日期',u'血糖'],axis=1)

    train = data_train.copy()#这个留着topN时候用
    y = label_train.copy()#这个留着topN时候用


    feature_name = data_train.columns

    test_sorce = pd.read_csv(data_path + 'd_test_A_20180102.csv', encoding = 'GBK')
    data_test = test_sorce.copy()
    data_test = data_test.drop([u'id',u'性别',u'体检日期'],axis=1)
    test = data_test.copy() #这个留着topN时候用

    data_train = data_train.values
    label_train = label_train.values
    print 'cv to find best num_boost_round'
    weight = label_train / np.sum(label_train)
    lgb_data_source = lgb.Dataset(data_train, label_train, weight=weight)
    bst = lgb.cv(
        param, lgb_data_source, num_boost_round=10000, nfold=10, stratified=False, shuffle=True,
        early_stopping_rounds=100, verbose_eval=50, show_stdv=True, seed=100)

    print u'迭代:', len(bst['l2-mean']), u' CV:', bst['l2-mean'][-1]

    gbm = lgb.train(param, lgb_data_source, num_boost_round=len(bst['l2-mean']))
    feat_imp = pd.Series(gbm.feature_importance(),index=feature_name).sort_values(ascending=False)

    #原始top20维特征：
    origin_top20 = list(feat_imp.index[:20])
    print ("初始特征top20:", origin_top20)
    ##原始特征top20相除特征共计380个

    divide_columns = []
    for i in origin_top20:
        for j in origin_top20:
            if j != i:
                divide = i + "/" + j
                divide_columns.append(divide)
                train[divide] = train[i] / train[j]
                test[divide] = test[i] / test[j]
    print test.shape
    divide_train = train[divide_columns]
    lgb_train_top_380 = lgb.Dataset(divide_train.values, y.values,weight=weight)

    bst2 = lgb.cv(
        param, lgb_train_top_380, num_boost_round=10000, nfold=10, stratified=False, shuffle=True,
        early_stopping_rounds=100, verbose_eval=50, show_stdv=True, seed=100)

    print u'迭代:', len(bst2['l2-mean']), u' CV:', bst2['l2-mean'][-1]

    gbm2 = lgb.train(param, lgb_train_top_380, num_boost_round=len(bst2['l2-mean']))
    feat_imp_topN = pd.Series(gbm2.feature_importance(), index=divide_train.columns).sort_values(ascending=False)
    print feat_imp_topN

    divide_top50 = list(feat_imp_topN.index[:50])
    train_topN= pd.concat([train_sorce,train[divide_top50]],axis=1)

    train_topN[u'性别'] = train_topN[u'性别'].map(get_gender)
    train_topN[u'体检日期'] = (pd.to_datetime(train_topN[u'体检日期']) - parse('2017-10-09')).dt.days
    train_topN.to_csv(data_path + 'train_divide_topN.csv', index=False, encoding='GBK')

    test_topN = pd.concat([test_sorce,test[divide_top50]],axis=1)
    test_topN[u'性别'] = test_topN[u'性别'].map(get_gender)
    test_topN[u'体检日期'] = (pd.to_datetime(test_topN[u'体检日期']) - parse('2017-10-09')).dt.days

    test_topN.to_csv(data_path + 'test_divide_topN.csv', index=False,encoding = 'GBK')


    ##xgb_feature
    xgb_params = {
        'eta': 0.1,
        'max_depth': 3,
        'min_child_weight': 1,
        'gamma': 0.3,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'booster': 'gbtree',
        'objective': 'reg:linear',
        'lambda': 1,
        'seed': 100,
        'silent': 1,
        'eval_metric': 'rmse'
    }

    d_train = xgb.DMatrix(data_train, label=label_train, missing=np.NAN)
    d_test = xgb.DMatrix(data_test, missing=np.NAN)
    model_bst = xgb.train(xgb_params, d_train, num_boost_round=50)

    train_new_feature = model_bst.predict(d_train, pred_leaf=True)
    test_new_feature = model_bst.predict(d_test, pred_leaf=True)
    train_new_feature1 = pd.DataFrame(train_new_feature)
    test_new_feature1 = pd.DataFrame(test_new_feature)

    train_df = pd.concat([train_topN, train_new_feature1],axis=1)
    test_df = pd.concat([test_topN, test_new_feature1],axis=1)
    print train_df.shape
    print test_df.shape
    train_df.to_csv(data_path + 'train_topN_and_xgb_feature.csv', index=False, encoding='GBK')
    test_df.to_csv(data_path + 'test_topN_and_xgb_feature.csv', index=False, encoding='GBK')

if __name__  ==  "__main__":
    get_top50_feature()

