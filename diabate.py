#!/usr/bin/env python
# coding=utf-8
import pdb
import time
import datetime
import numpy as np

import pandas as pd
import lightgbm as lgb
from dateutil.parser import parse
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
try:
    import matplotlib.pyplot as plt
except ImportError:
    raise ImportError('You need to install matplotlib for plot_example.py.')

data_path = 'F:/work/xuetang/old/data/'


def get_gender(data_x):
    if data_x == u'男':
        return 1
    else:
        return 2
train = pd.read_csv(data_path+'train_divide.csv', encoding='GBK')
Y_train = train[u'血糖']
train = train.drop([u'血糖'], axis=1)
#train[u'性别'] = train[u'性别'].map(get_gender)
#del train[u'高密度脂蛋白胆固醇']
#del train[u'红细胞体积分布宽度']
#del train[u'年龄']
#del train[u'乙肝e抗体']
#del train[u'乙肝表面抗体']
#del train[u'乙肝表面抗原']
#del train[u'乙肝e抗原']
#del train[u'体检日期']
#del train[u'男']
test = pd.read_csv(data_path+'test_divide.csv', encoding='GBK')
#test[u'性别'] = test[u'性别'].map(get_gender)
#del test[u'高密度脂蛋白胆固醇']
#del test[u'红细胞体积分布宽度']
#del test[u'年龄']
#del test[u'乙肝e抗体']
#del test[u'乙肝表面抗体']
#del test[u'乙肝表面抗原']
#del test[u'乙肝e抗原']
#del test[u'体检日期']
#del test[u'男']
param = {
    'boosting_type': 'gbdt',
    'objective': 'regression',
    'metric': 'l2',
    'num_leaves': 12,
    #'min_data_in_leaf': 5,
    #'min_sum_hessian_in_leaf': 1e-3,
    'learning_rate': 0.01,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 10,
    'reg_alpha': 0,
    #'reg_lambda': 100,
    'verbose': 0,
    #'max_bin': 300,

}
#kf = KFold(n_splits=10, shuffle=True, random_state=520)
# pdb.set_trace()
#t0 = time.time()
#train_preds = np.zeros(train.shape[0])
#test_preds = np.zeros((test.shape[0], 10))
#result_feat_imp = pd.DataFrame()
#cv_mean = 0
# for i, (train_index, test_index) in enumerate(kf.split(train)):
#    print 'strat training', i
#    train_feat1 = train.iloc[train_index]
#    label_feat1 = Y_train.iloc[train_index]
#    train_feat2 = train.iloc[test_index]
#    label_feat2 = Y_train.iloc[test_index]
#    weight1 = label_feat1/np.sum(label_feat1)
#    #weight2 = weight.iloc[test_index]
#    lgb_train1 = lgb.Dataset(
#        train_feat1.values, label_feat1.values)#

#    # K折预测取平均
#    print 'cv to find best num_boost_round'
#    bst = lgb.cv(
#        param, lgb_train1, num_boost_round=10000, nfold=10, stratified=False, shuffle=True,
#        early_stopping_rounds=100, verbose_eval=100, show_stdv=True, seed=100)
#    print u'迭代:', len(bst['l2-mean']), u' CV:', bst['l2-mean'][-1]
#    cv_mean += bst['l2-mean'][-1]
#    gbm = lgb.train(params=param,
#                    train_set=lgb_train1,
#                    num_boost_round=len(bst['l2-mean']),
#                    # num_boost_round=20000,
#                    # early_stopping_rounds=100
#                    )
#    feat_imp = pd.Series(gbm.feature_importance(
#    ), name=i, index=train.columns).sort_values(ascending=False)
#    result_feat_imp = pd.concat([result_feat_imp, feat_imp], axis=1)
#    train_preds[test_index] += gbm.predict(train_feat2.values)
#    test_preds[:, i] = gbm.predict(test.values)
# print 'cv mean ', cv_mean/10
# print '线下得分：    {}'.format(mean_squared_error(Y_train, train_preds)*0.5)
# print 'CV训练用时{}秒'.format(time.time() - t0)
# mean_feat_imp = pd.Series(result_feat_imp.mean(
#    axis=1), name="mean", index=train.columns).sort_values(ascending=False)
# result_feat_imp = pd.concat(
#    [result_feat_imp, mean_feat_imp], axis=1).sort_values(by='mean', ascending=False)
# result_feat_imp.to_csv(r'./feature_importance{}.csv'.format(datetime.datetime.now().strftime('%Y%m%d_%H%M%S')),
#                       header=True, encoding='GBK', float_format='%.3f')
#result = pd.DataFrame({'pred': test_preds.mean(axis=1)})
# result.to_csv(r'./sub{}.csv'.format(datetime.datetime.now().strftime('%Y%m%d_%H%M%S')),
#              header=None, index=False, float_format='%.3f')
#train_preds = pd.DataFrame({'result': train_preds, 'raw': Y_train})
# train_preds.to_csv(r'./test_sub{}.csv'.format(datetime.datetime.now().strftime('%Y%m%d_%H%M%S')),
#                   header=None, index=False, float_format='%.3f')
# pdb.set_trace()

x_train, x_test, y_train, y_test = train_test_split(
    train.values, Y_train.values, test_size=0.2, random_state=100)
weight = y_train/np.sum(y_train)
weight2 = y_test/np.sum(y_test)

#lgb_train = lgb.Dataset(x_train, y_train)
lgb_train = lgb.Dataset(x_train, y_train, weight=weight)
#lgb_train2 = lgb.Dataset(x_test, y_test, weight=weight2)

print('cv to find best num_boost_round')
bst = lgb.cv(
    param, lgb_train, num_boost_round=10000, nfold=10, stratified=False, shuffle=True,
    early_stopping_rounds=100, verbose_eval=100, show_stdv=True, seed=100)
print(u'迭代:', len(bst['l2-mean']), u' CV:', bst['l2-mean'][-1])
gbm = lgb.train(params=param,
                train_set=lgb_train,
                num_boost_round=len(bst['l2-mean']),
                # valid_sets=lgb_train2,
                )
#print('Plot feature importances...')
#ax = lgb.plot_importance(gbm, max_num_features=50)
# plt.show()

print('Start predicting...')

y_pred = gbm.predict(x_test, num_iteration=gbm.best_iteration)
y_pred2 = np.round(y_pred, 3)


result = pd.DataFrame(
    {'result': list(y_pred), 'result2': list(y_pred2), 'raw': y_test})
result[['result', 'result2', 'raw']].to_csv(
    './blood_glucosebase1.csv', index=None, header=False)
# 用scikit-learn计算MSE
print('线下得分：    {}'.format(mean_squared_error(y_test, y_pred)*0.5))

print('Start predicting...')
y_pred = gbm.predict(test.values, num_iteration=gbm.best_iteration)
y_pred2 = np.round(y_pred, 3)
y_pred3 = pd.DataFrame(y_pred2)
y_pred3.to_csv(r'./div_sub_{}.csv'.format(datetime.datetime.now().strftime('%Y%m%d_%H%M%S')), header=None,
               index=False, float_format='%.3f')
