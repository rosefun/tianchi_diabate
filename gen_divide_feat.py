
# coding: utf-8

# In[ ]:


import time
import datetime
import numpy as np
import pandas as pd
import lightgbm as lgb
from dateutil.parser import parse
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
try:
    import matplotlib.pyplot as plt
except ImportError:
    raise ImportError('You need to install matplotlib for plot_example.py.')


# In[ ]:


data_path = 'D:/tianchi/diabate/'


# In[ ]:


def make_train_set():
    train = pd.read_csv(data_path+'d_train_20180102.csv',encoding='gb2312')
    
    
    ulimit = np.percentile(train[u'血糖'].values, 99.5)
    llimit = np.percentile(train[u'血糖'].values, 0.3)

    
    train = train[train[u'血糖'] < ulimit]
    train = train[train[u'血糖'] > llimit]
    
    
    train['性别'] = train['性别'].map(lambda x: '男' if x not in ['男','女'] else x)
    sex_dummy = pd.get_dummies(train["性别"])
    train = pd.concat([train, sex_dummy],axis=1)
    
    train = train.drop(["id","性别","体检日期"],axis=1)
    
    train.fillna(train.median(axis=0),inplace=True)
    
    tarin_y = train['血糖']
    train_x = train.drop(['血糖'],axis=1)
    
    return train_x,tarin_y

def make_test_set():
    
    test = pd.read_csv(data_path+'d_test_A_20180102.csv',encoding='gb2312')
    test['性别'] = test['性别'].map(lambda x: '男' if x not in ['男','女'] else x)
    sex_dummy = pd.get_dummies(test["性别"])
    test = pd.concat([test, sex_dummy],axis=1)
    
    test = test.drop(["id","性别","体检日期"],axis=1)
    
    test.fillna(test.median(axis=0),inplace=True)
    
    return test



train, Y_train = make_train_set()
test = make_test_set()

origin_columns = [f for f in train.columns]
print("初始特征：",origin_columns)


# In[ ]:


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


# In[ ]:



lgb_train = lgb.Dataset(train.values,  Y_train.values)

kf = KFold(n_splits=10, shuffle=True, random_state=520)
t0 = time.time()
result_feat_imp = pd.DataFrame()


for i, (train_index, test_index) in enumerate(kf.split(train)):
    print ( 'strat training', i)
    train_feat1 = train.iloc[train_index]
    label_feat1 = Y_train.iloc[train_index]
    train_feat2 = train.iloc[test_index]
    label_feat2 = Y_train.iloc[test_index]

    lgb_train1 = lgb.Dataset(
        train_feat1.values, label_feat1.values)

    # K折预测取平均
    print  ( 'cv to find best num_boost_round')
    bst = lgb.cv(
        param, lgb_train1, num_boost_round=10000, nfold=10, stratified=False, shuffle=True,
        early_stopping_rounds=100, verbose_eval=100, show_stdv=True, seed=100)
    print ( u'迭代:', len(bst['l2-mean']), u' CV:', bst['l2-mean'][-1])
    gbm = lgb.train(params=param,
                    train_set=lgb_train1,
                    num_boost_round=len(bst['l2-mean']),
                    # num_boost_round=20000,
                    # early_stopping_rounds=100
                    )
    feat_imp = pd.Series(gbm.feature_importance(
    ), name=i, index=train.columns).sort_values(ascending=False)
    result_feat_imp = pd.concat([result_feat_imp, feat_imp], axis=1)


# In[ ]:


mean_feat_imp = pd.Series(result_feat_imp.mean(
    axis=1), name="mean", index=train.columns).sort_values(ascending=False)
result_feat_imp = pd.concat(
    [result_feat_imp, mean_feat_imp], axis=1).sort_values(by='mean', ascending=False)

##原始特征的top20
origin_top20 = list(result_feat_imp.index[:20])
print ("初始特征top20:",origin_top20)


# In[ ]:


##原始特征top20相除特征共计380个
divide_columns = []
for i in origin_top20:
    for j in origin_top20:
        if j!=i:
            divide = i+"/"+j
            divide_columns.append(divide)
            train[divide] = train[i]/train[j]
            test[divide] = test[i]/test[j]
  


# In[ ]:


###得到仅相除特征的top50
divide_train = train[divide_columns]
lgb_train = lgb.Dataset(divide_train.values,  Y_train.values)

kf = KFold(n_splits=10, shuffle=True, random_state=520)
# pdb.set_trace()
t0 = time.time()
result_feat_imp = pd.DataFrame()

print ( "获取仅相除特征top50")
for i, (train_index, test_index) in enumerate(kf.split(divide_train)):
    print ( 'strat training', i)
    train_feat1 = divide_train.iloc[train_index]
    label_feat1 = Y_train.iloc[train_index]
    train_feat2 = divide_train.iloc[test_index]
    label_feat2 = Y_train.iloc[test_index]

    lgb_train1 = lgb.Dataset(
        train_feat1.values, label_feat1.values)

    # K折预测取平均
    print  ( 'cv to find best num_boost_round')
    bst = lgb.cv(
        param, lgb_train1, num_boost_round=10000, nfold=10, stratified=False, shuffle=True,
        early_stopping_rounds=100, verbose_eval=100, show_stdv=True, seed=100)
    print ( u'迭代:', len(bst['l2-mean']), u' CV:', bst['l2-mean'][-1])
    gbm = lgb.train(params=param,
                    train_set=lgb_train1,
                    num_boost_round=len(bst['l2-mean']),
                    # num_boost_round=20000,
                    # early_stopping_rounds=100
                    )
    feat_imp = pd.Series(gbm.feature_importance(), name=i, index=divide_train.columns).sort_values(ascending=False)
    result_feat_imp = pd.concat([result_feat_imp, feat_imp], axis=1)


# In[ ]:


mean_feat_imp = pd.Series(result_feat_imp.mean(
    axis=1), name="mean", index=divide_train.columns).sort_values(ascending=False)
result_feat_imp = pd.concat(
    [result_feat_imp, mean_feat_imp], axis=1).sort_values(by='mean', ascending=False)


divide_top50 = list(result_feat_imp.index[:50])
print("相除特征top50",divide_top50 )


# In[ ]:


train_divide = pd.concat([train[origin_columns],train[divide_top50],Y_train],axis=1)
train_divide.to_csv('./train_divide.csv',index=False)
test_divide = pd.concat([test[origin_columns],test[divide_top50]],axis=1)
test.to_csv('./test_divide.csv',index=False)

