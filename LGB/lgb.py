from sklearn.model_selection import train_test_split
import lightgbm as lgb
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd
import math
import os

def split_tr_val_ts():
    train = pd.read_csv('train.csv')
    val = pd.read_csv('val.csv')
    test = pd.read_csv('ts_ds.csv')
    train_X = train.drop(['area_id','label'],axis=1)
    train_Y = train['label']
    val_X = val.drop(['area_id','label'],axis=1)
    val_Y = val['label']
    test_X = test.drop(['area_id'],axis=1)
    return train_X,train_Y,val_X,val_Y,test_X
    
def lgb(train_X,train_Y):
    params = {
    'boosting_type': 'gbdt',
    'objective': 'multiclassova',  
    'num_class': 9,  
    'metric': 'multi_error', 
    'num_leaves': 63,
    'learning_rate': 0.01,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.9,
    'bagging_seed':0,
    'bagging_freq': 1,
    'verbose': -1,
    'reg_alpha':1,
    'reg_lambda':2,
    'lambda_l1': 0,
    'lambda_l2': 1,
    'num_threads': 16,
    }
    ############################################################################ 35万验证集拿一部分训练并验证
    X_train, X_valid, y_train, y_valid = train_test_split(train_X, train_Y-1, test_size=0.2, random_state=2019)
    lgb_train = lgb.Dataset(X_train, y_train)
    lgb_evals = lgb.Dataset(X_valid, y_valid , reference=lgb_train)
    gbm = lgb.train(params,
                    lgb_train,
                    num_boost_round=10000,
                    valid_sets=[lgb_train,lgb_evals],
                    valid_names=['train','valid'],
                     early_stopping_rounds=300,
                    verbose_eval=100,
                    )
    train_pro = gbm.predict(X_train, num_iteration=gbm.best_iteration)
    val_pro = gbm.predict(X_valid, num_iteration=gbm.best_iteration)
    print ('train_acc：',pro2label(y_train,train_pro),'val_acc：',pro2label(y_valid,val_pro))
    ############################################################################ 5万验证集全部用于训练
    lgb_train = lgb.Dataset(train_X, train_Y-1)
    gbm = lgb.train(params,
                    lgb_train,
                    num_boost_round=gbm.best_iteration,
                    valid_sets=[lgb_train],
                    valid_names=['train'],
                    verbose_eval=100,
                    )
    return gbm

def bn(temp):
    shp=temp.shape
    #MEAN=np.mean(temp,axis=1)
    #STD=np.std(temp,axis=1)
    #for i in range(shp[0]):
     #   temp.loc[i,:]=(temp.loc[i,:]-MEAN[i])/(STD[i]+1e-10)
    temp0=np.zeros([temp.shape[0],24])
    for j in range(temp.shape[0]):
        SUM=0.0
        for i in range(24):
            SUM+=math.exp(float(temp.loc[j,i]))
        for i in range(24):
            temp0[j,i]=math.exp(float(temp.loc[j,i]))/SUM
    return temp0

train_X,train_Y,val_X,val_Y,test_X = split_tr_val_ts()
train_X = bn(train_X)
val_X = bn(val_X)
test_X = bn(test_X)
gbm = lgb(train_X,train_Y)
val_pro = gbm.predict(val_X, num_iteration=gbm.best_iteration)
print (val_pro)
np.save('val_pro.npy',val_pro)

