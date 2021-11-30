import os
import pandas as pd
import numpy as np
import warnings
import math

from sklearn.ensemble import ExtraTreesRegressor, AdaBoostRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import KFold, LeaveOneGroupOut, GroupKFold
from sklearn.model_selection import cross_val_score, cross_validate, cross_val_predict
from sklearn.metrics import r2_score, make_scorer, roc_auc_score, precision_recall_curve, auc
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectFromModel



def spl(loss, lambd):
    """The function of hard SPL function"""
    sortedselectedidx = np.argsort(loss)[0:lambd].values.tolist()
    
    return sortedselectedidx


def spld(loss, Group, lam, gamma, num):
    """The function of Self-paced learning with Diversity"""

    groups_labels = np.array(list(set(Group)))
    b = len(groups_labels)
    selected_idx = []
    selected_score = [0] * len(loss)
    for i in range(b):
        idx_in_group = np.where(Group == groups_labels[i])[0]

        loss_in_group = []
        for idx in idx_in_group:
            loss_in_group.append(loss[idx])
    
        idx_loss_dict = dict()
        for j in idx_in_group:
            idx_loss_dict[j] = loss[j]

        sorted_idx_in_group = sorted(idx_loss_dict.keys(), key=lambda s: idx_loss_dict[s])
        sorted_idx_in_group_arr = np.array(sorted_idx_in_group)
    
        for (index, sample_idx) in enumerate(sorted_idx_in_group_arr):
            if loss[sample_idx] < (lam + gamma / (np.sqrt(index + 1) + np.sqrt(index))):
                selected_idx.append(sample_idx)        
            else:
                pass
            selected_score[sample_idx] = loss[sample_idx] - (lam + gamma / (np.sqrt(index + 1) + np.sqrt(index)))
    selected_idx_arr = np.array(selected_idx)

    selected_idx_and_new_loss_dict = dict()
    for idx in selected_idx_arr:
        selected_idx_and_new_loss_dict[idx] = selected_score[idx]
    
    sorted_idx_in_selected_samples = sorted(selected_idx_and_new_loss_dict.keys(),
                                            key=lambda s: selected_idx_and_new_loss_dict[s])
    sorted_idx_in_selected_samples_arr = np.array(sorted_idx_in_selected_samples)
    selected_sample = sorted_idx_in_selected_samples_arr[:num]

    return selected_sample



def run_spl(X_train, Y_train, X_test, Y_test, lambd, lambd_add):
    # create model
    model = ExtraTreesRegressor(n_estimators=200, max_depth = None, min_samples_split = 2, 
                            bootstrap = True, oob_score = True, n_jobs = -1)
    loss_s = []
    reg = model.fit(X_train, Y_train)
    y_pred = reg.predict(X_train)
    loss_s = (Y_train - y_pred)**2
    
    # running spl stage
    iter_num = math.ceil(X_train.shape[0]/lambd_add)
    loss_train_iter = []
    loss_iter = []
    selectedidx_iter = []
    y_pred_train_iter = []
    y_pred_test_iter = []
    
    for i in range(iter_num):
        # step 1: selet high-confidence smaples
        selectedidx = spl(loss_s, lambd)
        selectedidx_iter.append(selectedidx)
                
        x_train = X_train.loc[selectedidx]
        y_train = Y_train.loc[selectedidx]
               
        # step 2: training the model
        reg = model.fit(x_train, y_train)
        Y_pred = reg.predict(X_train)
        loss_train = ((Y_train - Y_pred)**2).sum()
        
        Y_evl = reg.predict(X_test)
        loss_test = ((Y_test - Y_evl)**2).sum()
        loss_s = (Y_train - Y_pred)**2
        
        # step 3: add the sample size
        lambd = lambd + lambd_add
        
        # store the medinum value
        loss_train_iter.append(loss_train)
        loss_iter.append(loss_test)
        y_pred_train_iter.append(Y_pred)
        y_pred_test_iter.append(Y_evl)
        
        if(i%50==1):
            print("Iteration times:", i)
    
    index = loss_iter.index(min(loss_iter))
    
    return y_pred_test_iter[index]



def run_spld(X_train, Y_train, X_test, Y_test, Group, lam, gamma, u1, u2, num, num_add):

    # running spl stage
    loss_train_iter = []
    loss_iter = []
    selectedidx_iter = []
    y_pred_train_iter = []
    y_pred_test_iter = []
    
    iter = math.ceil(X_train.shape[0]/num_add) * 5
    
    # Create model
    model = ExtraTreesRegressor(n_estimators=200, max_depth = None, min_samples_split = 2,
                                bootstrap = True, oob_score = True, n_jobs = -1)
    
    loss_s = []
    reg = model.fit(X_train, Y_train)
    y_pred = reg.predict(X_train)
    loss_s = (Y_train - y_pred)**2
    
    for i in range(iter):
        # step 1: selet high-confidence smaples
        selectedidx = spld(loss_s, Group, lam, gamma, num)
        
        selectedidx_iter.append(selectedidx)
        x_train = X_train.loc[selectedidx]
        y_train = Y_train.loc[selectedidx]
        
        # step 2: training the model
        reg = model.fit(x_train, y_train)
        Y_pred = reg.predict(X_train)
        loss_train = ((Y_train - Y_pred)**2).sum()
        
        Y_evl = reg.predict(X_test)
        loss_test = ((Y_test - Y_evl)**2).sum()
        
        loss_s = (Y_train - Y_pred)**2
        
        # step 3: add the sample size
        lam = u1 * lam
        gamma = u2 * gamma
        if(i == 0):
            num = num
        elif(i%5==0):
            num = num + num_add
            
        if(i%50==1):
            print("Iteration times:", i)

        # store the medinum value
        loss_train_iter.append(loss_train)
        loss_iter.append(loss_test)
        y_pred_train_iter.append(Y_pred)
        y_pred_test_iter.append(Y_evl)

    
    index = loss_iter.index(min(loss_iter))
    
    return y_pred_test_iter[index]
