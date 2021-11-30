import pandas as pd
from pandas import DataFrame
import numpy as np
from sklearn import metrics
import heapq
import random
from random import choice
from copy import deepcopy


def calclass(value, threshold):
    new_value = []
    for i in value:
        if i > threshold:
            new_value.append(1)
        else:
            new_value.append(0)
    return new_value


def binary_perf_ddG(Y_true, Y_pred, threshold=1.36):

    y_true = calclass(Y_true, 1.36)
    y_pred = calclass(Y_pred, threshold)

    # calculate the precision, recall and F1
    F1_score = metrics.f1_score(y_true, y_pred)
    Recall_score = metrics.recall_score(y_true, y_pred)
    Precision_score = metrics.precision_score(y_true, y_pred)
    Balanced_accuracy_score = metrics.balanced_accuracy_score(y_true, y_pred)
    MMC = metrics.matthews_corrcoef(y_true, y_pred)

    # record the performance
    perf = {
        'Recall': Recall_score,
        'Precision': Precision_score,
        'Balanced Accuracy': Balanced_accuracy_score,
        'F1 Score': F1_score,
        'Matthews Correlation Coefficient': MMC
        }

    return perf


def binary_perf_top(Y_true, Y_pred, threshold=0.15):
    
    top = int(len(Y_pred)*threshold)
    top_index = heapq.nlargest(top, range(len(Y_pred)), Y_pred.__getitem__)
    top_pred = []

    for i in range(len(Y_pred)):
        if i in top_index:
            top_pred.append(1)
        else:
            top_pred.append(0)

    y_true = calclass(Y_true, 1.36)

    perf = {
        'Recall': metrics.recall_score(y_true, top_pred),
        'Precision': metrics.precision_score(y_true, top_pred),
        'Balanced Accuracy': metrics.balanced_accuracy_score(y_true, top_pred),
        'F1 Score': metrics.f1_score(y_true, top_pred),
        'Matthews Correlation Coefficient': metrics.matthews_corrcoef(y_true, top_pred)
        }

    return perf

def cal_performance(result_list):
    result_min = min(result_list)
    result_max = max(result_list)
    result_mean = np.mean(result_list)
    result_var = np.var(result_list)
    result_median = np.median(result_list)
    result_std = np.std(result_list, ddof = 1)
    
    results = pd.DataFrame(columns=['value'])
    results.loc['min'] = result_min
    results.loc['max'] = result_max
    results.loc['mean'] = result_mean
    results.loc['var'] = result_var
    results.loc['median'] = result_median
    results.loc['std'] = result_std
    
    return results



def random_select_samples_group(X_sel, Y, tki, Y_tki, group_tki, group_train, feature_name):
    group_dict = {}
    select_num = 2
    
    mask = [c for c in feature_name]
    X_test = tki[mask]
    Y_test = Y_tki
    
    for index, value in enumerate(group_tki):
        if value not in group_dict.keys():
            group_dict[value] = []
            group_dict[value].append(index)
        else:
            group_dict[value].append(index)
            
    selected_tki = []
    for key in group_dict:
        slice = random.sample(group_dict[key], select_num)
        selected_tki.extend(slice)
        
    print("Selected sample list:", selected_tki)
    
    tki_list = [i for i in range(len(Y_test))]
    tki_rest = list(set(tki_list).difference(set(selected_tki)))
    
    X_test_s = X_test.loc[selected_tki]
    Y_test_s = Y_test.loc[selected_tki]
    
    X_test_new = X_test.loc[tki_rest]
    Y_test_new = Y_test.loc[tki_rest]
    
    X_sel.columns = feature_name
    
    # Reset the group information
    group_tki_select = group_tki.loc[selected_tki]
    group_tki_new = ['Abl' for i in group_tki_select]
    group_tki_new = pd.Series(group_tki_new)
    
    X_train = pd.concat([X_sel, X_test_s], axis=0, ignore_index=True)
    Y_train = pd.concat([Y, Y_test_s], axis=0, ignore_index=True)
    Group = pd.concat([group_train, group_tki_new], axis=0, ignore_index=True)
    
    pTest = deepcopy(tki.loc[tki_rest][['PDB_ID', 'MUTATION','DDG.EXP']]).reset_index()
    
    return X_train, Y_train, X_test_new, Y_test_new, pTest, Group


def random_select_samples(X_sel, Y, tki, Y_tki, group_tki, feature_name):
    group_dict = {}
    select_num = 2
    mask = [c for c in feature_name]
    X_test = tki[mask]
    Y_test = Y_tki
    
    for index, value in enumerate(group_tki):
        if value not in group_dict.keys():
            group_dict[value] = []
            group_dict[value].append(index)
        else:
            group_dict[value].append(index)
            
    selected_tki = []
    for key in group_dict:
        slice = random.sample(group_dict[key], select_num)
        selected_tki.extend(slice)
        
    print("Selected sample list:", selected_tki)
    
    tki_list = [i for i in range(len(Y_test))]
    tki_rest = list(set(tki_list).difference(set(selected_tki)))
    
    X_test_s = X_test.loc[selected_tki]
    Y_test_s = Y_test.loc[selected_tki]
    
    X_test_new = X_test.loc[tki_rest]
    Y_test_new = Y_test.loc[tki_rest]
    
    X_sel.columns = feature_name
    
    X_train = pd.concat([X_sel, X_test_s], axis=0, ignore_index=True)
    Y_train = pd.concat([Y, Y_test_s], axis=0, ignore_index=True)
    
    pTest = deepcopy(tki.loc[tki_rest][['PDB_ID', 'MUTATION','DDG.EXP']]).reset_index()
    
    return X_train, Y_train, X_test_new, Y_test_new, pTest