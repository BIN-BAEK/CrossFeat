#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 20 11:09:49 2023

@author: binnie
"""

import pandas as pd
import easydict
import random
import sys
import pickle
import numpy as np
from utils import train_test

with open('./data/sample_5fold_idx.pkl', 'rb') as f:
    sample_5fold_idx = pickle.load(f)
    
lrs = [0.0001, 0.0005, 0.00001]
batchs = [64, 128, 256]
droprates = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
wds = [0.001, 0.0001, 0.00001]
heads = [1,2,4,8]

fold = 1
total_auc, total_pr_auc, total_rmse, total_mae, total_f1 = [], [], [], [], []
max_iters = 10
for k in range(0,5):
    print("==================================fold "+ str(k+1)+" start" )
    trainsample = sample_5fold_idx[k][0]
    validsample = sample_5fold_idx[k][1]
    testsample = sample_5fold_idx[k][2]
    dff = pd.DataFrame()
    for iters in range(max_iters):    
        lr = random.choice(lrs)
        batchsize = random.choice(batchs)
        droprate = random.choice(droprates)
        droprate2 = random.choice(droprates)
        wd = random.choice(wds)
        n_head = random.choice(heads)
        
        hyperparams = easydict.EasyDict({
            "epochs": 100,
            "lr": lr,
            "embed_dim": 128,
            "weight_decay": wd,
            "N": 30000, #L0 parameter
            "droprate": droprate,
            "droprate2": droprate2,
            "batch_size": batchsize,
            "test_batch_size": 128,
            "n_head": n_head
            })
        
        
        print('-------------------- Hyperparameters --------------------')
        print('dropout rate: ' + str(hyperparams.droprate))
        print('dropout rate2: ' + str(hyperparams.droprate2))
        print('learning rate: ' + str(hyperparams.lr))
        print('batch size: ' + str(hyperparams.batch_size))
        print('weight decay: ' + str(hyperparams.weight_decay))
        print('N: ' + str(hyperparams.N))
        print('dimension of embedding: ' + str(hyperparams.embed_dim))
        print('Transformer head: ' + str(hyperparams.n_head))
        
        train_data = np.array(trainsample)
        val_data = np.array(validsample)
        test_data = np.array(testsample)
        val_auc, val_PR_auc, val_rmse, val_mae, auc, PR_auc, rmse, mae, f1, time_cost, epoch, pred1, pred2, fpr, tpr = train_test(train_data.tolist(), val_data.tolist(), test_data.tolist(), fold, hyperparams)

        df = pd.DataFrame(data=[[str(k+1), str(iters), time_cost, epoch, val_rmse, val_mae, val_auc, val_PR_auc, rmse, mae, auc, PR_auc, lr, batchsize, 
                                 droprate, droprate2, wd, n_head]], columns=['Fold', 'iteration', 'time_cost', 'epoch', 'valRMSE', 'valMAE', 'valAUROC', 'valAUPRC', 
                                                                             'RMSE', 'MAE', 'AUROC', 'AUPRC', 'LearningRate', 'Batchsize', 'Dropout_rate', 
                                                                             'Dropout_rate2', 'Weight_decay', 'N_heads'])
        dff = dff.append(df)
        sys.stdout.flush()
        save_path = './'
        pd.DataFrame(pred1).to_csv(save_path + 'fold' + str(k+1) + '_iters' + str(iters) + '_pred1.csv', index=None)
        pd.DataFrame(pred2).to_csv(save_path + 'fold' + str(k+1) + '_iters' + str(iters) + '_pred2.csv', index=None)
        pd.DataFrame(fpr).to_csv(save_path + 'fold' + str(k+1) + '_iters' + str(iters) + '_fpr.csv', index=None)
        pd.DataFrame(tpr).to_csv(save_path + 'fold' + str(k+1) + '_iters' + str(iters) + '_tpr.csv', index=None)
        dff.to_csv(save_path + 'fold' + str(k+1) + '_prediction_result.csv', index=None)
