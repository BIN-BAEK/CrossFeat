#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 20 11:09:49 2023

@author: binnie
"""

import os
import torch
import pickle
import numpy as np
import torch.nn as nn
import time
from math import sqrt
import torch.utils.data
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import f1_score
from sklearn import metrics
from sklearn.metrics.pairwise import cosine_similarity
from network_main import CrossFeat


def read_raw_data(rawdata_dir, data_train, data_val, data_test):
    with open(rawdata_dir + 'data/' + 'drug_word.pkl', 'rb') as f:
        drug_mol_wordvec = pickle.load(f)
    drug_mol_wordvec_sim = cosine_similarity(drug_mol_wordvec)
    with open(rawdata_dir + 'data/' + 'drug_fingerprint_similarity.pkl', 'rb') as f:
        drug_fingerprint_sim = pickle.load(f)
    
    with open(rawdata_dir + 'data/' + 'side_sem.pkl', 'rb') as f:
        side_semantic = pickle.load(f)
    with open(rawdata_dir + 'data/' + 'side_word.pkl', 'rb') as f:
        side_glove_wordvec = pickle.load(f)
    side_wordvec_sim = cosine_similarity(side_glove_wordvec)

    for i in np.unique(data_train[:,0]):
        for j in np.unique(data_test[:,0]):
            drug_fingerprint_sim[i,j] = 0
            drug_mol_wordvec_sim[i,j] = 0
        
    drug_features, side_features = [], []
    drug_features.append(drug_mol_wordvec_sim)
    drug_features.append(drug_fingerprint_sim)

    side_features.append(side_semantic)
    side_features.append(side_wordvec_sim)

    return drug_features, side_features, drug_mol_wordvec, side_glove_wordvec

def fold_files(data_train, data_val, data_test, hyperparams):
    rawdata_dir = './'
    data_train = np.array(data_train)
    data_val = np.array(data_val)
    data_test = np.array(data_test)

    drug_features, side_features, drug_mol_wordvec, side_glove_wordvec = read_raw_data(rawdata_dir, data_train, data_val, data_test)

    drug_features_matrix = drug_features[0]
    for i in range(1, len(drug_features)):
        drug_features_matrix = np.hstack((drug_features_matrix, drug_features[i]))

    side_features_matrix = side_features[0]
    for i in range(1, len(side_features)):
        side_features_matrix = np.hstack((side_features_matrix, side_features[i]))
        
    drug_train = drug_features_matrix[data_train[:, 0]]
    side_train = side_features_matrix[data_train[:, 1]]
    f_train = data_train[:, 2]

    drug_val = drug_features_matrix[data_val[:, 0]]
    side_val = side_features_matrix[data_val[:, 1]]
    f_val = data_val[:, 2]
    
    drug_test = drug_features_matrix[data_test[:, 0]]
    side_test = side_features_matrix[data_test[:, 1]]
    f_test = data_test[:, 2]

    mlp_drug_train = drug_mol_wordvec[data_train[:, 0]]
    mlp_side_train = side_glove_wordvec[data_train[:, 1]]

    mlp_drug_val = drug_mol_wordvec[data_val[:, 0]]
    mlp_side_val = side_glove_wordvec[data_val[:, 1]]
    
    mlp_drug_test = drug_mol_wordvec[data_test[:, 0]]
    mlp_side_test = side_glove_wordvec[data_test[:, 1]]
    
    return drug_train, side_train, f_train, mlp_drug_train, mlp_side_train, drug_val, side_val, f_val, mlp_drug_val, mlp_side_val, drug_test, side_test, f_test, mlp_drug_test, mlp_side_test

def train_test(data_train, data_val, data_test, fold, hyperparams):
    drug_train, side_train, f_train, mlp_drug_train, mlp_side_train, drug_val, side_val, f_val, mlp_drug_val, mlp_side_val, drug_test, side_test, f_test, mlp_drug_test, mlp_side_test = fold_files(data_train, data_val, data_test, hyperparams)
    trainset = torch.utils.data.TensorDataset(torch.FloatTensor(drug_train), torch.FloatTensor(side_train),
                                              torch.FloatTensor(f_train), torch.FloatTensor(mlp_drug_train), torch.FloatTensor(mlp_side_train))
    valset = torch.utils.data.TensorDataset(torch.FloatTensor(drug_val), torch.FloatTensor(side_val),
                                              torch.FloatTensor(f_val), torch.FloatTensor(mlp_drug_val), torch.FloatTensor(mlp_side_val))
    testset = torch.utils.data.TensorDataset(torch.FloatTensor(drug_test), torch.FloatTensor(side_test),
                                             torch.FloatTensor(f_test), torch.FloatTensor(mlp_drug_test), torch.FloatTensor(mlp_side_test))
    _train = torch.utils.data.DataLoader(trainset, batch_size=hyperparams.batch_size, shuffle=True,
                                         num_workers=16, pin_memory=True)
    _valid = torch.utils.data.DataLoader(valset, batch_size=hyperparams.batch_size, shuffle=True,
                                         num_workers=16, pin_memory=True)
    _test = torch.utils.data.DataLoader(testset, batch_size=hyperparams.test_batch_size, shuffle=True,
                                        num_workers=16, pin_memory=True)
    
    torch.backends.cudnn.benchmark = True
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    use_cuda = False
    if torch.cuda.is_available():
        use_cuda = True
    device = torch.device("cuda" if use_cuda else "cpu")

    model = CrossFeat(1472, 1988, hyperparams.embed_dim, hyperparams.batch_size, hyperparams.n_head, hyperparams.droprate, hyperparams.droprate2, hyperparams.droprate2).to(device)
    Classification_criterion = nn.BCELoss() 
    Regression_criterion = nn.L1Loss()
    class_optimizer = torch.optim.Adam(model.parameters(), lr=hyperparams.lr, weight_decay=hyperparams.weight_decay)
    regress_optimizer = torch.optim.Adam(model.parameters(), lr=hyperparams.lr, weight_decay=hyperparams.weight_decay)
    
    AUC_mn = 0
    AUPR_mn = 0
    endure_count = 0

    start = time.time()

    for epoch in range(1, hyperparams.epochs + 1):
        # ====================   training    ====================
        training(model, _train, class_optimizer, regress_optimizer, Classification_criterion, Regression_criterion, device)
        # ====================     validation       ====================
        t_i_auc, t_iPR_auc, t_rmse, t_mae, t_f1, t_ground_i, t_ground_u, t_ground_truth, t_pred1, t_pred2, t_fpr, t_tpr = test(model, _valid, device)
        if AUC_mn < t_i_auc and AUPR_mn < t_iPR_auc:
            AUC_mn = t_i_auc
            AUPR_mn = t_iPR_auc
            endure_count = 0

        else:
            endure_count += 1

        print("Epoch: %d <Validation> RMSE: %.3f, MAE: %.3f, AUC: %.3f, AUPR: %.3f " % (
        epoch, t_rmse, t_mae, t_i_auc, t_iPR_auc))

        if endure_count > 10:
            break
    i_auc, iPR_auc, rmse, mae, f1, ground_i, ground_u, ground_truth, pred1, pred2, fpr, tpr = test(model, _test, device)

    time_cost = time.time() - start

    print("Time: %.2f Epoch: %d <Test> RMSE: %.3f, MAE: %.3f, AUC: %.3f, AUPR: %.3f " % (
        time_cost, epoch, rmse, mae, i_auc, iPR_auc))
    print('The best AUC/AUPR: %.3f / %.3f' % (i_auc, iPR_auc))
    print('The best RMSE/MAE: %.3f / %.3f' % (rmse, mae))

    return t_i_auc, t_iPR_auc, t_rmse, t_mae, i_auc, iPR_auc, rmse, mae, f1, time_cost, epoch, pred1, pred2, fpr, tpr


def training(model, train_loader, optimizer1, optimizer2, lossfunction1, lossfunction2, device):

    model.train()
        
    for i, data in enumerate(train_loader, 0):
        batch_drug, batch_side, batch_ratings, batch_mlp_drug, batch_mlp_side = data
        batch_labels = batch_ratings.clone().float()
        for k in range(batch_ratings.data.size()[0]):
            if batch_ratings.data[k] > 0:
                batch_labels.data[k] = 1
        optimizer1.zero_grad()
        optimizer2.zero_grad()
        
        one_label_index = np.nonzero(batch_labels.data.numpy())
        logits, reconstruction = model(batch_drug, batch_side, batch_mlp_drug, batch_mlp_side, device)
        logit_binary = np.zeros(logits.data.size()[0])
        for k in range(reconstruction.data.size()[0]):
            if reconstruction.data[k] > 0:
                logit_binary.data[k] = 1
            else:
                logit_binary.data[k] = 0
        
        loss1 = lossfunction1(logits, batch_labels.to(device))
        loss1.backward(retain_graph = True)
        optimizer1.step()
        loss2 = lossfunction2(reconstruction[one_label_index], batch_ratings[one_label_index].to(device))
        loss2 = loss2*loss2
        loss2.backward(retain_graph = True)
        optimizer2.step()
        
    return 0

def test(model, test_loader, device):
    model.eval()
    pred1 = []
    pred2 = []
    pred1_binary = []
    ground_truth = []
    label_truth = []
    ground_u = []
    ground_i = []

    for test_drug, test_side, test_ratings, test_mlp_drug, test_mlp_side in test_loader:

        test_labels = test_ratings.clone().long()
        for k in range(test_ratings.data.size()[0]):
            if test_ratings.data[k] > 0:
                test_labels.data[k] = 1
        ground_i.append(list(test_drug.data.cpu().numpy()))
        ground_u.append(list(test_side.data.cpu().numpy()))
        test_u, test_i, test_ratings = test_drug.to(device), test_side.to(device), test_ratings.to(device)
        scores_one, scores_two = model(test_drug, test_side, test_mlp_drug, test_mlp_side, device)
        scores_one_binary = np.zeros(scores_one.data.size()[0])
        for k in range(scores_two.data.size()[0]):
            if scores_two.data[k] > 0:
                scores_one_binary.data[k] = 1
            else:
                scores_one_binary.data[k] = 0
        pred1.append(list(scores_one.data.cpu().numpy()))
        pred2.append(list(scores_two.data.cpu().numpy()))
        ground_truth.append(list(test_ratings.data.cpu().numpy()))
        label_truth.append(list(test_labels.data.cpu().numpy()))
        pred1_binary.append(list(scores_one_binary))

    pred1 = np.array(sum(pred1, []), dtype = np.float32)
    pred2 = np.array(sum(pred2, []), dtype=np.float32)
    pred1_binary = np.array(sum(pred1_binary, []), dtype=np.float32)

    ground_truth = np.array(sum(ground_truth, []), dtype = np.float32)
    label_truth = np.array(sum(label_truth, []), dtype=np.float32)

    pred1[np.isnan(pred1)] = 0
    pred2[np.isnan(pred2)] = 0
    pred1_binary[np.isnan(pred1_binary)] = 0

    iprecision, irecall, ithresholds = metrics.precision_recall_curve(label_truth, pred1, pos_label=1, sample_weight=None)
    iPR_auc = metrics.auc(irecall, iprecision)
    
    fpr, tpr, _ = metrics.roc_curve(label_truth, pred1)

    try:
        i_auc = metrics.roc_auc_score(label_truth, pred1)
    except ValueError:
        i_auc = 0

    one_label_index = np.nonzero(label_truth)
    rmse = sqrt(mean_squared_error(pred2[one_label_index], ground_truth[one_label_index]))
    mae = mean_absolute_error(pred2[one_label_index], ground_truth[one_label_index])
    f1 = f1_score(label_truth, pred1_binary, average='binary')

    return i_auc, iPR_auc, rmse, mae, f1, ground_i, ground_u, ground_truth, pred1, pred2, fpr, tpr