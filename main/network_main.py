#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 20 11:09:49 2023

@author: binnie
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    def __init__(self, n_head, d_model, dropout):
        super().__init__()

        self.n_head = n_head
        self.d_model = d_model
        #self.d_k = d_k
        #self.d_v = d_v
        self.d_k = d_model // n_head
        self.w_qs = nn.Linear(d_model, d_model)
        self.w_ks = nn.Linear(d_model, d_model)
        self.w_vs = nn.Linear(d_model, d_model)
        self.fc = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

        if torch.cuda.is_available():
            self.scale = torch.sqrt(torch.FloatTensor([d_model // n_head])).cuda()
        else:
            self.scale = torch.sqrt(torch.FloatTensor([d_model // n_head]))
            
    def forward(self, q, k, v, mask=None):
        
        n_head = self.n_head
        sz_b = q.shape[0]
        d_k = self.d_k

        q = self.w_qs(q).view(sz_b, -1, n_head, d_k).permute(0, 2, 1, 3)
        k = self.w_ks(k).view(sz_b, -1, n_head, d_k).permute(0, 2, 1, 3)
        v = self.w_vs(v).view(sz_b, -1, n_head, d_k).permute(0, 2, 1, 3)
        qq = torch.matmul(q, k.permute(0, 1, 3, 2)) / self.scale
        q,k = q.cpu(),k.cpu()

        del q,k
        return self.fc(torch.matmul(self.dropout(F.softmax(qq, dim=-1)), v).permute(0, 2, 1, 3).contiguous().view(sz_b, -1, self.n_head * (self.d_model // self.n_head)))

class PositionwiseFeedforward(nn.Module):
    def __init__(self, d_model, d_hid, dropout):
        super().__init__()

        self.d_model = d_model
        self.d_hid = d_hid

        self.w_1 = nn.Conv1d(d_model, d_hid, 1) # position-wise 8 -> 8
        self.w_2 = nn.Conv1d(d_hid, d_model, 1) # position-wise 8 -> 8

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x2 = self.w_2(self.dropout(F.relu(self.w_1(x.permute(0, 2, 1))))).permute(0, 2, 1)

        return x2
    
class EncoderLayer(nn.Module):
    def __init__(self, d_model, d_hid, n_head, MultiHeadAttention, PositionwiseFeedforward, dropout):
        super().__init__()

        self.layer_norm = nn.LayerNorm(d_model)
        self.slf_attn = MultiHeadAttention(n_head, d_model, dropout)
        self.cros_learn = MultiHeadAttention(n_head, d_model, dropout)
        self.pos_ffn = PositionwiseFeedforward(d_model, d_hid, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, drug, side, drug_mask=None, side_mask=None):
        drug1 = self.layer_norm(drug + self.dropout(self.slf_attn(drug, drug, drug, drug_mask)))
        side1 = self.layer_norm(side + self.dropout(self.slf_attn(side, side, side, side_mask)))
        drug1 = self.layer_norm(drug1 + self.dropout(self.cros_learn(drug1, side1, side1, side_mask)))
        drug1 = self.layer_norm(drug1 + self.dropout(self.pos_ffn(drug1)))
        side1 = self.layer_norm(side1 + self.dropout(self.cros_learn(side1, drug1, drug1, drug_mask)))
        side1 = self.layer_norm(side1 + self.dropout(self.pos_ffn(side1)))
        drug,side= drug.cpu(), side.cpu()

        del drug, side, drug_mask, side_mask
        return drug1, side1
    
class Encoder(nn.Module):
    def __init__(self, feature_dim, d_model, n_layers, n_head, d_hid, EncoderLayer, MultiHeadAttention,
                  PositionwiseFeedforward, dropout):
        super().__init__()
        self.layer_norm = nn.LayerNorm(d_model)
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_head = n_head
        self.d_hid = d_hid
        self.EncoderLayer = EncoderLayer
        self.MultiHeadAttention = MultiHeadAttention
        self.PositionwiseFeedforward = PositionwiseFeedforward
        self.dropout = dropout
        self.slf_attn = MultiHeadAttention(n_head, d_model, dropout)
        self.layers = nn.ModuleList(
            [EncoderLayer(d_model, d_hid, n_head, MultiHeadAttention, PositionwiseFeedforward, dropout)
             for _ in range(n_layers)])
        self.ft = nn.Linear(feature_dim, d_model)
        self.dropout = nn.Dropout(dropout)
        self.w_1 = nn.Linear(d_model, 2)
        self.gn = nn.GroupNorm(8, 256)

    def forward(self, src, trg, src_mask=None, trg_mask=None):
        for layer in self.layers:
            src,trg = layer(src, trg)

        del src_mask,trg_mask
        return src, trg
    
class CrossFeat(nn.Module):
    def __init__(self, drugs_dim, sides_dim, embed_dim, batchsize, n_head, dropout = 0.2, dropout1=0.8, dropout2=0.8):
        super(CrossFeat, self).__init__()

        self.drugs_dim = drugs_dim
        self.sides_dim = sides_dim
        self.batchsize = batchsize
        self.drug_dim = self.drugs_dim//2
        self.side_dim = self.sides_dim//2
        
        self.embed_dim = embed_dim
        self.n_head = n_head
        self.dropout = dropout
        self.dropout1 = dropout1
        self.dropout2 = dropout2
        
        self.drug_layer1 = nn.Linear(self.drug_dim, self.embed_dim)
        self.drug_layer1_1 = nn.Linear(self.embed_dim, self.embed_dim)
        self.drug_layer2 = nn.Linear(self.drug_dim, self.embed_dim)
        self.drug_layer2_1 = nn.Linear(self.embed_dim, self.embed_dim)

        self.side_layer1 = nn.Linear(self.side_dim, self.embed_dim)
        self.side_layer1_1 = nn.Linear(self.embed_dim, self.embed_dim)
        self.side_layer2 = nn.Linear(self.side_dim, self.embed_dim)
        self.side_layer2_1 = nn.Linear(self.embed_dim, self.embed_dim)

        self.drug1_bn = nn.BatchNorm1d(self.embed_dim, momentum=0.5)
        self.drug2_bn = nn.BatchNorm1d(self.embed_dim, momentum=0.5)

        self.side1_bn = nn.BatchNorm1d(self.embed_dim, momentum=0.5)
        self.side2_bn = nn.BatchNorm1d(self.embed_dim, momentum=0.5)
        
        #mlp layer
        self.mlp_drug_layer = nn.Linear(100, self.embed_dim)
        self.mlp_drug_layer_1 = nn.Linear(self.embed_dim, self.embed_dim)
        self.mlp_side_layer = nn.Linear(300, self.embed_dim)
        self.mlp_side_layer_1 = nn.Linear(self.embed_dim, self.embed_dim)
        self.mlp_drug_bn = nn.BatchNorm1d(self.embed_dim, momentum=0.5)
        self.mlp_side_bn = nn.BatchNorm1d(self.embed_dim, momentum=0.5)
    
        # cnn setting
        self.output_channel = 32
        self.kernel_size = 2
        self.stride = 2
        self.input_channel = 1 * 1
        
        # transformer encoder setting
        self.n_layers = 2
        self.d_model = 8
        self.node_feat_size = 32
        self.d_hid = self.d_model
        
        self.CNN1 = nn.Sequential(
            # batch_size * 1 * 64 * 64
            nn.Conv2d(self.input_channel, self.output_channel, self.kernel_size, stride=self.stride),
            nn.BatchNorm2d(self.output_channel),
            nn.ReLU(),
            # batch_size * 32 * 32 * 32
            nn.Conv2d(self.output_channel, self.output_channel, self.kernel_size, stride=self.stride),
            nn.BatchNorm2d(self.output_channel),
            nn.ReLU(),
            # batch_size * 32 * 16 * 16
            nn.Conv2d(self.output_channel, self.output_channel, self.kernel_size, stride=self.stride),
            nn.BatchNorm2d(self.output_channel),
            nn.ReLU(),
            # batch_size * 32 * 8 * 8
            nn.Conv2d(self.output_channel, self.output_channel, self.kernel_size, stride=self.stride),
            nn.BatchNorm2d(self.output_channel),
            nn.ReLU())
        
        self.CNN2 = nn.Sequential(
            # batch_size * 1 * 64 * 64
            nn.Conv2d(self.input_channel, self.output_channel, self.kernel_size, stride=self.stride),
            nn.BatchNorm2d(self.output_channel),
            nn.ReLU(),
            # batch_size * 32 * 32 * 32
            nn.Conv2d(self.output_channel, self.output_channel, self.kernel_size, stride=self.stride),
            nn.BatchNorm2d(self.output_channel),
            nn.ReLU(),
            # batch_size * 32 * 16 * 16
            nn.Conv2d(self.output_channel, self.output_channel, self.kernel_size, stride=self.stride),
            nn.BatchNorm2d(self.output_channel),
            nn.ReLU(),
            # batch_size * 32 * 8 * 8
            nn.Conv2d(self.output_channel, self.output_channel, self.kernel_size, stride=self.stride),
            nn.BatchNorm2d(self.output_channel),
            nn.ReLU())
        
        self.encoder = Encoder(self.node_feat_size, self.d_model, self.n_layers, self.n_head, self.d_hid, EncoderLayer, 
                       MultiHeadAttention, PositionwiseFeedforward, self.dropout)

        self.total_layer = nn.Linear(768, 256)  #768->256
        self.total_bn = nn.BatchNorm1d(768, momentum=0.5)

        self.classifier = nn.Linear(256, 1)  #256->1
        self.con_layer = nn.Linear(256, 1)  #256->1

    def forward(self, drug_features, side_features, mlpdrug, mlpside, device):
        drug1, drug2 = drug_features.chunk(2, 1)
        side1, side2 = side_features.chunk(2, 1)

        mlp_drug = F.relu(self.mlp_drug_bn(self.mlp_drug_layer(mlpdrug.to(device))), inplace=True)
        mlp_drug = F.dropout(mlp_drug, training=self.training, p=self.dropout)
        mlp_drug = self.mlp_drug_layer_1(mlp_drug)

        mlp_side = F.relu(self.mlp_side_bn(self.mlp_side_layer(mlpside.to(device))), inplace=True)
        mlp_side = F.dropout(mlp_side, training=self.training, p=self.dropout)
        mlp_side = self.mlp_side_layer_1(mlp_side)

        x_drug1 = F.relu(self.drug1_bn(self.drug_layer1(drug1.to(device))), inplace=True)
        x_drug1 = F.dropout(x_drug1, training=self.training, p=self.dropout1)
        x_drug1 = self.drug_layer1_1(x_drug1)
        x_drug2 = F.relu(self.drug2_bn(self.drug_layer2(drug2.to(device))), inplace=True)
        x_drug2 = F.dropout(x_drug2, training=self.training, p=self.dropout1)
        x_drug2 = self.drug_layer2_1(x_drug2)
        drugs = [x_drug1, x_drug2]

        x_side1 = F.relu(self.side1_bn(self.side_layer1(side1.to(device))), inplace=True)
        x_side1 = F.dropout(x_side1, training=self.training, p=self.dropout1)
        x_side1 = self.side_layer1_1(x_side1)
        x_side2 = F.relu(self.side2_bn(self.side_layer2(side2.to(device))), inplace=True)
        x_side2 = F.dropout(x_side2, training=self.training, p=self.dropout1)
        x_side2 = self.side_layer2_1(x_side2)
        sides = [x_side1, x_side2]
        
        drug_matrix = torch.bmm(drugs[0].unsqueeze(2), drugs[1].unsqueeze(1))  #batch*128*128       
        drug_matrix1 = drug_matrix.view((-1, 1, self.embed_dim, self.embed_dim)) #batch*1*128*128

        side_matrix = torch.bmm(sides[0].unsqueeze(2), sides[1].unsqueeze(1))  #batch*128*128       
        side_matrix1 = side_matrix.view((-1, 1, self.embed_dim, self.embed_dim)) #batch*1*128*128
        
        #각각 drug matrix, side effect matrix고 featuremap -> drug feqture map, side feature map
        drug_feature_map = self.CNN1(drug_matrix1)  # output: batch_size * 32 * 8 * 8
        drug_feature_map = torch.mean(drug_feature_map,2) # batch_size * 32 * 8

        side_feature_map = self.CNN2(side_matrix1)  # output: batch_size * 32 * 8 * 8
        side_feature_map = torch.mean(side_feature_map,2) # batch_size * 32 * 8
        
        drug_feature_map,side_feature_map=self.encoder(drug_feature_map,side_feature_map)    # batch_size * 32 * 8

        drug_feature = drug_feature_map.view((-1, 256)) # batch_size * 256
        side_feature = side_feature_map.view((-1, 256)) # batch_size * 256
        final_feature = torch.cat((mlp_drug, mlp_side, drug_feature, side_feature), dim=1)# 768
        
        last = F.relu(self.total_layer(final_feature), inplace=True) # batch_size * 128
        last = F.dropout(last, training=self.training, p=self.dropout2)

        classification = torch.sigmoid(self.classifier(last))
        regression = self.con_layer(last)

        return classification.squeeze(), regression.squeeze()
