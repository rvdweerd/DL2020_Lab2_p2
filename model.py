# MIT License
#
# Copyright (c) 2019 Tom Runia
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to conditions.
#
# Author: Deep Learning Course | Fall 2019
# Date Created: 2019-09-06
###############################################################################

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.nn as nn
import torch

class TextGenerationModel(nn.Module):

    def __init__(self, batch_size, seq_length, vocabulary_size,
                 lstm_num_hidden=256, lstm_num_layers=2, device='cuda:0'):

        super(TextGenerationModel, self).__init__()
        # Initialization here...
        self.batch_size = batch_size
        #self.seq_length = seq_length
        self.voc_size = vocabulary_size
        self.hidden_dim = lstm_num_hidden
        self.num_layers = lstm_num_layers
        self.device = device
        self.input_dim = lstm_num_hidden // 2
        self.embed = nn.Embedding(self.voc_size,self.input_dim)
        self.lstm = nn.LSTM(self.input_dim,self.hidden_dim,self.num_layers,batch_first=False) 
        self.fc = nn.Linear(self.hidden_dim,self.voc_size) # From hidden vector to output p_t
        self.lsm=nn.LogSoftmax(dim=2) ###CHECK

    def forward(self, x, h, C): # h=hidden tensor, C=cell state tensor
        # Implementation here...
        if len(x.shape) == 0:
            x=x.unsqueeze(0).unsqueeze(0)   # if sequence length ==1 and batch size ==1, add two dimensions to tensor
        if len(x.shape) == 1:               # if sequence length >1 and batch size == 1, add batch dimension 
            x=x.unsqueeze(1)                # in dim 1, following the default nn.LSTM input, this makes x (seq_len,batch_size=1)
        seq_len = x.size(0)
        out = self.embed(x).to(self.device) # out (seq_len,batch_size,input_dim)
        out, (h,C) = self.lstm(out,(h,C))
        out = self.fc(out.reshape(-1,self.hidden_dim))  # linear layer acts on one character at a time
        out = out.reshape(seq_len,self.batch_size,-1) # shape back to output tensor (seq_len,batch_size,voc_size)
        out = self.lsm(out) # apply the log softmax
        return out
    
    def init_cell(self, bsize):
        h = torch.zeros(self.num_layers,bsize,self.hidden_dim).to(self.device)
        C = torch.zeros(self.num_layers,bsize,self.hidden_dim).to(self.device)
        self.batch_size=bsize # overwrite in preparation of new data coming in
        return h,C