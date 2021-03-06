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
import numpy as np
import torch.nn as nn
import torch

class TextGenerationModel(nn.Module):

    def __init__(self,config,vocabulary_size,device):

        super(TextGenerationModel, self).__init__()
        # Initialization here...
        self.batch_size = config.batch_size
        self.voc_size = vocabulary_size
        self.hidden_dim = config.lstm_num_hidden
        self.num_layers = config.lstm_num_layers
        self.device = device
        self.input_dim = config.lstm_num_hidden
        self.embed = nn.Embedding(self.voc_size,self.input_dim)
        self.lstm = nn.LSTM(self.input_dim,self.hidden_dim,self.num_layers,batch_first=False,dropout=config.dropout_keep_prob) 
        self.fc = nn.Linear(self.hidden_dim,self.voc_size) # From hidden vector to output p_t
        self.lsm=nn.LogSoftmax(dim=2)
        self.temp=1 # used to enable softmax with temperature after training
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
        out = self.lsm(out/self.temp) # apply the log softmax
        return out,h,C

    def init_cell(self, bsize):
        h = torch.zeros(self.num_layers,bsize,self.hidden_dim).to(self.device)
        C = torch.zeros(self.num_layers,bsize,self.hidden_dim).to(self.device)
        self.batch_size=bsize # overwrite in preparation of new data coming in
        return h,C

    def numTrainableParameters(self):
        #return sum(p.numel() for p in self.parameters() if p.requires_grad)
  
        total = 0
        for name, p in self.named_parameters():
            total += np.prod(p.shape)
            print("{:24s} {:12s} requires_grad={}".format(name, str(list(p.shape)), p.requires_grad))
        print("\nTotal number of parameters: {}\n".format(total))
        assert total == sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total
