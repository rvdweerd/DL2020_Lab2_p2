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
        self.seq_length = seq_length
        self.voc_size = vocabulary_size
        self.hidden_dim = lstm_num_hidden
        self.num_layers = lstm_num_layers
        self.device = device
        self.input_dim = lstm_num_hidden // 2
        self.embed = nn.Embedding(self.voc_size,self.input_dim)
        self.lstm = nn.LSTM(self.input_dim,self.hidden_dim,self.num_layers,batch_first=False) 
        #self.fc = nn.Linear(self.hidden_dim*self.seq_length,self.voc_size*self.seq_length)
        self.fc = nn.Linear(self.hidden_dim*self.batch_size,self.voc_size*self.batch_size)
        self.lsm=nn.LogSoftmax(dim=2) ###CHECK

    def forward(self, x):
        # Implementation here...
        out = self.embed(x).to(self.device)
        h0 = torch.zeros(self.num_layers,self.batch_size,self.hidden_dim).to(self.device)
        C0 = torch.zeros(self.num_layers,self.batch_size,self.hidden_dim).to(self.device)
        out, (hidden,cell) = self.lstm(out,(h0,C0))
        #out = out.permute(1,0,2)
        out = self.fc(out.reshape(out.shape[0],-1))
        out = out = out.reshape(self.seq_length,self.batch_size,-1)
        out = self.lsm(out)
        return out
