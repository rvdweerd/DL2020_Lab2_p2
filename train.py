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

import os
import time
from datetime import datetime
import argparse

import numpy as np

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from dataset import TextDataset
from model import TextGenerationModel

###############################################################################


def train(config):

    # Initialize the device which to run the model on
    device = torch.device(config.device)
    #device = torch.device('cpu')

    # Initialize the dataset and data loader (note the +1)
    dataset = TextDataset(filename="./assets/book_NL_tolstoy_anna_karenina.txt",seq_length=config.seq_length)  # fixme
    data_loader = DataLoader(dataset, config.batch_size)

    # Initialize the model that we are going to use
    model = TextGenerationModel(
        config.batch_size,
        config.seq_length,
        dataset._vocab_size,
        config.lstm_num_hidden,
        config.lstm_num_layers,
        device#config.device
        ).to(device)  # 

    # Setup the loss and optimizer
    criterion = torch.nn.NLLLoss() 
    optimizer = optim.AdamW(model.parameters(),lr=1e-4)

    (x,t) = next(iter(data_loader))  # x and t are lists (len=seq_len) of tensors (bsize)
    X = torch.stack(x).to(device)    # (seq_len,bsize)
    T = torch.stack(t).to(device)
    T_onehot = torch.nn.functional.one_hot(T,num_classes=dataset._vocab_size)   # (seq_len,bsize,voc_size)

    logprobs = model(X)              # (seq_len,bsize,voc_size)

    Loss_sum_total = 0 # sum over all [sequences in batch] and all [timesteps]
    for i in range(logprobs.size(0)):
        for j in range(logprobs.size(1)):
            Loss_sum_total += logprobs[i][j][T[i][j]]
    # Sanity check: same result when using one hot vectors?
    Loss_sum_total_check = torch.sum(T_onehot*logprobs)
    assert abs(Loss_sum_total_check - Loss_sum_total)<1e-1
    
    loss = criterion(logprobs.reshape(config.batch_size * config.seq_length,dataset._vocab_size),T.reshape(-1))


    Loss_sum_avg = Loss_sum_total / (config.batch_size * config.seq_length)

    predchar = torch.argmax(logprobs,dim=2) # (seq_len,bsize) the predicted characters: selected highest logprob for each sequence and example in the mini batch
    accuracy = torch.sum(predchar==T).item() / (config.batch_size * config.seq_length)
    
    #for step in range(1000):
    for step, (batch_inputs, batch_targets) in enumerate(data_loader):

        # Only for time measurement of step through network
        t1 = time.time()

        #######################################################
        # Add more code here ...
        #######################################################
        X=torch.stack(batch_inputs).to(device)
        T=torch.stack(batch_targets).to(device)
        
        model.zero_grad()
        #hidden,cell = model.init_hidden(config.batch_size)
        logprobs = model(X)              # (seq_len,bsize,voc_size)
        Loss_sum_total = 0 # sum over all [sequences in batch] and all [timesteps]
        for i in range(logprobs.size(0)):
            for j in range(logprobs.size(1)):
                Loss_sum_total += logprobs[i][j][T[i][j]]
        loss = criterion(logprobs.reshape(config.seq_length*config.batch_size,dataset.vocab_size),T.reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),
                                       max_norm=10)
        optimizer.step()
        
        predchar = torch.argmax(logprobs,dim=2) # (seq_len,bsize) the predicted characters: selected highest logprob for each sequence and example in the mini batch
        accuracy = torch.sum(predchar==T).item() / (config.batch_size * config.seq_length)
    
        #print('loss:',loss,', acc: ',accuracy)
        # Just for time measurement
        t2 = time.time()
        examples_per_second = config.batch_size/float(t2-t1)

        if (step + 1) % config.print_every == 0:

            print("[{}] Train Step {:04d}/{:04d}, Batch Size = {}, \
                    Examples/Sec = {:.2f}, "
                  "Accuracy = {:.2f}, Loss = {:.3f}".format(
                    datetime.now().strftime("%Y-%m-%d %H:%M"), step,
                    config.train_steps, config.batch_size, examples_per_second,
                    accuracy, loss
                    ))

        if (step + 1) % config.sample_every == 0:
            # Generate some sentences by sampling from the model
            pass

        if step == config.train_steps:
            # If you receive a PyTorch data-loader error,
            # check this bug report:
            # https://github.com/pytorch/pytorch/pull/9655
            break

    print('Done training.')


###############################################################################
###############################################################################

if __name__ == "__main__":

    # Parse training configuration
    parser = argparse.ArgumentParser()

    # Model params
    parser.add_argument('--txt_file', type=str, required=False, default="./assets/book_NL_tolstoy_anna_karenina.txt",
                        help="Path to a .txt file to train on")
    parser.add_argument('--seq_length', type=int, default=30,
                        help='Length of an input sequence')
    parser.add_argument('--lstm_num_hidden', type=int, default=128,
                        help='Number of hidden units in the LSTM')
    parser.add_argument('--lstm_num_layers', type=int, default=2,
                        help='Number of LSTM layers in the model')

    # Training params
    parser.add_argument('--batch_size', type=int, default=32,#64,
                        help='Number of examples to process in a batch')
    parser.add_argument('--learning_rate', type=float, default=2e-3,
                        help='Learning rate')

    # It is not necessary to implement the following three params,
    # but it may help training.
    parser.add_argument('--learning_rate_decay', type=float, default=0.96,
                        help='Learning rate decay fraction')
    parser.add_argument('--learning_rate_step', type=int, default=5000,
                        help='Learning rate step')
    parser.add_argument('--dropout_keep_prob', type=float, default=1.0,
                        help='Dropout keep probability')

    parser.add_argument('--train_steps', type=int, default=int(1e6),
                        help='Number of training steps')
    parser.add_argument('--max_norm', type=float, default=5.0, help='--')

    # Misc params
    parser.add_argument('--summary_path', type=str, default="./summaries/",
                        help='Output path for summaries')
    parser.add_argument('--print_every', type=int, default=5,
                        help='How often to print training progress')
    parser.add_argument('--sample_every', type=int, default=100,
                        help='How often to sample from the model')
    parser.add_argument('--device', type=str, default=("cpu" if not torch.cuda.is_available() else "cuda"),
                        help="Device to run the model on.")

    # If needed/wanted, feel free to add more arguments

    config = parser.parse_args()

    # Train the model
    train(config)