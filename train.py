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
def printSequence(sequenceTensor,itemInBatch,textdataset):
    # Prints sequence in column c=itemInBatch contained in stacked sequenceTensor (seq_len,batch_size)
    seq=sequenceTensor[:,itemInBatch]
    for ch in seq:
        if ch.item()==1:
            print('+',end="")
        else:
            print(textdataset.convert_to_string([ch.item()]),end="")
    #   INDICES
    print('[',end="")
    for ch in seq:
        print(ch.item(),',',end="")
    print(']')


def testLSTM(dataset,data_loader,model,config,device):
    ###################
    # Running some tests to see if model works for all input options
    ###################
    #############
    # First Test: Forward pass and manual loss calculation on one minibatch (our training setup)
    #############
    (x,t) = next(iter(data_loader))  # x and t are lists (len=seq_len) of tensors (bsize)
    X = torch.stack(x).to(device)    # (seq_len,bsize)
    T = torch.stack(t).to(device)
    T_onehot = torch.nn.functional.one_hot(T,num_classes=dataset._vocab_size)   # (seq_len,bsize,voc_size)
    h,C = model.init_cell(config.batch_size)
    logprobs = model(X,h,C)          # (seq_len,bsize,voc_size)
    assert (logprobs.size(0)==config.seq_length and logprobs.size(1)==config.batch_size and logprobs.size(2)==dataset._vocab_size)
    # Test manual Loss calculation
    Loss_sum_total = 0  
    for i in range(logprobs.size(0)):       # sum over all [characters in a sequency] ('timesteps')...
        for j in range(logprobs.size(1)):   # and all [sequences in batch]
            Loss_sum_total += logprobs[i][j][T[i][j]]   # and add the logprobs for that particular predicted character
    # Sanity check: same result when using one hot vectors
    Loss_sum_total_check = torch.sum(T_onehot*logprobs)
    assert abs(Loss_sum_total_check - Loss_sum_total)<1e-1
    #############
    # Second Test: try forward pass for only one training sequence (batch size = 1, sequence length remains the same)
    #############
    X_test = X[:,1].to(device)
    h,C = model.init_cell(1)
    logprobs = model(X_test,h,C) # (seq_len,1,voc_size)
    assert (logprobs.size(0)==config.seq_length and logprobs.size(1)==1 and logprobs.size(2)==dataset._vocab_size)

    #############
    # Third Test: try forward pass for only one training sequence and one character
    #############
    X_test2 = X[0,0].to(device)
    h,C = model.init_cell(1)
    logprobs = model(X_test2,h,C) # (1,1,voc_size)
    assert (logprobs.size(0)==1 and logprobs.size(1)==1 and logprobs.size(2)==dataset._vocab_size)
    ####################
    print('Model tests passed..')
    ####################
    # End of tests
    ####################

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
    print('device:',device.type)
    testLSTM(dataset,data_loader,model,config,device)
    
    # Setup the loss and optimizer
    criterion = torch.nn.NLLLoss() 
    optimizer = optim.AdamW(model.parameters(),lr=1e-4)
   
    schedSwitch=0 # simple LR scheduler
    maxAcc=0
    for step, (batch_inputs, batch_targets) in enumerate(data_loader):
        # Only for time measurement of step through network
        t1 = time.time()
        #######################################################
        # Add more code here ...
        #######################################################
        X=torch.stack(batch_inputs).to(device)  # (seq_len,bsize), input sequence
        T=torch.stack(batch_targets).to(device) # (seq_len,bsize), ground truth sequence
       
        model.zero_grad()
        h,C = model.init_cell(config.batch_size)
        logprobs = model(X,h,C) # (seq_len,bsize,voc_size)
        
        loss = criterion(logprobs.reshape(config.seq_length*config.batch_size,dataset.vocab_size),T.reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),max_norm=10)
        optimizer.step()
        
        if (schedSwitch==0 and loss<1.7):
          optimizer=optim.AdamW(model.parameters(),lr=config.learning_rate/10)
          schedSwitch=1
          print('LR reduced to:',config.learning_rate/10)

        predchar = torch.argmax(logprobs,dim=2) # (seq_len,bsize) the predicted characters: selected highest logprob for each sequence and example in the mini batch
        accuracy = torch.sum(predchar==T).item() / (config.batch_size * config.seq_length)
        
        # Save model with max accuracy
        if accuracy > maxAcc:
            maxAcc=accuracy
            torch.save({
            'step': step,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            'accuracy' : accuracy
            }, "best_model.tar")

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
            print('Input sentence (characters , [charcodes]):')
            printSequence(X,0,dataset)
            print('Target sentence (characters , [charcodes]):')
            printSequence(T,0,dataset)
            print('Predicted sentence (characters , [charcodes]):')
            printSequence(predchar,0,dataset)

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
