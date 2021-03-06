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

# Utility Functions
from utils import *

###############################################################################

def train(config):
    seed=config.seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic=True
        torch.backends.cudnn.benchmark=False

    # Initialize the device which to run the model on
    device = torch.device(config.device)
    #device = torch.device('cpu')

    # Initialize the dataset and data loader (note the +1)
    dataset = TextDataset(filename=config.txt_file,seq_length=config.seq_length)  
    data_loader = DataLoader(dataset, config.batch_size,num_workers=0)

    # Initialize the model that we are going to use
    model = TextGenerationModel(config,dataset._vocab_size,device).to(device)
    print('device:',device.type)
    print('Model defined. Number of trainable params:',model.numTrainableParameters())
    print(model)
    model.numTrainableParameters()
    testLSTM(dataset,data_loader,model,config,device)
    
    # Setup the loss and optimizer
    criterion = torch.nn.NLLLoss() 
    optimizer = optim.AdamW(model.parameters(),config.learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=config.learning_rate_step,gamma=config.learning_rate_decay)

    selfGenTHRES=0
    maxTrainAcc=0
    acc_plt=[]
    loss_plt=[]
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
        logprobs,_,_ = model(X,h,C) # (seq_len,bsize,voc_size)
        
        loss = criterion(logprobs.reshape(config.seq_length*config.batch_size,dataset.vocab_size),T.reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),max_norm=config.max_norm)
        optimizer.step()
        scheduler.step()

        predchar = torch.argmax(logprobs,dim=2) # (seq_len,bsize) the predicted characters: selected highest logprob for each sequence and example in the mini batch
        accuracy = torch.sum(predchar==T).item() / (config.batch_size * config.seq_length)
        loss_plt.append(loss)
        acc_plt.append(accuracy)

        # Save model with max train accuracy (I will use this for this toy example with batch_size*seq_len character predictions.
        # Of course this should be based on a larger test dataset
        if accuracy > maxTrainAcc:
            maxTrainAcc=accuracy
            torch.save({
            'step': step,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            'accuracy' : accuracy
            }, "saved_model.tar")
            # If a new accuracy level (steps of 0.1) is reached, print five self-generated sentences
            if accuracy > selfGenTHRES:
                selfGenTHRES+=0.1
                print('\n#################################### SAMPLE SELF GENERATED SEQUENCES #######################################')
                print('# Step:',step,', loss:',loss,'accuracy',accuracy)
                print('# ')
                print('# Greedy sampling [a...]:',generateSequenceGreedy(dataset,model,device,70,'a'))
                print('# Greedy sampling [b...]:',generateSequenceGreedy(dataset,model,device,70,'b'))
                print('# Greedy sampling [c...]:',generateSequenceGreedy(dataset,model,device,70,'c'))
                print('# Greedy sampling [d...]:',generateSequenceGreedy(dataset,model,device,70,'d'))
                print('# Greedy sampling [e...]:',generateSequenceGreedy(dataset,model,device,70,'e'))
                print('#')
                print('# Output of last training example:')
                print('# INPUT....: ',end="")
                printSequence(X,0,dataset)
                print('# TARGET...: ',end="")
                printSequence(T,0,dataset)
                print('# PREDICTED: ',end="")
                printSequence(predchar,0,dataset)
                print('#')
                print('############################################################################################################\n')


        # Just for time measurement
        t2 = time.time()
        examples_per_second = config.batch_size/float(t2-t1)

        if (step + 1) % config.print_every == 0:
            # Print training update
            print("[{}] Train Step {:04d}/{:04d}, Batch Size = {}, \
                    Examples/Sec = {:.2f}, "
                  "Accuracy = {:.2f}, Loss = {:.3f}".format(
                    datetime.now().strftime("%Y-%m-%d %H:%M"), step,
                    config.train_steps, config.batch_size, examples_per_second,
                    accuracy, loss
                    ))
            print('best training acc',maxTrainAcc)

        
        if (step + 1) % (config.train_steps//3) == 0:
            # Generate some sentences by sampling from the model
            print('\n#################################### SAMPLE SELF GENERATED SEQUENCES #######################################')
            print('# Step:',step,', loss:',loss,'accuracy',accuracy)
            print('# Greedy sampling [a...]:',generateSequenceGreedy(dataset,model,device,30,'a'))
            print('############################################################################################################\n')

        if step == config.train_steps:
            # If you receive a PyTorch data-loader error,
            # check this bug report:
            # https://github.com/pytorch/pytorch/pull/9655
            break
    print('Done training.')
    Testaccuracy=getTestAccuracy(dataset,data_loader,model,config,device,200)
    pltLossAcc(loss_plt,acc_plt,config)

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
    parser.add_argument('--seed', type=int, default=42,
                        help='Seed')

    # It is not necessary to implement the following three params,
    # but it may help training.
    parser.add_argument('--learning_rate_decay', type=float, default=0.96,
                        help='Learning rate decay fraction')
    parser.add_argument('--learning_rate_step', type=int, default=5000,
                        help='Learning rate step')
    parser.add_argument('--dropout_keep_prob', type=float, default=0.1,
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
