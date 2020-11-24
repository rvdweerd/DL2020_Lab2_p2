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

def testLSTM(dataset,data_loader,model,config,device):
    # check model performance
    correct=0
    total=0
    evalBatches=200
    model.eval()
    with torch.no_grad():
        for i in range(evalBatches):
            (x,t) = next(iter(data_loader))  # x and t are lists (len=seq_len) of tensors (bsize)
            X = torch.stack(x).to(device)    # (seq_len,bsize)
            T = torch.stack(t).to(device)
            h,C = model.init_cell(config.batch_size)
            logprobs,_,_ = model(X,h,C)          # (seq_len,bsize,voc_size)
            predchar = torch.argmax(logprobs,dim=2) # (seq_len,bsize) the predicted characters: selected highest logprob for each sequence and example in the mini batch
            correct+=torch.sum(predchar==T).item()
            total+=(config.batch_size * config.seq_length)
        accuracy =correct / total
    print('accuracy over ',evalBatches*config.batch_size,' sequences:',accuracy)
    model.train()
    ####################
    # End of tests
    ####################

def generateSequenceGreedy(dataset,model,device,length=10,startString='A'):
    model.eval()
    seq_out=startString
    h,C = model.init_cell(1)
    # First, prep the cell with our starting sequence
    for i in range(len(startString)):
        charId=torch.tensor(dataset._char_to_ix[startString[i]]).to(device)
        logprobs,h,C = model(charId,h,C)
    # Now, run the cell independently (its output is fed back into the cell to self-generate)
    for i in range(length-len(startString)):
        predchar=torch.argmax(logprobs,dim=2)
        if predchar.item()==1:
            seq_out+='+'
        else:
            seq_out+=dataset._ix_to_char[predchar.item()]
        startId=predchar
        logprobs,h,C = model(startId,h,C)
    model.train()
    return seq_out

def generateSequenceRandom(temp,dataset,model,device,length=10,startString='A'):
    model.eval()
    model.temp=temp # Set temperature model in logprob calculation
    seq_out=startString
    h,C = model.init_cell(1)
    # First, prep the cell with our starting sequence
    for i in range(len(startString)):
        charId=torch.tensor(dataset._char_to_ix[startString[i]]).to(device)
        logprobs,h,C = model(charId,h,C)
    # Now, run the cell independently (its output is fed back into the cell to self-generate)
    for i in range(length-len(startString)):
        probs = torch.exp(logprobs)
        predchar = torch.multinomial(probs.squeeze(),1)
        if predchar.item()==1:
            seq_out+='+'
        else:
            seq_out+=dataset._ix_to_char[predchar.item()]
        startId=predchar
        logprobs,h,C = model(startId,h,C)
    model.train()
    return seq_out

def test(config):
    # Initialize the device which to run the model on
    device = torch.device("cpu" if not torch.cuda.is_available() else "cuda")
    #device = torch.device('cpu')

    # Initialize the dataset and data loader (note the +1)
    #dataset = TextDataset(filename="./assets/book_NL_tolstoy_anna_karenina.txt",seq_length=30)  
    dataset = TextDataset(filename=config.txt_file,seq_length=30)  
    data_loader = DataLoader(dataset, 32)

    model=TextGenerationModel(config,dataset._vocab_size,device).to(device)
    
    #checkpoint=torch.load("saved_model.tar")
    checkpoint=torch.load("AnnaK_607.tar")
    model.load_state_dict(checkpoint['model_state_dict'])
    print(model)
    testLSTM(dataset,data_loader,model,config,device)

    startStr='anna'
    seq_out=generateSequenceGreedy(dataset,model,device,length=100,startString=startStr)
    print('Example sequence started with [',startStr,']:',seq_out)
    temp=1
    seq_out=generateSequenceRandom(temp,dataset,model,device,length=100,startString=startStr)
    print('Temperature=',temp,'- Example sequence started with [',startStr,']:',seq_out)
    return
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
    parser.add_argument('--lstm_temperature',type=int,default=1,help='temperature used in softmax')

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
    test(config)
