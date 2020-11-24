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

from utils import *
# This demo file loads a pre-trained SLTM model, tests its accuracy and produces self-generated sentences

def testAccuracyLSTM(dataset,data_loader,model,config,device):
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
    return accuracy
    ####################
    # End of tests
    ####################


def test(config):
    seed=config.seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic=True
        torch.backends.cudnn.benchmark=False
    # Initialize the device which to run the model on
    device = torch.device("cpu" if not torch.cuda.is_available() else "cuda")
    #device = torch.device('cpu')

    # Initialize the dataset and data loader (note the +1)
    #dataset = TextDataset(filename="./assets/book_NL_tolstoy_anna_karenina.txt",seq_length=30)  
    dataset = TextDataset(filename=config.txt_file,seq_length=30)  
    data_loader = DataLoader(dataset, 32)

    model=TextGenerationModel(config,dataset._vocab_size,device).to(device)
    
    #checkpoint=torch.load("saved_model.tar")
    checkpoint=torch.load("AnnaK_0.55.tar")
    model.load_state_dict(checkpoint['model_state_dict'])
    print(model)
    testLSTM(dataset,data_loader,model,config,device)
    accuracy=testAccuracyLSTM(dataset,data_loader,model,config,device)

    startStr='Anna'
    print('########### SAMPLE SELF GENERATED SEQUENCE ###############')
    print('# Test accuracy:',accuracy)
    print('# Self generated sentences, start string = Anna')
    print('#')
    print('# Greedy sampling           :',generateSequenceGreedy(dataset,model,device,length=100,startString=startStr))
    print('# Random sampling, temp=0.01:',generateSequenceRandom(0.01,dataset,model,device,length=100,startString=startStr))
    print('# Random sampling, temp=0.10:',generateSequenceRandom(0.5,dataset,model,device,length=100,startString=startStr))
    print('# Random sampling, temp=0.50:',generateSequenceRandom(0.5,dataset,model,device,length=100,startString=startStr))
    print('# Random sampling, temp=1.50:',generateSequenceRandom(1.5,dataset,model,device,length=100,startString=startStr))
    print('# Random sampling, temp=2.00:',generateSequenceRandom(2.0,dataset,model,device,length=100,startString=startStr))
    print('#')
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
    test(config)
