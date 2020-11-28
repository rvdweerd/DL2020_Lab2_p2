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

# This demo file loads a pre-trained SLTM model, tests its accuracy and produces self-generated sentences
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
    dataset = TextDataset(filename=config.txt_file,seq_length=30)  
    data_loader = DataLoader(dataset, 32)

    model=TextGenerationModel(config,dataset._vocab_size,device).to(device)
    
    if device.type=='cpu':
        checkpoint=torch.load("AnnaK_0.48_cpu.tar")
    else:
        checkpoint=torch.load("AnnaK_0.59_cuda.tar")
    model.load_state_dict(checkpoint['model_state_dict'])
    print(model)
    testLSTM(dataset,data_loader,model,config,device)
    accuracy=getTestAccuracy(dataset,data_loader,model,config,device,200)

    startStr=config.startstring
    print('########### SAMPLE SELF GENERATED SEQUENCE ###############')
    print('# Test accuracy:',accuracy)
    print('# Self generated sentences, start string = Anna')
    print('#')
    print('# Greedy sampling           :',generateSequenceGreedy(dataset,model,device,length=100,startString=startStr))
    print('# Random sampling, temp=0.01:',generateSequenceRandom(0.01,dataset,model,device,length=100,startString=startStr))
    print('# Random sampling, temp=0.10:',generateSequenceRandom(0.1,dataset,model,device,length=100,startString=startStr))
    print('# Random sampling, temp=0.50:',generateSequenceRandom(0.5,dataset,model,device,length=100,startString=startStr))
    print('# Random sampling, temp=1.00:',generateSequenceRandom(1.0,dataset,model,device,length=100,startString=startStr))
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
    parser.add_argument('--startstring', type=str, default="Anna",
                        help="Start string to prompt sentence generation.")
    # If needed/wanted, feel free to add more arguments

    config = parser.parse_args()

    # Train the model
    test(config)
