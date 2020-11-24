import matplotlib.pyplot as plt
import torch

def printSequence(sequenceTensor,itemInBatch,textdataset):
    # Printing function for our text dataset
    # Prints sequence in column c=itemInBatch contained in stacked sequenceTensor (seq_len,batch_size)
    # Used for Q2.1a
    seq=sequenceTensor[:,itemInBatch]
    print('[',end="")
    for ch in seq:
        if ch.item()==1:
            print('+',end="")
        else:
            print(textdataset.convert_to_string([ch.item()]),end="")
    print(']')

def pltLossAcc(loss_plt,acc_plt,config):
    plt.plot(loss_plt,label='Train loss',color='tab:orange')
    plt.title('Train loss (NLL) curve',fontsize=15)
    plt.xlabel('Training step (mini batch)',fontsize=15)
    plt.ylabel('Loss (NLL)',fontsize=15)
    note1 = 'seq_len='+str(config.seq_length)+', LSTM layers/num_hidden='+str(config.lstm_num_layers)+'/'+str(config.lstm_num_hidden)
    note2 = 'bsize=' + str(config.batch_size) + ', lr=%.1E' %config.learning_rate
    plt.text(0,4.5, note1)
    plt.text(0,4.35, note2)
    plt.legend()
    axes=plt.axes()
    plt.show()

    plt.plot(acc_plt,label='Train accuracy')
    plt.title('Train accuracy curves',fontsize=15)
    plt.xlabel('Training step (mini batch)',fontsize=15)
    plt.ylabel('Accuracy',fontsize=15)
    note1 = 'seq_len='+str(config.seq_length)+', LSTM layers/num_hidden='+str(config.lstm_num_layers)+'/'+str(config.lstm_num_hidden)
    note2 = 'bsize=' + str(config.batch_size) + ', lr=%.1E' %config.learning_rate
    plt.text(0,0.95, note1)
    plt.text(0,0.9, note2)
    plt.legend()
    axes=plt.axes()
    axes.set_ylim(0,1.05)
    plt.show()

def getTestAccuracy(dataset,data_loader,model,config,device,numEvalBatches=100):
    # Check model performance: takes a number of batches from dataset and calculates accuracy
    correct=0
    total=0
    model.eval()
    with torch.no_grad():
        for i in range(numEvalBatches):
            (x,t) = next(iter(data_loader))  # x and t are lists (len=seq_len) of tensors (bsize)
            X = torch.stack(x).to(device)    # (seq_len,bsize)
            T = torch.stack(t).to(device)
            h,C = model.init_cell(config.batch_size)
            logprobs,_,_ = model(X,h,C)          # (seq_len,bsize,voc_size)
            predchar = torch.argmax(logprobs,dim=2) # (seq_len,bsize) the predicted characters: selected highest logprob for each sequence and example in the mini batch
            correct+=torch.sum(predchar==T).item()
            total+=(config.batch_size * config.seq_length)
        accuracy =correct / total
    print('Test accuracy over ',numEvalBatches*config.batch_size,' sequences:',accuracy)
    model.train()
    return accuracy

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
        logprobs,h,C = model(charId,h,C) # forward pass for one character charId
    # Now, run the cell independently (its output is fed back into the cell to self-generate)
    for i in range(length-len(startString)):
        probs = torch.exp(logprobs) # convert back to probs from logprobs
        predchar = torch.multinomial(probs.squeeze(),1) # sample according to PMF
        if predchar.item()==1:
            seq_out+='+'
        else:
            seq_out+=dataset._ix_to_char[predchar.item()]
        startId=predchar
        logprobs,h,C = model(startId,h,C)
    model.train()
    return seq_out

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
    logprobs,_,_ = model(X,h,C)          # (seq_len,bsize,voc_size)
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
    logprobs,_,_ = model(X_test,h,C) # (seq_len,1,voc_size)
    assert (logprobs.size(0)==config.seq_length and logprobs.size(1)==1 and logprobs.size(2)==dataset._vocab_size)

    #############
    # Third Test: try forward pass for only one training sequence and one character
    #############
    X_test2 = X[0,0].to(device)
    h,C = model.init_cell(1)
    logprobs,_,_ = model(X_test2,h,C) # (1,1,voc_size)
    assert (logprobs.size(0)==1 and logprobs.size(1)==1 and logprobs.size(2)==dataset._vocab_size)
    ####################
    print('Model tests passed..')
    ####################
    # End of tests
    ####################