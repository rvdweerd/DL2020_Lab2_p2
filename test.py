from dataset import TextDataset
from torch.utils.data import DataLoader
import torch

def printSequence(sequenceTensor,itemInBatch):
    seq=sequenceTensor[:,b]
    for ch in seq:
        if ch.item()==1:
            print('+',end="")
        else:
            print(tdset.convert_to_string([ch.item()]),end="")
    #   INDICES
    print('[',end="")
    for ch in seq:
        print(ch.item(),',',end="")
    print(']')

bsize=5
slen=20

#tdset = TextDataset(filename="./book_EN_grimms_fairy_tails.txt",seq_length=10)
tdset = TextDataset(filename="./assets/book_NL_tolstoy_anna_karenina.txt",seq_length=slen)
data_loader = DataLoader(tdset, bsize)

test=next(iter(data_loader))
x=test[0]
t=test[1]
X=torch.stack(x)
T=torch.stack(t)

for b in range(bsize):
    printSequence(X,b)
    printSequence(T,b)

# for b in range(bsize):
#     # print input sequence for batch b
#     #   CHARACTERS
#     for t in test[0]:
#         if t[b].item()==1:
#             print('+',end="")
#         else:
#             print(tdset.convert_to_string([t[b].item()]),end="")
#     #   INDICES
#     for t in test[0]:
#         print('[',t[b].item(),']',end="")
#     print('\n')

#     # print target sequence for batch b
#     #   CHARACTERS
#     for t in test[1]:
#         if t[b].item()==1:
#             print('+',end="")
#         else:
#             print(tdset.convert_to_string([t[b].item()]),end="")
#     #   INDICES
#     for t in test[1]:
#         print('[',t[b].item(),']',end="")
#     print('\n-------------')
