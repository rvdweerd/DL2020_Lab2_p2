from dataset import TextDataset
from torch.utils.data import DataLoader

bsize=5
slen=20

#tdset = TextDataset(filename="./book_EN_grimms_fairy_tails.txt",seq_length=10)
tdset = TextDataset(filename="./assets/book_NL_tolstoy_anna_karenina.txt",seq_length=slen)
data_loader = DataLoader(tdset, bsize)

test=next(iter(data_loader))

for b in range(bsize):
    # print input sequence for batch b
    #   CHARACTERS
    for t in test[0]:
        if t[b].item()==1:
            print('+',end="")
        else:
            print(tdset.convert_to_string([t[b].item()]),end="")
    #   INDICES
    for t in test[0]:
        print('[',t[b].item(),']',end="")
    print('\n')

    # print target sequence for batch b
    #   CHARACTERS
    for t in test[1]:
        if t[b].item()==1:
            print('+',end="")
        else:
            print(tdset.convert_to_string([t[b].item()]),end="")
    #   INDICES
    for t in test[1]:
        print('[',t[b].item(),']',end="")
    print('\n-------------')
