import pandas as pd
import numpy as np
import torch 
import random
from torch.autograd import Variable
label_to_index = {"SEKER":0,"BARBUNYA":1, "BOMBAY":2,"CALI":3,"HOROZ":4,"SIRA":5,"DERMASON":6}
index_to_label = {label_to_index[key]:key for key in label_to_index}
labeledindex_to_clusterindex = {}
def read_data(path):
    df = pd.read_excel(path).values.tolist()
    print(random.shuffle(df))
    print(df)
    values = list([sublist[:-1] for sublist in df])
    tags = list([label_to_index[sublist[-1]] for sublist in df])
    print(tags)
    # print(data_array)
    # print(len(data_array))
    # print(values)
    return values, tags

def to_var(tensor, opts):
    """Wraps a Tensor in a Variable, optionally placing it on the GPU.

        Arguments:
            tensor: A Tensor object.
            cuda: A boolean flag indicating whether to use the GPU.

        Returns:
            A Variable object, on the GPU if cuda==True.
    """
    if opts.cuda:
        return Variable(tensor.cuda())
    elif opts.mps:
        return Variable(tensor.to("mps"))
    else:
        return Variable(tensor)
if __name__ == '__main__':
    read_data('DryBeanDataset/Dry_Bean_Dataset.xlsx')

