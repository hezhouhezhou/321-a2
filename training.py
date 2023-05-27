import os
import argparse
import pickle as pkl

import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import utils 
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import math
import random
import pandas as pd
from torch.autograd import Variable
# Local imports

import model

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

def checkpoint(model):

    with open(os.path.join('modelcp', 'model.pt'), 'wb') as f:
        torch.save(model, f)

def evaluate(features, target, used_model, criterion, opts):

    losses = []
    input_tensors = [torch.Tensor(w) for w in features]
    target_tensors = [torch.LongTensor([w]) for w in target]
    
    num_tensors = len(input_tensors)
    num_batches = int(np.ceil(num_tensors / float(opts.batch_size)))
    num_met = 0.0
    for i in range(num_batches):
        start = i * opts.batch_size
        #print("batch size "+ str(opts.batch_size))
        adder = min(opts.batch_size, num_tensors-start)
        end = start + adder
        inputs = utils.to_var(torch.stack(input_tensors[start:end]), opts)
        targets = utils.to_var(torch.stack(target_tensors[start:end]), opts)
        outputs = used_model(inputs)
        predicted = torch.argmax(outputs, dim=1)
        equal_elements = torch.eq(predicted, targets.squeeze(1))
        loss = 0.0
        num_met += torch.sum(equal_elements).item()
        loss += criterion(outputs, targets.squeeze(1))  # cross entropy between the decoder distribution and GT
        losses.append(loss.item())

    rate = num_met / num_tensors
    mean_loss = np.mean(losses)
    return mean_loss,rate

def training_loop(features_train, features_eval, target_train, target_eval, used_model, criterion, optimizer, opts):
    
    num_batches = int(np.ceil(len(features_train) / float(opts.batch_size)))
    best_val_loss = 1e6
    train_losses = []
    val_losses = []
    loss_log = open(os.path.join('loss_log', 'loss_log.txt'), 'w')

    for epoch in range(opts.nepochs):
        print("epoch: " + str(epoch))
        optimizer.param_groups[0]['lr'] *= opts.lr_decay
        input_tensors = [torch.Tensor(w) for w in features_train]
        target_tensors = [torch.LongTensor([int(w)]) for w in target_train]
        epoch_losses = []

        for i in range(num_batches):  
            start = i * opts.batch_size
            end = start + opts.batch_size if i < num_batches - 1 else len(features_train) - 1
            inputs = utils.to_var(torch.stack(input_tensors[start:end]), opts)
            targets = utils.to_var(torch.stack(target_tensors[start:end]), opts)
            outputs = used_model(inputs)
            loss = criterion(outputs, targets.squeeze(1))
            epoch_losses.append(loss.item())
            optimizer.zero_grad()

            # Compute gradients
            loss.backward()

            # Update the parameters of the model
            optimizer.step()
        train_loss = np.mean(epoch_losses)
        #print(train_loss)
        val_loss,rate = evaluate(features_eval,target_eval,used_model, criterion, opts)
        #print(val_loss)
        print(rate)
        if val_loss < best_val_loss:
            checkpoint(used_model)

        loss_log.write('{} {} {} {}\n'.format(epoch, train_loss, val_loss, rate))
        loss_log.flush()

        train_losses.append(train_loss)
        val_losses.append(val_loss)

def create_parser():
    """Creates a parser for command-line arguments.
    """
    parser = argparse.ArgumentParser()

    # Training hyper-parameters
    parser.add_argument('--nepochs', type=int, default=100,
                        help='The max number of epochs.')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='The number of examples in a batch.')
    parser.add_argument('--learning_rate', type=float, default=0.01,
                        help='The learning rate (default 0.01)')
    parser.add_argument('--lr_decay', type=float, default=0.99,
                        help='Set the learning rate decay factor.')
    parser.add_argument('--hidden_size', type=int, default=10,
                        help='The size of the GRU hidden state.')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints',
                        help='Set the directry to store the best model checkpoints.')
    parser.add_argument('--cuda', action='store_true', default=False,
                        help='Choose whether to use GPU.')
    parser.add_argument('--mps', action='store_true', default=False, 
                        help='choose whether to use mps')

    return parser


def print_opts(opts):
    """Prints the values of all command-line arguments.
    """
    print('=' * 80)
    print('Opts'.center(80))
    print('-' * 80)
    for key in opts.__dict__:
        print('{:>30}: {:<30}'.format(key, opts.__dict__[key]).center(80))
    print('=' * 80)


if __name__ == '__main__':

    parser = create_parser()
    opts = parser.parse_args()
    features, targets = utils.read_data('DryBeanDataset/Dry_Bean_Dataset.xlsx')
    #print(targets.size())
    num = len(features)
    numtrain = math.floor(num * 0.8)
    numeval = num-numtrain

    features_train,features_eval = features[:numtrain],features[numtrain:]
    target_train, target_eval = targets[:numtrain],targets[numtrain:]

    mod = model.DryBeanModel()
    criterion = nn.CrossEntropyLoss()
    #print(list(mod.parameters()))
    optimizer = optim.Adam(mod.parameters(), lr=opts.learning_rate)

    if opts.cuda:
        mod.cuda()
        print("Moved models to GPU!")
    elif opts.mps:
        mod.to("mps")
        print("Moved models to apple chip")
    
    try:
        training_loop(features_train,features_eval, target_train, target_eval, mod, criterion, optimizer, opts)
    except KeyboardInterrupt:
        print('Exiting early from training.')
