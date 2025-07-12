import math
import pickle

import sys
import json

import torch
import torch.nn as nn
import os

from d2l.torch import grad_clipping
from torch.nn import functional as F
from tqdm import tqdm

from d2l import torch as d2l
from Model import predict, RNN

num_hidden = 512
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
batch_size = 32
num_steps = 32
if os.path.exists('imdb_vocab.pkl') and os.path.exists('imdb_train.pkl'):
    with open('imdb_vocab.pkl', 'rb') as f:
        vocab = pickle.load(f)
    with open('imdb_train.pkl', 'rb') as f:
        train_iter = pickle.load(f)
else:
    train_iter, vocab= d2l.load_data_time_machine(batch_size, num_steps)
    with open('imdb_vocab.pkl', 'wb') as f:
        pickle.dump(vocab, f)
    with open('imdb_train.pkl', 'wb') as f:
        pickle.dump(train_iter, f)



def train_epoch(net, train_iter, criterion, updater, device,):
    state = None
    metric = d2l.Accumulator(2)
    for X, Y in train_iter:
        if state is None:
            state = net.init_state(batch_size=X.shape[0], num_hidden=num_hidden, device=device)
        else:
            if isinstance(net, nn.Module) and not isinstance(state, tuple):
                state.detach_()
            else:
                for param in state:
                    param.detach_()

        y = Y.T.reshape(-1)
        X, y = X.to(device), y.to(device)
        y_hat, state = net(X, state)
        lo = criterion(y_hat, y.long()).mean()
        if isinstance(updater, torch.optim.Optimizer):
            updater.zero_grad()
            lo.backward()
            grad_clipping(net, 1)
            updater.step()
        else:
            lo.backward()
            grad_clipping(net, 1)
            updater(batch_size=1)
        metric.add(lo * y.numel(), y.numel())
    return math.exp(metric[0] / metric[1])

def train_net(net, train_iter, vocab, lr=0.5, num_epochs=500, device=device,):
    criterion = nn.CrossEntropyLoss().to(device)

    if isinstance(net, nn.Module):
        updater = torch.optim.SGD(net.parameters(), lr=lr)
    else:
        updater = lambda batch_size: d2l.sgd(net.params, lr, batch_size)
    pred = lambda prefix: predict(prefix, num_pred=30, net=net, vocab=vocab, device=device)

    for epoch in range(num_epochs):

        train_bar = tqdm(train_iter, file=sys.stdout)
        ppl = train_epoch(net, train_iter, criterion, updater, device)

        if (epoch + 1) % 10 == 0:
            print('epoch %d, ppl=%g' % (epoch + 1, ppl))
        train_bar.desc = "train epoch [{}/{}]".format(epoch + 1, num_epochs)

    print(pred('time traveller'))
    #print('without save')
    model_path = 'rnn_model.pth'
    torch.save(net.state_dict(), model_path)
    print(f"模型已保存到 {model_path}")

def main():
    net = RNN(len(vocab), num_hidden)
    net = net.to(device)
    train_net(net, train_iter, vocab)


if __name__ == '__main__':
    main()
