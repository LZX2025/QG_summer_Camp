import math
import pickle

import torch
import torch.nn as nn
import os

from torch.nn import functional as F

from d2l import torch as d2l

from Model import LSTM, predict
from d2l.torch import grad_clipping

num_hidden = 512
batch_size = 32
num_steps = 32
vocab_path = 'data_vocab.pkl'
train_path = 'data_train.pkl'
if os.path.exists(vocab_path) and os.path.exists(train_path):
    with open(vocab_path, 'rb') as f:
        vocab = pickle.load(f)
    with open(train_path, 'rb') as f:
        train_iter = pickle.load(f)
else:
    train_iter, vocab = d2l.load_data_time_machine(batch_size, num_steps)
    with open(vocab_path, 'wb') as f:
        pickle.dump(vocab, f)
    with open(train_path, 'wb') as f:
        pickle.dump(train_iter, f)

print('vocab_len = ', len(vocab))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

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

        ppl = train_epoch(net, train_iter, criterion, updater, device)
        if (epoch + 1) % 10 == 0:
            print('epoch %d, ppl=%g' % (epoch + 1, ppl))


    #print('without save')
    model_path = 'lstm_model.pth'
    torch.save(net.state_dict(), model_path)
    print(f"模型已保存到 {model_path}")

def main():
    net = LSTM(vocab_size=len(vocab), hidden_size=num_hidden, device=device)
    net = net.to(device)
    train_net(net, train_iter, vocab)


if __name__ == '__main__':
    main()