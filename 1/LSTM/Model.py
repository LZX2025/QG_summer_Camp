import math
import pickle

import torch
import torch.nn as nn
import os

import torch.nn.functional as F

from d2l import torch as d2l


def get_param(vocab_size, num_hidden, device):

    def normal(shape):
        return torch.randn(size=shape, device=device) * 0.01
    def init_3():
        return normal((vocab_size, num_hidden)), normal((num_hidden, num_hidden)), torch.zeros(num_hidden, device=device)

    W_xi, W_hi, b_i = init_3()  # 输入门参数
    W_xf, W_hf, b_f = init_3()  # 遗忘门参数
    W_xo, W_ho, b_o = init_3() # 输出门参数
    W_xc, W_hc, b_c = init_3() # 候选记忆元参数

    W_hy = normal((num_hidden, vocab_size))
    b_y = torch.zeros(vocab_size, device=device)   # 输出层参数

    params = [W_xi, W_hi, b_i, W_xf, W_hf, b_f, W_xo, W_ho, b_o, W_xc, W_hc, b_c, W_hy, b_y]
    for param in params:
        param.requires_grad_(True)
    params = nn.ParameterList(params)
    return params

def init_state_(batch_size, num_hidden, device):
    return torch.zeros((batch_size, num_hidden), device=device), torch.zeros((batch_size, num_hidden), device=device)


class LSTM(nn.Module):
    def __init__(self, vocab_size, hidden_size, device, get_param=get_param, init_state=init_state_, ):
        super(LSTM, self).__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.params = get_param(vocab_size, hidden_size, device=device)
        self.init_state = init_state
        self.device = device

    def init_state(self, batch_size):
        return self.init_state(batch_size, self.hidden_size, self.device)

    def forward(self, x, state):
        input = nn.functional.one_hot(x.T, self.vocab_size).float()
        W_xi, W_hi, b_i, W_xf, W_hf, b_f, W_xo, W_ho, b_o, W_xc, W_hc, b_c, W_hy, b_y = self.params
        (H, C) = state
        outputs = []
        for X in input:
            I = torch.sigmoid((X @ W_xi) + (H @ W_hi) + b_i)
            F = torch.sigmoid((X @ W_xf) + (H @ W_hf) + b_f)
            O = torch.sigmoid((X @ W_xo) + (H @ W_ho) + b_o)
            C_t = torch.tanh((X @ W_xc) + (H @ W_hc) + b_c)
            C = I * C_t + F * O
            H = O * torch.tanh(C)
            output = (H @ W_hy + b_y)
            outputs.append(output)
        return torch.cat(outputs, dim=0), (H, C)

def predict(prefix, num_pred, net, vocab, device):
    state = net.init_state(batch_size=1, num_hidden=net.hidden_size, device=device)
    print(prefix[0])
    outputs = [vocab[prefix[0]]]
    get_input = lambda: torch.tensor([outputs[-1]], device=device).reshape((1, 1))
    # rank1
    for y in prefix[1:]:
        _, state = net(get_input(), state)
        outputs.append(vocab[y])
    # rank2
    for _ in range(num_pred):
        y, state = net(get_input(), state)
        outputs.append(int(y.argmax(dim=1).reshape(1)))
    return ''.join([vocab.idx_to_token[i] for i in outputs])

