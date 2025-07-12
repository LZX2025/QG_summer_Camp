import math
import pickle

import torch
import torch.nn as nn
import os

from torch.nn import functional as F

from d2l import torch as d2l

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

"""
# 假设vocab是通过d2l.load_data_imdb()加载的词汇表
word = "example"  # 要查询的单词
idx = vocab[word]  # 获取单词对应的索引

print(f"单词 '{word}' 对应的索引是: {idx}")
"""




device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

X = torch.arange(10).reshape((2, 5))
#print(F.one_hot(X.T, 28).shape)



def get_param(vocab_size, num_hidden, device):
    def normal(shape):
        return torch.randn(shape, device=device)

    W_xh = normal((vocab_size, num_hidden))
    W_hh = normal((num_hidden, num_hidden))
    W_hy = normal((num_hidden, vocab_size))
    b_h = torch.zeros(num_hidden, device=device)
    b_y = torch.zeros(vocab_size, device=device)
    params = [W_xh, W_hh, b_h, W_hy, b_y]
    for param in params:
        param.requires_grad_(True)
    params = nn.ParameterList(params)
    return params

def init_state_(batch_size, num_hidden, device):
    return (torch.zeros((batch_size, num_hidden), device=device), )


class RNN(nn.Module):
    def __init__(self, vocab_size, hidden_size, get_param=get_param, init_state=init_state_, device=device):
        super(RNN, self).__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.params = get_param(vocab_size, hidden_size, device=device)
        self.init_state = init_state



    def forward(self, x, state):
        input = F.one_hot(x.T, self.vocab_size).float()
        W_xh, W_hh, b_h, W_hy, b_y = self.params
        H, = state
        outputs = []
        for x in input:
            H = torch.tanh(torch.mm(x, W_xh) + torch.mm(H, W_hh) + b_h)
            out = torch.mm(H, W_hy) + b_y
            outputs.append(out)

        return torch.cat(outputs, 0), (H,)

    def init_state(self, batch_size):
        return self.init_state(batch_size, self.hidden_size, self.device)

def grad_clip(net, max_norm = 1.0):
    nn.utils.clip_grad_norm_(net.params, max_norm)

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







