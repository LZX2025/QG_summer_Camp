import math
import pickle

import torch
import torch.nn as nn
import os

from d2l import torch as d2l
from Model import predict, RNN

num_hidden = 512
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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

def main():
    net = RNN(vocab_size=len(vocab), hidden_size=num_hidden)
    model_path = 'rnn_model.pth'
    assert os.path.exists(model_path),"{} does not exist".format(model_path)
    net.load_state_dict(torch.load(model_path))
    net.to(device)
    print(predict('Time traveller', 20, net, vocab, device))


if __name__ == '__main__':
    main()