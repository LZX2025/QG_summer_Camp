import sys
import os

import torch
from tqdm import tqdm
from torch import nn
from d2l import torch as d2l


batch_size = 128
num_steps = 64

train_iter, vocab = d2l.load_data_time_machine(batch_size, num_steps)

vocab_size = len(vocab)
hidden_size = 256
num_layers = 3
save_path = 'DLstm.pth'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

class LSTM(nn.Module):
    def __init__(self, vocab_size, hidden_size, num_layers):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=hidden_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True).to(device)
        self.embedding = nn.Embedding(vocab_size, hidden_size).to(device)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.device = device
        self.vocab_size = vocab_size

    def forward(self, x, state = None):
        x = self.embedding(x)
        if state is None:
            state = (torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=device),
                     torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=device))

        x, state = self.lstm(x, state)
        x = self.linear(x)
        return x, state

def train(net):
    optimizer = torch.optim.Adam(net.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',factor=0.5, patience=5)    # 调整学习率
    criterion = nn.CrossEntropyLoss()
    num_epochs = 500
    net.train()
    best_loss = float('inf')
    for epoch in range(num_epochs):
        e_loss = 0

        for x, y in train_iter:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            output, _ = net(x)
            output = output.view(-1, vocab_size)
            y = y.view(-1)
            loss = criterion(output, y)
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1)
            optimizer.step()
            e_loss += loss.item()
        avg_loss = e_loss / (epoch + 1)
        if epoch + 1 % 10 == 0:
            print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch + 1, num_epochs, avg_loss))
        scheduler.step(avg_loss)
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(net.state_dict(), save_path)

    print('Finished Training')

def predict(prefix, net, num_predict, vocab, device):
    net.eval()
    state = None
    output = [vocab[prefix[0]]]
    get_input = lambda: torch.tensor([output[-1]], device=device).reshape((1, 1))
    for y in prefix[1:]:
        _, state = net(get_input(), state)
        output.append(vocab[y])
    for _ in range(num_predict):
        y, state = net(get_input(), state)
        pred = y.argmax().item()
        output.append(pred)
    return ''.join([vocab.idx_to_token[i] for i in output])

def main():
    model = LSTM(vocab_size, hidden_size, num_layers).to(device)
    if os.path.exists(save_path):
        train(model)

    model.load_state_dict(torch.load(save_path))
    print(predict('The name of time traveller is ', model, 50, vocab, device))


if __name__ == '__main__':
    main()


