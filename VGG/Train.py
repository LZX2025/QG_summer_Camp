import os
import sys
import json

import torch
import torch.nn as nn

from torchvision import transforms, datasets
import torch.optim as optim
from tqdm import tqdm

from vggModel import vgg



def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("device: ", device)
    transform = {
        'train' : transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]),
        'val' : transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
    }

    data_root = "../" + "data_set/flower_data"
    assert os.path.exists(data_root), "{} doesn't exist".format(data_root)
    batch_size = 32
    train_dataset = datasets.ImageFolder(root=data_root + '/train', transform=transform['train'])
    val_dataset = datasets.ImageFolder(root=data_root + '/val', transform=transform['val'])
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    print("train {}, val {}".format(len(train_dataset), len(val_dataset)))

    cla_path = 'cla_dict.json'
    cla_list = train_dataset.class_to_idx
    cla_dict = dict((idx, name) for name, idx in cla_list.items())
    with open(cla_path, 'w') as f:
        f.write(json.dumps(cla_dict, indent=4))

    net = vgg('vgg16', num_classes=len(cla_list))
    net.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.0002)

    epochs = 20
    s_path = 'vgg_model.pth'
    best_accuracy = 0.0

    for epoch in range(epochs):
        net.train()
        running_loss = 0.0
        train_bar = tqdm(train_loader, file=sys.stdout)
        for i, data in enumerate(train_bar):
            inputs, labels = data[0].to(device), data[1].to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels.to(device))
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            train_bar.desc = "train epoch [{}/{}], loss:{:.3f}".format(epoch + 1, epochs, loss)

        net.eval()
        accuracy = 0.0
        with torch.no_grad():
            val_bar = tqdm(val_loader, file=sys.stdout)
            for data in val_bar:
                inputs, labels = data
                outputs = net(inputs.to(device))
                pred = torch.argmax(outputs, dim=1)
                accuracy += torch.sum(pred == labels.to(device)).item()

        val_accuracy = accuracy / len(val_dataset)
        print('epoch %d, loss %.3f, accuracy %.3f' % (epoch + 1, running_loss / len(train_loader), val_accuracy))
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            torch.save(net.state_dict(), s_path)

    print("Training done")

if __name__ == '__main__':
    main()





