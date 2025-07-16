import os
import sys
import json

import torch
import torch.nn as nn

from torchvision import transforms, datasets
import torch.optim as optim
from tqdm import tqdm

from Model import make_model
#from Model import res_next50_32x4d

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("device: ", device)
    transform = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
    }

    data_root = '../data/cassava-leaf-disease-classification/departed_images'
    assert os.path.exists(data_root), 'cannot find data root: {}'.format(data_root)
    batch_size = 32
    train_dataset = datasets.ImageFolder(root=data_root + '/train', transform=transform['train'])
    val_dataset = datasets.ImageFolder(root=data_root + '/val', transform=transform['val'])
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    print("train {}, val {}".format(len(train_dataset), len(val_dataset)))

    cla_path = '../data/cassava-leaf-disease-classification/label_num_to_disease_map.json'
    cla_dic = json.load(open(cla_path, 'r'))

    model = make_model()
    #model = res_next50_32x4d()
    pre_model_path = "../Pre_model/resnext50_32x4d_pre.pth"
    _, _ = model.load_state_dict(torch.load(pre_model_path), strict=False)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, len(cla_dic))

    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0002)

    epochs = 5
    s_path = 'model.pth'
    best_accuracy = 0.0

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        train_bar = tqdm(train_loader, file=sys.stdout)
        for i, data in enumerate(train_bar):
            inputs, labels = data[0].to(device), data[1].to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels.to(device))
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            train_bar.desc = "train epoch [{}/{}], loss:{:.3f}".format(epoch + 1, epochs, loss)

        model.eval()
        accuracy = 0.0
        with torch.no_grad():
            val_bar = tqdm(val_loader, file=sys.stdout)
            for data in val_bar:
                inputs, labels = data
                outputs = model(inputs.to(device))
                pred = torch.argmax(outputs, dim=1)
                accuracy += torch.sum(pred == labels.to(device)).item()

        val_accuracy = accuracy / len(val_dataset)
        print('epoch %d, loss %.3f, accuracy %.3f' % (epoch + 1, running_loss / len(train_loader), val_accuracy))
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            torch.save(model.state_dict(), s_path)

    print("Training done")

if __name__ == '__main__':
    main()
