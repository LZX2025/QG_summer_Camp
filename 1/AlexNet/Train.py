
import json
import sys


import torch
import torch.nn as nn
from torchvision import transforms, datasets, utils

import torch.optim as optim
from tqdm import tqdm

from anModel import AlexNet



def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("device: ", device)

    data_transforms = {
        "train":transforms.Compose([
                        transforms.RandomResizedCrop(224),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),]),
        "val":transforms.Compose([
                        transforms.Resize((224, 224)),
                        transforms.ToTensor(),
                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),])
    }

    data_root = "../" + "data_set/flower_data"
    train_dataset = datasets.ImageFolder(root=data_root + "/train", transform=data_transforms["train"])
    val_dataset = datasets.ImageFolder(root=data_root + "/val", transform=data_transforms["val"])

    flower_list = train_dataset.class_to_idx
    cla_dict = dict((idx, name) for name, idx in flower_list.items())
    json_str = json.dumps(cla_dict, indent=4)
    with open('cla_dict.json', 'w') as f:
        f.write(json_str)

    batch_size = 32

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=4, shuffle=False)

    print("ues {} for training, use {} for validation".format(len(train_dataset), len(val_dataset)))

    net = AlexNet()
    net.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.0002)

    epochs = 10
    savepath = "./alexnet.pth"
    best_acc = 0.0
    train_step = len(train_loader)


    for epoch in range(epochs):
        #train
        net.train()
        running_loss = 0.0
        train_bar = tqdm(train_loader, file=sys.stdout)
        for step, data in enumerate(train_bar):
            inputs, labels = data
            optimizer.zero_grad()
            outputs = net(inputs.to(device))
            loss = criterion(outputs, labels.to(device))
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            train_bar.desc = "train epoch [{}/{}], loss:{:.3f}".format(epoch + 1, epochs, loss)

        # validate
        net.eval()
        acc = 0.0
        with torch.no_grad():
            val_bar = tqdm(val_loader, file=sys.stdout)
            for data in val_bar:
                inputs, labels = data
                outputs = net(inputs.to(device))
                pred = torch.argmax(outputs, dim=1)
                acc += torch.sum(pred == labels.to(device)).item()

        val_acc = acc / len(val_dataset)
        print("epoch: ", epoch+1,
              "val_acc: ", val_acc,
              "loss: ", running_loss/train_step)
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(net.state_dict(), savepath)

    print("Finished Training")

if __name__ == "__main__":
    main()
