import torch
import torchvision
import torch.nn as nn


from lenetmodel import LeNet
import torch.optim as optim
import torchvision.transforms as transforms



def mian():
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=False, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=36, shuffle=True)

    test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=100, shuffle=True)

    test_iter = iter(test_loader)
    test_image, test_label = next(test_iter)

    # classes = ('plane', 'car', 'bird', 'cat',
    #            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    net = LeNet()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    for epoch in range(5):
        r_loss = 0.0
        for step, (inputs, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            r_loss += loss.item()

            if step % 500 == 0:
                with torch.no_grad():
                    outputs = net(test_image)
                    predicted = torch.argmax(outputs, 1)
                    accuracy = torch.eq(predicted, test_label).sum().item() / test_label.size(0)
                    print('[%d, %5d] train_loss: %.3f  test_accuracy: %.3f' %
                          (epoch + 1, step + 1, r_loss / 500, accuracy))
                    r_loss = 0.0

    print('Finished Training')
    savepath = './Lenet.pth'
    torch.save(net.state_dict(), savepath)

if __name__ == '__main__':
    mian()