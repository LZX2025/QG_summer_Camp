import torch
import torchvision.transforms as transforms
from PIL import Image

from lenetmodel import LeNet

def main():
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),])

    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    net = LeNet()
    net.load_state_dict(torch.load('Lenet.pth'))

    im = Image.open('test.jpg')
    im = transform(im)
    im = torch.unsqueeze(im, dim=0)

    with torch.no_grad():
        out = net(im)
        pre = torch.argmax(out, 1)
        print(classes[int(pre.item())])

if __name__ == '__main__':
    main()
