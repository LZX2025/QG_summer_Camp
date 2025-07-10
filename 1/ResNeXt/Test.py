import os
import json

import torch
from PIL import Image
from torchvision import transforms

from Model import res_next50_32x4d

def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225]),
    ])

    json_path = 'cla_dict.json'
    assert os.path.exists(json_path),"{} does not exist".format(json_path)
    with open(json_path, 'r') as f:
        cla_dict = json.load(f)

    im_path = 'test.jpg'
    assert os.path.exists(im_path),"{} does not exist".format(im_path)
    im = Image.open(im_path)
    im = transform(im)
    im = torch.unsqueeze(im, 0)

    model_path = 'resnext_model.pth'
    assert os.path.exists(model_path),"{} does not exist".format(model_path)

    net = res_next50_32x4d(num_classes=5)
    net.load_state_dict(torch.load(model_path))
    net.to(device)
    net.eval()
    with torch.no_grad():
        out = net(im.to(device)).cpu()
        out = out.squeeze(0)
        pred = torch.argmax(out).numpy()

    print(cla_dict[str(pred)])

if __name__ == '__main__':
    main()