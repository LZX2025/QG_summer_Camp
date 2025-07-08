
import json
import os
import torch
from PIL import Image
from torchvision import transforms

from vggModel import vgg

def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
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

    model_path = 'vgg_model.pth'
    assert os.path.exists(model_path),"{} does not exist".format(model_path)

    net = vgg(model_name='vgg16', num_classes=len(cla_dict))
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
