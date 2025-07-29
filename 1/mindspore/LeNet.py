import mindspore
from mindspore import nn



class LeNet5(nn.Cell):
    def __init__(self, num_class = 10, num_channel = 1):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(num_channel, 6, 5, pad_mode='valid')
        self.conv2 = nn.Conv2d(6, 16, 5, pad_mode='valid')
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()
        self.fc1 = nn.Dense(16 * 4 * 4, 120)
        self.fc2 = nn.Dense(120, 84)
        self.fc3 = nn.Dense(84, num_class)
        self.flatten = nn.Flatten()

    def construct(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x


from mindspore import Model

net = LeNet5()
criterion = nn.CrossEntropyLoss()
optim = nn.Adam(params=net.trainable_params(), learning_rate=0.001)
model = Model(network=net, loss_fn=criterion, optimizer=optim, metrics={'accuracy'})

import mindspore.dataset as ds
import mindspore.dataset.transforms.c_transforms as C

from mindspore.train.callback import LossMonitor
from mindspore.dataset import vision, transforms

train_path = 'data/MNIST_Data/train'
test_path = 'data/MNIST_Data/test'
batch_size = 32
def datapipe(path, batch_size):
    img_trans = [
        ds.vision.Resize((28,28)),
        ds.vision.Rescale(1/255, 0),
        ds.vision.HWC2CHW()
    ]
    label_trans = transforms.TypeCast(mindspore.int32)
    dataset = ds.MnistDataset(path)
    dataset = dataset.map(img_trans, 'image')
    dataset = dataset.map(label_trans, 'label')
    dataset = dataset.batch(batch_size)
    return dataset

train_dataset = datapipe(train_path, batch_size)
test_dataset = datapipe(test_path, batch_size)

loss_cb = LossMonitor(per_print_times=1000)
model.train(epoch=2, train_dataset=train_dataset, callbacks=[loss_cb])

model.eval(test_dataset)
save_path = 'model.ckpt'
mindspore.save_checkpoint(net, save_path)