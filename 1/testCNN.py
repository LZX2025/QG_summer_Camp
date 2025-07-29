

import mindspore as ms
from mindspore import nn, ops
from mindspore import jit

ms.set_context(mode=ms.GRAPH_MODE, device_target="CPU")


class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()
        self.flatten = nn.Flatten()
        self.layer1 = nn.SequentialCell(
            nn.Dense(28*28, 512),
            nn.ReLU(),
            nn.Dense(512, 512),
            nn.ReLU(),
            nn.Dense(512, 10),
        )

    def construct(self, x):
        x = self.flatten(x)
        logits = self.layer1(x)
        return logits




from mindspore.dataset import vision, transforms
from mindspore.dataset import MnistDataset

# Download data from open datasets
from download import download

#url = "https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/" \
#      "notebook/datasets/MNIST_Data.zip"
#path = download(url, "./", kind="zip", replace=True)


def datapipe(path, batch_size):
    image_transforms = [
        vision.Rescale(1.0 / 255.0, 0),
        vision.Normalize(mean=(0.1307,), std=(0.3081,)),
        vision.HWC2CHW()
    ]
    label_transform = transforms.TypeCast(ms.int32)

    dataset = MnistDataset(path)
    dataset = dataset.map(image_transforms, 'image')
    dataset = dataset.map(label_transform, 'label')
    dataset = dataset.batch(batch_size)
    return dataset


batch_size = 64
epochs = 3
learning_rate = 0.001
model = Net()
train_dataset = datapipe('data/MNIST_Data/train', batch_size=batch_size)
test_dataset = datapipe('data/MNIST_Data/test', batch_size=batch_size)
criterion = nn.CrossEntropyLoss()
optimizer = nn.optim.Adam(params=model.trainable_params(), learning_rate=learning_rate)


def forward_fn(data, label):
    logits = model(data)
    loss = criterion(logits, label)
    return logits, loss

grad_fn = ms.value_and_grad(forward_fn)
params = model.trainable_params()
cast = ops.Cast()

def train_step(data, label):


    (loss, logits), grads = grad_fn(data, label)
    # grads = tuple([cast(grad, param.dtype) for grad, param in zip(grads, params)])

    optimizer(grads)
    return loss, logits.step()



def train_loop(model, dataset):

    size = dataset.get_dataset_size()
    model.set_train(True)
    for batch, (data, label) in enumerate(dataset.create_tuple_iterator()):
        loss, logits = train_step(data, label)

        if batch % 100 == 0:
            loss, current = loss.asnumpy(), batch
            print('batch:', batch)

def test_loop(model, dataset, criterion):
    num_batches = dataset.get_dataset_size()
    model.set_train(False)
    total, test_loss, correct = 0, 0, 0
    for data, label in dataset.create_tuple_iterator():
        pred = model(data)
        total += len(data)
        test_loss += criterion(pred, label).asnumpy()
        correct += ms.Tensor(pred.argmax(1) == label).asnumpy().sum()
    test_loss /= num_batches
    correct /= total
    print(correct, end=' ')


def main():
    for epoch in range(epochs):
        print("epoch ", epoch)
        train_loop(model, train_dataset)
        test_loop(model, test_dataset, criterion)
    print("Finished")

if __name__ == '__main__':
    main()




