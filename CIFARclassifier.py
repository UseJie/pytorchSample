# 使用AlexNet对CIFAR进行分类

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, Resize
import time

# prepare dataset
# The CIFAR-10 dataset consists of 60000 32x32 colour images in 10 classes,
# with 6000 images per class.
# There are 50000 training images and 10000 test images.
# https://www.cs.toronto.edu/~kriz/cifar.html
train_dataset = datasets.CIFAR10(
    root="data",
    download=True,
    train=True,
    transform=ToTensor(),
)
test_dataset = datasets.CIFAR10(
    root="data",
    download=True,
    train=False,
    transform=ToTensor()
)


# hyperparameter
learning_rate = 1e-5
batch_size = 32
epoch = 10
# Model
# [227x227x3] INPUT
# [55x55x96] CONV1: 96 11x11 filters at stride 4, pad 0
# [27x27x96] MAX POOL1: 3x3 filters at stride 2
# [27x27x96] NORM1: Normalization layer
# [27x27x256] CONV2: 256 5x5 filters at stride 1, pad 2
# [13x13x256] MAX POOL2: 3x3 filters at stride 2
# [13x13x256] NORM2: Normalization layer
# [13x13x384] CONV3: 384 3x3 filters at stride 1, pad 1
# [13x13x384] CONV4: 384 3x3 filters at stride 1, pad 1
# [13x13x256] CONV5: 256 3x3 filters at stride 1, pad 1
# [6x6x256] MAX POOL3: 3x3 filters at stride 2
# [4096] FC6: 4096 neurons
# [4096] FC7: 4096 neurons
# [1000] FC8: 1000 neurons (class scores)
class AlexNetModel(nn.Module):
    def __init__(self, num_classes=10):
        super(AlexNetModel, self).__init__()
        self.feature = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=0),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(96, 256, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1,),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        self.fullyConnect = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256*6*6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            #nn.Softmax(4096, 10),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = Resize([227, 227])(x)
        x = self.feature(x)
        x = x.view(x.size(0), 256*6*6)
        x = self.fullyConnect(x)
        return x

model = AlexNetModel()

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
def train(dataloader, model, lossFun, optimizer):
    size = len(dataloader.dataset)
    train_loss = 0
    for batch, (X, y) in enumerate(dataloader):
        time_start = time.time()
        pred = model(X)
        time_end = time.time()
        loss = lossFun(pred, y)
        train_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch*len(X)
            print(f'loss: {loss:>7f} [{current:>5d}/{size:5d}] [per image time: {(time_end-time_start)/len(X)}s]')
    return train_loss

def test_loop(test_dataloader, model, loss_fn):
    size = len(test_dataloader.dataset)
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in test_dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1)==y).type(torch.float).sum().item()

    test_loss /= size
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    return test_loss, correct

for t in range(epoch):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loss = train(train_loader, model, loss_fn, optimizer)
    test_loss, correct = test_loop(test_loader, model, loss_fn)
torch.save(model, "model.pth")
print("Done!")

# 记录损失值
