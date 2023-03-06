import torch.nn.functional as F
from torch import nn


def get_model(dataset_name):
    if dataset_name == "MNIST":
        return Mnist_Net()
    elif dataset_name == "FashionMNIST":
        return FashionMnist_Net()
    elif dataset_name == "CHMNIST":
        return CHMnist_Net(3*64*64, 15)


class FashionMnist_Net(nn.Module):
    def __init__(self):
        super(FashionMnist_Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 5)
        self.fc = nn.Sequential(
            nn.Linear(64 * 4 * 4, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 4 * 4 * 64)
        x = self.fc(x)
        return x


class Mnist_Net(nn.Module):
    def __init__(self):
        super(Mnist_Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 5)
        self.fc = nn.Sequential(
            nn.Linear(64 * 4 * 4, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 4 * 4 * 64)
        x = self.fc(x)
        return x


class CHMnist_Net(nn.Module):
    def __init__(self, input_size, output_size):
        super(CHMnist_Net, self).__init__()

        self.in_layer = nn.Linear(input_size, 2096)
        self.hidden1 = nn.Linear(2096, 1048)
        self.out_layer = nn.Linear(1048, output_size)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.in_layer(x)
        x = self.hidden1(F.relu(x))
        x = self.out_layer(F.relu(x))
        return x
