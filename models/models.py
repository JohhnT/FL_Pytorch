import torch.nn.functional as F
from torch import nn


def get_model(dataset_name):
    if dataset_name == "MNIST":
        return Mnist_Net()
    elif dataset_name == "FashionMNIST":
        return FashionMnist_Net()
    elif dataset_name == "CHMNIST":
        return CHMnist_Net(3*64*64, 15)
    elif dataset_name == "BreastCancer":
        return BreastCancer_Net()


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
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 30, 3),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(30, 50, 3),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.fc = nn.Sequential(
            nn.Linear(5*5*50, 200),
            nn.ReLU(),
            nn.Linear(200, 10),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = x.view(x.size(0), -1)
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


class BreastCancer_Net(nn.Module):
    def __init__(self, i_dim=29, o_dim=2):
        super(BreastCancer_Net, self).__init__()
        self.i_dim = i_dim
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=30,
                      kernel_size=(1, 3), padding=(1, 1)),
            nn.ReLU()
        )
        self.layer2 = nn.MaxPool2d(kernel_size=(2, 2))
        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=30, out_channels=50,
                      kernel_size=(3, 3), padding=(1, 1)),
            nn.ReLU()
        )
        self.layer4 = nn.MaxPool2d(kernel_size=(2, 2), padding=(1, 0))
        self.layer5 = nn.Sequential(
            nn.Linear(in_features=50 * 7, out_features=200),
            nn.ReLU()
        )
        self.layer6 = nn.Sequential(
            nn.Linear(in_features=200, out_features=o_dim),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = x.view(-1, 1, 1, self.i_dim)
        x = self.layer1(x)
        x = x.view(x.size(0), 30, -1, self.i_dim)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = x.view(x.size(0), -1)
        x = self.layer5(x)
        x = self.layer6(x)
        return x
