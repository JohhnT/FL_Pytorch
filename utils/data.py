import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import random_split, DataLoader, Subset
from torch._utils import _accumulate
import random
import numpy as np
import pandas as pd
import cv2


class BuiltinTorchDataset:
    def __init__(self, config):
        self.config = config
        print(self.config.paths.data)
        self.path = str(self.config.paths.data) + '/' + self.config.dataset
        self.trainset = None
        self.testset = None
        self.Dataset = None

    def load_data(self, IID=True):
        self.trainset, self.testset = self.download_data()
        if self.trainset is not None and self.testset is not None:
            total_clients = self.config.clients.total
            total_sample = self.trainset.data.shape[0]
            # number of samples on each client
            length = [total_sample // total_clients] * total_clients
            if IID:
                spilted_train = random_split(self.trainset, length)

            else:
                print("None-IID")
                if sum(length) != len(self.trainset):
                    raise ValueError(
                        "Sum of input lengths does not equal the length of the input dataset!")
                index = []
                for i in range(10):
                    index.append([])

                i = 0
                for img, label in self.trainset:
                    index[label].append(i)
                    i += 1

                indices = np.array(
                    [elem for c_list in index for elem in c_list]).reshape(-1, 200)

                np.random.shuffle(indices)
                indices = indices.flatten()
                print(indices.shape)

                spilted_train = [Subset(self.trainset, indices[offset - length:offset]) for offset, length in
                                 zip(_accumulate(length), length)]
                print(len(spilted_train))
            return spilted_train, self.testset
        else:
            return self.trainset, self.testset

    def download_data(self):
        if self.Dataset is None:
            return None, None
        else:
            trainset = self.Dataset(
                self.path, train=True, download=True, transform=transforms.Compose([
                    transforms.RandomRotation(15),
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,))
                ]))
            testset = self.Dataset(
                self.path, train=False, transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,))
                ]))
            return trainset, testset


class FashionMNIST(BuiltinTorchDataset):
    def __init__(self, config):
        super().__init__(config)
        self.Dataset = datasets.FashionMNIST


class MNIST(BuiltinTorchDataset):
    def __init__(self, config):
        super().__init__(config)
        self.Dataset = datasets.MNIST


class CHMNIST(BuiltinTorchDataset):
    pass


def get_data(dataset, config):
    if dataset == "MNIST":
        return MNIST(config).load_data(IID=config.data.IID)
    elif dataset == "FashionMNIST":
        return FashionMNIST(config).load_data(IID=config.data.IID)
    elif dataset == "CHMNIST":
        # NOT IMPLEMENTED
        return CHMNIST(config).load_data(IID=config.data.IID)
