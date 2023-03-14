import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import random_split, Subset, TensorDataset
from torch._utils import _accumulate
from utils.datasets import CHMNISTDataset
import random
import numpy as np
import pandas as pd
import cv2
from sklearn.model_selection import train_test_split


class FLDataset:
    def __init__(self, config):
        self.config = config
        # print(self.config.paths.data)
        self.path = str(self.config.paths.data) + '/' + self.config.dataset
        self.trainset = None
        self.testset = None

    def download_data(self):
        pass

    def load_data(self, IID=True):
        self.trainset, self.testset = self.download_data()
        if self.trainset is not None and self.testset is not None:
            total_clients = self.config.clients.total
            try:
                # builtin torch dataset
                total_sample = self.trainset.data.shape[0]
            except:
                # CHMNIST
                total_sample = len(self.trainset)
            # number of samples on each client
            length = [total_sample // total_clients] * total_clients
            if IID:
                spilted_train = random_split(self.trainset, length)

            else:
                # print("None-IID")
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
                # print(indices.shape)

                spilted_train = [Subset(self.trainset, indices[offset - length:offset]) for offset, length in
                                 zip(_accumulate(length), length)]
                # print(len(spilted_train))
            return spilted_train, self.testset
        else:
            return self.trainset, self.testset


class BuiltinTorchDataset(FLDataset):
    def __init__(self, config):
        super().__init__(config)
        self.Dataset = None

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


class CHMNIST(FLDataset):
    """
    Dataset: https://www.kaggle.com/datasets/gpreda/chinese-mnist
    """

    def download_data(self):
        """
        Resource: https://www.kaggle.com/code/stpeteishii/chinese-mnist-classify-torch-linear
        """
        data_dir = "{}/data/data".format(self.path)
        labels = pd.read_csv("{}/chinese_mnist.csv".format(self.path))
        labels['file'] = labels[['sample_id', 'code']].apply(
            lambda x: 'input_100_'+x['sample_id'].astype(str)+'_'+x['code'].astype(str)+'.jpg', axis=1)

        dataset = []
        for i in range(len(labels)):
            codei = labels.loc[i, 'code']
            filei = labels.loc[i, 'file']
            path = os.path.join(data_dir, filei)
            image = cv2.imread(path)
            image = image.astype(np.float32)
            image2 = torch.from_numpy(image)
            dataset += [[image2, codei-1]]

        random.shuffle(dataset)

        train_size = int(0.8 * len(dataset))
        test_size = len(dataset) - train_size
        train_set, test_set = random_split(dataset, [train_size, test_size])

        # Convert each set to a PyTorch Dataset object
        trainset = CHMNISTDataset(train_set)
        testset = CHMNISTDataset(test_set)

        return trainset, testset


class BreastCancer(FLDataset):
    """
    Dataset: https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data
    """

    def download_data(self):
        """
        Resource: https://www.kaggle.com/code/gamerplayer/cancer-detection-using-pytorch
        """
        data = pd.read_csv("{}/data.csv".format(self.path))

        data.drop(['id', data.columns[-1]], axis=1, inplace=True)
        data['diagnosis'] = data['diagnosis'].astype(
            'category').cat.as_ordered()
        Y = data['diagnosis'].cat.codes
        X = data[data.columns[1:-1]]
        X = (X-X.mean())/X.std()
        X = torch.from_numpy(X.to_numpy())
        X = X.float()
        Y = torch.from_numpy(Y.to_numpy().copy())

        X_train, X_test, Y_train, Y_test = train_test_split(
            X, Y, test_size=0.12)

        trainset = TensorDataset(torch.Tensor(
            np.array(X_train)), torch.Tensor(np.array(Y_train)))
        testset = TensorDataset(torch.Tensor(
            np.array(X_test)), torch.Tensor(np.array(Y_test)))
        return trainset, testset


def get_data(dataset, config):
    if dataset == "MNIST":
        return MNIST(config).load_data(IID=config.data.IID)
    elif dataset == "FashionMNIST":
        return FashionMNIST(config).load_data(IID=config.data.IID)
    elif dataset == "CHMNIST":
        return CHMNIST(config).load_data(IID=config.data.IID)
    elif dataset == "BreastCancer":
        return BreastCancer(config).load_data(IID=config.data.IID)
