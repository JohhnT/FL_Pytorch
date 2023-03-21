from torch import nn, optim
import torch
import copy
import logging
from threading import Thread
from utils.data import get_data
from torch.utils.data import DataLoader
import threading


class Client:
    def __init__(self, config):
        self.config = config
        self.num = self.config.clients.total
        self.compromised = self.config.clients.compromised
        self.compromised_id = [i for i in range(0, self.compromised)]
        # we initialize benign_iteration at 1st round only
        # other consequent rounds use this iteration to
        # compute malicious updates
        self.benign_iteration = [None for _ in range(0, self.compromised)]
        self.benign_mean = None
        self.compromised_attack = self.config.clients.compromised_attack
        self.client_id = [i for i in range(self.compromised, self.num)]
        self.model = None
        self.dataloaders = []
        self.weights = []
        self.epoch_loss = []
        self.running_corrects = []
        self.len_dataset = []

    def load_data(self):
        self.trainset, self.testset = get_data(
            self.config.dataset, self.config)
        for subset in self.trainset:
            loader = DataLoader(subset, batch_size=self.config.fl.batch_size)
            self.dataloaders.append(loader)

    def clients_to_server(self):
        return self.compromised_id + self.client_id

    def get_model(self, model):
        self.model = model

    def local_train(self, user_id, dataloaders, verbose=1):
        if user_id in self.compromised_id:
            self.malicious_local_train(user_id, dataloaders, verbose)
        else:
            self.benign_local_train(user_id, dataloaders, verbose)

    def malicious_local_train(self, user_id, dataloaders, verbose=1):
        """
        Compromised client's local train function
        """

        # key steps
        # 1. compute mean of compromised clients' benign updates (done)
        # 2. compute direction deviating from self.model
        # 3. find optimised lambda
        # 4. find w1 (malicious)
        # 5. randomly select w2..wc
        # 6. append updates
        #
        # We do step 1 during first iteration, since the benign iteration is only computed once
        # Step 2-5 is only done by the first selected compromised client in the current iteration (due to data structure)
        # List of crafted w1..wc is stored during the iteration
        # At step 6, compromised client select crafted updates based on their user_id

        if self.benign_iteration[user_id]:
            benign_iter = self.benign_iteration[user_id]
            model, epoch_loss, running_corrects, len_dataset = benign_iter["model"], benign_iter[
                "epoch_loss"], benign_iter["running_corrects"], benign_iter["len_dataset"]

            if self.benign_mean is None:
                self.compute_benign_mean()

            #
            lock = threading.Lock()
            lock.acquire()
            self.weights.append(self.benign_mean)
            self.epoch_loss.append(epoch_loss)
            self.running_corrects.append(running_corrects)
            self.len_dataset.append(len_dataset)
            lock.release()
        else:
            model, epoch_loss, running_corrects, len_dataset = self.benign_local_train(
                user_id, dataloaders, verbose)
            self.benign_iteration[user_id] = {
                "model": model,
                "epoch_loss": epoch_loss,
                "running_corrects": running_corrects,
                "len_dataset": len_dataset
            }

    def compute_benign_mean(self):
        weights = []
        length = []
        for d in self.benign_iteration:
            weights.append(d["model"])
            length.append(d["len_dataset"])
        w_avg = copy.deepcopy(weights[0])
        for k in w_avg.keys():
            w_avg[k] *= length[0]
            for i in range(1, len(weights)):
                w_avg[k] += weights[i][k] * length[i]
            w_avg[k] = w_avg[k] / (sum(length))
        self.benign_mean = w_avg

    def benign_local_train(self, user_id, dataloaders, verbose=1):
        """
        Benign client's local train function
        """
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = copy.deepcopy(self.model)
        model.to(device)
        model.train()
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(params=model.parameters(), lr=self.config.fl.lr)

        for e in range(self.config.fl.epochs):
            running_loss = 0
            running_corrects = 0
            epoch_loss = 0
            epoch_acc = 0

            for inputs, labels in dataloaders:
                inputs = inputs.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                # loss = criterion(outputs, labels)
                loss = criterion(outputs, labels.type(torch.LongTensor))
                _, preds = torch.max(outputs, 1)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders.dataset)
            epoch_acc = int(running_corrects) / len(dataloaders.dataset)

            logging.debug('User {}: {} Loss: {:.4f} Acc: {:.4f}'.format(
                user_id, "training", epoch_loss, epoch_acc))
        # need be locked
        lock = threading.Lock()
        lock.acquire()
        self.weights.append(copy.deepcopy(model.state_dict()))
        self.epoch_loss.append(epoch_loss)
        self.running_corrects.append(int(running_corrects))
        self.len_dataset.append(len(dataloaders.dataset))
        lock.release()

        return copy.deepcopy(model.state_dict()), epoch_loss, int(running_corrects), len(dataloaders.dataset)

    def upload(self, info):
        return info

    def update(self, glob_weights):
        self.model.load_state_dict(glob_weights)

    def train(self, selected_client):
        self.weights = []
        self.epoch_loss = []
        self.running_corrects = []

        self.len_dataset = []

        # multithreading
        threads = [Thread(target=self.local_train(user_id=client, dataloaders=self.dataloaders[client])) for client in
                   selected_client]
        [t.start() for t in threads]
        [t.join() for t in threads]
        # training details
        info = {"weights": self.weights, "loss": self.epoch_loss, "corrects": self.running_corrects,
                'len': self.len_dataset}
        return self.upload(info)

    def test(self):
        corrects = 0
        test_loss = 0
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = copy.deepcopy(self.model)
        model.to(device)
        model.eval()
        criterion = nn.CrossEntropyLoss()

        dataloader = DataLoader(self.testset, batch_size=32, shuffle=True)
        for batch_id, (inputs, labels) in enumerate(dataloader):
            loss = 0
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            # loss = criterion(outputs, labels)
            loss = criterion(outputs, labels.type(torch.LongTensor))
            _, preds = torch.max(outputs, 1)
            test_loss += loss.item() * inputs.size(0)
            corrects += torch.sum(preds == labels.data)

        acc = int(corrects) / len(dataloader.dataset)
        avg_loss = test_loss / len(dataloader.dataset)
        # print(corrects)
        # print(len(dataloader.dataset))
        # print(f"test_acc:{acc}",)
        return acc, avg_loss


if __name__ == "__main__":
    c = Client(100)
