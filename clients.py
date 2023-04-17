import copy
import logging
import random
import threading
from threading import Thread

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from utils.data import get_data
from utils.krum import Krum


class Client:
    def __init__(self, config):
        self.config = config
        self.num = self.config.clients.total
        self.compromised = self.config.clients.compromised
        self.compromised_id = [i for i in range(0, self.compromised)]
        self.compromised_round_updates = 0
        self.compromised_weights = []
        # we initialize benign_iteration at 1st round only
        # other consequent rounds use this iteration to
        # compute malicious updates
        self.benign_iteration = [None for _ in range(0, self.compromised)]
        self.benign_mean = {"weights": None, "length": None,
                            "running_correct": None, "epoch_loss": None}
        self.benign_cweights = None
        self.compromised_attack = self.config.clients.compromised_attack
        self.benign_means = None
        self.benign_stds = None
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
        # 2. compute direction deviating from self.model (done)
        # 3. find optimised lambda
        # 4. find w1 (malicious)
        # 5. randomly select w2..wc
        # 6. append updates
        #
        # We do step 1 during first iteration, since the benign iteration is only computed once
        # Step 2-5 is only done by the first selected compromised client in the current iteration (due to data structure)
        # List of crafted w1..wc is stored during the iteration
        # At step 6, compromised client select crafted updates based on their user_id

        if self.benign_mean["weights"] is None and self.compromised_round_updates < self.compromised:
            # first iteration, need to go over all compromised clients
            model, epoch_loss, running_corrects, len_dataset = self.benign_local_train(
                user_id, dataloaders, verbose)
            self.benign_iteration[user_id] = {
                "model": model,
                "epoch_loss": epoch_loss,
                "running_corrects": running_corrects,
                "len_dataset": len_dataset
            }
            weights = model
            self.compromised_round_updates += 1
            return
        elif self.compromised_round_updates >= self.compromised:
            # first compromised client of attack iteration
            self.compromised_round_updates = 0
            if self.benign_mean["weights"] is None:
                self.compute_benign_mean()

            # compute directions
            directions = self.compute_direction_change()

            # get randomized compromised samples
            if (self.compromised_attack == "krum"):
                # optimization
                _, w_1 = self.compute_optimized_lambda(directions=directions)

                self.compromised_weights = self.select_compromised_weights(w_1)
            elif self.compromised_attack == "krum_ext":
                # optimization
                _, w_1 = self.compute_optimized_lambda(directions=directions)

                self.compromised_weights = self.select_compromised_weights(
                    w_1, directions=directions)
            elif self.compromised_attack == "trimmed_mean":
                # self.compromised_weights =
                self.select_compromised_mean_weights(
                    directions=directions
                )
                self.compromised_weights = [
                    self.benign_mean["weights"] for _ in range(self.compromised)]
            else:
                self.compromised_weights = [
                    self.benign_mean["weights"] for _ in range(self.compromised)]

            weights, len_dataset, running_corrects, epoch_loss = self.compromised_weights[user_id], self.benign_mean[
                "length"], self.benign_mean["running_corrects"], self.benign_mean["epoch_loss"]
        else:
            # other client of attack iteration, used compute w1'

            weights, len_dataset, running_corrects, epoch_loss = self.compromised_weights[user_id], self.benign_mean[
                "length"], self.benign_mean["running_corrects"], self.benign_mean["epoch_loss"]

        lock = threading.Lock()
        lock.acquire()
        self.weights.append(weights)
        self.epoch_loss.append(epoch_loss)
        self.running_corrects.append(running_corrects)
        self.len_dataset.append(len_dataset)
        lock.release()

        self.compromised_round_updates += 1

    def select_compromised_mean_weights(self, directions):
        if self.benign_means is None or self.benign_stds is None:
            self.compute_benign_statistics()

    def compute_benign_statistics(self):
        weights = [copy.deepcopy(update["model"])
                   for update in self.benign_iteration]
        means = copy.deepcopy(weights[0])
        stds = copy.deepcopy(weights[0])
        for k in means.keys():
            means[k] = torch.stack([w[k] for w in weights]).mean(dim=0)
            stds[k] = torch.stack([w[k] for w in weights]).std(dim=0)

        self.benign_means = means
        self.benign_stds = stds

    def select_compromised_weights(self, w_1, epsilon=0.01, directions=None):
        krum = Krum()

        weights = [copy.deepcopy(w_1)]

        while len(weights) < self.compromised:
            w_c = copy.deepcopy(w_1)

            # keys = random.sample(
            #     list(w_1.keys()), k=random.randint(1, len(w_1.keys())))
            # for k in keys:
            #     w_c[k] += random.uniform(-epsilon, epsilon)

            k = random.choice(list(w_1.keys()))
            if directions is None:
                w_c[k] += random.uniform(-epsilon, epsilon)
            else:
                # krum_ext
                w_c[k][directions[k] == 1] += random.uniform(-epsilon, 0)
                w_c[k][directions[k] == -1] += random.uniform(0, epsilon)

            d = krum.distance(w_1, w_c)

            if d <= epsilon:
                weights.append(w_c)

        return weights

    def compute_lambda_upperbound(self):
        """
        Formula: lambda <= 1 / ((m - 2c - 1)sqrt(d)) * min<c + 1 <= i <= m>(Sum<?>(D(wl, wi))) + 1 / sqrt(d) * max<c + 1 <= i <= m>(D(wi, wRe))
        Where: 
            d is the number of parameters in global model, D(wl, wi) is the Euclidean distance between wl and wi,
            l <- <?> is the set of m - c - 2 benign local models that have the smallest Euclidean distance to wi.
        """

        # retreive global model weights
        global_weights = copy.deepcopy(self.model.state_dict())

        benign_weights = []
        for d in self.benign_iteration:
            benign_weights.append(d["model"])

        # number of worker devices
        m = self.num
        # number of compromised
        c = self.compromised
        # number of parameters
        d = len(global_weights.keys())

        krum = Krum()

        min_score = min(krum.scores(
            self.benign_cweights, num_nodes=self.compromised))

        max_score = max([krum.distance(
            benign_weights[i], global_weights) for i in range(len(benign_weights))])

        # logging.info(
        #     f"Total: {m}, Compromised: {c}, Parameters: {d}, "
        #     + f"Min distance score: {min_score}"
        #     + f"Max distance score: {max_score}")

        upperbound = 1 / ((m - 2 * c - 1) * np.sqrt(d)) * \
            min_score + 1 / np.sqrt(d) * max_score

        # logging.info(f"Lambda upperbound: {upperbound}")

        return upperbound, benign_weights

    def compute_optimized_lambda(self, directions):
        """
        Formula: 
            max <lambda> lambda
            subject to w1' = Krum(w1', w1, ..., wc),
                       w1' = wRe - lambda * s
        """
        upperbound, benign_weights = self.compute_lambda_upperbound()
        threshold = 0.00001
        lamda = upperbound

        # retreive global model weights
        global_weights = copy.deepcopy(self.model.state_dict())

        # logging.info(
        #     "[Lambda optimization] intialization: "
        #     + f"upperbound={upperbound} "
        #     + f"threshold={threshold}")

        krum = Krum()

        def compute_w_1(ld):
            """
            w_1 = global_weights - lamda * directions
            """
            res = copy.deepcopy(global_weights)
            for k in global_weights.keys():
                res[k] = global_weights[k] - lamda * directions[k]
            return res

        added_w_1 = 1
        found_lambda = False

        while not found_lambda:
            while lamda >= threshold:
                w_1 = compute_w_1(lamda)

                weights = [w_1 for _ in range(added_w_1)] + benign_weights
                selected_id = krum.aggregate(weights)

                # logging.info(
                #     f"[Lambda optimization] selected id = {selected_id}, lamda = {lamda}")

                if selected_id == 0:
                    found_lambda = True
                    break

                lamda = lamda / 2
            lamda = upperbound
            added_w_1 = added_w_1 + 1

        # logging.info(f"[Lambda optimization] result: lambda = {lamda}")

        w_1 = compute_w_1(lamda)

        return lamda, w_1

    def compute_direction_change(self):
        glob_weights = copy.deepcopy(self.model.state_dict())
        mean_weights = self.benign_mean["weights"]
        s_directions = copy.deepcopy(mean_weights)
        for k in s_directions.keys():
            s_directions[k] = (mean_weights[k] - glob_weights[k]) / \
                abs(mean_weights[k] - glob_weights[k])
            s_directions[k][torch.isnan(s_directions[k])] = -1

        return s_directions

    def compute_benign_mean(self):
        weights = []
        length = []
        running_corrects = []
        epoch_loss = []
        for d in self.benign_iteration:
            weights.append(d["model"])
            length.append(d["len_dataset"])
            running_corrects.append(d["running_corrects"])
            epoch_loss.append(d["epoch_loss"])
        w_avg = copy.deepcopy(weights[0])
        for k in w_avg.keys():
            w_avg[k] *= length[0]
            for i in range(1, len(weights)):
                w_avg[k] += weights[i][k] * length[i]
            w_avg[k] = w_avg[k] / (sum(length))
        self.benign_mean["weights"] = w_avg
        self.benign_mean["length"] = int(np.mean(length))
        self.benign_mean["running_corrects"] = int(np.mean(running_corrects))
        self.benign_mean["epoch_loss"] = np.mean(epoch_loss)

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

        if self.benign_cweights is None:
            self.benign_cweights = copy.deepcopy(self.weights)

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
