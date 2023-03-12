import torch
import logging
from models import models
from utils.config import Config
from clients import Client
import copy
import numpy as np


class Server:
    def __init__(self, config):
        self.config = config
        self.model = self.load_model()
        # print(self.model)
        self.clients = None
        self.client_index = []
        self.target_round = -1

    def run(self):
        self.connect_clients()
        # communication rounds
        for round in (range(1, self.config.fl.rounds + 1)):
            logging.info("-" * 22 + "round {}".format(round) + "-" * 30)
            # select clients which participate training
            selected = self.clients_selection()
            logging.info("selected clients:{}".format(selected))
            info = self.clients.train(selected)

            logging.info(f"aggregate weights ({self.config.fl.rule})")
            # update glob model
            if self.config.fl.rule == 'krum':
                glob_weights = self.krum(info)
            elif self.config.fl.rule == 'trimmed_mean':
                glob_weights = self.trimmed_mean(info)
            elif self.config.fl.rule == 'median':
                glob_weights = self.median(info)
            else:  # default to fed_avg
                glob_weights = self.fed_avg(info)
            self.model.load_state_dict(glob_weights)
            train_acc = self.getacc(info)
            test_acc, test_loss = self.test()
            logging.info(
                "training acc: {:.4f},test acc: {:.4f}, test_loss: {:.4f}\n".format(train_acc, test_acc, test_loss))
            if test_acc > self.config.fl.target_accuracy:
                self.target_round = round
                logging.info("target achieved")
                break

            # broadcast glob weights
            self.clients.update(glob_weights)

    def fed_avg(self, info):
        weights = info["weights"]
        length = info["len"]
        w_avg = copy.deepcopy(weights[0])
        for k in w_avg.keys():
            w_avg[k] *= length[0]
            for i in range(1, len(weights)):
                w_avg[k] += weights[i][k] * length[i]
            w_avg[k] = w_avg[k] / (sum(length))
        return w_avg

    def krum(self, info, k_param=2):
        """
        krum aggregation rule
        k_param: number of farthest nodes to be excluded
        """
        weights = info["weights"]
        num_nodes = len(weights)
        key_params = weights[0].keys()

        distances = np.zeros((num_nodes, num_nodes))
        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                distance = 0
                for k in key_params:
                    distance += torch.norm(
                        torch.cat([weights[i][k].flatten(), weights[j][k].flatten()]), p=2)
                distance = np.sqrt(distance)
                distances[i][j] = distance
                distances[j][i] = distance

        krum_scores = np.zeros(num_nodes)
        for i in range(num_nodes):
            if not weights[i]:
                # Skip empty weight tensors
                continue
            sorted_indices = np.argsort(distances[i])
            krum_distances = np.sum(
                distances[i][sorted_indices[:num_nodes - k_param]])
            krum_scores[i] = krum_distances

        selected_index = np.argmin(krum_scores)
        return weights[selected_index]

    def trimmed_mean(self, info, beta=0.1):
        """
        Trimmed mean aggregation rule
        beta: parameter for trimmed mean
        """
        weights = info["weights"]
        length = info["len"]
        key_params = weights[0].keys()

        num_to_trim = int(np.ceil(len(weights) * beta))

        # Compute the L2-norm of each weight tensor
        norms = [np.linalg.norm(np.concatenate([w[k].flatten()
                                for k in key_params])) for w in weights]

        # Sort the weights based on their L2-norm magnitude
        sorted_indices = np.argsort(norms)
        sorted_weights = [weights[i] for i in sorted_indices]

        # Trim the extreme weights
        trimmed_weights = sorted_weights[num_to_trim:-num_to_trim]

        # Calculate avg of the rest
        w_avg = copy.deepcopy(trimmed_weights[0])
        for k in w_avg.keys():
            w_avg[k] *= length[0]
            for i in range(1, len(trimmed_weights)):
                w_avg[k] += trimmed_weights[i][k] * length[i]
            w_avg[k] = w_avg[k] / (sum(length))
        return w_avg

    def median(self, info):
        return self.fed_avg(info)

    def clients_selection(self):
        # randomly selection
        frac = self.config.clients.fraction
        n_clients = max(1, int(self.config.clients.total * frac))
        training_clients = np.random.choice(
            self.client_index, n_clients, replace=False)
        return training_clients

    def load_model(self):
        model_path = self.config.paths.model
        dataset = self.config.dataset
        logging.info('dataset: {}'.format(dataset))

        # Set up global model
        model = models.get_model(dataset)
        logging.debug(model)
        return model

    def connect_clients(self):
        self.clients = Client(self.config)
        self.client_index = self.clients.clients_to_server()
        self.clients.get_model(self.model)
        self.clients.load_data()

    def test(self):
        return self.clients.test()

    def getacc(self, info):
        corrects = sum(info["corrects"])
        total_samples = sum(info["len"])
        return corrects / total_samples


if __name__ == "__main__":
    config = Config("configs/MNIST/mnist.json")
    server = Server(config)
    server.run()
