import torch
import numpy as np


class Krum:
    def __init__(self):
        pass

    def aggregate(self, weights, num_nodes=None, k_param=2):

        krum_scores = self.scores(
            weights, num_nodes=num_nodes, k_param=k_param)

        selected_index = np.argmin(krum_scores)
        return selected_index

    def scores(self, weights, num_nodes=None, k_param=2):
        if num_nodes is None:
            num_nodes = len(weights)

        distances = np.zeros((num_nodes, num_nodes))
        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                distance = self.distance(weights[i], weights[j])
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
        return krum_scores

    def distance(self, w1, w2):
        key_params = w1.keys()
        distance = 0
        for k in key_params:
            diff = w1[k].flatten() - w2[k].flatten()
            distance += torch.dot(diff, diff)
        distance = np.sqrt(distance.item())
        return distance
