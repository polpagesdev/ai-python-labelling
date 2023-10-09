__authors__ = '1494769'
__group__ = 'DL.15'

import numpy as np
import math
import operator
from scipy.spatial.distance import cdist

class KNN:
    def __init__(self, train_data, labels):

        self._init_train(train_data)
        self.labels = np.array(labels)

    def _init_train(self, train_data):
        """
        initializes the train data
        :param train_data: PxMxNx3 matrix corresponding to P color images
        :return: assigns the train set to the matrix self.train_data shaped as PxD (P points in a D dimensional space)
        """
        if train_data.dtype != 'float64':
            train_data = np.array(train_data, dtype=np.float64)
        self.train_data = train_data.reshape(train_data.shape[0], 14400)

        ################
        #   WORKING    #
        ################

    def get_k_neighbours(self, test_data, k):
        """
        given a test_data matrix calculates de k nearest neighbours at each point (row) of test_data on self.neighbors
        :param test_data:   array that has to be shaped to a NxD matrix ( N points in a D dimensional space)
        :param k:  the number of neighbors to look at
        :return: the matrix self.neighbors is created (NxK)
                 the ij-th entry is the j-th nearest train point to the i-th test point
        """
        if test_data.dtype != 'float64':
            test_data = np.array(test_data, dtype=np.float64)
        test_data = test_data.reshape(test_data.shape[0], 14400)

        neighbors = np.array([])
        dist = cdist(test_data, self.train_data)
        for entry in dist:
            idx = np.argsort(entry)[:k]
            idx = self.labels[idx]
            neighbors = np.append(neighbors, idx)

        self.neighbors = neighbors.reshape(dist.shape[0], k)

        ################
        #   WORKING    #
        ################

    def get_class(self):
        """
        Get the class by maximum voting
        :return: Numpy array of Nx1 elements.
                1st array For each of the rows in self.neighbors gets the most voted value
                            (i.e. the class at which that row belongs)
        """
        voted_values = np.array([])
        ones = np.ones(self.neighbors.shape[1], dtype=int)
        twos = np.ones(int(self.neighbors.shape[1]/2), dtype=int) * 2

        for row in self.neighbors:
            reps = np.unique(row, return_counts=True)
            if reps[1].tolist() == ones.tolist():
                voted_values = np.append(voted_values, row[0])
            elif reps[1].tolist() == twos.tolist():
                voted_values = np.append(voted_values, row[0])
            elif np.count_nonzero(reps[1] == 2) > 1 or np.count_nonzero(reps[1] == 3) > 1:
                pos = 0
                for elem in row:
                    pos += 1
                    if elem in row[:(pos-1)] or elem in row[pos:]:
                        voted_values = np.append(voted_values, elem)
                        break
            else:
                voted_values = np.append(voted_values, max(set(row.tolist()), key=row.tolist().count))

        return voted_values

        ################
        #   WORKING    #
        ################

    def predict(self, test_data, k):
        """
        predicts the class at which each element in test_data belongs to
        :param test_data: array that has to be shaped to a NxD matrix (N points in a D dimensional space)
        :param k: the number of neighbors to look at
        :return: the output from get_class, Nx1 vector
        """
        self.get_k_neighbours(test_data, k)
        return self.get_class()

        ################
        #   WORKING    #
        ################
