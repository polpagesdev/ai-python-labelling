__authors__ = ['1494769']
__group__ = 'DL.15'

import numpy as np
import utils
import math

class KMeans:

    def __init__(self, X, K, options=None):
        """
         Constructor of KMeans class
             Args:
                 K (int): Number of cluster
                 options (dict): dictionary with options
            """
        self.num_iter = 0
        self.K = K
        self._init_X(X)
        self._init_options(options)
        self.labels = []
        self._init_centroids()
        self.iter = 0

        ################
        #   WORKING    #
        ################

    def _init_X(self, X):
        """Initialization of all pixels, sets X as an array of data in vector form (PxD)
            Args:
                X (list or np.array): list(matrix) of all pixel values
                    if matrix has more than 2 dimensions, the dimensionality of the sample space is the length of
                    the last dimension
        """
        rows = X.shape[0] * X.shape[1]

        if X.dtype != 'float64':
            X = np.array(X, dtype=np.float64)

        self.X = X.reshape(rows, X.shape[2])

        ################
        #   WORKING    #
        ################

    def _init_options(self, options=None):
        """
        Initialization of options in case some fields are left undefined
        Args:
            options (dict): dictionary with options
        """
        if options is None:
            options = {}
        if 'km_init' not in options:
            options['km_init'] = 'first'
        if 'verbose' not in options:
            options['verbose'] = False
        if 'tolerance' not in options:
            options['tolerance'] = np.inf
        if 'max_iter' not in options:
            options['max_iter'] = 10
        if 'fitting' not in options:
            options['fitting'] = 'WCD'

        self.options = options

        ################
        #   WORKING    #
        ################

    def _init_centroids(self):
        """
        Initialization of centroids
        """
        if self.options['km_init'].lower() == 'first':
            centroids = [self.X[0]]
            i = 1
            for pixel in self.X[1:]:
                slc = self.X[:i]
                i += 1
                for pos_slc in range(len(slc)):
                    if np.array_equal(pixel, slc[pos_slc]):
                        break
                    elif pos_slc == (len(slc) - 1) and len(centroids) < self.K:
                        centroids.append(pixel)
                if len(centroids) == self.K:
                    break
            self.centroids = centroids[:]
            self.old_centroids = centroids[:]

        elif self.options['km_init'].lower() == 'random':
            self.centroids = np.random.random((self.K, self.X.shape[1])) * 100
            self.old_centroids = np.random.random((self.K, self.X.shape[1])) * 200

        ################
        #   WORKING    #
        ################

    def get_labels(self):
        """
        Calculates the closest centroid of all points in X and assigns each point to the closest centroid
        """
        centroids = self.centroids[:]
        dist = distance(self.X, centroids)
        labels = []
        for pixel in dist:
            aux = pixel.tolist()
            labels.append(aux.index(min(aux)))

        self.labels = labels

        ################
        #   WORKING    #
        ################

    def get_centroids(self):
        """
        Calculates coordinates of centroids based on the coordinates of all the points assigned to the centroid
        """
        self.old_centroids = self.centroids[:]
        labels = np.array([self.labels])
        new_centroids = np.array([])

        for cnt in range(self.K):
            indexes = np.where(labels == cnt)
            p_sum = self.X[indexes[1]]
            centroid = sum(p_sum)/p_sum.shape[0]
            new_centroids = np.append(new_centroids, centroid, axis=0)
        new_centroids = new_centroids.reshape(self.K, 3)

        self.centroids = new_centroids[:]

        ################
        #   WORKING    #
        ################

    def converges(self):
        """
        Checks if there is a difference between current and old centroids
        """
        return np.allclose(self.centroids, self.old_centroids, rtol=1e-03, atol=1e-05)

        ################
        #   WORKING    #
        ################

    def fit(self):
        """
        Runs K-Means algorithm until it converges or until the number
        of iterations is smaller than the maximum number of iterations.
        """
        self.iter = 0
        while self.iter <= self.options['max_iter'] or self.converges() is False:
            self.get_labels()
            self.get_centroids()
            self.iter += 1

        ################
        #   WORKING    #
        ################

    def whitinClassDistance(self):
        """
        Returns the within class distance of the current clustering
        """
        labels = np.array([self.labels])
        cluster = np.array([])

        for cnt in range(self.K):
            indexes = np.where(labels == cnt)
            pixels = self.X[indexes[1]]
            centroid = self.centroids[cnt]
            for pix in pixels:
                dist = (pix - centroid).dot((pix - centroid).transpose())
                cluster = np.append(cluster, dist)
        wcd = sum(cluster) / len(cluster)

        return wcd

    def find_bestK(self, max_K):
        """
        Sets the best k analysing the results up to 'max_K' clusters
        """
        self.K = 2
        wcd = np.array([])
        pos = 0

        while self.K < max_K:
            self._init_centroids()
            self.fit()
            wcd = np.append(wcd, self.whitinClassDistance())
            if len(wcd) > 1:
                dec_per = 100 * (wcd[pos]/wcd[pos-1])
                if (100 - dec_per) < 17:
                    break
            self.K += 1
            pos += 1
        self.K -= 1

        return self.K


def distance(X, C):
    """
    Calculates the distance between each pixel and each centroid
    Args:
        X (numpy array): PxD 1st set of data points (usually data points)
        C (numpy array): KxD 2nd set of data points (usually cluster centroids points)

    Returns:
        dist: PxK numpy array position ij is the distance between the
        i-th point of the first set an the j-th point of the second set
    """
    dist = np.array([])
    p = X.shape[0]
    if type(C) is np.ndarray:
        k = C.shape[0]
    if type(C) is list:
        k = len(C)

    for i in range(k):
        aux_dist = X - C[i]
        aux_dist = pow(aux_dist, 2)
        aux_dist = np.sum(aux_dist, axis=1).reshape(p, 1)
        aux_dist = np.sqrt(aux_dist)
        if i > 0:
            dist = np.append(dist, aux_dist, axis=1)
        else:
            dist = aux_dist

    return dist

    ################
    #   WORKING    #
    ################


def get_colors(centroids):
    """
    for each row of the numpy matrix 'centroids' returns the color laber folllowing the 11 basic colors as a LIST
    Args:
        centroids (numpy array): KxD 1st set of data points (usually centroind points)

    Returns:
        labels: list of K labels corresponding to one of the 11 basic colors
    """
    color_probs = utils.get_color_prob(centroids)
    labels = np.array([])

    for centroid in color_probs:
        color = utils.colors[centroid.tolist().index(max(centroid))]
        labels = np.append(labels, color)

    return labels

    ################
    #   WORKING    #
    ################
