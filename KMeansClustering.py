'''
This is a simple script that implements the K-means clustering algorithm.
Reference link: https://dev.to/mxl/dijkstras-algorithm-in-python-algorithms-for-beginners-dkc
Another good reference: https://www.geeksforgeeks.org/printing-paths-dijkstras-shortest-path-algorithm/
=========================
Author  :  Muhan Zhao
Date    :  Aug. 12, 2019
Location:  West Hill, LA, CA
=========================
'''

import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm

class KMeans:
    def __init__(self, data, K):
        '''

        :param data :  Input data, R^(n*m), m data point each with dimension n
        :param K    :  The number of clusters
        '''
        self.n, self.m = data.shape[0], data.shape[1]
        self.data = data
        self.K = K
        self.centroids = np.zeros((self.n, self.K))
        self.labels = np.zeros(self.m)
        self.old_centroids = []
        self.error = 1e-6

    def cluster(self):
        # Randomly initialize K classes centroids
        random_list = list(range(self.m + 1))
        random.shuffle(random_list)
        self.centroids = self.data[:, random_list[:self.K]]
        print('Begin K means clustering...')
        self.iteration_plot2D()
        while 1:
            # copy the old centroids
            self.old_centroids = np.copy(self.centroids)
            # create the cluster labels for all the data points
            for j in range(self.m):
                self.labels[j] = self.assign_cluster(self.data[:, j])
            # assign all the data points to the new cluster
            for k in range(self.K):
                self.centroids[:, k] = np.mean(self.data[:, np.where(self.labels==k)[0]], axis=1)
            if np.sum(np.linalg.norm(self.centroids - self.old_centroids, axis=0)) < 1e-10:
                print('K means clustering finished')
                break
            else:
                self.iteration_plot2D()
        self.final_plot2D()

    def assign_cluster(self, point):
        point = point.reshape(-1, 1)
        distance = np.linalg.norm(point - self.centroids, axis=0)
        return np.argmin(distance)

    def iteration_plot2D(self):
        plt.figure()
        frame = plt.gca()
        plt.scatter(self.data[0, :], self.data[1, :], c='k', marker='o', label='Data')
        plt.scatter(self.centroids[0, :], self.centroids[1, :], c='r', marker='*', label='Centroids')
        frame.axes.get_xaxis().set_visible(False)
        frame.axes.get_yaxis().set_visible(False)
        plt.show()

    def final_plot2D(self):
        plt.figure()
        frame = plt.gca()
        color = iter(cm.rainbow(np.linspace(0, 1, self.K + 1)))
        for k in range(self.K):
            c = next(color)
            plt.scatter(self.data[0, np.where(self.labels == k)[0]], self.data[1, np.where(self.labels == k)[0]], c=c, label='Class %i'%k)
        c = next(color)
        plt.scatter(self.centroids[0, :], self.centroids[1, :], c=c, label='Centroids')
        frame.axes.get_xaxis().set_visible(False)
        frame.axes.get_yaxis().set_visible(False)
        plt.show()


if __name__ == '__main__':
    class_num = 5
    # generate 3 classes data
    x = np.random.multivariate_normal((0.5, 0.9), [[1, 0], [0, 1]], 10)
    y = np.random.multivariate_normal((0.2, 0.3), [[1, 0], [0, 1]], 10)
    z = np.random.multivariate_normal((0.95, 0.2), [[1, 0], [0, 1]], 10)
    u = np.random.multivariate_normal((0.1, 0.9), [[1, 0], [0, 1]], 10)
    v = np.random.multivariate_normal((0.9, 0.9), [[1, 0], [0, 1]], 10)
    d = np.hstack((x.T, y.T, z.T, u.T, v.T))

    # create class
    graph = KMeans(d, class_num)
    graph.cluster()

