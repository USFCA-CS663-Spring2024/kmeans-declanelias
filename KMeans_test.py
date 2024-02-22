import unittest

from cluster import *
import numpy as np


class MyTestCase(unittest.TestCase):

    def test_2Dassignment(self):
        X = [[1, 1],
             [1.1, 1.1],
             [2, 2],
             [25, 25],
             [26, 26],
             [27, 27]]

        centroids = [[1.5, 1.5],
                     [25, 25]]

        expected = [0, 0, 0, 1, 1, 1]

        kmeans = My_KMeans()
        self.assertEqual(expected, assign_points(X, centroids))

    def test_3Dassignment(self):
        X = [[1, 1, 1],
             [1.1, 1.1, 1.1],
             [2, 2, 2],
             [25, 25, 25],
             [26, 26, 26],
             [27, 27, 27]]

        centroids = [[1.5, 1.5, 1.5],
                     [25, 25, 25]]

        expected = [0, 0, 0, 1, 1, 1]

        kmeans = My_KMeans()
        self.assertEqual(expected, assign_points(X, centroids))

    def test_cluster_update(self):
        X = [[1, 1, 1],
             [1.1, 1.1, 1.1],
             [2, 2, 2],
             [25, 25, 25],
             [26, 26, 26],
             [27, 27, 27]]

        cluster_assignments = [0, 0, 0, 1, 1, 1]
        expected = [[(1 + 1.1 + 2) / 3, (1 + 1.1 + 2) / 3, (1 + 1.1 + 2) / 3],
                    [(25 + 26 + 27) / 3, (25 + 26 + 27) / 3, (25 + 26 + 27) / 3]]

        kmeans = My_KMeans(k=2)
        clusters = compute_centroids(X, cluster_assignments, 2)
        print(clusters)
        print(expected)

        self.assertEqual(expected, clusters)


    def test_clustering(self):
        X = [[0, 0], [2, 2], [0, 2], [2, 0], [10, 10], [8, 8], [10, 8], [8, 10]]

        kmeans = My_KMeans(k=2)
        (assignment, centroids) = kmeans.fit(X)

        centroids = np.sort(centroids)
        expected_centroids = [[1, 1], [9, 9]]
        self.assertEqual(expected_centroids, centroids)

    def test_inertia(self):
        X = [[0, 0], [2, 2], [0, 2], [2, 0], [10, 10], [8, 8], [10, 8], [8, 10]]
        kmeans = My_KMeans(k=2)
        kmeans.fit(X)
        self.assertEqual(16, kmeans.intertia_)





if __name__ == '__main__':
    unittest.main()
