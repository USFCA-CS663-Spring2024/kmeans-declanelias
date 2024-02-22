import numpy as np
import random


def assign_points(X, centroids):
    cluster_assignments = []
    for point in X:
        distances = []
        for centroid in centroids:
            distance = np.linalg.norm(np.array(point) - np.array(centroid))
            distances.append(distance)

        assignment = np.argmin(distances)
        cluster_assignments.append(assignment)

    return cluster_assignments


def compute_centroids(X, cluster_assignments, num_clusters):
    new_centroids = []
    for k in range(0, num_clusters):
        points = []
        for i in range(0, len(cluster_assignments)):
            if cluster_assignments[i] == k:
                points.append(X[i])

        new_centroid = np.mean(points, axis=0)
        new_centroids.append(new_centroid)

    return new_centroids


def calc_inertia(X, cluster_assignments, centroids):
    inertia = 0
    for i in range(0, len(X)):
        cluster = cluster_assignments[i]
        inertia = inertia + np.sum((X[i] - centroids[cluster]) ** 2)

    return inertia


class My_KMeans:

    def __init__(self, k=5, max_iterations=100):
        self.intertia_ = None
        self.k = k
        self.max_iterations = max_iterations

    def init_centroids(self, X):
        index = np.random.randint(len(X), size=self.k)
        return X[index, :]

    def fit(self, X):
        X = np.array(X)
        iteration = 0

        centroids = self.init_centroids(X)
        previous_centroids = []
        cluster_assignments = []

        while iteration < self.max_iterations:
            cluster_assignments = assign_points(X, centroids)
            centroids = compute_centroids(X, cluster_assignments, self.k)

            if np.array_equal(previous_centroids, centroids):
                break

            previous_centroids = centroids

        self.intertia_ = calc_inertia(X, cluster_assignments, centroids)

        return cluster_assignments, centroids

