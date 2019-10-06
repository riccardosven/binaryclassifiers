"Test binary classifiers on IRIS dataset"

from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
import numpy as np
from binaryclassifiers import Perceptron, LogisticRegression, FisherLinearDiscriminant


def main():
    "Main script for testing classifiers"
    iris = load_iris() # We only consider setosa and virginica
    target = iris.target[iris.target != 2]
    data = iris.data[iris.target != 2]

    idx = np.random.permutation(len(target))

    data = data[idx, 2:]  # petal length and petal width (for good class separation)
    target = target[idx]

    plt.scatter(data[target == 0, 0], data[target == 0, 1])
    plt.scatter(data[target == 1, 0], data[target == 1, 1])

    for classifier in [Perceptron, LogisticRegression, FisherLinearDiscriminant]:
        mdl = classifier()
        mdl.train(data, target)

        boundary_x = np.linspace(min(data[:, 0]), max(data[:, 0]), 100)
        boundary_y = -(mdl.weights[0] * boundary_x + mdl.weights[2])/mdl.weights[1]

        plt.plot(boundary_x, boundary_y)

    plt.show()


if __name__ == "__main__":
    main()
