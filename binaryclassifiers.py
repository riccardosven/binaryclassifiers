"Textbook binary classifiers"

import numpy as np


class Perceptron:
    "Textbook perceptron learning"

    def __init__(self):
        self.weights = None

    def train(self, data, targets):
        "Train using the perceptron algorithm"
        nfeatures = len(data[0])
        self.weights = np.random.randn(nfeatures + 1)
        converged = False
        rate = 1
        while not converged:
            converged = True
            for example, target in zip(data, targets):
                example = np.append(example, 1)
                if target == 1 and self.weights.dot(example) < 0:
                    converged = False
                    self.weights += rate*example
                elif target == 0 and self.weights.dot(example) >= 0:
                    converged = False
                    self.weights -= rate*example

    def predict(self, feature):
        "Predict class of new featuren vector"
        feature = np.append(feature, 1)
        return 1 if self.weights.dot(feature) >= 0 else 0


class LogisticRegression:
    "Textbook Logistic Regression"

    def __init__(self, learning_rate=0.2, max_iter=400):
        self.weights = None
        self.learning_rate = learning_rate
        self.max_iter = max_iter

    def _sigmoid(self, x):
        "Sigmoid function"
        return 1/(1 + np.exp(-self.weights.dot(x)))

    def train(self, data, targets):
        "Train using gradient descent optimization"
        nfeatures = len(data[0])
        self.weights = np.random.randn(nfeatures + 1)
        for _ in range(self.max_iter):
            gradient = np.zeros_like(self.weights)
            for example, target in zip(data, targets):
                example = np.append(example, 1)
                if target == 0:
                    gradient -= self._sigmoid(example)*example
                else:
                    gradient += (1 - self._sigmoid(example))*example
            self.weights += self.learning_rate*gradient

    def predict(self, feature):
        "Predict class of new feature vector"
        feature = np.append(feature, 1)
        return 1 if self._sigmoid(feature) > 0.5 else 0


class FisherLinearDiscriminant:
    "Textbook Linear discriminant classifier"

    def __init__(self):
        self.weights = None

    def train(self, data, targets):
        "Train using maximum margin"
        mu0 = np.mean(data[targets == 0, :], axis=0)
        mu1 = np.mean(data[targets == 1, :], axis=0)
        sigma0 = np.cov(data[targets == 0, :].T)
        sigma1 = np.cov(data[targets == 1, :].T)

        self.weights = np.linalg.solve(sigma1 + sigma0, mu1 - mu0)
        self.weights = np.append(
            self.weights, -0.5*self.weights.dot(mu0 + mu1))

    def predict(self, feature):
        "Predict class of new featuren vector"
        feature = np.append(feature, 1)
        return 1 if self.weights.dot(feature) > 0 else 0
