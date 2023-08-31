import numpy as np


class Perceptron(object):
    def __init__(self, eta=0.01, n_iter=50, random_state=42):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, X, y):
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=1 + X.shape[1])
        self.errors = []
        for i in range(self.n_iter):
            errors = 0
            for j, target in zip(X, y):
                update = self.eta * (target - self.predict(j))
                self.w_[1:] += update * j
                self.w_[0] += update
                errors += int(update != 0.0)
                print(f"{update}\n{self.w_[1:]}\n{self.w_[0]}\n{errors}")
            self.errors.append(errors)
            print("-" * 50)
            print(self.errors)
        return self

    def net_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def predict(self, X):
        return np.where(self.net_input(X) >= 0.0, 1, -1)
