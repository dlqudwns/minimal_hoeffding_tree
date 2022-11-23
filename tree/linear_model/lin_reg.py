import numpy as np

class LinearRegression:

    def __init__(self):
        self.lr = 0.01
        self._weights = None
        self._bias = 0.0

    def _raw_dot_one(self, x: dict) -> float:
        return np.dot(self._weights, x) + self._bias

    def _eval_gradient_one(self, x, y, w):
        loss_gradient = float(2 * (self._raw_dot_one(x) - y) * w)
        return (loss_gradient * x, loss_gradient)

    def learn_one(self, x, y, w=1.0):
        gradient, loss_gradient = self._eval_gradient_one(x, y, w)
        self._bias -= self.lr * loss_gradient
        self._weights -= self.lr * gradient
        return self

    def predict_one(self, x):
        if self._weights is None:
            self._weights = np.zeros_like(x)
        return self._raw_dot_one(x)

