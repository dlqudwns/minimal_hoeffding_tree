

import numpy as np
from river.tree.hoeffding_tree_regressor import HoeffdingTreeRegressor
from tree.hoeffding_tree_regressor import HoeffdingTreeRegressor as mytree
from tree.hoeffding_tree.stats import Var

class R2:

    def __repr__(self):
        """Return the class name along with the current value of the metric."""
        return f"{self.__class__.__name__}: {self.get():,.6f}".rstrip("0")

    def __init__(self):
        self._y_var = Var()
        self._total_sum_of_squares = 0
        self._residual_sum_of_squares = 0

    @property
    def bigger_is_better(self):
        return True

    def update(self, y_true, y_pred, sample_weight=1.0):
        self._y_var.update(y_true, w=sample_weight)
        squared_error = (y_true - y_pred) * (y_true - y_pred) * sample_weight
        self._residual_sum_of_squares += squared_error
        return self

    def get(self):
        if self._y_var.mean.n > 1:
            try:
                total_sum_of_squares = (self._y_var.mean.n - 1) * self._y_var.get()
                return 1 - (self._residual_sum_of_squares / total_sum_of_squares)
            except ZeroDivisionError:
                return 0.0
        return 0.0


def main():
    model = HoeffdingTreeRegressor(delta=1e-1, grace_period=100)
    model2 = mytree(delta=1e-1, grace_period=100)

    metric = R2()
    np.random.seed(0)
    num_data = 1000
    dx = 400
    shift_interval = 100
    fields = list(range(dx))
    for i in range(num_data):
        x_numpy = np.random.rand(dx)
        if i % shift_interval == 0:
            w = np.random.rand(dx)
        y = np.dot(x_numpy, w)
        x = {k:v for k, v in zip(fields, x_numpy)}

        y_pred = model.predict_one(x)
        y_pred2 = model2.predict_one(x_numpy)
        
        if np.abs(y_pred - y_pred2) > 1e-6:
            print("test fails; two model differs from each other")
            print(y_pred, y_pred2)
            model.draw().render("output.png", format="png")
            model2.draw().render("output2.png", format="png")
            import sys
            sys.exit()

        metric = metric.update(y, y_pred2)
        model = model.learn_one(x, y)
        model2 = model2.learn_one(x_numpy, y)

        print(i, metric)
    model.draw().render("output.png", format="png")
    model2.draw().render("output2.png", format="png")

main()