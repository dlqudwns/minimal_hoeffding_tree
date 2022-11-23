

import numpy as np
from river.tree.hoeffding_tree_regressor import HoeffdingTreeRegressor
from tree.hoeffding_tree_regressor import HoeffdingTreeRegressor as mytree
from tree.metrics.r2 import R2


#for i, (x, y) in enumerate(dataset):

def main():
    model = HoeffdingTreeRegressor(delta=1e-3, grace_period=100)
    model2 = mytree(delta=1e-3, grace_period=100)

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

main()

# WIth basic vectordict - based: 
# node.learnone: 400    2181414.9   5453.5     74.2
# attempt_to_split: 753271.0 376635.5     25.6 

# Changing to numpy-based - similar performance
# node.learnone: 400    2029851.1   5074.6     73.2
# attempt_to_split: 737562.2 368781.1     26.6 

