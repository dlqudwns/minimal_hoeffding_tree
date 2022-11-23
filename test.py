
"""
from chick_weights import ChickWeights

dataset = ChickWeights()
print(dataset)

from tree.hoeffding_tree_regressor import HoeffdingTreeRegressor
from tree.metrics.r2 import R2

model = HoeffdingTreeRegressor()

metric = R2()

for x, y in dataset:
    y_pred = model.predict_one(x)
    metric = metric.update(y, y_pred)
    model = model.learn_one(x, y)

    print(metric)

"""
from chick_weights import ChickWeights

dataset = ChickWeights()
print(dataset)

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
    w = np.random.rand(100)
    fields = [np.random.rand() for _ in range(400)]
    for i in range(400):
        x = np.random.rand(400)
        if i % 100 == 0:
            w = np.random.rand(400)
        y = np.dot(x, w)
        x = {k:v for k, v in zip(x, fields)}

        y_pred = model.predict_one(x)
        y_pred2 = model2.predict_one(x)

        if y_pred != y_pred2:
            print(model._root)
            print(model2._root)
            import sys
            sys.exit()

        metric = metric.update(y, y_pred2)
        model = model.learn_one(x, y)
        model2 = model2.learn_one(x,y)


        print(i, metric)
    model.draw().render("output.png", format="png")

main()

# WIth basic vectordict - based: 
# node.learnone: 400    7393005.8  18482.5     17.2
# attempt_to_split: 35693821.3 8923455.3     82.8

# Changing to numpy-based