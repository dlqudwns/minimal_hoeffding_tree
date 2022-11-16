
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

from river.tree.hoeffding_tree_regressor import HoeffdingTreeRegressor
from tree.hoeffding_tree_regressor import HoeffdingTreeRegressor as mytree
from tree.metrics.r2 import R2

model = HoeffdingTreeRegressor(delta=1e-3, grace_period=100)
model2 = mytree(delta=1e-3, grace_period=100)

metric = R2()

for i, (x, y) in enumerate(dataset):
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
