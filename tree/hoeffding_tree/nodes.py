import copy
import numpy as np

from .stats import Var
from .tree_utils import BranchFactory

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


class LeafAdaptive:
    """Learning Node for regression tasks that dynamically selects between predictors and
        might behave as a regression tree node or a model tree node, depending on which predictor
        is the best one."""

    def __init__(self, stats, splitter, parent=None):
        self.stats = Var() if stats is None else stats
        self.depth = 0 if parent is None else parent.depth + 1
        self.parent = None

        self.splitter = splitter

        self.splitters = {}
        self.last_split_attempt_at = self.total_weight

        self._leaf_model = LinearRegression() if parent is None else copy.deepcopy(parent._leaf_model)
        self._fmse_mean = 0.0 if parent is None else parent._fmse_mean
        self._fmse_model = 0.0 if parent is None else parent._fmse_model
        self.model_selector_decay = 0.95

    def learn_one(self, x, y, *, sample_weight=1.0, tree=None):
        pred_mean = self.stats.mean.get()
        pred_model = self._leaf_model.predict_one(x)

        self._fmse_mean = self.model_selector_decay * self._fmse_mean + (y - pred_mean) ** 2
        self._fmse_model = self.model_selector_decay * self._fmse_model + (y - pred_model) ** 2

        self.stats.update(y, sample_weight)
        if self.is_active():
            self.update_splitters(x, y, sample_weight)
        self._leaf_model.learn_one(x, y, sample_weight)

    def prediction(self, x):
        if self._fmse_mean < self._fmse_model:  # Act as a regression tree
            return self.stats.mean.get()
        else:  # Act as a model tree
            return self._leaf_model.predict_one(x)

    def update_splitters(self, x, y, sample_weight):
        for i, val in enumerate(x):
            try:
                splitter = self.splitters[i]
            except KeyError:
                splitter = copy.deepcopy(self.splitter)
                self.splitters[i] = splitter
            splitter.update(val, y, sample_weight)

    def best_split_suggestions(self):
        """Find possible split candidates."""
        best_suggestions = [BranchFactory()]
        pre_split_dist = self.stats
        for att_id, splitter in self.splitters.items():
            best_suggestion = splitter.best_evaluated_split_suggestion(pre_split_dist, att_id)
            best_suggestions.append(best_suggestion)
        best_suggestions.sort()
        return best_suggestions

    def manage_memory(self, merit_lower_bound):
        """Trigger Attribute Observers' memory management routines."""
        for splitter in self.splitters.values():
            splitter.remove_bad_splits(
                merit_lower_bound=merit_lower_bound,
                pre_split_dist=self.stats,
            )

    @property
    def total_weight(self):
        """Calculate the total weight seen by the node."""
        return self.stats.mean.n

    def walk(self, x, until_leaf=True):  # noqa
        yield self

    def traverse(self, x):
        return self

    def is_active(self):
        return self.splitters is not None

    def activate(self):
        if not self.is_active():
            self.splitters = {}

    def deactivate(self):
        self.splitters = None


class NumericBinaryBranch:
    def __init__(self, stats, feature, threshold, depth, parent, left, right, **attributes):
        self.stats = stats
        self.feature = feature
        self.threshold = threshold
        self.depth = depth
        self.parent = parent
        self.children = [left, right]
        self.__dict__.update(attributes)

    def walk(self, x, until_leaf=True):
        """Iterate over the nodes of the path induced by x."""
        yield self
        yield from self.next(x).walk(x, until_leaf)

    def traverse(self, x, until_leaf=True):
        """Return the leaf corresponding to the given input."""
        for node in self.walk(x, until_leaf):
            pass
        return node

    @property
    def total_weight(self):
        return sum(child.total_weight for child in filter(None, self.children))

    def next(self, x):
        return self.children[self.branch_no(x)]

    def branch_no(self, x):
        if x[self.feature] <= self.threshold:
            return 0
        return 1

    def repr_branch(self, index: int, shorten=False):
        if shorten:
            if index == 0:
                return f"≤ {round(self.threshold, 4)}"
            return f"> {round(self.threshold, 4)}"
        else:
            if index == 0:
                return f"{self.feature} ≤ {self.threshold}"
            return f"{self.feature} > {self.threshold}"
