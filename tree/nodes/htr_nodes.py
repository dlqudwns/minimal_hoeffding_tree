import copy

from ..stats.var import Var
from ..tree_utils import BranchFactory

class LeafAdaptive:
    """Learning Node for regression tasks that dynamically selects between predictors and
        might behave as a regression tree node or a model tree node, depending on which predictor
        is the best one.

    Parameters
    ----------
    stats
        In regression tasks the node keeps an instance of `river.stats.Var` to estimate
        the target's statistics.
    depth
        The depth of the node.
    splitter
        The numeric attribute observer algorithm used to monitor target statistics
        and perform split attempts.
    leaf_model
        A `river.base.Regressor` instance used to learn from instances and provide
        responses.
    kwargs
        Other parameters passed to the learning node.
    """

    def __init__(self, stats, depth, splitter, leaf_model):
        self.stats = Var() if stats is None else stats
        self.depth = depth

        self.splitter = splitter

        self.splitters = {}
        self._disabled_attrs = set()
        self.last_split_attempt_at = self.total_weight

        self._leaf_model = leaf_model
        self._fmse_mean = 0.0
        self._fmse_model = 0.0
        self.model_selector_decay = 0.95

    def is_active(self):
        return self.splitters is not None

    def activate(self):
        if not self.is_active():
            self.splitters = {}

    def deactivate(self):
        self.splitters = None

    def learn_one(self, x, y, *, sample_weight=1.0, tree=None):
        pred_mean = self.stats.mean.get()
        pred_model = self._leaf_model.predict_one(x)

        self._fmse_mean = self.model_selector_decay * self._fmse_mean + (y - pred_mean) ** 2
        self._fmse_model = self.model_selector_decay * self._fmse_model + (y - pred_model) ** 2

        self.stats.update(y, sample_weight)
        if self.is_active():
            self.update_splitters(x, y, sample_weight)
        self._leaf_model.learn_one(x, y, sample_weight)

    def prediction(self, x, *, tree=None):
        if self._fmse_mean < self._fmse_model:  # Act as a regression tree
            return self.stats.mean.get()
        else:  # Act as a model tree
            return self._leaf_model.predict_one(x)

    def update_splitters(self, x, y, sample_weight):
        for i, val in enumerate(x):
            if i in self._disabled_attrs:
                continue

            try:
                splitter = self.splitters[i]
            except KeyError:
                splitter = copy.deepcopy(self.splitter)
                self.splitters[i] = splitter
            splitter.update(val, y, sample_weight)

    def best_split_suggestions(self, tree):
        """Find possible split candidates."""
        best_suggestions = []
        pre_split_dist = self.stats
        if tree.merit_preprune:
            # Add null split as an option
            null_split = BranchFactory()
            best_suggestions.append(null_split)
        for att_id, splitter in self.splitters.items():
            best_suggestion = splitter.best_evaluated_split_suggestion(pre_split_dist, att_id)
            best_suggestions.append(best_suggestion)

        return best_suggestions

    def disable_attribute(self, att_id):
        """Disable an attribute observer."""
        if att_id in self.splitters:
            del self.splitters[att_id]
            self._disabled_attrs.add(att_id)

    @property
    def total_weight(self):
        """Calculate the total weight seen by the node."""
        return self.stats.mean.n

    def calculate_promise(self) -> int:
        """Uses the node's depth as a heuristic to estimate how likely the leaf is going to become a decision node."""
        return -self.depth

    def walk(self, x, until_leaf=True):  # noqa
        yield self

    def manage_memory(self, criterion, last_check_ratio, last_check_vr, last_check_e):
        """Trigger Attribute Observers' memory management routines.

        Currently, only `EBSTSplitter` and `TEBSTSplitter` have support to this feature.

        Parameters
        ----------
        criterion
            Split criterion
        last_check_ratio
            The ratio between the second best candidate's merit and the merit of the best
            split candidate.
        last_check_vr
            The best candidate's split merit.
        last_check_e
            Hoeffding bound value calculated in the last split attempt.
        """
        for splitter in self.splitters.values():
            splitter.remove_bad_splits(
                criterion=criterion,
                last_check_ratio=last_check_ratio,
                last_check_vr=last_check_vr,
                last_check_e=last_check_e,
                pre_split_dist=self.stats,
            )