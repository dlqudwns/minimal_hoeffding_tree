import math
from copy import deepcopy

from .linear_model.lin_reg import LinearRegression

from .nodes.branch import NumericBinaryBranch
from .nodes.htr_nodes import LeafAdaptive
from .split_criterion.variance_reduction_split_criterion import VarianceReductionSplitCriterion
from .splitter.tebst_splitter import TEBSTSplitter

import functools
import collections
try:
    import graphviz

    GRAPHVIZ_INSTALLED = True
except ImportError:
    GRAPHVIZ_INSTALLED = False


class HoeffdingTreeRegressor:
    """Hoeffding Tree regressor.

    Parameters
    ----------
    grace_period
        Number of instances a leaf should observe between split attempts.
    max_depth
        The maximum depth a tree can reach. If `None`, the tree will grow indefinitely.
    delta
        Significance level to calculate the Hoeffding bound. The significance level is given by
        `1 - delta`. Values closer to zero imply longer split decision delays.
    tau
        Threshold below which a split will be forced to break ties.
    leaf_model
        The regression model used to provide responses if `leaf_prediction='model'`. If not
        provided an instance of `river.linear_model.LinearRegression` with the default
        hyperparameters is used.
    model_selector_decay
        The exponential decaying factor applied to the learning models' squared errors, that
        are monitored if `leaf_prediction='adaptive'`. Must be between `0` and `1`. The closer
        to `1`, the more importance is going to be given to past observations. On the other hand,
        if its value approaches `0`, the recent observed errors are going to have more influence
        on the final decision.
    splitter
        The Splitter or Attribute Observer (AO) used to monitor the class statistics of numeric
        features and perform splits. Splitters are available in the `tree.splitter` module.
        Different splitters are available for classification and regression tasks. Classification
        and regression splitters can be distinguished by their property `is_target_class`.
        This is an advanced option. Special care must be taken when choosing different splitters.
        By default, `tree.splitter.TEBSTSplitter` is used if `splitter` is `None`.
    min_samples_split
        The minimum number of samples every branch resulting from a split candidate must have
        to be considered valid.
    binary_split
        If True, only allow binary splits.
    remove_poor_attrs
        If True, disable poor attributes to reduce memory usage.
    merit_preprune
        If True, enable merit-based tree pre-pruning.

    Notes
    -----
    HTR uses the Hoeffding bound to control its split decisions.
    HTR relies on calculating the reduction of variance in the target space to decide among the
    split candidates. The smallest the variance at its leaf nodes, the more homogeneous the
    partitions are. At its leaf nodes, HTR fits either linear models or uses the target
    average as the predictor.

    """

    def __init__(
        self,
        grace_period: int = 200,
        max_depth: int = None,
        delta: float = 1e-7,
        tau: float = 0.05,
        leaf_model = None,
        min_samples_split: int = 5
    ):
        self.max_depth: float = max_depth if max_depth is not None else math.inf
        self._root = None
        self.n_active_leaves: int = 0
        self.n_inactive_leaves: int = 0
        self._train_weight_seen_by_model: float = 0.0
        self.merit_preprune = True
        self.binary_split = False
        self.remove_poor_attrs = False

        self.grace_period = grace_period
        self.delta = delta
        self.tau = tau
        self.leaf_model = leaf_model if leaf_model else LinearRegression()
        self.split_criterion = VarianceReductionSplitCriterion(min_samples_split)

        self.splitter = TEBSTSplitter(self.split_criterion)

    @staticmethod
    def _hoeffding_bound(range_val, confidence, n):
        return math.sqrt((range_val * range_val * math.log(1.0 / confidence)) / (2.0 * n))

    def _new_leaf(self, initial_stats=None, parent=None):
        """Create a new learning node."""
        depth = 0 if parent is None else parent.depth + 1
        leaf_model = deepcopy(self.leaf_model) if parent is None else deepcopy(parent._leaf_model)

        new_adaptive = LeafAdaptive(initial_stats, depth, self.splitter, leaf_model)
        if parent is not None and isinstance(parent, LeafAdaptive):
            new_adaptive._fmse_mean = parent._fmse_mean  # noqa
            new_adaptive._fmse_model = parent._fmse_model  # noqa

        return new_adaptive

    def learn_one(self, x, y, *, sample_weight=1.0):
        """Train the tree model on sample x and corresponding target y.

        Parameters
        ----------
        x
            Instance attributes.
        y
            Target value for sample x.
        sample_weight
            The weight of the sample.

        Returns
        -------
        self
        """

        self._train_weight_seen_by_model += sample_weight

        if self._root is None:
            self._root = self._new_leaf()
            self._n_active_leaves = 1

        p_node = None
        node = None
        if isinstance(self._root, NumericBinaryBranch):
            path = iter(self._root.walk(x, until_leaf=False))
            while True:
                aux = next(path, None)
                if aux is None:
                    break
                p_node = node
                node = aux
        else:
            node = self._root

        if isinstance(node, LeafAdaptive):
            node.learn_one(x, y, sample_weight=sample_weight, tree=self)
            if node.is_active():
                if node.depth >= self.max_depth:  # Max depth reached
                    node.deactivate()
                    self._n_active_leaves -= 1
                    self._n_inactive_leaves += 1
                else:
                    weight_seen = node.total_weight
                    weight_diff = weight_seen - node.last_split_attempt_at
                    if weight_diff >= self.grace_period:
                        p_branch = p_node.branch_no(x) if isinstance(p_node, NumericBinaryBranch) else None
                        self._attempt_to_split(node, p_node, p_branch)
                        node.last_split_attempt_at = weight_seen
        else:
            while True:
                # Split node encountered a previously unseen categorical value (in a multi-way
                #  test), so there is no branch to sort the instance to
                if node.max_branches() == -1 and node.feature in x:
                    # Create a new branch to the new categorical value
                    leaf = self._new_leaf(parent=node)
                    node.add_child(x[node.feature], leaf)
                    self._n_active_leaves += 1
                    node = leaf
                # The split feature is missing in the instance. Hence, we pass the new example
                # to the most traversed path in the current subtree
                else:
                    _, node = node.most_common_path()
                    # And we keep trying to reach a leaf
                    if isinstance(node, NumericBinaryBranch):
                        node = node.traverse(x, until_leaf=False)
                # Once a leaf is reached, the traversal can stop
                if isinstance(node, LeafAdaptive):
                    break
            # Learn from the sample
            node.learn_one(x, y, sample_weight=sample_weight, tree=self)

        return self

    def predict_one(self, x):
        """Predict the target value using one of the leaf prediction strategies.

        Parameters
        ----------
        x
            Instance for which we want to predict the target.

        Returns
        -------
        Predicted target value.

        """
        pred = 0.0
        if self._root is not None:
            leaf = self._root.traverse(x, until_leaf=True) if isinstance(self._root, NumericBinaryBranch) else self._root
            pred = leaf.prediction(x, tree=self)
        return pred

    def _attempt_to_split(self, leaf, parent, parent_branch: int, **kwargs):
        """Attempt to split a node.

        If the target's variance is high at the leaf node, then:

        1. Find split candidates and select the top 2.
        2. Compute the Hoeffding bound.
        3. If the ratio between the merit of the second best split candidate and the merit of the
        best one is smaller than 1 minus the Hoeffding bound (or a tie breaking decision
        takes place), then:
           3.1 Replace the leaf node by a split node.
           3.2 Add a new leaf node on each branch of the new split node.
           3.3 Update tree's metrics

        Optional: Disable poor attribute. Depends on the tree's configuration.

        Parameters
        ----------
        leaf
            The node to evaluate.
        parent
            The node's parent in the tree.
        parent_branch
            Parent node's branch index.
        kwargs
            Other parameters passed to the new branch.

        """
        best_split_suggestions = leaf.best_split_suggestions(self)
        best_split_suggestions.sort()
        should_split = False
        if len(best_split_suggestions) < 2:
            should_split = len(best_split_suggestions) > 0
        else:
            hoeffding_bound = self._hoeffding_bound(
                self.split_criterion.range_of_merit(leaf.stats),
                self.delta,
                leaf.total_weight,
            )
            best_suggestion = best_split_suggestions[-1]
            second_best_suggestion = best_split_suggestions[-2]
            if best_suggestion.merit > 0.0 and (
                second_best_suggestion.merit / best_suggestion.merit < 1 - hoeffding_bound
                or hoeffding_bound < self.tau
            ):
                should_split = True
            if self.remove_poor_attrs:
                poor_attrs = set()
                best_ratio = second_best_suggestion.merit / best_suggestion.merit

                # Add any poor attribute to set
                for suggestion in best_split_suggestions:
                    if (
                        suggestion.feature
                        and suggestion.merit / best_suggestion.merit
                        < best_ratio - 2 * hoeffding_bound
                    ):
                        poor_attrs.add(suggestion.feature)
                for poor_att in poor_attrs:
                    leaf.disable_attribute(poor_att)
        if should_split:
            split_decision = best_split_suggestions[-1]
            if split_decision.feature is None:
                # Pre-pruning - null wins
                leaf.deactivate()
                self._n_inactive_leaves += 1
                self._n_active_leaves -= 1
            else:
                branch = NumericBinaryBranch
                leaves = tuple(
                    self._new_leaf(initial_stats, parent=leaf)
                    for initial_stats in split_decision.children_stats  # type: ignore
                )

                new_split = split_decision.assemble(
                    branch, leaf.stats, leaf.depth, *leaves, **kwargs
                )

                self._n_active_leaves -= 1
                self._n_active_leaves += len(leaves)
                if parent is None:
                    self._root = new_split
                else:
                    parent.children[parent_branch] = new_split
        elif (
            len(best_split_suggestions) >= 2
            and best_split_suggestions[-1].merit > 0
            and best_split_suggestions[-2].merit > 0
        ):
            last_check_ratio = best_split_suggestions[-2].merit / best_split_suggestions[-1].merit
            last_check_vr = best_split_suggestions[-1].merit

            leaf.manage_memory(  # type: ignore
                self.split_criterion, last_check_ratio, last_check_vr, hoeffding_bound
            )


    def draw(self, max_depth: int = None):
        """Draw the tree using the `graphviz` library.

        Since the tree is drawn without passing incoming samples, classification trees
        will show the majority class in their leaves, whereas regression trees will
        use the target mean.

        Parameters
        ----------
        max_depth
            Only the root will be drawn when set to `0`. Every node will be drawn when
            set to `None`.

        Notes
        -----
        Currently, Label Combination Hoeffding Tree Classifier (for multi-label
        classification) is not supported.

        Examples
        --------
        >>> from river import datasets
        >>> from river import tree
        >>> model = tree.HoeffdingTreeClassifier(
        ...    grace_period=5,
        ...    delta=1e-5,
        ...    split_criterion='gini',
        ...    max_depth=10,
        ...    tau=0.05,
        ... )
        >>> for x, y in datasets.Phishing():
        ...    model = model.learn_one(x, y)
        >>> dot = model.draw()

        .. image:: ../../docs/img/dtree_draw.svg
            :align: center
        """
        counter = 0

        def iterate(node=None):
            if node is None:
                yield None, None, self._root, 0, None
                yield from iterate(self._root)

            nonlocal counter
            parent_no = counter

            if isinstance(node, NumericBinaryBranch):
                for branch_index, child in enumerate(node.children):
                    counter += 1
                    yield parent_no, node, child, counter, branch_index
                    if isinstance(child, NumericBinaryBranch):
                        yield from iterate(child)

        if max_depth is None:
            max_depth = -1

        dot = graphviz.Digraph(
            graph_attr={"splines": "ortho", "forcelabels": "true", "overlap": "false"},
            node_attr={
                "shape": "box",
                "penwidth": "1.2",
                "fontname": "trebuchet",
                "fontsize": "11",
                "margin": "0.1,0.0",
            },
            edge_attr={"penwidth": "0.6", "center": "true", "fontsize": "7  "},
        )

        n_colors = 1

        # Pick a color palette which maps classes to colors
        new_color = functools.partial(next, iter(_color_brew(n_colors)))
        palette = collections.defaultdict(new_color)

        for parent_no, parent, child, child_no, branch_index in iterate():
            if child.depth > max_depth and max_depth != -1:
                continue

            if isinstance(child, NumericBinaryBranch):
                text = f"{child.feature}"  # type: ignore
            else:
                text = f"{repr(child)}\nsamples: {int(child.total_weight)}"

            fillcolor = "#FFFFFF"

            dot.node(f"{child_no}", text, fillcolor=fillcolor, style="filled")

            if parent_no is not None:
                dot.edge(
                    f"{parent_no}",
                    f"{child_no}",
                    xlabel=parent.repr_branch(branch_index, shorten=True),
                )

        return dot


# Utility adapted from the original creme's implementation
def _color_brew(n: int):
    """Generate n colors with equally spaced hues.

    Parameters
    ----------
    n
        The number of required colors.

    Returns
    -------
        List of n tuples of form (R, G, B) being the components of each color.
    References
    ----------
    https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/tree/_export.py
    """
    colors = []

    # Initialize saturation & value; calculate chroma & value shift
    s, v = 0.75, 0.9
    c = s * v
    m = v - c

    for h in [i for i in range(25, 385, int(360 / n))]:

        # Calculate some intermediate values
        h_bar = h / 60.0
        x = c * (1 - abs((h_bar % 2) - 1))

        # Initialize RGB with same hue & chroma as our color
        rgb = [
            (c, x, 0),
            (x, c, 0),
            (0, c, x),
            (0, x, c),
            (x, 0, c),
            (c, 0, x),
            (c, x, 0),
        ]
        r, g, b = rgb[int(h_bar)]

        # Shift the initial RGB values to match value and store
        colors.append(((int(255 * (r + m))), (int(255 * (g + m))), (int(255 * (b + m)))))

    return colors


# Utility adapted from the original creme's implementation
def transparency_hex(color, alpha: float) -> str:
    """Apply alpha coefficient on hexadecimal color."""
    return "#%02x%02x%02x" % tuple([int(round(alpha * c + (1 - alpha) * 255, 0)) for c in color])
