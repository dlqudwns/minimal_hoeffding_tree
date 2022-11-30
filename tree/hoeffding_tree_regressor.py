import math

from .hoeffding_tree.nodes import NumericBinaryBranch, LeafAdaptive
from .hoeffding_tree.variance_reduction_split_criterion import VarianceReductionSplitCriterion
from .hoeffding_tree.splitter import TEBSTSplitter

import functools
import collections


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
    min_samples_split
        The minimum number of samples every branch resulting from a split candidate must have
        to be considered valid.
    """

    def __init__(
        self,
        grace_period: int = 200,
        max_depth: int = None,
        delta: float = 1e-7,
        tau: float = 0.05,
        min_samples_split: int = 5
    ):
        self.max_depth: float = max_depth if max_depth is not None else math.inf
        self._root = None
        self.binary_split = False

        self.grace_period = grace_period
        self.delta = delta
        self.tau = tau
        self.split_criterion = VarianceReductionSplitCriterion(min_samples_split)
        self.splitter = TEBSTSplitter(self.split_criterion)

    @staticmethod
    def _hoeffding_bound(range_val, confidence, n):
        return math.sqrt((range_val * range_val * math.log(1.0 / confidence)) / (2.0 * n))

    def _new_leaf(self, initial_stats=None, parent=None):
        return LeafAdaptive(initial_stats, self.splitter, parent)

    def learn_one(self, x, y, *, sample_weight=1.0):
        """Train the tree model on sample x and corresponding target y"""

        if self._root is None:
            self._root = self._new_leaf()
        node = self._root.traverse(x)
        p_node = node.parent

        node.learn_one(x, y, sample_weight=sample_weight, tree=self)
        if node.is_active():
            if node.depth >= self.max_depth:  # Max depth reached
                node.deactivate()
            else:
                if node.total_weight - node.last_split_attempt_at >= self.grace_period:
                    p_branch = None if p_node is None else p_node.branch_no(x)
                    self._attempt_to_split(node, p_node, p_branch)
                    node.last_split_attempt_at = node.total_weight
        return self

    def predict_one(self, x):
        """Predict the target value using one of the leaf prediction strategies"""
        if self._root is None:
            return 0.0
        return self._root.traverse(x).prediction(x)

    def _attempt_to_split(self, leaf, parent, parent_branch: int, **kwargs):
        """Attempt to split a node."""

        best_split_suggestions = leaf.best_split_suggestions()
        if len(best_split_suggestions) >= 2:
            hoeffding_bound = self._hoeffding_bound(
                self.split_criterion.range_of_merit(leaf.stats),
                self.delta,
                leaf.total_weight,
            )
            merit_1 = best_split_suggestions[-1].merit
            merit_2 = best_split_suggestions[-2].merit
            if merit_1 <= 0.0:
                return
            if merit_2 / merit_1 >= 1 - hoeffding_bound and hoeffding_bound >= self.tau:
                if merit_2 > 0:
                    merit_lower_bound = (merit_2 / merit_1 - 2 * hoeffding_bound) * merit_1
                    leaf.manage_memory(merit_lower_bound)
                return

        split_decision = best_split_suggestions[-1]
        if split_decision.feature is None:
            leaf.deactivate()
        else:
            leaves = tuple(
                self._new_leaf(initial_stats, parent=leaf)
                for initial_stats in split_decision.children_stats  # type: ignore
            )
            new_split = split_decision.assemble(
                NumericBinaryBranch, leaf.stats, leaf.depth, parent, *leaves, **kwargs
            )
            if parent is None:
                self._root = new_split
            else:
                parent.children[parent_branch] = new_split


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
        import graphviz
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
