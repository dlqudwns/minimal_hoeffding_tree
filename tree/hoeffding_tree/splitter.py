from .stats import Var
from .tree_utils import BranchFactory

class EBSTSplitter:
    """iSOUP-Tree's Extended Binary Search Tree (E-BST)"""

    def __init__(self, split_criterion):
        self.split_criterion = split_criterion
        self._root = None

    def update(self, att_val, target_val, sample_weight):
        if att_val is not None:
            if self._root is None:
                self._root = EBSTNode(att_val, target_val, sample_weight)
            else:
                self._root.insert_value(att_val, target_val, sample_weight)

    def best_evaluated_split_suggestion(self, pre_split_dist, att_idx):
        candidate = BranchFactory()

        if self._root is None:
            return candidate
        return self._find_best_split(self._root, candidate, pre_split_dist, att_idx, Var())

    def _compute_merit(self, node, aux_estimator, pre_split_dist):
        left_dist = node.estimator + aux_estimator
        right_dist = pre_split_dist - left_dist
        post_split_dists = [left_dist, right_dist]
        merit = self.split_criterion.merit_of_split(pre_split_dist, post_split_dists)
        return merit, post_split_dists

    def _find_best_split(self, node, candidate, pre_split_dist, att_idx, aux_estimator):
        if node._left is not None:
            candidate = self._find_best_split(node._left, candidate, pre_split_dist, att_idx, aux_estimator)
        merit, post_split_dists = self._compute_merit(node, aux_estimator, pre_split_dist)
        if merit > candidate.merit:
            candidate = BranchFactory(merit, att_idx, node.att_val, post_split_dists)

        if node._right is not None:
            right_candidate = self._find_best_split(node._right, candidate, pre_split_dist, att_idx, aux_estimator + node.estimator)
            if right_candidate.merit > candidate.merit:
                candidate = right_candidate
        return candidate

    def remove_bad_splits(self, merit_lower_bound, pre_split_dist):
        if self._root is None:
            return
        self._remove_bad_split_nodes(self._root, pre_split_dist, merit_lower_bound, Var())

    def _remove_bad_split_nodes(self, node, pre_split_dist, merit_lower_bound, aux_estimator):
        if node._left is not None:
            if self._remove_bad_split_nodes(node._left, pre_split_dist, merit_lower_bound, aux_estimator):
                node._left = None

        if node._left is None and node._right is not None:
            if self._remove_bad_split_nodes(
                node._right, pre_split_dist, merit_lower_bound, aux_estimator + node.estimator):
                node._right = None

        if node._left is None and node._right is None:
            merit, _ = self._compute_merit(node, aux_estimator, pre_split_dist)
            if merit < merit_lower_bound:
                if node == self._root:
                    self._root = None
                return True

        return False


class EBSTNode:
    """
    Binary search tree that tracks variance of target (y)
    of data points that some attribute (1 dimension of x) is smaller than
    some value (att_val of some node)
    """
    def __init__(self, att_val, target_val, sample_weight, parent=None):
        self.att_val = att_val

        self.estimator = Var()
        self.estimator.update(target_val, sample_weight)

        self._left = None
        self._right = None

    def insert_value(self, att_val, target_val, sample_weight):
        current = self
        while current is not None:
            if att_val == current.att_val:
                current.estimator.update(target_val, sample_weight)
                return
            elif att_val < current.att_val:
                current.estimator.update(target_val, sample_weight)
                if current._left is None:
                    current._left = EBSTNode(att_val, target_val, sample_weight)
                    return
                current = current._left
            else:
                if current._right is None:
                    current._right = EBSTNode(att_val, target_val, sample_weight)
                    return
                current = current._right


class TEBSTSplitter(EBSTSplitter):
    """Variation of E-BST that rounds the incoming feature values before passing them to the binary
    search tree (BST)."""

    def __init__(self, split_criterion, digits: int = 1):
        super().__init__(split_criterion)
        self.digits = digits

    def update(self, att_val, target_val, sample_weight):
        try:
            att_val = round(att_val, self.digits)
            super().update(att_val, target_val, sample_weight)
        except TypeError:  # feature value is None
            pass
