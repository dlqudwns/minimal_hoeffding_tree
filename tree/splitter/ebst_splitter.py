import functools
import typing

from ..stats.var import Var

from ..tree_utils import BranchFactory

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

        aux_estimator = Var()
        return self._find_best_split(self._root, candidate, pre_split_dist, att_idx, aux_estimator)

    def _find_best_split(self, node, candidate, pre_split_dist, att_idx, aux_estimator):
        if node._left is not None:
            candidate = self._find_best_split(node._left, candidate, pre_split_dist, att_idx, aux_estimator)
        # Left post split distribution
        left_dist = node.estimator + aux_estimator

        # The right split distribution is calculated as the difference between the total
        # distribution (pre split distribution) and the left distribution
        right_dist = pre_split_dist - left_dist

        post_split_dists = [left_dist, right_dist]

        merit = self.split_criterion.merit_of_split(pre_split_dist, post_split_dists)
        if merit > candidate.merit:
            candidate = BranchFactory(merit, att_idx, node.att_val, post_split_dists)

        if node._right is not None:
            right_candidate = self._find_best_split(node._right, candidate, pre_split_dist, att_idx, aux_estimator + node.estimator)
            if right_candidate.merit > candidate.merit:
                candidate = right_candidate
        return candidate

    def remove_bad_splits(
        self,
        criterion,
        last_check_ratio: float,
        last_check_vr: float,
        last_check_e: float,
        pre_split_dist: typing.Union[typing.List, typing.Dict],
    ):
        """Remove bad splits.

        Based on FIMT-DD's [^1] procedure to remove bad split candidates from the E-BST. This
        mechanism is triggered every time a split attempt fails. The rationale is to remove
        points whose split merit is much worse than the best candidate overall (for which the
        growth decision already failed).

        Let $m_1$ be the merit of the best split point and $m_2$ be the merit of the
        second best split candidate. The ratio $r = m_2/m_1$ along with the Hoeffding bound
        ($\\epsilon$) are used to decide upon creating a split. A split occurs when
        $r < 1 - \\epsilon$. A split candidate, with merit $m_i$, is considered badr
        if $m_i / m_1 < r - 2\\epsilon$. The rationale is the following: if the merit ratio
        for this point is smaller than the lower bound of $r$, then the true merit of that
        split relative to the best one is small. Hence, this candidate can be safely removed.

        To avoid excessive and costly manipulations of the E-BST to update the stored statistics,
        only the nodes whose children are all bad split points are pruned, as defined in [^1].

        Parameters
        ----------
        criterion
            The split criterion used by the regression tree.
        last_check_ratio
            The ratio between the merit of the second best split candidate and the merit of the
            best split candidate observed in the last failed split attempt.
        last_check_vr
            The merit (variance reduction) of the best split candidate observed in the last
            failed split attempt.
        last_check_e
            The Hoeffding bound value calculated in the last failed split attempt.
        pre_split_dist
            The complete statistics of the target observed in the leaf node.

        References
        ----------
        [^1]: Ikonomovska, E., Gama, J., & Džeroski, S. (2011). Learning model trees from evolving
        data streams. Data mining and knowledge discovery, 23(1), 128-168.
        """

        if self._root is None:
            return

        # Auxiliary variables
        self._criterion = criterion
        self._pre_split_dist = pre_split_dist
        self._last_check_ratio = last_check_ratio
        self._last_check_vr = last_check_vr
        self._last_check_e = last_check_e

        self._aux_estimator = Var()

        self._remove_bad_split_nodes(self._root)

        # Delete auxiliary variables
        del self._criterion
        del self._pre_split_dist
        del self._last_check_ratio
        del self._last_check_vr
        del self._last_check_e
        del self._aux_estimator

    def _remove_bad_split_nodes(self, current_node, parent=None, is_left_child=True):
        is_bad = False

        if current_node._left is not None:
            is_bad = self._remove_bad_split_nodes(current_node._left, current_node, True)
        else:  # Every leaf node is potentially a bad candidate
            is_bad = True

        if is_bad:
            if current_node._right is not None:
                self._aux_estimator += current_node.estimator

                is_bad = self._remove_bad_split_nodes(current_node._right, current_node, False)

                self._aux_estimator -= current_node.estimator
            else:  # Every leaf node is potentially a bad candidate
                is_bad = True

        if is_bad:
            # Left post split distribution
            left_dist = current_node.estimator + self._aux_estimator

            # The right split distribution is calculated as the difference between the total
            # distribution (pre split distribution) and the left distribution
            right_dist = self._pre_split_dist - left_dist

            post_split_dists = [left_dist, right_dist]
            merit = self._criterion.merit_of_split(self._pre_split_dist, post_split_dists)
            if (merit / self._last_check_vr) < (self._last_check_ratio - 2 * self._last_check_e):
                # Remove children nodes
                current_node._left = None
                current_node._right = None

                # Root node
                if parent is None:
                    self._root = None
                else:  # Root node
                    # Remove bad candidate
                    if is_left_child:
                        parent._left = None
                    else:
                        parent._right = None
                return True

        return False


class EBSTNode:
    """
    Binary search tree that tracks variance of target (y)
    of data points that some attribute (1 dimension of x) is smaller than
    some value (att_val of some node)
    """
    def __init__(self, att_val, target_val, sample_weight):
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
