import copy
import dataclasses
import functools
import math
import typing

from .stats.var import Var


def do_naive_bayes_prediction(x, observed_class_distribution: dict, splitters: dict):
    """Perform Naive Bayes prediction

    Parameters
    ----------
    x
        The feature values.

    observed_class_distribution
        Observed class distribution.

    splitters
        Attribute (features) observers.

    Returns
    -------
    The probabilities related to each class.

    Notes
    -----
    This method is not intended to be used as a stand-alone method.
    """
    total_weight = sum(observed_class_distribution.values())
    if not observed_class_distribution or total_weight == 0:
        # No observed class distributions, all classes equal
        return {}

    votes = {}
    for class_index, class_weight in observed_class_distribution.items():
        # Prior
        if class_weight > 0:
            votes[class_index] = math.log(class_weight / total_weight)
        else:
            votes[class_index] = 0.0
            continue

        if splitters:
            for att_idx in splitters:
                if att_idx not in x:
                    continue
                obs = splitters[att_idx]
                # Prior plus the log likelihood
                tmp = obs.cond_proba(x[att_idx], class_index)
                votes[class_index] += math.log(tmp) if tmp > 0 else 0.0

    # Max log-likelihood
    max_ll = max(votes.values())
    # Apply the log-sum-exp trick (https://stats.stackexchange.com/a/253319)
    lse = max_ll + math.log(sum(math.exp(log_proba - max_ll) for log_proba in votes.values()))

    for class_index in votes:
        votes[class_index] = math.exp(votes[class_index] - lse)

    return votes


@functools.total_ordering
@dataclasses.dataclass
class BranchFactory:
    """Helper class used to assemble branches designed by the splitters.

    If constructed using the default values, a null-split suggestion is assumed.
    """

    merit: float = -math.inf
    feature: ... = None
    split_info: typing.Optional[
        typing.Union[
            typing.Hashable,
            typing.List[typing.Hashable],
            typing.Tuple[typing.Hashable, typing.List[typing.Hashable]],
        ]
    ] = None
    children_stats: typing.Optional[typing.List] = None
    numerical_feature: bool = True
    multiway_split: bool = False

    def assemble(
        self,
        branch,  # typing.Type[DTBranch],
        stats: typing.Union[typing.Dict, Var],
        depth: int,
        *children,
        **kwargs,
    ):
        return branch(stats, self.feature, self.split_info, depth, *children, **kwargs)

    def __lt__(self, other):
        return self.merit < other.merit

    def __eq__(self, other):
        return self.merit == other.merit


class GradHess:
    """The most basic inner structure of the Stochastic Gradient Trees that carries information
    about the gradient and hessian of a given observation.
    """

    __slots__ = ["gradient", "hessian"]

    def __init__(self, gradient: float = 0.0, hessian: float = 0.0, *, grad_hess=None):
        if grad_hess:
            self.gradient = grad_hess.gradient
            self.hessian = grad_hess.hessian
        else:
            self.gradient = gradient
            self.hessian = hessian

    def __iadd__(self, other):
        self.gradient += other.gradient
        self.hessian += other.hessian

        return self

    def __isub__(self, other):
        self.gradient -= other.gradient
        self.hessian -= other.hessian

        return self

    def __add__(self, other):
        new = copy.deepcopy(self)
        new += other
        return new

    def __sub__(self, other):
        new = copy.deepcopy(self)
        new -= other
        return new


@functools.total_ordering
@dataclasses.dataclass
class GradHessMerit:
    """Class used to keep the split merit of each split candidate, accordingly to its
    gradient and hessian information.

    In Stochastic Gradient Trees, the split merit is given by a combination of the loss mean and
    variance. Additionally, the loss in each resulting tree branch is also accounted.
    """

    loss_mean: float = 0.0
    loss_var: float = 0.0
    delta_pred: typing.Optional[typing.Union[float, typing.Dict]] = None

    def __lt__(self, other):
        return self.loss_mean < other.loss_mean

    def __eq__(self, other):
        return self.loss_mean == other.loss_mean

