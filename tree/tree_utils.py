
import dataclasses
import functools
import math
import typing

from .stats.var import Var

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

