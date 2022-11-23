
class NumericBinaryBranch:
    def __init__(self, stats, feature, threshold, depth, left, right, **attributes):
        self.stats = stats
        self.feature = feature
        self.threshold = threshold
        self.depth = depth
        self.children = [left, right]
        self.__dict__.update(attributes)

    def walk(self, x, until_leaf=True):
        """Iterate over the nodes of the path induced by x."""
        yield self
        try:
            yield from self.next(x).walk(x, until_leaf)
        except KeyError:
            if until_leaf:
                _, node = self.most_common_path()
                yield node
                yield from node.walk(x, until_leaf)

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

    def max_branches(self):
        return 2

    def most_common_path(self):
        left, right = self.children

        if left.total_weight < right.total_weight:
            return 1, right
        return 0, left

    def repr_branch(self, index: int, shorten=False):
        if shorten:
            if index == 0:
                return f"≤ {round(self.threshold, 4)}"
            return f"> {round(self.threshold, 4)}"
        else:
            if index == 0:
                return f"{self.feature} ≤ {self.threshold}"
            return f"{self.feature} > {self.threshold}"

    @property
    def repr_split(self):
        return f"{self.feature} ≤ {self.threshold}"
