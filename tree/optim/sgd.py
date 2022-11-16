
import numpy as np

from ..utils.vectordict import VectorDict

class SGD:

    def __init__(self, lr=0.01):
        self.lr = lr
        self.n_iterations = 0

    def _step_with_dict(self, w, g):
        for i, gi in g.items():
            w[i] -= self.lr * gi
        return w

    def _step_with_vector(self, w, g):
        w -= self.lr * g
        return w

    def look_ahead(self, w):
        return w

    def step(self, w, g):
        """Updates a weight vector given a gradient.

        Parameters
        ----------
        w
            A vector-like object containing weights. The weights are modified in-place.
        g
            A vector-like object of gradients.

        Returns
        -------
        The updated weights.

        """

        if isinstance(w, (VectorDict, np.ndarray)) and isinstance(
            g, (VectorDict, np.ndarray)
        ):
            try:
                w = self._step_with_vector(w, g)
                self.n_iterations += 1
                return w
            except NotImplementedError:
                pass

        w = self._step_with_dict(w, g)
        self.n_iterations += 1
        return w