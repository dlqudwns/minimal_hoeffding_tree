from __future__ import annotations

import abc
import typing

import numpy as np

from ..base.base import Base
from ..optim.schedulers import Constant
from ..utils.vectordict import VectorDict


__all__ = ["Initializer", "Scheduler", "Optimizer", "Loss"]

VectorLike = typing.Union[VectorDict, np.ndarray]




class Optimizer(Base):
    """Optimizer interface.

    Every optimizer inherits from this base interface.

    Parameters
    ----------
    lr

    Attributes
    ----------
    learning_rate : float
        Returns the current learning rate value.

    """

    def __init__(self, lr: int | float | Scheduler):
        if isinstance(lr, (int, float)):
            lr = Constant(lr)
        self.lr = lr
        self.n_iterations = 0

    @property
    def _mutable_attributes(self):
        return {"lr"}

    @property
    def learning_rate(self) -> float:
        return self.lr.get(self.n_iterations)

    def look_ahead(self, w: dict) -> dict:
        """Updates a weight vector before a prediction is made.

        Parameters:
            w (dict): A dictionary of weight parameters. The weights are modified in-place.

        Returns:
            The updated weights.

        """
        return w

    def _step_with_dict(self, w: dict | VectorLike, g: dict | VectorLike) -> dict:
        raise NotImplementedError

    def _step_with_vector(self, w: VectorLike, g: VectorLike) -> VectorLike:
        raise NotImplementedError

    def step(self, w: dict | VectorLike, g: dict | VectorLike) -> dict | VectorLike:
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

    def __repr__(self):
        return f"{self.__class__.__name__}({vars(self)})"

