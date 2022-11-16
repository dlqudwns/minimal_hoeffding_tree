"""Utilities for inspecting a model's type.

Sometimes we need to check if a model can perform regression, classification, etc. However, for
some models the model's type is only known at runtime. For instance, we can't do
`isinstance(pipeline, base.Regressor)` or `isinstance(wrapper, base.Regressor)`. This submodule
thus provides utilities for determining an arbitrary model's type.

"""
import inspect

from ..base.estimator import Estimator
from ..base.classifier import Classifier
from ..base.clusterer import Clusterer
from ..base.multi_output import MultiOutputMixin
from ..base.regressor import Regressor
from ..base.transformer import Transformer
from ..base.drift_detector import DriftDetector

# TODO: maybe all of this could be done by monkeypatching isintance for pipelines?


__all__ = [
    "extract_relevant",
    "isanomalydetector",
    "isclassifier",
    "isregressor",
    "ismoclassifier",
    "ismoregressor",
    "isdriftdetector",
]


def extract_relevant(model: Estimator):
    """Extracts the relevant part of a model.

    Parameters
    ----------
    model

    """

    if ischildobject(obj=model, class_name="Pipeline"):
        return extract_relevant(model._last_step)  # type: ignore[attr-defined]
    return model


def ischildobject(obj: object, class_name: str) -> bool:
    """Checks weather or not the given object inherits from a class with the given class name.

    Workaround isinstance function to not have to import modules defining classes and become
    dependent on them. class_name is case-sensitive.

    Examples
    --------

    >>> from river import anomaly
    >>> from river import utils

    >>> class_name = "AnomalyDetector"

    >>> utils.inspect.ischildobject(obj=anomaly.HalfSpaceTrees(), class_name=class_name)
    True

    >>> utils.inspect.ischildobject(obj=anomaly.OneClassSVM(), class_name=class_name)
    True

    >>> utils.inspect.ischildobject(obj=anomaly.GaussianScorer(), class_name=class_name)
    False

    """
    parent_classes = inspect.getmro(obj.__class__)
    return any(cls.__name__ == class_name for cls in parent_classes)


def isanomalydetector(model):
    model = extract_relevant(model)
    return ischildobject(obj=model, class_name="AnomalyDetector")


def isclassifier(model):
    return isinstance(extract_relevant(model), Classifier)


def isclusterer(model):
    return isinstance(extract_relevant(model), Clusterer)


def ismoclassifier(model):
    return isclassifier(model) and isinstance(extract_relevant(model), MultiOutputMixin)


def isregressor(model):
    return isinstance(extract_relevant(model), Regressor)


def istransformer(model):
    return isinstance(extract_relevant(model), Transformer)


def ismoregressor(model):
    return isregressor(model) and isinstance(extract_relevant(model), MultiOutputMixin)


def isdriftdetector(model):
    return isinstance(extract_relevant(model), DriftDetector)