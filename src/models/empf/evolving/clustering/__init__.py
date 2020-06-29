"""That package provide implementation of evolving clustering methods."""

from . import _base
from . import esbm
from . import autocloud
from . import oec
from . import microtedaclus

__all__ = ['esbm', 'autocloud', 'oec', 'microtedaclus']

ON_FIRST_CLUSTER = _base.BaseClusteringMethod.ON_FIRST_CLUSTER
ON_KNOWED_CLUSTER = _base.BaseClusteringMethod.ON_KNOWED_CLUSTER
ON_TRANSITION_CLUSTER = _base.BaseClusteringMethod.ON_TRANSITION_CLUSTER
