"""That package provide implementation of methods for help the others.

Contains methods for:
    - compute distances
    - compute similarity
    - manage data (local and remote)
    - incremental update of statistics values

"""

from . import data
from . import distance
from . import incremental
from . import similarity
from . import benchmarks

__all__ = ['data', 'distance', 'incremental', 'similarity', 'benchmarks']
