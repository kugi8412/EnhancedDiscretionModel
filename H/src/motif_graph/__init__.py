"""Motif-graph representation of DNA sequences.

Nodes are motifs / k-mers, edges are the distance and purine-pyrimidine
composition between them.  See :mod:`motif_graph.vocab` for the node set and
:mod:`motif_graph.featurize` for the sequence -> graph transform.
"""

from .vocab import (
    IUPAC,
    KNOWN_MOTIFS,
    MotifVocabulary,
    canonical_pattern,
    is_palindromic,
    reverse_complement_pattern,
    sample_vocabulary,
)
from .featurize import (
    EDGE_DIM,
    EDGE_FEATURE_NAMES,
    HELICAL_PERIOD,
    NODE_DIM,
    NODE_FEATURE_NAMES,
    MotifGraphFeaturizer,
)

__all__ = [
    "IUPAC", "KNOWN_MOTIFS", "MotifVocabulary", "canonical_pattern",
    "is_palindromic", "reverse_complement_pattern", "sample_vocabulary",
    "EDGE_DIM", "EDGE_FEATURE_NAMES", "HELICAL_PERIOD", "NODE_DIM",
    "NODE_FEATURE_NAMES", "MotifGraphFeaturizer",
]
