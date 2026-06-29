from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class EdgeRecord:
    """One undirected edge in the neighbor graph."""

    i: int
    j: int
    pair_key: str        # canonical sorted pair, e.g. "Au-Pt"
    pair_type_idx: int   # integer index into the global pair_types list
    raw_value: float     # interatomic distance in Å
    bin_idx: int         # bin assigned by PairBinScheme
    cut_cost: float      # s_ij = d^2 / (2*sigma^2)
