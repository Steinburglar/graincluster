"""Shared fixtures and synthetic graph builders for graincluster tests."""

from __future__ import annotations

import numpy as np
import pytest

from graincluster.graph.edge import EdgeRecord
from graincluster.features.binning import BinScheme, PairBinScheme
from graincluster.model.cluster import ClusterState
from graincluster.model.partition import partition_from_labels


# ---------------------------------------------------------------------------
# Minimal bin scheme helpers
# ---------------------------------------------------------------------------

def make_bin_scheme(pair_keys: list[str], n_bins: int = 10, lo: float = 1.0, hi: float = 5.0) -> BinScheme:
    """Uniform bin scheme for tests (not FD — keeps tests deterministic)."""
    edges = np.linspace(lo, hi, n_bins + 1)
    schemes = {pk: PairBinScheme(pair_key=pk, edges=edges, range_lo=lo, range_hi=hi)
               for pk in pair_keys}
    return BinScheme(schemes=schemes, _pair_types=sorted(pair_keys))


# ---------------------------------------------------------------------------
# Synthetic edge list builders
# ---------------------------------------------------------------------------

def make_ring_edges(
    n_atoms: int,
    pair_key: str,
    pair_type_idx: int,
    raw_value: float,
    bin_idx: int,
    sigma: float = 1.0,
) -> list[EdgeRecord]:
    """Ring graph: atom i connected to i+1 (wrapping)."""
    edges = []
    for i in range(n_atoms):
        j = (i + 1) % n_atoms
        cut_cost = raw_value ** 2 / (2 * sigma ** 2)
        edges.append(EdgeRecord(
            i=i, j=j,
            pair_key=pair_key,
            pair_type_idx=pair_type_idx,
            raw_value=raw_value,
            bin_idx=bin_idx,
            cut_cost=cut_cost,
        ))
    return edges


def make_two_domain_edges(
    n_a: int,
    n_b: int,
    pair_key: str,
    pair_type_idx: int,
    raw_a: float,
    raw_b: float,
    bridge_raw: float,
    bin_scheme: BinScheme,
    sigma: float = 1.0,
) -> list[EdgeRecord]:
    """Two cliques (domain A: atoms 0..n_a-1, domain B: atoms n_a..n_a+n_b-1)
    connected by one bridge edge.

    Each domain is fully connected. Bridge between atom n_a-1 and atom n_a.
    """
    edges = []
    pk = pair_key
    pt = pair_type_idx

    def _edge(i, j, rv):
        b = bin_scheme.assign_one(pk, rv)
        return EdgeRecord(i=i, j=j, pair_key=pk, pair_type_idx=pt,
                          raw_value=rv, bin_idx=b,
                          cut_cost=rv**2 / (2 * sigma**2))

    # Domain A clique
    for i in range(n_a):
        for j in range(i + 1, n_a):
            edges.append(_edge(i, j, raw_a))

    # Domain B clique
    for i in range(n_a, n_a + n_b):
        for j in range(i + 1, n_a + n_b):
            edges.append(_edge(i, j, raw_b))

    # Bridge
    edges.append(_edge(n_a - 1, n_a, bridge_raw))
    return edges


# ---------------------------------------------------------------------------
# Pytest fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def single_pair_scheme():
    return make_bin_scheme(["A-A"], n_bins=10, lo=1.0, hi=5.0)


@pytest.fixture
def two_pair_scheme():
    return make_bin_scheme(["A-A", "A-B"], n_bins=10, lo=1.0, hi=5.0)
