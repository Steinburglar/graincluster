"""Build edge lists from Frame + neighbor list."""

from __future__ import annotations

import numpy as np
from graphcluster.io.frame import Frame
from scipy.spatial import cKDTree

from .edge import EdgeRecord
from ..features.species import canonical_pair_key
from ..features.binning import BinScheme


def build_edges(
    frame: Frame,
    cutoff: float,
    bin_scheme: BinScheme,
    sigma: float = 1.0,
) -> list[EdgeRecord]:
    """Return undirected EdgeRecord list for one frame.

    Neighbor pairs found via cKDTree within cutoff.
    Periodic images handled when frame.cell is provided.
    """
    positions = np.asarray(frame.positions, dtype=float)
    symbols = list(frame.chemical_symbols or frame.atom_types or [])
    n_atoms = len(positions)

    if frame.box is not None:
        cell = np.asarray(frame.box, dtype=float)
        if cell.shape == (3,):
            cell = np.diag(cell)
        tree = cKDTree(positions, boxsize=None)
        pairs, dists = _pairs_periodic(positions, cell, cutoff)
    else:
        tree = cKDTree(positions)
        pairs = tree.query_pairs(cutoff, output_type="ndarray")
        i_idx, j_idx = pairs[:, 0], pairs[:, 1]
        dists = np.linalg.norm(positions[i_idx] - positions[j_idx], axis=1)
        pairs = list(zip(i_idx.tolist(), j_idx.tolist()))

    edges: list[EdgeRecord] = []
    pair_types = bin_scheme.pair_types

    for (i, j), d in zip(pairs, dists):
        si = symbols[i] if symbols else str(i)
        sj = symbols[j] if symbols else str(j)
        pk = canonical_pair_key(si, sj)
        if pk not in bin_scheme.schemes:
            continue
        pt_idx = pair_types.index(pk)
        b_idx = bin_scheme.assign_one(pk, d)
        cut_cost = d * d / (2.0 * sigma * sigma)
        edges.append(EdgeRecord(
            i=i, j=j,
            pair_key=pk,
            pair_type_idx=pt_idx,
            raw_value=d,
            bin_idx=b_idx,
            cut_cost=cut_cost,
        ))

    return edges


def _pairs_periodic(
    positions: np.ndarray,
    cell: np.ndarray,
    cutoff: float,
) -> tuple[list[tuple[int, int]], list[float]]:
    """Brute-force periodic pairs for orthorhombic cells."""
    n = len(positions)
    diag = np.diag(cell)
    pairs = []
    dists = []
    for i in range(n):
        for j in range(i + 1, n):
            delta = positions[i] - positions[j]
            delta -= np.round(delta / diag) * diag
            d = float(np.linalg.norm(delta))
            if d < cutoff:
                pairs.append((i, j))
                dists.append(d)
    return pairs, dists
