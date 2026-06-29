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
    """Periodic pairs via fractional-coordinate minimum image.

    Works for orthorhombic and triclinic cells. Uses a cKDTree on a
    replicated (2×2×2) supercell to avoid O(N²) overhead.
    """
    from scipy.spatial import cKDTree

    inv_cell = np.linalg.inv(cell)
    frac = positions @ inv_cell
    frac -= np.floor(frac)  # wrap to [0, 1)

    n = len(positions)
    # Replicate ±1 images in each direction; collect (image_cartesian, original_index)
    images = []
    image_idx = []
    shifts = [0, 1, -1]
    for s0 in shifts:
        for s1 in shifts:
            for s2 in shifts:
                shift = np.array([s0, s1, s2], dtype=float)
                cart = (frac + shift) @ cell
                images.append(cart)
                image_idx.extend(range(n))

    images_arr = np.vstack(images)   # shape (27*n, 3)
    image_idx_arr = np.array(image_idx)

    # Build tree on images, query from original positions only
    wrapped_cart = frac @ cell
    tree = cKDTree(images_arr)
    query_tree = cKDTree(wrapped_cart)
    raw_pairs = query_tree.query_ball_tree(tree, cutoff)

    pairs = []
    dists = []
    seen: set[tuple[int, int]] = set()
    for i, neighbours in enumerate(raw_pairs):
        for img_k in neighbours:
            j = int(image_idx_arr[img_k])
            if j <= i:
                continue
            key = (i, j)
            if key in seen:
                continue
            seen.add(key)
            dr = images_arr[img_k] - wrapped_cart[i]
            d = float(np.linalg.norm(dr))
            if d < cutoff:
                pairs.append((i, j))
                dists.append(d)
    return pairs, dists
