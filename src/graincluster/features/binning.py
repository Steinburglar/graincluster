"""Per-species-pair bin schemes using Freedman-Diaconis rule."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np


@dataclass
class PairBinScheme:
    """Bin scheme for one species-pair type.

    Bin edges are fixed at fit time and never updated during optimization.
    The range is clipped to [p1, p99] of the reference data.
    Bin count set by Freedman-Diaconis: h = 2 * IQR * N^(-1/3).
    """

    pair_key: str
    edges: np.ndarray   # shape (n_bins + 1,)
    range_lo: float
    range_hi: float

    @property
    def n_bins(self) -> int:
        return len(self.edges) - 1

    def assign(self, values: np.ndarray) -> np.ndarray:
        """Assign bin indices. Values outside range clipped to boundary bins."""
        indices = np.digitize(values, self.edges[1:-1])
        return np.clip(indices, 0, self.n_bins - 1).astype(np.int32)

    def assign_one(self, value: float) -> int:
        idx = int(np.searchsorted(self.edges[1:-1], value, side="right"))
        return max(0, min(self.n_bins - 1, idx))


def _freedman_diaconis_bins(
    data: np.ndarray,
    p_lo: float = 1.0,
    p_hi: float = 99.0,
    min_bins: int = 4,
    max_bins: int = 256,
) -> tuple[np.ndarray, float, float]:
    """Compute linear bin edges via Freedman-Diaconis rule.

    Returns (edges, range_lo, range_hi).
    """
    data = np.asarray(data, dtype=float)
    if len(data) < 4:
        raise ValueError(f"Need at least 4 samples, got {len(data)}")

    q25, q75 = np.percentile(data, [25.0, 75.0])
    iqr = q75 - q25

    range_lo = float(np.percentile(data, p_lo))
    range_hi = float(np.percentile(data, p_hi))
    span = range_hi - range_lo

    if span <= 0.0:
        raise ValueError("Zero-span range after clipping — all values identical?")

    if iqr < 1e-6:
        # Degenerate distribution (IQR numerically zero): fall back to sqrt(N) bins
        n_bins = max(min_bins, min(max_bins, int(math.ceil(math.sqrt(len(data))))))
    else:
        h = 2.0 * iqr * (len(data) ** (-1.0 / 3.0))
        n_bins = int(math.ceil(span / h))
        n_bins = max(min_bins, min(max_bins, n_bins))

    edges = np.linspace(range_lo, range_hi, n_bins + 1)
    return edges, range_lo, range_hi


def fit_pair_bin_scheme(
    pair_key: str,
    values: np.ndarray,
    p_lo: float = 1.0,
    p_hi: float = 99.0,
    min_bins: int = 4,
    max_bins: int = 256,
) -> PairBinScheme:
    """Fit a PairBinScheme from a reference sample of edge values."""
    edges, rlo, rhi = _freedman_diaconis_bins(
        values, p_lo=p_lo, p_hi=p_hi, min_bins=min_bins, max_bins=max_bins
    )
    return PairBinScheme(pair_key=pair_key, edges=edges, range_lo=rlo, range_hi=rhi)


@dataclass
class BinScheme:
    """Collection of PairBinScheme for all pair types in a system."""

    schemes: dict[str, PairBinScheme] = field(default_factory=dict)
    _pair_types: list[str] = field(default_factory=list)

    @property
    def pair_types(self) -> list[str]:
        return self._pair_types

    @property
    def n_bins_per_type(self) -> dict[int, int]:
        return {i: self.schemes[pk].n_bins for i, pk in enumerate(self._pair_types)}

    def total_categories(self) -> int:
        return sum(s.n_bins for s in self.schemes.values())

    def assign_one(self, pair_key: str, value: float) -> int:
        return self.schemes[pair_key].assign_one(value)

    def assign(self, pair_key: str, values: np.ndarray) -> np.ndarray:
        return self.schemes[pair_key].assign(values)


def fit_bin_scheme(
    pair_values: dict[str, np.ndarray],
    p_lo: float = 1.0,
    p_hi: float = 99.0,
    min_bins: int = 4,
    max_bins: int = 256,
) -> BinScheme:
    """Fit a BinScheme from a dict mapping pair_key -> array of edge values.

    pair_values should come from a representative reference set (e.g. one
    or more trajectory frames). Keys are canonical pair keys like 'Au-Pt'.
    """
    pair_types = sorted(pair_values.keys())
    schemes: dict[str, PairBinScheme] = {}
    for pk in pair_types:
        vals = np.asarray(pair_values[pk], dtype=float)
        schemes[pk] = fit_pair_bin_scheme(
            pk, vals, p_lo=p_lo, p_hi=p_hi, min_bins=min_bins, max_bins=max_bins
        )
    bs = BinScheme(schemes=schemes, _pair_types=pair_types)
    return bs
