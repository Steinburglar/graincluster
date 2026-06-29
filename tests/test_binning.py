"""Tests for Freedman-Diaconis binning."""

from __future__ import annotations

import math

import numpy as np
import pytest

from graincluster.features.binning import (
    PairBinScheme,
    fit_pair_bin_scheme,
    fit_bin_scheme,
    _freedman_diaconis_bins,
)


class TestFreedmanDiaconis:
    def test_normal_distribution(self):
        rng = np.random.default_rng(0)
        data = rng.normal(3.0, 0.2, size=1000)
        edges, lo, hi = _freedman_diaconis_bins(data)
        assert lo < hi
        assert len(edges) >= 3                     # at least 2 bins
        assert edges[0] == pytest.approx(lo)
        assert edges[-1] == pytest.approx(hi)
        # All edges monotone
        assert np.all(np.diff(edges) > 0)

    def test_bin_count_formula(self):
        # Known IQR → verify bin count matches formula.
        rng = np.random.default_rng(1)
        data = rng.uniform(1.0, 5.0, size=500)
        q25, q75 = np.percentile(data, [25, 75])
        iqr = q75 - q25
        span = np.percentile(data, 99) - np.percentile(data, 1)
        h = 2.0 * iqr * (500 ** (-1.0 / 3.0))
        expected_bins = max(4, min(256, math.ceil(span / h)))

        edges, _, _ = _freedman_diaconis_bins(data)
        assert len(edges) - 1 == expected_bins

    def test_min_bins_floor(self):
        # Highly concentrated data → IQR near 0 → fallback to sqrt(N).
        data = np.array([2.5] * 8 + [2.500001])
        edges, _, _ = _freedman_diaconis_bins(data, min_bins=4)
        assert len(edges) - 1 >= 4

    def test_too_few_samples_raises(self):
        with pytest.raises(ValueError, match="at least 4"):
            _freedman_diaconis_bins(np.array([1.0, 2.0, 3.0]))

    def test_zero_span_raises(self):
        with pytest.raises(ValueError, match="Zero-span"):
            _freedman_diaconis_bins(np.ones(100))

    def test_max_bins_ceiling(self):
        # Tiny IQR with large N → would produce enormous B without ceiling.
        rng = np.random.default_rng(2)
        data = rng.normal(3.0, 1e-4, size=10000)
        edges, _, _ = _freedman_diaconis_bins(data, max_bins=64)
        assert len(edges) - 1 <= 64


class TestPairBinScheme:
    def _make_scheme(self, n_bins=10):
        edges = np.linspace(1.0, 5.0, n_bins + 1)
        return PairBinScheme(pair_key="Au-Au", edges=edges, range_lo=1.0, range_hi=5.0)

    def test_n_bins(self):
        s = self._make_scheme(10)
        assert s.n_bins == 10

    def test_assign_interior(self):
        s = self._make_scheme(10)
        # Midpoint of each bin should land in correct bin.
        for b in range(10):
            lo = 1.0 + b * 0.4
            hi = lo + 0.4
            mid = (lo + hi) / 2
            assert s.assign_one(mid) == b

    def test_assign_below_range_clips_to_zero(self):
        s = self._make_scheme(10)
        assert s.assign_one(0.0) == 0

    def test_assign_above_range_clips_to_last(self):
        s = self._make_scheme(10)
        assert s.assign_one(99.0) == 9

    def test_assign_array_shape(self):
        s = self._make_scheme(10)
        values = np.array([1.5, 2.5, 3.5, 4.5])
        result = s.assign(values)
        assert result.shape == (4,)
        assert result.dtype == np.int32

    def test_assign_array_values_match_scalar(self):
        s = self._make_scheme(10)
        values = np.linspace(1.1, 4.9, 50)
        arr_result = s.assign(values)
        scalar_result = np.array([s.assign_one(v) for v in values])
        np.testing.assert_array_equal(arr_result, scalar_result)

    def test_assign_boundary_edges(self):
        s = self._make_scheme(4)  # bins: [1,2), [2,3), [3,4), [4,5]
        # Exact bin edge values: should go into the higher bin (searchsorted right).
        assert s.assign_one(1.0) == 0   # left edge of first bin
        assert s.assign_one(5.0) == 3   # right edge of last bin (clips)


class TestFitBinScheme:
    def test_fit_pair_bin_scheme_roundtrip(self):
        rng = np.random.default_rng(3)
        values = rng.normal(2.8, 0.3, size=500)
        scheme = fit_pair_bin_scheme("Au-Au", values)
        assert scheme.pair_key == "Au-Au"
        assert scheme.n_bins >= 4
        # All fitted values should assign to valid bins.
        bins = scheme.assign(values)
        assert np.all(bins >= 0)
        assert np.all(bins < scheme.n_bins)

    def test_fit_bin_scheme_multiple_pairs(self):
        rng = np.random.default_rng(4)
        pair_values = {
            "Au-Au": rng.normal(2.8, 0.2, size=500),
            "Au-Pt": rng.normal(2.9, 0.3, size=300),
            "Pt-Pt": rng.normal(2.7, 0.15, size=400),
        }
        bs = fit_bin_scheme(pair_values)
        assert set(bs.pair_types) == {"Au-Au", "Au-Pt", "Pt-Pt"}
        assert bs.total_categories() == sum(
            bs.schemes[pk].n_bins for pk in bs.pair_types
        )

    def test_fit_bin_scheme_pair_types_sorted(self):
        rng = np.random.default_rng(5)
        pair_values = {
            "Pt-Pt": rng.uniform(2.5, 3.5, 200),
            "Au-Au": rng.uniform(2.7, 3.7, 200),
            "Au-Pt": rng.uniform(2.6, 3.6, 200),
        }
        bs = fit_bin_scheme(pair_values)
        assert bs.pair_types == sorted(bs.pair_types)

    def test_fit_bin_scheme_pair_specific_ranges(self):
        """Different pair types should have different ranges, not forced shared."""
        rng = np.random.default_rng(6)
        pair_values = {
            "A-A": rng.normal(2.0, 0.1, 300),   # mean 2.0
            "B-B": rng.normal(4.0, 0.1, 300),   # mean 4.0
        }
        bs = fit_bin_scheme(pair_values)
        aa = bs.schemes["A-A"]
        bb = bs.schemes["B-B"]
        assert aa.range_hi < 3.0   # A-A range near 2.0
        assert bb.range_lo > 3.0   # B-B range near 4.0
