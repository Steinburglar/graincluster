"""ClusterState: joint count table and bookkeeping for one cluster."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ClusterState:
    """Mutable state for one cluster.

    counts maps (pair_type_idx, bin_idx) -> raw edge count.
    N is total internal edge count (sum of all counts).
    """

    cluster_id: int
    atom_ids: set[int] = field(default_factory=set)
    counts: dict[tuple[int, int], int] = field(default_factory=dict)
    N: int = 0
    _entropy: float | None = field(default=None, repr=False, compare=False)

    def add_edge(self, pair_type_idx: int, bin_idx: int) -> None:
        key = (pair_type_idx, bin_idx)
        self.counts[key] = self.counts.get(key, 0) + 1
        self.N += 1
        self._entropy = None

    def remove_edge(self, pair_type_idx: int, bin_idx: int) -> None:
        key = (pair_type_idx, bin_idx)
        c = self.counts.get(key, 0)
        if c <= 0:
            raise ValueError(f"Cannot remove edge: count for {key} is {c}")
        if c == 1:
            del self.counts[key]
        else:
            self.counts[key] = c - 1
        self.N -= 1
        self._entropy = None

    def invalidate_entropy(self) -> None:
        self._entropy = None

    def is_empty(self) -> bool:
        return len(self.atom_ids) == 0
