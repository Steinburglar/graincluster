"""Species-pair utilities."""

from __future__ import annotations

from itertools import combinations_with_replacement


def canonical_pair_key(s1: str, s2: str) -> str:
    """Return sorted canonical pair key, e.g. 'Au-Pt'."""
    a, b = (s1, s2) if s1 <= s2 else (s2, s1)
    return f"{a}-{b}"


def all_pair_keys(species: list[str]) -> list[str]:
    """All canonical pair keys for a species list (sorted, deduplicated)."""
    unique = sorted(set(species))
    return [canonical_pair_key(a, b) for a, b in combinations_with_replacement(unique, 2)]
