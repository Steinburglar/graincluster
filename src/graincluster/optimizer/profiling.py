"""Lightweight live profiling helpers for graincluster optimization."""

from __future__ import annotations

from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field
from time import perf_counter


@dataclass
class LiveProfiler:
    """Collect cumulative and checkpointed timing/counter statistics."""

    times: dict[str, float] = field(default_factory=lambda: defaultdict(float))
    counts: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    extras: dict[str, float] = field(default_factory=lambda: defaultdict(float))
    _last_times: dict[str, float] = field(default_factory=dict, repr=False)
    _last_counts: dict[str, int] = field(default_factory=dict, repr=False)
    _last_extras: dict[str, float] = field(default_factory=dict, repr=False)
    start_time: float = field(default_factory=perf_counter)

    @contextmanager
    def time_block(self, name: str):
        t0 = perf_counter()
        try:
            yield
        finally:
            self.times[name] += perf_counter() - t0
            self.counts[f"{name}:calls"] += 1

    def add_count(self, name: str, value: int = 1) -> None:
        self.counts[name] += value

    def add_extra(self, name: str, value: float) -> None:
        self.extras[name] += value

    def elapsed(self) -> float:
        return perf_counter() - self.start_time

    def checkpoint(self) -> tuple[dict[str, float], dict[str, int], dict[str, float]]:
        time_delta = {
            key: value - self._last_times.get(key, 0.0)
            for key, value in self.times.items()
        }
        count_delta = {
            key: value - self._last_counts.get(key, 0)
            for key, value in self.counts.items()
        }
        extra_delta = {
            key: value - self._last_extras.get(key, 0.0)
            for key, value in self.extras.items()
        }
        self._last_times = dict(self.times)
        self._last_counts = dict(self.counts)
        self._last_extras = dict(self.extras)
        return time_delta, count_delta, extra_delta

    def format_checkpoint(self, header: str) -> list[str]:
        time_delta, count_delta, extra_delta = self.checkpoint()
        lines = [header]

        def fmt_entry(name: str) -> str | None:
            dt = time_delta.get(name, 0.0)
            if dt <= 0.0:
                return None
            calls = count_delta.get(f"{name}:calls", 0)
            suffix = f" calls={calls}" if calls else ""
            return f"  {name}: {dt:.3f}s{suffix}"

        ordered = [
            "score_move",
            "exact_data_delta",
            "exact_data_delta_with_split",
            "source_after_removal_state",
            "cut_cost_delta_for_move",
            "data_term_from_parts",
            "apply_move",
            "split_cluster_if_disconnected",
            "score_cluster_merge",
            "apply_cluster_merge",
            "greedy_pass",
            "merge_sweep",
            "louvain_round",
        ]
        for name in ordered:
            entry = fmt_entry(name)
            if entry is not None:
                lines.append(entry)

        split_events = count_delta.get("split_events", 0)
        split_components = extra_delta.get("split_components_created", 0.0)
        if split_events or split_components:
            lines.append(
                f"  split_events={split_events} new_components={int(split_components)}"
            )

        accepted_moves = count_delta.get("accepted_moves", 0)
        if accepted_moves:
            lines.append(f"  accepted_moves={accepted_moves}")
        merge_accepts = count_delta.get("accepted_cluster_merges", 0)
        if merge_accepts:
            lines.append(f"  accepted_cluster_merges={merge_accepts}")

        return lines

