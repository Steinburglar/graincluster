"""Run normalized parameterized MAP graincluster jobs on SiC frame 0."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import numpy as np
from ase.io import read
from ase.io import write as ase_write
from ase.io.trajectory import Trajectory

from graphcluster.io.frame import Frame
from graincluster.features.binning import fit_bin_scheme_quantile
from graincluster.features.species import canonical_pair_key
from graincluster.graph.builder import build_edges
from graincluster.model.entropy import cluster_entropy
from graincluster.model.partition import OTHER_ID, partition_from_labels
from graincluster.optimizer.louvain import louvain_optimize


ROOT = Path("/n/holylabs/kozinsky_lab/Users/lsteinberger/systems/sic")
INPUT = ROOT / "data" / "trajectories" / "8katoms" / "19.lammpstrj"
OUTDIR = ROOT / "analysis" / "graincluster"


def collect_reference_edges(atoms, cutoff: float, pair_cutoffs: dict[str, float]) -> dict[str, np.ndarray]:
    """Collect reference pair distances using the same pair cutoffs as the graph."""
    from scipy.spatial import cKDTree

    pos = np.asarray(atoms.positions, dtype=float)
    symbols = list(atoms.get_chemical_symbols())
    cell = atoms.cell.array
    inv_cell = np.linalg.inv(cell)
    frac = pos @ inv_cell
    frac -= np.floor(frac)

    max_cutoff = max(cutoff, max(pair_cutoffs.values(), default=cutoff))
    n_atoms = len(pos)
    images: list[np.ndarray] = []
    image_idx: list[int] = []
    for s0 in (0, 1, -1):
        for s1 in (0, 1, -1):
            for s2 in (0, 1, -1):
                shift = np.array([s0, s1, s2], dtype=float)
                images.append((frac + shift) @ cell)
                image_idx.extend(range(n_atoms))

    images_arr = np.vstack(images)
    image_idx_arr = np.array(image_idx)
    wrapped_cart = frac @ cell
    tree = cKDTree(images_arr)
    query_tree = cKDTree(wrapped_cart)
    raw_pairs = query_tree.query_ball_tree(tree, max_cutoff)

    pair_dists: dict[str, list[float]] = {}
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
            pk = canonical_pair_key(symbols[i], symbols[j])
            if d < pair_cutoffs.get(pk, cutoff):
                pair_dists.setdefault(pk, []).append(d)
    return {pk: np.array(v) for pk, v in pair_dists.items()}


def color_codes(labels: np.ndarray) -> np.ndarray:
    return ((labels.astype(np.int64) * 2654435761) % 9973).astype(np.int32)


def _slug_float(value: float) -> str:
    text = f"{value:g}"
    return text.replace("-", "m").replace(".", "p")


def _prior_label(args: argparse.Namespace) -> str:
    labels: list[str] = []
    if args.tau_k is not None or args.cluster_count_prior_strength is not None:
        labels.append("K")
    if args.cut_prior_beta0 is not None:
        labels.append("cut")
    if not labels:
        return "legacy"
    return "plus".join(labels)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cutoff", type=float, default=3.6)
    parser.add_argument("--cc-cutoff", type=float, default=3.6)
    parser.add_argument("--frame-index", type=int, default=0)
    parser.add_argument("--beta", type=float, default=0.75)
    parser.add_argument("--gamma", type=float, default=0.0)
    parser.add_argument("--cluster-count-prior-mean", type=float, default=None)
    parser.add_argument("--cluster-count-prior-strength", type=float, default=None)
    parser.add_argument("--tau-k", type=float, default=None)
    parser.add_argument("--cut-prior-beta0", type=float, default=None)
    parser.add_argument("--sigma", type=float, default=1.0)
    parser.add_argument("--n-quantile-bins", type=int, default=50)
    parser.add_argument("--tau-species", type=float, default=0.0)
    parser.add_argument("--tau-edge", type=float, default=0.0)
    parser.add_argument("--max-rounds", type=int, default=20)
    parser.add_argument("--max-atom-passes", type=int, default=50)
    parser.add_argument("--tag", type=str, default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cutoff = args.cutoff
    pair_cutoffs = {"C-C": args.cc_cutoff}
    beta = args.beta
    gamma = args.gamma
    sigma = args.sigma
    n_quantile_bins = args.n_quantile_bins
    n_species = 2.0
    kappa_species = n_species * (10.0 ** args.tau_species)
    kappa_edge = float(n_quantile_bins) * (10.0 ** args.tau_edge)
    tau_species = args.tau_species
    tau_edge = args.tau_edge

    OUTDIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading {INPUT} frame={args.frame_index}")
    atoms = read(
        str(INPUT),
        format="lammps-dump-text",
        index=args.frame_index,
        specorder=["C", "Si"],
    )
    symbols = list(atoms.get_chemical_symbols())
    print(f"Atoms: {len(atoms)} {dict(Counter(symbols))}")

    pair_dists = collect_reference_edges(atoms, cutoff=cutoff, pair_cutoffs=pair_cutoffs)
    pair_dists = {pk: vals for pk, vals in pair_dists.items() if len(vals) >= 4}
    for pk, vals in sorted(pair_dists.items()):
        print(f"Reference {pk}: {len(vals)}")

    bin_scheme = fit_bin_scheme_quantile(pair_dists, n_bins=n_quantile_bins)
    for pk in sorted(bin_scheme.pair_types):
        scheme = bin_scheme.schemes[pk]
        print(f"Bins {pk}: {scheme.n_bins} [{scheme.range_lo:.3f}, {scheme.range_hi:.3f}]")

    frame = Frame(
        index=args.frame_index,
        positions=atoms.positions,
        box=atoms.cell.array,
        chemical_symbols=symbols,
        atom_types=symbols,
    )
    edges = build_edges(
        frame,
        cutoff=cutoff,
        bin_scheme=bin_scheme,
        sigma=sigma,
        pbc=(True, True, True),
        pair_cutoffs=pair_cutoffs,
    )
    print(f"Edges: {len(edges)}")

    labels = np.full(len(atoms), OTHER_ID, dtype=int)
    labels[np.array(symbols) == "C"] = 0
    partition = partition_from_labels(
        labels,
        edges,
        bin_scheme,
        atom_species=symbols,
        gamma=gamma,
        beta=beta,
        cluster_count_prior_mean=args.cluster_count_prior_mean,
        cluster_count_prior_strength=args.cluster_count_prior_strength,
        cluster_count_prior_tau=args.tau_k,
        cut_prior_beta0=args.cut_prior_beta0,
        kappa_species=kappa_species,
        kappa_edge=kappa_edge,
        parameter_estimator="constrained_map",
    )
    print(
        "Init speciesSiOther "
        f"K={partition.n_clusters()} obj={partition.objective():.4f} "
        f"priors={_prior_label(args)} "
        f"tau_k={(args.tau_k if args.tau_k is not None else 'NA')} "
        f"cut_beta0={(args.cut_prior_beta0 if args.cut_prior_beta0 is not None else 'NA')} "
        f"tau_species={tau_species:g} tau_edge={tau_edge:g} "
        f"kappa_species={kappa_species:g} kappa_edge={kappa_edge:g}"
    )

    result = louvain_optimize(
        partition,
        max_rounds=args.max_rounds,
        max_atom_passes=args.max_atom_passes,
    )
    print(
        f"Result rounds={result.n_rounds} atom_moves={result.n_atom_moves} "
        f"merges={result.n_cluster_merges}"
    )
    print(f"Objective {result.objective_initial:.4f} -> {result.objective_final:.4f}")
    print(f"Final K={partition.n_clusters()}")

    real_clusters = sorted(
        [(cid, len(c.atom_ids)) for cid, c in partition.clusters.items() if cid != OTHER_ID],
        key=lambda x: -x[1],
    )
    id_rank = {cid: rank for rank, (cid, _) in enumerate(real_clusters)}
    other_tag = len(real_clusters)

    lines = [
        "Label Size N_edges Entropy Species",
        f"other {len(partition.clusters[OTHER_ID].atom_ids)} {partition.clusters[OTHER_ID].N} "
        f"{cluster_entropy(partition.clusters[OTHER_ID], bin_scheme.total_categories(), alpha=0.5):.4f} "
        + " ".join(f"{sp}:{n}" for sp, n in sorted(Counter(symbols[i] for i in partition.clusters[OTHER_ID].atom_ids).items())),
    ]
    for rank, (cid, size) in enumerate(real_clusters[:50]):
        c = partition.clusters[cid]
        sp = Counter(symbols[i] for i in c.atom_ids)
        lines.append(
            f"{rank} {size} {c.N} "
            f"{cluster_entropy(c, bin_scheme.total_categories(), alpha=0.5):.4f} "
            + " ".join(f"{k}:{v}" for k, v in sorted(sp.items()))
        )

    stem = (
        f"19__f{args.frame_index}_normMAP_init_speciesSiOther"
        f"_prior{_prior_label(args)}"
        f"_tauS{_slug_float(tau_species)}"
        f"_tauE{_slug_float(tau_edge)}"
        f"_tauK{_slug_float(args.tau_k if args.tau_k is not None else 0.0)}"
        f"_cutB{_slug_float(args.cut_prior_beta0 if args.cut_prior_beta0 is not None else 0.0)}"
        f"_b{_slug_float(beta)}"
        f"_g{_slug_float(gamma)}"
        f"_c{_slug_float(cutoff)}"
        f"_C-C{_slug_float(args.cc_cutoff)}"
        "_q_CSi"
    )
    if args.tag:
        stem += f"_{args.tag}"
    summary = OUTDIR / f"{stem}_summary.txt"
    traj = OUTDIR / f"{stem}.traj"
    xyz = OUTDIR / f"{stem}.extxyz"
    summary.write_text("\n".join(lines) + "\n")

    boundary_set: set[int] = set()
    for e in edges:
        if partition.atom_labels[e.i] != partition.atom_labels[e.j]:
            boundary_set.add(e.i)
            boundary_set.add(e.j)

    cluster_id_int = np.array(
        [
            other_tag if int(partition.atom_labels[i]) == OTHER_ID else id_rank[int(partition.atom_labels[i])]
            for i in range(len(atoms))
        ],
        dtype=int,
    )
    atoms.new_array("cluster_label", cluster_id_int)
    atoms.new_array("cluster_color_code", color_codes(cluster_id_int))
    atoms.new_array("is_boundary", np.array([1 if i in boundary_set else 0 for i in range(len(atoms))], dtype=int))
    atoms.set_tags(cluster_id_int)
    atoms.set_initial_charges(cluster_id_int.astype(float))
    atoms.wrap()

    with Trajectory(str(traj), "w") as out:
        out.write(atoms)
    ase_write(str(xyz), atoms)

    print(f"Boundary atoms: {len(boundary_set)}")
    print(f"Summary: {summary}")
    print(f"Trajectory: {traj}")
    print(f"ExtXYZ: {xyz}")
    print("\n".join(lines[:20]))


if __name__ == "__main__":
    main()
