#!/usr/bin/env python3
"""Large-neighbourhood fleet optimizer with a validated incumbent.

SciPy's high-level MILP interface cannot accept a warm start.  Solving the
entire accumulated CTOC14 route pool therefore often spends its time finding
an incumbent worse than the already validated fleet.  This optimizer keeps
that fleet explicitly, destroys a subset of routes, and solves only the
resulting residual set-cover problem.  Strict improvements are checkpointed.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
from typing import Sequence

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp

from ctoc14_core import write_submission
from search_low_mass_global import (
    RouteOption,
    base_full_route_options,
    base_route_options,
    deduplicate_options,
    load_option_cache,
    remove_dominated_options,
    renumber_rows,
)
from validate_ctoc14 import check


ALL_TARGETS = frozenset(range(1, 301))


def option_key(option: RouteOption) -> tuple[frozenset[int], float]:
    return option.covered, round(option.cost, 12)


def objective(options: Sequence[RouteOption]) -> tuple[float, set[int]]:
    covered = set().union(*(option.covered for option in options)) if options else set()
    return float(sum(option.cost for option in options) + 300 - len(covered)), covered


def prune_redundant(options: list[RouteOption]) -> list[RouteOption]:
    """Remove routes whose loss costs no more than their objective charge."""
    result = list(options)
    while True:
        counts = Counter(asteroid_id for option in result for asteroid_id in option.covered)
        removable = []
        for index, option in enumerate(result):
            unique_count = sum(counts[asteroid_id] == 1 for asteroid_id in option.covered)
            saving = option.cost - unique_count
            if saving > 1e-10:
                removable.append((saving, index))
        if not removable:
            return result
        _saving, index = max(removable)
        result.pop(index)


def shortlist_options(
    universe: Sequence[RouteOption],
    kept: Sequence[RouteOption],
    deficits: set[int],
    *,
    limit: int,
    per_target: int,
) -> list[RouteOption]:
    kept_keys = {option_key(option) for option in kept}
    scored = []
    by_target: dict[int, list[tuple[float, RouteOption]]] = defaultdict(list)
    for option in universe:
        if option_key(option) in kept_keys:
            continue
        gain = len(option.covered & deficits)
        if gain == 0:
            continue
        score = option.cost / gain
        scored.append((score, -gain, option.cost, option))
        for asteroid_id in option.covered & deficits:
            by_target[asteroid_id].append((score, option))

    chosen: dict[tuple[frozenset[int], float], RouteOption] = {}
    for _score, _negative_gain, _cost, option in sorted(
        scored, key=lambda item: (item[0], item[1], item[2])
    )[:limit]:
        chosen[option_key(option)] = option
    for asteroid_id in deficits:
        for _score, option in sorted(
            by_target.get(asteroid_id, ()), key=lambda item: (item[0], item[1].cost)
        )[:per_target]:
            chosen[option_key(option)] = option
    return list(chosen.values())


def solve_residual(
    candidates: Sequence[RouteOption],
    deficits: set[int],
    *,
    time_limit: float,
    mip_gap: float,
) -> list[RouteOption] | None:
    target_ids = sorted(deficits)
    route_count = len(candidates)
    target_count = len(target_ids)
    if target_count == 0:
        return []
    variable_count = route_count + target_count
    costs = np.r_[
        np.asarray([option.cost for option in candidates], dtype=float),
        np.ones(target_count),
    ]
    coverage = np.zeros((target_count, variable_count), dtype=float)
    target_row = {asteroid_id: index for index, asteroid_id in enumerate(target_ids)}
    for route_index, option in enumerate(candidates):
        for asteroid_id in option.covered & deficits:
            coverage[target_row[asteroid_id], route_index] = 1.0
    coverage[:, route_count:] = np.eye(target_count)
    result = milp(
        costs,
        integrality=np.ones(variable_count),
        bounds=Bounds(np.zeros(variable_count), np.ones(variable_count)),
        constraints=LinearConstraint(
            coverage,
            np.ones(target_count),
            np.full(target_count, np.inf),
        ),
        options={"time_limit": time_limit, "mip_rel_gap": mip_gap},
    )
    if result.x is None:
        return None
    return [
        option
        for index, option in enumerate(candidates)
        if result.x[index] > 0.5
    ]


def destroy_indices(
    incumbent: Sequence[RouteOption],
    random: np.random.Generator,
    size: int,
    iteration: int,
) -> np.ndarray:
    counts = Counter(asteroid_id for option in incumbent for asteroid_id in option.covered)
    pressure = []
    for option in incumbent:
        unique_count = sum(counts[asteroid_id] == 1 for asteroid_id in option.covered)
        overlap = len(option.covered) - unique_count
        # High-cost, overlapping routes are natural replacement candidates.
        pressure.append(option.cost + 0.08 * overlap - 0.05 * unique_count)
    weights = np.asarray(pressure, dtype=float)
    weights -= weights.min()
    weights += 0.05
    if iteration % 3 == 0:
        weights[:] = 1.0
    weights /= weights.sum()
    return random.choice(len(incumbent), size=size, replace=False, p=weights)


def write_checkpoint(
    path: Path,
    options: Sequence[RouteOption],
    iteration: int,
    score: float,
) -> dict:
    rows = renumber_rows(options)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_submission(
        path,
        rows,
        comments=[
            "CTOC14 route-pool large-neighbourhood fleet optimization",
            f"iteration={iteration} routes={len(options)} objective={score:.12f}",
        ],
    )
    validation = check(path)
    if not validation["valid"]:
        raise RuntimeError(f"checkpoint validation failed: {validation['errors']}")
    return validation


def run(args: argparse.Namespace) -> dict:
    baseline_validation = check(args.base)
    if not baseline_validation["valid"]:
        raise RuntimeError("base fleet is invalid")
    incumbent = base_full_route_options(args.base)
    incumbent = prune_redundant(incumbent)
    incumbent_score, incumbent_covered = objective(incumbent)
    if abs(incumbent_score - baseline_validation["J"]) > 2e-7:
        raise RuntimeError(
            f"base reconstruction mismatch: {incumbent_score} vs {baseline_validation['J']}"
        )

    universe = []
    universe.extend(load_option_cache(args.route_cache))
    universe.extend(base_route_options(args.base))
    if args.original.exists() and args.original.resolve() != args.base.resolve():
        universe.extend(base_route_options(args.original))
    universe = remove_dominated_options(deduplicate_options(universe))
    print(
        f"incumbent J={incumbent_score:.12f} routes={len(incumbent)} "
        f"covered={len(incumbent_covered)} universe={len(universe)}",
        flush=True,
    )

    random = np.random.default_rng(args.seed)
    history = []
    validation = baseline_validation
    for iteration in range(1, args.iterations + 1):
        maximum = min(args.destroy_max, len(incumbent) - 1)
        minimum = min(args.destroy_min, maximum)
        if maximum < 1:
            break
        size = int(random.integers(minimum, maximum + 1))
        indices = set(destroy_indices(incumbent, random, size, iteration))
        kept = [option for index, option in enumerate(incumbent) if index not in indices]
        kept_covered = set().union(*(option.covered for option in kept)) if kept else set()
        deficits = set(ALL_TARGETS - kept_covered)
        candidates = shortlist_options(
            universe,
            kept,
            deficits,
            limit=args.subproblem_limit,
            per_target=args.per_target,
        )
        replacements = solve_residual(
            candidates,
            deficits,
            time_limit=args.subproblem_time_limit,
            mip_gap=args.subproblem_mip_gap,
        )
        if replacements is None:
            continue
        trial = prune_redundant(kept + replacements)
        trial_score, trial_covered = objective(trial)
        accepted = bool(trial_score < incumbent_score - 1e-9)
        if accepted:
            old_score = incumbent_score
            incumbent = trial
            incumbent_score = trial_score
            incumbent_covered = trial_covered
            validation = write_checkpoint(
                args.output, incumbent, iteration, incumbent_score
            )
            print(
                f"IMPROVED iteration={iteration} {old_score:.12f} -> "
                f"{incumbent_score:.12f} routes={len(incumbent)} "
                f"covered={len(incumbent_covered)}",
                flush=True,
            )
        elif iteration % args.progress_every == 0:
            print(
                f"iteration={iteration}/{args.iterations} incumbent={incumbent_score:.12f} "
                f"destroy={size} candidates={len(candidates)} trial={trial_score:.12f}",
                flush=True,
            )
        history.append(
            {
                "iteration": iteration,
                "destroy": size,
                "candidates": len(candidates),
                "trial_J": trial_score,
                "accepted": accepted,
                "incumbent_J": incumbent_score,
                "routes": len(incumbent),
                "covered": len(incumbent_covered),
            }
        )

    if not args.output.exists() or incumbent_score >= baseline_validation["J"] - 1e-9:
        validation = write_checkpoint(args.output, incumbent, 0, incumbent_score)
    summary = {
        "base_J": baseline_validation["J"],
        "candidate_J": validation["J"],
        "improved": bool(validation["J"] < baseline_validation["J"] - 1e-9),
        "valid": bool(validation["valid"]),
        "spacecraft": int(validation["spacecraft"]),
        "covered": int(validation["covered"]),
        "missing": sorted(ALL_TARGETS - set(validation["covered_ids"])),
        "universe": len(universe),
        "accepted_moves": sum(item["accepted"] for item in history),
        "history": history,
        "worst": validation["worst"],
        "errors": validation["errors"],
        "output": str(args.output),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in summary.items() if key != "history"}, indent=2))
    return summary


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument(
        "--original", type=Path, default=Path("output/CTOC14_Result_TeamID.txt")
    )
    parser.add_argument(
        "--route-cache", type=Path, default=Path("tmp/global_route_pool.pkl")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("output/candidate_fleet_lns.txt")
    )
    parser.add_argument(
        "--report", type=Path, default=Path("tmp/fleet_lns_report.json")
    )
    parser.add_argument("--iterations", type=int, default=120)
    parser.add_argument("--destroy-min", type=int, default=4)
    parser.add_argument("--destroy-max", type=int, default=12)
    parser.add_argument("--subproblem-limit", type=int, default=320)
    parser.add_argument("--per-target", type=int, default=4)
    parser.add_argument("--subproblem-time-limit", type=float, default=2.0)
    parser.add_argument("--subproblem-mip-gap", type=float, default=0.005)
    parser.add_argument("--progress-every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    return parser


def main() -> None:
    args = make_parser().parse_args()
    if args.iterations < 1 or args.destroy_min < 1 or args.destroy_max < args.destroy_min:
        raise SystemExit("invalid iteration or destroy range")
    summary = run(args)
    raise SystemExit(0 if summary["valid"] else 1)


if __name__ == "__main__":
    main()
