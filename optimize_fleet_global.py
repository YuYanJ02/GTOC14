#!/usr/bin/env python3
"""Solve the accumulated CTOC14 route pool as one global set-cover MILP.

Each cached route is already a dynamically feasible, independently scalable
spacecraft trajectory.  The binary program jointly chooses routes and optional
one-point miss penalties, exactly matching the competition objective for this
fixed route pool.  A validated incumbent is retained whenever the time-limited
MILP does not improve it.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ctoc14_core import write_submission
from search_low_mass_global import (
    base_full_route_options,
    base_route_options,
    deduplicate_options,
    load_option_cache,
    remove_dominated_options,
    renumber_rows,
    select_fleet,
)
from validate_ctoc14 import check


def route_objective(options) -> tuple[float, set[int]]:
    covered = set().union(*(option.covered for option in options)) if options else set()
    return float(sum(option.cost for option in options) + 300 - len(covered)), covered


def run(args: argparse.Namespace) -> dict:
    baseline = check(args.base)
    if not baseline["valid"]:
        raise RuntimeError(f"invalid base fleet: {baseline['errors']}")

    incumbent = base_full_route_options(args.base)
    incumbent_j, incumbent_covered = route_objective(incumbent)
    if abs(incumbent_j - baseline["J"]) > 2e-7:
        raise RuntimeError(
            f"base reconstruction mismatch: {incumbent_j} vs {baseline['J']}"
        )

    universe = []
    universe.extend(load_option_cache(args.route_cache))
    universe.extend(base_route_options(args.base))
    if args.original.exists() and args.original.resolve() != args.base.resolve():
        universe.extend(base_route_options(args.original))
    deduplicated = deduplicate_options(universe)
    nondominated = remove_dominated_options(deduplicated)
    print(
        f"global MILP: incumbent J={incumbent_j:.12f}, "
        f"deduplicated={len(deduplicated)}, nondominated={len(nondominated)}",
        flush=True,
    )

    selected, covered, selected_j = select_fleet(
        nondominated,
        time_limit=args.time_limit,
        mip_rel_gap=args.mip_gap,
    )
    improved = selected_j < incumbent_j - 1e-9
    if not improved:
        selected = incumbent
        covered = incumbent_covered
        selected_j = incumbent_j

    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_submission(
        args.output,
        renumber_rows(selected),
        comments=[
            "CTOC14 full route-pool global set-cover MILP",
            f"routes={len(selected)} covered={len(covered)} objective={selected_j:.12f}",
            f"columns={len(nondominated)} time_limit={args.time_limit:g}s",
        ],
    )
    validation = check(args.output)
    if not validation["valid"]:
        raise RuntimeError(f"global fleet failed validation: {validation['errors']}")

    summary = {
        "base_J": baseline["J"],
        "candidate_J": validation["J"],
        "improved": bool(validation["J"] < baseline["J"] - 1e-9),
        "valid": bool(validation["valid"]),
        "spacecraft": int(validation["spacecraft"]),
        "covered": int(validation["covered"]),
        "missing": sorted(set(range(1, 301)) - set(validation["covered_ids"])),
        "deduplicated_columns": len(deduplicated),
        "nondominated_columns": len(nondominated),
        "selected_routes": [
            {
                "source": option.source,
                "targets": len(option.covered),
                "cost": option.cost,
                "initial_mass_kg": float(option.rows[0].state[6]),
                "covered": sorted(option.covered),
            }
            for option in selected
        ],
        "worst": validation["worst"],
        "errors": validation["errors"],
        "output": str(args.output),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {key: value for key, value in summary.items() if key != "selected_routes"},
            indent=2,
        ),
        flush=True,
    )
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
        "--output", type=Path, default=Path("output/candidate_fleet_global.txt")
    )
    parser.add_argument(
        "--report", type=Path, default=Path("tmp/fleet_global_report.json")
    )
    parser.add_argument("--time-limit", type=float, default=300.0)
    parser.add_argument("--mip-gap", type=float, default=0.001)
    return parser


def main() -> None:
    args = make_parser().parse_args()
    if args.time_limit <= 0.0 or not 0.0 <= args.mip_gap < 1.0:
        raise SystemExit("invalid time limit or MIP gap")
    summary = run(args)
    raise SystemExit(0 if summary["valid"] else 1)


if __name__ == "__main__":
    main()
