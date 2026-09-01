#!/usr/bin/env python3
"""Add several disjoint greedy spacecraft and retain every valid checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ctoc14_core import MDRY_KG, EventRow, write_submission
from extend_ctoc14_fleet import build_spacecraft, load_rows
from optimize_submission import optimize_mass_scaling
from validate_ctoc14 import check


def mission_cost(rows: list[EventRow], covered: set[int]) -> float:
    cost = 300.0 - len(covered)
    for row in rows:
        if row.event != 0:
            continue
        x = (float(row.state[6]) - MDRY_KG) / 1400.0
        cost += 1.0 + x + x * x
    return cost


def write_checkpoint(path: Path, rows: list[EventRow], covered: set[int]) -> dict:
    spacecraft = max(row.sc_id for row in rows)
    cost = mission_cost(rows, covered)
    write_submission(
        path,
        rows,
        comments=[
            "CTOC14 iterative multi-spacecraft submission",
            f"spacecraft={spacecraft} covered={len(covered)} J={cost:.12f}",
            f"covered_ids={','.join(map(str, sorted(covered)))}",
        ],
    )
    result = check(path)
    if not result["valid"]:
        raise RuntimeError(json.dumps(result, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("base", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--add-spacecraft", type=int, default=1)
    parser.add_argument("--initial-mass", type=float, default=2000.0)
    parser.add_argument("--max-new-targets", type=int, default=20)
    parser.add_argument("--stop-reserve-kg", type=float, default=5.0)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--dry-margin-kg", type=float, default=0.1)
    args = parser.parse_args()
    if args.add_spacecraft < 1:
        parser.error("--add-spacecraft must be positive")

    base_result = check(args.base)
    if not base_result["valid"]:
        raise SystemExit("base submission is not valid")
    rows = load_rows(args.base)
    covered = set(base_result["covered_ids"])
    best_cost = float(base_result["J"])
    args.output.parent.mkdir(parents=True, exist_ok=True)

    latest_unscaled = args.output.with_name(args.output.stem + "_unscaled" + args.output.suffix)
    for _ in range(args.add_spacecraft):
        sc_id = max(row.sc_id for row in rows) + 1
        new_rows, new_targets = build_spacecraft(
            sc_id,
            covered,
            initial_mass=args.initial_mass,
            max_new_targets=args.max_new_targets,
            stop_reserve_kg=args.stop_reserve_kg,
            workers=args.workers,
        )
        if not new_targets:
            print(f"SC {sc_id} found no new targets; stopping", flush=True)
            break

        proposed_rows = rows + new_rows
        proposed_covered = covered | new_targets
        checkpoint = args.output.with_name(
            args.output.stem + f"_sc{sc_id}_unscaled" + args.output.suffix
        )
        write_checkpoint(checkpoint, proposed_rows, proposed_covered)
        scaled_checkpoint = checkpoint.with_name(
            checkpoint.stem.removesuffix("_unscaled")
            + "_scaled"
            + checkpoint.suffix
        )
        scaled_result = optimize_mass_scaling(
            checkpoint,
            scaled_checkpoint,
            dry_margin_kg=args.dry_margin_kg,
        )

        # Decide with the actual scaled objective.  A route adding only two or
        # three targets can still be worthwhile when its scaled launch mass is
        # low, so target count alone is not a valid stopping rule.
        if scaled_result["J"] >= best_cost - 1e-12:
            print(
                f"SC {sc_id} adds {len(new_targets)} targets but changes "
                f"J from {best_cost:.12f} to {scaled_result['J']:.12f}; "
                "retaining the previous fleet",
                flush=True,
            )
            break
        rows = proposed_rows
        covered = proposed_covered
        latest_unscaled = checkpoint
        best_cost = float(scaled_result["J"])
        print(json.dumps(scaled_result, indent=2), flush=True)

    if latest_unscaled == args.output.with_name(
        args.output.stem + "_unscaled" + args.output.suffix
    ):
        write_checkpoint(latest_unscaled, rows, covered)
    result = optimize_mass_scaling(
        latest_unscaled,
        args.output,
        dry_margin_kg=args.dry_margin_kg,
    )
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
