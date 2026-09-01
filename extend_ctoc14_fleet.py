#!/usr/bin/env python3
"""Extend a valid CTOC14 submission with a disjoint greedy spacecraft route."""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from ctoc14_core import DAY_S, MDRY_KG, EventRow, load_asteroids, write_submission
from solve_ctoc14 import (
    append_leg_rows,
    ballistic_prefix,
    candidate_list,
    optimize_candidate,
)
from validate_ctoc14 import check, parse


def _optimize_task(payload):
    """Process-pool entry point for one independent endpoint optimization."""
    state, current_time, candidate, asteroids, max_nfev = payload
    try:
        return optimize_candidate(
            state,
            current_time,
            candidate,
            asteroids,
            max_nfev=max_nfev,
        )
    except (RuntimeError, ValueError, FloatingPointError):
        # Some trial controls can drive an optimizer through a near-Sun
        # singularity.  That invalidates only this endpoint candidate, not the
        # rest of the spacecraft search (or its process pool).
        return None


def load_rows(path: Path) -> list[EventRow]:
    result = []
    for row in parse(path):
        result.append(
            EventRow(
                row["sc"],
                row["event"],
                row["time"],
                row["state"],
                row["force"],
                row["asteroid"],
            )
        )
    return result


def build_spacecraft(
    sc_id: int,
    excluded: set[int],
    *,
    initial_mass: float,
    max_new_targets: int,
    stop_reserve_kg: float = 5.0,
    workers: int = 1,
) -> tuple[list[EventRow], set[int]]:
    asteroids = load_asteroids()
    state, current_time, prefix = ballistic_prefix(initial_mass)
    for row in prefix:
        row.sc_id = sc_id
    visited_for_search = set(excluded)
    visited_for_search.add(174)
    new_targets: set[int] = set()
    if 174 not in excluded:
        new_targets.add(174)
    legs = []

    if workers < 1:
        raise ValueError("workers must be positive")
    executor = ProcessPoolExecutor(max_workers=workers) if workers > 1 else None

    def evaluate_candidates(candidates, *, long_search: bool = False):
        feasible = []
        max_nfev = 70 if long_search else 55
        prefix = "L" if long_search else " "
        batch_size = workers if executor is not None else 1
        for batch_start in range(0, len(candidates), batch_size):
            batch = candidates[batch_start : batch_start + batch_size]
            payloads = [
                (state, current_time, candidate, asteroids, max_nfev)
                for candidate in batch
            ]
            if executor is None:
                results = [_optimize_task(payloads[0])]
            else:
                results = list(executor.map(_optimize_task, payloads))
            for offset, (candidate, optimized) in enumerate(zip(batch, results), 1):
                index = batch_start + offset
                status = "fail" if optimized is None else (
                    f"ok fuel={optimized.fuel_kg:.2f}kg "
                    f"miss={optimized.flyby_km:.4f}km"
                )
                distance = "" if long_search else f" d0={candidate.distance_au:.4f}AU"
                print(
                    f"  {prefix}{index:02d} id={candidate.asteroid_id:3d} "
                    f"day={candidate.encounter_time:8.3f}{distance} {status}",
                    flush=True,
                )
                if optimized is not None:
                    feasible.append(optimized)
                if len(feasible) >= 4:
                    return feasible
        return feasible

    try:
        while (
            len(new_targets) < max_new_targets
            and state[6] > MDRY_KG + stop_reserve_kg
        ):
            candidates = candidate_list(
                state,
                current_time,
                asteroids,
                visited_for_search,
                keep=18,
            )
            print(
                f"SC {sc_id} leg {len(legs)+1}: day={current_time:.3f} "
                f"mass={state[6]:.3f} new={len(new_targets)}",
                flush=True,
            )
            feasible = evaluate_candidates(candidates)
            if not feasible:
                print("  retrying with 900-day candidate horizon", flush=True)
                candidates = candidate_list(
                    state,
                    current_time,
                    asteroids,
                    visited_for_search,
                    minimum_days=260.0,
                    maximum_days=900.0,
                    step_days=40.0,
                    keep=36,
                )
                feasible = evaluate_candidates(candidates, long_search=True)
            if not feasible:
                break
            chosen = min(feasible, key=lambda item: item.objective)
            legs.append(chosen)
            target_id = chosen.leg.asteroid_id
            new_targets.add(target_id)
            visited_for_search.add(target_id)
            state = chosen.leg.state_separator
            current_time = chosen.leg.separator_time
            print(
                f"CHOSEN SC={sc_id} id={target_id} new={len(new_targets)} "
                f"day={current_time:.3f} mass={state[6]:.3f}",
                flush=True,
            )
    finally:
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)

    rows = prefix[:2]
    if legs:
        rows.append(prefix[2])
        for index, optimized in enumerate(legs):
            before = len(rows)
            append_leg_rows(rows, optimized.leg, is_last=index == len(legs) - 1)
            for row in rows[before:]:
                row.sc_id = sc_id
    else:
        final = prefix[2]
        final.event = 4
        rows.append(final)
    return rows, new_targets


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("base", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--initial-mass", type=float, default=2000.0)
    parser.add_argument("--max-new-targets", type=int, default=17)
    parser.add_argument("--stop-reserve-kg", type=float, default=5.0)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()

    before = check(args.base)
    if not before["valid"]:
        raise SystemExit("base submission is not valid")
    rows = load_rows(args.base)
    excluded = set(before["covered_ids"])
    sc_id = before["spacecraft"] + 1
    new_rows, new_targets = build_spacecraft(
        sc_id,
        excluded,
        initial_mass=args.initial_mass,
        max_new_targets=args.max_new_targets,
        stop_reserve_kg=args.stop_reserve_kg,
        workers=args.workers,
    )
    rows.extend(new_rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    all_covered = excluded | new_targets
    initial_masses = [
        row.state[6] for row in rows if row.event == 0
    ]
    cost = 300 - len(all_covered)
    for mass in initial_masses:
        x = (mass - MDRY_KG) / 1400.0
        cost += 1.0 + x + x * x
    write_submission(
        args.output,
        rows,
        comments=[
            "CTOC14 problem A multi-spacecraft submission",
            f"spacecraft={sc_id} covered={len(all_covered)} J={cost:.12f}",
            f"covered_ids={','.join(map(str, sorted(all_covered)))}",
        ],
    )
    after = check(args.output)
    print(json.dumps(after, indent=2), flush=True)
    raise SystemExit(0 if after["valid"] else 1)


if __name__ == "__main__":
    main()
