#!/usr/bin/env python3
"""Extend a CTOC14 submission with a direct Earth-launch spacecraft.

Unlike :mod:`extend_ctoc14_fleet`, this builder does not force the published
ballistic Earth-to-174 prefix.  It may launch at any supplied mission epoch,
optionally optimize the allowed launch excess velocity, and then use the same
independently validated two-sample low-thrust legs.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares, minimize

from ctoc14_core import (
    AU_KM,
    DAY_S,
    MDRY_KG,
    EventRow,
    asteroid_state,
    bounded_vector,
    coast,
    earth_state,
    inverse_bounded_vector,
    load_asteroids,
    propagate_leg,
    write_submission,
)
from extend_ctoc14_fleet import _optimize_task, load_rows
from solve_ctoc14 import (
    CONTROL_LIMIT,
    Candidate,
    _linear_thrust_fuel_kg,
    append_leg_rows,
    candidate_list,
    initial_controls,
)
from validate_ctoc14 import check


VINF_LIMIT_KMS = 3.99


@dataclass
class OptimizedLaunchLeg:
    """First leg plus the launch state selected by its optimizer."""

    leg: object
    launch_state: np.ndarray
    flyby_km: float
    fuel_kg: float
    objective: float
    vinf_kms: float


def optimize_launch_candidate(
    launch_time_days: float,
    initial_mass: float,
    candidate: Candidate,
    asteroids: dict[int, np.ndarray],
    *,
    max_nfev: int = 90,
) -> OptimizedLaunchLeg | None:
    """Optimize launch excess velocity and the first two-sample thrust arc."""
    base_state = np.r_[earth_state(launch_time_days), initial_mass]
    target = asteroid_state(
        asteroids, candidate.asteroid_id, candidate.encounter_time
    )
    duration = candidate.encounter_time - launch_time_days

    ballistic = coast(
        base_state, launch_time_days, candidate.encounter_time, rtol=3e-9
    )
    velocity_guess_kms = (
        (target[:3] - ballistic[:3]) / duration * AU_KM / DAY_S
    )
    guess_norm = float(np.linalg.norm(velocity_guess_kms))
    if guess_norm > 3.5:
        velocity_guess_kms *= 3.5 / guess_norm
    raw_velocity = inverse_bounded_vector(velocity_guess_kms, VINF_LIMIT_KMS)
    guessed_state = np.array(base_state, copy=True)
    guessed_state[3:6] += velocity_guess_kms * DAY_S / AU_KM
    x0 = np.r_[
        raw_velocity,
        initial_controls(
            guessed_state, launch_time_days, candidate, asteroids
        ),
    ]

    def unpack(raw: np.ndarray):
        velocity_kms = bounded_vector(raw[:3], VINF_LIMIT_KMS)
        launch_state = np.array(base_state, copy=True)
        launch_state[3:6] += velocity_kms * DAY_S / AU_KM
        force_start = bounded_vector(raw[3:6], CONTROL_LIMIT)
        force_end = bounded_vector(raw[6:9], CONTROL_LIMIT)
        return launch_state, velocity_kms, force_start, force_end

    def residual(raw: np.ndarray, rtol: float) -> np.ndarray:
        launch_state, _velocity, force_start, force_end = unpack(raw)
        leg = propagate_leg(
            launch_state,
            launch_time_days,
            candidate.asteroid_id,
            candidate.encounter_time,
            force_start,
            force_end,
            rtol=rtol,
        )
        return (leg.state_encounter[:3] - target[:3]) * AU_KM / 1000.0

    solution = least_squares(
        lambda raw: residual(raw, 2e-8),
        x0,
        method="trf",
        max_nfev=max_nfev,
        xtol=2e-10,
        ftol=2e-10,
        gtol=2e-10,
    )
    search_miss_km = float(np.linalg.norm(residual(solution.x, 2e-8)) * 1000.0)
    if search_miss_km > 80.0:
        return None

    refined = least_squares(
        lambda raw: residual(raw, 2e-12),
        solution.x,
        method="trf",
        max_nfev=55,
        xtol=2e-12,
        ftol=2e-12,
        gtol=2e-12,
    )
    launch_state, velocity_kms, force_start, force_end = unpack(refined.x)
    leg = propagate_leg(
        launch_state,
        launch_time_days,
        candidate.asteroid_id,
        candidate.encounter_time,
        force_start,
        force_end,
        rtol=8e-13,
    )
    flyby_km = float(
        np.linalg.norm(leg.state_encounter[:3] - target[:3]) * AU_KM
    )
    fuel_kg = float(initial_mass - leg.state_separator[6])
    if flyby_km > 0.25 or leg.state_separator[6] < MDRY_KG + 0.1:
        return None
    objective = fuel_kg + 0.025 * duration
    return OptimizedLaunchLeg(
        leg,
        launch_state,
        flyby_km,
        fuel_kg,
        objective,
        float(np.linalg.norm(velocity_kms)),
    )


def refine_launch_candidate_fuel(
    launch_time_days: float,
    initial_mass: float,
    candidate: Candidate,
    asteroids: dict[int, np.ndarray],
    seed: OptimizedLaunchLeg,
    *,
    maxiter: int = 60,
) -> OptimizedLaunchLeg:
    """Use launch-energy freedom to reduce first-leg propellant consumption."""
    base_state = np.r_[earth_state(launch_time_days), initial_mass]
    target = asteroid_state(
        asteroids, candidate.asteroid_id, candidate.encounter_time
    )
    thrust_days = candidate.encounter_time - launch_time_days - 0.1 - 1.0 / DAY_S
    seed_vinf = (
        seed.launch_state[3:6] - base_state[3:6]
    ) * AU_KM / DAY_S
    raw0 = np.r_[
        inverse_bounded_vector(seed_vinf, VINF_LIMIT_KMS),
        inverse_bounded_vector(seed.leg.force_start, CONTROL_LIMIT),
        inverse_bounded_vector(seed.leg.force_end, CONTROL_LIMIT),
    ]
    cache_raw: np.ndarray | None = None
    cache_leg = None
    cache_launch = None

    def unpack(raw: np.ndarray):
        velocity_kms = bounded_vector(raw[:3], VINF_LIMIT_KMS)
        launch_state = np.array(base_state, copy=True)
        launch_state[3:6] += velocity_kms * DAY_S / AU_KM
        force_start = bounded_vector(raw[3:6], CONTROL_LIMIT)
        force_end = bounded_vector(raw[6:9], CONTROL_LIMIT)
        return launch_state, velocity_kms, force_start, force_end

    def evaluate(raw: np.ndarray):
        nonlocal cache_raw, cache_leg, cache_launch
        if (
            cache_raw is None
            or cache_leg is None
            or not np.array_equal(raw, cache_raw)
        ):
            launch_state, _velocity, force_start, force_end = unpack(raw)
            cache_leg = propagate_leg(
                launch_state,
                launch_time_days,
                candidate.asteroid_id,
                candidate.encounter_time,
                force_start,
                force_end,
                rtol=3e-10,
            )
            cache_launch = launch_state
            cache_raw = np.array(raw, copy=True)
        return cache_leg, cache_launch

    def position_constraint(raw: np.ndarray) -> np.ndarray:
        leg, _launch_state = evaluate(raw)
        return (leg.state_encounter[:3] - target[:3]) * AU_KM / 1000.0

    def fuel_objective(raw: np.ndarray) -> float:
        _launch_state, _velocity, force_start, force_end = unpack(raw)
        return _linear_thrust_fuel_kg(
            force_start, force_end, thrust_days
        ) / 1000.0

    result = minimize(
        fuel_objective,
        raw0,
        method="SLSQP",
        constraints={"type": "eq", "fun": position_constraint},
        options={"maxiter": maxiter, "ftol": 2e-10, "disp": False},
    )
    corrected = least_squares(
        lambda raw: (
            propagate_leg(
                unpack(raw)[0],
                launch_time_days,
                candidate.asteroid_id,
                candidate.encounter_time,
                unpack(raw)[2],
                unpack(raw)[3],
                rtol=2e-12,
            ).state_encounter[:3]
            - target[:3]
        )
        * AU_KM
        / 1000.0,
        result.x,
        method="trf",
        max_nfev=40,
        xtol=2e-12,
        ftol=2e-12,
        gtol=2e-12,
    )
    launch_state, velocity_kms, force_start, force_end = unpack(corrected.x)
    leg = propagate_leg(
        launch_state,
        launch_time_days,
        candidate.asteroid_id,
        candidate.encounter_time,
        force_start,
        force_end,
        rtol=8e-13,
    )
    flyby_km = float(
        np.linalg.norm(leg.state_encounter[:3] - target[:3]) * AU_KM
    )
    fuel_kg = float(initial_mass - leg.state_separator[6])
    if (
        flyby_km > 0.25
        or leg.state_separator[6] < MDRY_KG + 0.1
        or fuel_kg > seed.fuel_kg + 1e-7
    ):
        return seed
    duration = candidate.encounter_time - launch_time_days
    return OptimizedLaunchLeg(
        leg,
        launch_state,
        flyby_km,
        fuel_kg,
        fuel_kg + 0.025 * duration,
        float(np.linalg.norm(velocity_kms)),
    )


def _optimize_launch_task(payload):
    launch_time_days, initial_mass, candidate, asteroids, max_nfev = payload
    try:
        return optimize_launch_candidate(
            launch_time_days,
            initial_mass,
            candidate,
            asteroids,
            max_nfev=max_nfev,
        )
    except (RuntimeError, ValueError, FloatingPointError):
        return None


def build_direct_spacecraft(
    sc_id: int,
    excluded: set[int],
    *,
    launch_time_days: float,
    initial_mass: float,
    max_new_targets: int,
    stop_reserve_kg: float = 5.0,
    workers: int = 1,
    optimize_vinf: bool = False,
    first_target: int | None = None,
    second_target: int | None = None,
    long_minimum_days: float = 300.0,
    long_horizon_days: float = 1200.0,
) -> tuple[list[EventRow], set[int]]:
    """Build one greedy route beginning directly at Earth.

    Returning an empty row list means that no feasible first encounter was
    found and therefore no spacecraft should be added to the submission.
    """
    if workers < 1:
        raise ValueError("workers must be positive")

    asteroids = load_asteroids()
    state = np.r_[earth_state(launch_time_days), initial_mass]
    selected_launch_state = np.array(state, copy=True)
    current_time = launch_time_days
    visited_for_search = set(excluded)
    new_targets: set[int] = set()
    legs = []
    executor = ProcessPoolExecutor(max_workers=workers) if workers > 1 else None

    def evaluate_candidates(candidates, *, long_search: bool = False):
        feasible = []
        max_nfev = 85 if long_search else 65
        marker = "L" if long_search else " "
        batch_size = workers if executor is not None else 1
        for batch_start in range(0, len(candidates), batch_size):
            batch = candidates[batch_start : batch_start + batch_size]
            if optimize_vinf and not legs:
                task = _optimize_launch_task
                payloads = [
                    (
                        launch_time_days,
                        initial_mass,
                        candidate,
                        asteroids,
                        max_nfev,
                    )
                    for candidate in batch
                ]
            else:
                task = _optimize_task
                payloads = [
                    (state, current_time, candidate, asteroids, max_nfev)
                    for candidate in batch
                ]
            if executor is None:
                results = [task(payloads[0])]
            else:
                results = list(executor.map(task, payloads))
            for offset, (candidate, optimized) in enumerate(zip(batch, results), 1):
                index = batch_start + offset
                if optimized is None:
                    status = "fail"
                else:
                    vinf = getattr(optimized, "vinf_kms", None)
                    vinf_text = "" if vinf is None else f" vinf={vinf:.3f}km/s"
                    status = (
                        f"ok fuel={optimized.fuel_kg:.2f}kg "
                        f"miss={optimized.flyby_km:.4f}km{vinf_text}"
                    )
                print(
                    f"  {marker}{index:02d} id={candidate.asteroid_id:3d} "
                    f"day={candidate.encounter_time:8.3f} "
                    f"d0={candidate.distance_au:.4f}AU {status}",
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
            search_asteroids = asteroids
            if len(legs) == 1 and second_target is not None:
                search_asteroids = {second_target: asteroids[second_target]}
            candidates = candidate_list(
                state,
                current_time,
                search_asteroids,
                visited_for_search,
                minimum_days=100.0 if not legs else 80.0,
                maximum_days=600.0 if not legs else 440.0,
                step_days=25.0 if not legs else 30.0,
                keep=28 if not legs else 18,
            )
            if not legs and first_target is not None:
                candidates = [
                    item for item in candidates
                    if item.asteroid_id == first_target
                ]
            print(
                f"SC {sc_id} direct leg {len(legs) + 1}: "
                f"day={current_time:.3f} mass={state[6]:.3f} "
                f"new={len(new_targets)}",
                flush=True,
            )
            feasible = evaluate_candidates(candidates)
            if not feasible:
                print(
                    f"  retrying with {long_horizon_days:g}-day candidate horizon",
                    flush=True,
                )
                candidates = candidate_list(
                    state,
                    current_time,
                    search_asteroids,
                    visited_for_search,
                    minimum_days=long_minimum_days,
                    maximum_days=long_horizon_days,
                    step_days=40.0,
                    keep=48,
                )
                if not legs and first_target is not None:
                    candidates = [
                        item for item in candidates
                        if item.asteroid_id == first_target
                    ]
                feasible = evaluate_candidates(candidates, long_search=True)
            if not feasible:
                break

            chosen = min(feasible, key=lambda item: item.objective)
            if not legs and isinstance(chosen, OptimizedLaunchLeg):
                selected_launch_state = np.array(chosen.launch_state, copy=True)
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

    if not legs:
        return [], set()

    rows = [
        EventRow(
            sc_id,
            0,
            launch_time_days,
            selected_launch_state,
            np.zeros(3),
            0,
        )
    ]
    for index, optimized in enumerate(legs):
        before = len(rows)
        append_leg_rows(rows, optimized.leg, is_last=index == len(legs) - 1)
        for row in rows[before:]:
            row.sc_id = sc_id
    return rows, new_targets


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("base", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--launch-day", type=float, required=True)
    parser.add_argument("--initial-mass", type=float, default=2000.0)
    parser.add_argument("--max-new-targets", type=int, default=20)
    parser.add_argument("--stop-reserve-kg", type=float, default=20.0)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--optimize-vinf",
        action="store_true",
        help="optimize first-leg launch excess velocity up to 3.99 km/s",
    )
    parser.add_argument(
        "--first-target",
        type=int,
        help="restrict the first-leg search to this asteroid ID",
    )
    parser.add_argument(
        "--second-target",
        type=int,
        help="restrict the second-leg search to this asteroid ID",
    )
    parser.add_argument(
        "--long-minimum-days",
        type=float,
        default=300.0,
        help="minimum duration considered by the fallback leg search",
    )
    parser.add_argument(
        "--long-horizon-days",
        type=float,
        default=1200.0,
        help="maximum duration considered by the fallback leg search",
    )
    args = parser.parse_args()

    before = check(args.base)
    if not before["valid"]:
        raise SystemExit("base submission is not valid")
    if not (0.0 <= args.launch_day < 5478.75):
        parser.error("launch day must be inside the mission window")
    if not (MDRY_KG <= args.initial_mass <= 2000.0):
        parser.error("initial mass must be in [600, 2000] kg")

    rows = load_rows(args.base)
    excluded = set(before["covered_ids"])
    sc_id = before["spacecraft"] + 1
    new_rows, new_targets = build_direct_spacecraft(
        sc_id,
        excluded,
        launch_time_days=args.launch_day,
        initial_mass=args.initial_mass,
        max_new_targets=args.max_new_targets,
        stop_reserve_kg=args.stop_reserve_kg,
        workers=args.workers,
        optimize_vinf=args.optimize_vinf,
        first_target=args.first_target,
        second_target=args.second_target,
        long_minimum_days=args.long_minimum_days,
        long_horizon_days=args.long_horizon_days,
    )
    if not new_rows:
        print(json.dumps({"added": False, "new_targets": []}, indent=2))
        raise SystemExit(2)

    rows.extend(new_rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    all_covered = excluded | new_targets
    initial_masses = [row.state[6] for row in rows if row.event == 0]
    cost = 300 - len(all_covered)
    for mass in initial_masses:
        x = (mass - MDRY_KG) / 1400.0
        cost += 1.0 + x + x * x
    write_submission(
        args.output,
        rows,
        comments=[
            "CTOC14 multi-spacecraft submission with a direct Earth launch",
            f"spacecraft={sc_id} covered={len(all_covered)} J={cost:.12f}",
            f"direct_launch_day={args.launch_day:.9f}",
        ],
    )
    after = check(args.output)
    after["new_targets"] = sorted(new_targets)
    print(json.dumps(after, indent=2), flush=True)
    raise SystemExit(0 if after["valid"] else 1)


if __name__ == "__main__":
    main()
