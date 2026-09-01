#!/usr/bin/env python3
"""Build a valid, deterministic CTOC14 submission with greedy low-thrust legs.

The first encounter (asteroid 174) is the ballistic example disclosed in the
problem statement.  Later legs use two thrust samples, hence the official
interpolation is exactly linear and cannot overshoot the 0.5 N bound.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import sys
import time

DEPS = Path(__file__).resolve().parent / ".deps"
if DEPS.exists():
    sys.path.insert(0, str(DEPS))

import numpy as np
from scipy.optimize import least_squares, minimize

from ctoc14_core import (
    AU_KM,
    DAY_S,
    MDOT,
    MDRY_KG,
    MISSION_DAYS,
    EventRow,
    LegResult,
    asteroid_state,
    bounded_vector,
    coast,
    earth_state,
    inverse_bounded_vector,
    load_asteroids,
    propagate_leg,
    write_submission,
)


BALLISTIC_TIME_DAYS = 1.0008479152e7 / DAY_S
BALLISTIC_POSITION_KM = np.array(
    [-1.9500328647e7, 1.4581848347e8, -7.6372048832e3]
)
BALLISTIC_VELOCITY_KMS = np.array(
    [-2.9389477847e1, -4.3166912970e0, -3.9417153392e0]
)
CONTROL_LIMIT = 0.49


@dataclass
class Candidate:
    asteroid_id: int
    encounter_time: float
    score: float
    distance_au: float


@dataclass
class OptimizedLeg:
    leg: LegResult
    raw_controls: np.ndarray
    flyby_km: float
    fuel_kg: float
    objective: float


def _linear_thrust_fuel_kg(
    force_start: np.ndarray,
    force_end: np.ndarray,
    duration_days: float,
) -> float:
    """Integrate the propellant used by one linearly interpolated thrust arc.

    A fixed Gauss-Legendre rule keeps this objective cheap and smooth enough
    for the constrained fuel-refinement step.  Final acceptance still uses a
    high-accuracy dynamics propagation, so this quadrature is only an
    optimization aid.
    """
    # A high-order rule remains inexpensive for a two-vector control profile
    # and stays accurate when the interpolated force passes close to zero,
    # where the Euclidean norm has a sharp change in derivative.
    nodes, weights = np.polynomial.legendre.leggauss(64)
    fractions = 0.5 * (nodes + 1.0)
    forces = (
        (1.0 - fractions[:, None]) * force_start[None, :]
        + fractions[:, None] * force_end[None, :]
    )
    mean_force = 0.5 * float(weights @ np.linalg.norm(forces, axis=1))
    return duration_days * MDOT * mean_force


def refine_candidate_fuel(
    state: np.ndarray,
    start_time: float,
    candidate: Candidate,
    asteroids: dict[int, np.ndarray],
    seed: OptimizedLeg,
    *,
    maxiter: int = 45,
) -> OptimizedLeg:
    """Minimize propellant while retaining the candidate encounter position.

    Position matching supplies three equality constraints for six thrust
    parameters.  The original shooting solve leaves those three remaining
    degrees of freedom largely unused; SLSQP exploits them to reduce the
    integral thrust without changing the target or encounter epoch.
    """
    target = asteroid_state(asteroids, candidate.asteroid_id, candidate.encounter_time)
    thrust_days = candidate.encounter_time - start_time - 0.1 - 1.0 / DAY_S
    cache_raw: np.ndarray | None = None
    cache_leg: LegResult | None = None

    def unpack(raw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return (
            bounded_vector(raw[:3], CONTROL_LIMIT),
            bounded_vector(raw[3:], CONTROL_LIMIT),
        )

    def evaluate(raw: np.ndarray, *, rtol: float = 3e-10) -> LegResult:
        nonlocal cache_raw, cache_leg
        if (
            cache_raw is None
            or cache_leg is None
            or not np.array_equal(raw, cache_raw)
        ):
            f0, f1 = unpack(raw)
            cache_leg = propagate_leg(
                state,
                start_time,
                candidate.asteroid_id,
                candidate.encounter_time,
                f0,
                f1,
                rtol=rtol,
            )
            cache_raw = np.array(raw, copy=True)
        return cache_leg

    def position_constraint(raw: np.ndarray) -> np.ndarray:
        leg = evaluate(raw)
        # Thousands of kilometres gives SLSQP derivatives a useful scale.
        return (leg.state_encounter[:3] - target[:3]) * AU_KM / 1000.0

    def fuel_objective(raw: np.ndarray) -> float:
        f0, f1 = unpack(raw)
        return _linear_thrust_fuel_kg(f0, f1, thrust_days) / 1000.0

    result = minimize(
        fuel_objective,
        np.array(seed.raw_controls, copy=True),
        method="SLSQP",
        constraints={"type": "eq", "fun": position_constraint},
        options={"maxiter": maxiter, "ftol": 2e-10, "disp": False},
    )

    # SLSQP can stop after a useful fuel step with a small residual.  Restore
    # the exact encounter using the same high-accuracy shooting formulation;
    # starting from the refined point normally requires only a few evaluations.
    corrected = least_squares(
        lambda raw: (
            propagate_leg(
                state,
                start_time,
                candidate.asteroid_id,
                candidate.encounter_time,
                *unpack(raw),
                rtol=2e-12,
            ).state_encounter[:3]
            - target[:3]
        )
        * AU_KM
        / 1000.0,
        result.x,
        method="trf",
        max_nfev=30,
        xtol=2e-12,
        ftol=2e-12,
        gtol=2e-12,
    )
    f0, f1 = unpack(corrected.x)
    leg = propagate_leg(
        state,
        start_time,
        candidate.asteroid_id,
        candidate.encounter_time,
        f0,
        f1,
        rtol=8e-13,
    )
    flyby_km = float(np.linalg.norm(leg.state_encounter[:3] - target[:3]) * AU_KM)
    fuel_kg = float(state[6] - leg.state_separator[6])
    if (
        flyby_km > 0.25
        or leg.state_separator[6] < MDRY_KG + 0.1
        or fuel_kg > seed.fuel_kg + 1e-7
    ):
        return seed
    duration = candidate.encounter_time - start_time
    return OptimizedLeg(
        leg,
        corrected.x,
        flyby_km,
        fuel_kg,
        fuel_kg + 0.025 * duration,
    )


def make_launch_state(initial_mass: float) -> np.ndarray:
    state = np.empty(7)
    state[:3] = BALLISTIC_POSITION_KM / AU_KM
    state[3:6] = BALLISTIC_VELOCITY_KMS * DAY_S / AU_KM
    state[6] = initial_mass
    launch_position_error = np.linalg.norm(state[:3] - earth_state(0.0)[:3]) * AU_KM
    if launch_position_error > 1.0:
        raise RuntimeError("published launch state no longer matches Earth ephemeris")
    return state


def ballistic_prefix(initial_mass: float) -> tuple[np.ndarray, float, list[EventRow]]:
    launch = make_launch_state(initial_mass)
    encounter = coast(launch, 0.0, BALLISTIC_TIME_DAYS, rtol=2e-12)
    separator_time = BALLISTIC_TIME_DAYS + 1.0 / DAY_S
    separator = coast(encounter, BALLISTIC_TIME_DAYS, separator_time, rtol=2e-12)
    rows = [
        EventRow(1, 0, 0.0, launch, np.zeros(3), 0),
        EventRow(1, 3, BALLISTIC_TIME_DAYS, encounter, np.zeros(3), 174),
        EventRow(1, 2, separator_time, separator, np.zeros(3), 0),
    ]
    return separator, separator_time, rows


def candidate_list(
    state: np.ndarray,
    start_time: float,
    asteroids: dict[int, np.ndarray],
    visited: set[int],
    *,
    minimum_days: float = 80.0,
    maximum_days: float = 440.0,
    step_days: float = 30.0,
    keep: int = 18,
) -> list[Candidate]:
    """Rank candidate endpoints by a displacement/time proxy."""
    candidates: list[Candidate] = []
    trial_state = np.array(state, copy=True)
    previous = start_time
    trial_times = start_time + np.arange(minimum_days, maximum_days + 0.1, step_days)
    for encounter_time in trial_times:
        if encounter_time >= MISSION_DAYS - 1.0:
            break
        trial_state = coast(trial_state, previous, float(encounter_time), rtol=3e-9)
        previous = float(encounter_time)
        duration = encounter_time - start_time
        local = []
        for asteroid_id in asteroids:
            if asteroid_id in visited:
                continue
            target = asteroid_state(asteroids, asteroid_id, float(encounter_time))
            distance = float(np.linalg.norm(trial_state[:3] - target[:3]))
            # Approximate fuel-like proxy, with a small duration penalty so a
            # long easy leg does not consume the entire mission window.
            score = distance / duration + 2.5e-5 * duration
            local.append((score, distance, asteroid_id))
        for score, distance, asteroid_id in sorted(local)[:3]:
            candidates.append(
                Candidate(asteroid_id, float(encounter_time), score, distance)
            )
    best_by_pair = {}
    for item in candidates:
        key = (item.asteroid_id, item.encounter_time)
        best_by_pair[key] = item
    return sorted(best_by_pair.values(), key=lambda item: item.score)[:keep]


def initial_controls(
    state: np.ndarray,
    start_time: float,
    candidate: Candidate,
    asteroids: dict[int, np.ndarray],
) -> np.ndarray:
    ballistic = coast(state, start_time, candidate.encounter_time, rtol=3e-9)
    target = asteroid_state(asteroids, candidate.asteroid_id, candidate.encounter_time)
    duration = candidate.encounter_time - start_time
    acceleration = 2.0 * (target[:3] - ballistic[:3]) / duration**2
    from ctoc14_core import THRUST_ACCEL

    force = acceleration * state[6] / THRUST_ACCEL
    force_norm = np.linalg.norm(force)
    if force_norm > 0.35:
        force *= 0.35 / force_norm
    raw = inverse_bounded_vector(force, CONTROL_LIMIT)
    return np.r_[raw, raw]


def optimize_candidate(
    state: np.ndarray,
    start_time: float,
    candidate: Candidate,
    asteroids: dict[int, np.ndarray],
    *,
    max_nfev: int = 55,
) -> OptimizedLeg | None:
    target = asteroid_state(asteroids, candidate.asteroid_id, candidate.encounter_time)
    x0 = initial_controls(state, start_time, candidate, asteroids)

    def residual(raw: np.ndarray, rtol: float) -> np.ndarray:
        f0 = bounded_vector(raw[:3], CONTROL_LIMIT)
        f1 = bounded_vector(raw[3:], CONTROL_LIMIT)
        leg = propagate_leg(
            state,
            start_time,
            candidate.asteroid_id,
            candidate.encounter_time,
            f0,
            f1,
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

    # The loose integration used during the search can move the endpoint by a
    # few km.  Refine with the same tolerance used for final file generation.
    refined = least_squares(
        lambda raw: residual(raw, 2e-12),
        solution.x,
        method="trf",
        max_nfev=45,
        xtol=2e-12,
        ftol=2e-12,
        gtol=2e-12,
    )
    f0 = bounded_vector(refined.x[:3], CONTROL_LIMIT)
    f1 = bounded_vector(refined.x[3:], CONTROL_LIMIT)
    leg = propagate_leg(
        state,
        start_time,
        candidate.asteroid_id,
        candidate.encounter_time,
        f0,
        f1,
        rtol=8e-13,
    )
    flyby_km = float(np.linalg.norm(leg.state_encounter[:3] - target[:3]) * AU_KM)
    fuel = float(state[6] - leg.state_separator[6])
    if flyby_km > 0.25 or leg.state_separator[6] < MDRY_KG + 0.1:
        return None
    duration = candidate.encounter_time - start_time
    # Prefer low fuel, but value mission time as well.
    objective = fuel + 0.025 * duration
    return OptimizedLeg(leg, refined.x, flyby_km, fuel, objective)


def append_leg_rows(rows: list[EventRow], leg: LegResult, is_last: bool = False) -> None:
    rows.extend(
        [
            EventRow(1, 1, leg.sample_start, leg.state_sample_start, leg.force_start, 0),
            EventRow(1, 1, leg.sample_end, leg.state_sample_end, leg.force_end, 0),
            EventRow(1, 3, leg.encounter_time, leg.state_encounter, np.zeros(3), leg.asteroid_id),
            EventRow(
                1,
                4 if is_last else 2,
                leg.separator_time,
                leg.state_separator,
                np.zeros(3),
                0,
            ),
        ]
    )


def build_solution(
    initial_mass: float,
    max_targets: int,
    output_path: Path,
    checkpoint_path: Path,
) -> dict:
    asteroids = load_asteroids()
    state, current_time, prefix_rows = ballistic_prefix(initial_mass)
    ballistic_distance = np.linalg.norm(
        state[:3] - asteroid_state(asteroids, 174, current_time)[:3]
    ) * AU_KM
    if ballistic_distance > 1000.0:
        raise RuntimeError("published ballistic flyby of asteroid 174 does not reproduce")

    visited = {174}
    legs: list[OptimizedLeg] = []
    started = time.time()
    while len(visited) < max_targets and current_time < MISSION_DAYS - 90.0:
        candidates = candidate_list(state, current_time, asteroids, visited)
        feasible: list[OptimizedLeg] = []
        print(
            f"search leg {len(legs) + 1}: day={current_time:.3f} "
            f"mass={state[6]:.3f} candidates={len(candidates)}",
            flush=True,
        )
        for index, candidate in enumerate(candidates, 1):
            result = optimize_candidate(state, current_time, candidate, asteroids)
            status = "fail" if result is None else (
                f"ok fuel={result.fuel_kg:.2f}kg miss={result.flyby_km:.4f}km"
            )
            print(
                f"  {index:02d} id={candidate.asteroid_id:3d} "
                f"day={candidate.encounter_time:8.3f} d0={candidate.distance_au:.4f}AU {status}",
                flush=True,
            )
            if result is not None:
                feasible.append(result)
            if len(feasible) >= 4:
                break
        if not feasible:
            print("no feasible continuation", flush=True)
            break
        chosen = min(feasible, key=lambda item: item.objective)
        legs.append(chosen)
        visited.add(chosen.leg.asteroid_id)
        state = chosen.leg.state_separator
        current_time = chosen.leg.separator_time
        checkpoint = {
            "initial_mass_kg": initial_mass,
            "visited": sorted(visited),
            "current_time_days": current_time,
            "current_mass_kg": float(state[6]),
            "legs": [
                {
                    "asteroid_id": item.leg.asteroid_id,
                    "encounter_time_days": item.leg.encounter_time,
                    "force_start_N": item.leg.force_start.tolist(),
                    "force_end_N": item.leg.force_end.tolist(),
                    "fuel_kg": item.fuel_kg,
                    "flyby_km": item.flyby_km,
                }
                for item in legs
            ],
        }
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint_path.write_text(json.dumps(checkpoint, indent=2) + "\n", encoding="utf-8")
        print(
            f"CHOSEN id={chosen.leg.asteroid_id} covered={len(visited)} "
            f"day={current_time:.3f} mass={state[6]:.3f}",
            flush=True,
        )

    rows = prefix_rows[:2]  # replace the prefix separator when there are no later legs
    if legs:
        rows.append(prefix_rows[2])
        for index, optimized in enumerate(legs):
            append_leg_rows(rows, optimized.leg, is_last=index == len(legs) - 1)
    else:
        final_time = BALLISTIC_TIME_DAYS + 1.0 / DAY_S
        final_state = coast(prefix_rows[1].state, BALLISTIC_TIME_DAYS, final_time, rtol=8e-13)
        rows.append(EventRow(1, 4, final_time, final_state, np.zeros(3), 0))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fuel_used = initial_mass - rows[-1].state[6]
    x = (initial_mass - MDRY_KG) / 1400.0
    cost = 1.0 + x + x * x + 300 - len(visited)
    write_submission(
        output_path,
        rows,
        comments=[
            "CTOC14 problem A submission generated by solve_ctoc14.py",
            f"covered={len(visited)} ids={','.join(map(str, sorted(visited)))}",
            f"initial_mass_kg={initial_mass:.10f} fuel_used_kg={fuel_used:.10f} J={cost:.12f}",
        ],
    )
    summary = {
        "validity_pending": True,
        "spacecraft": 1,
        "initial_mass_kg": initial_mass,
        "final_mass_kg": float(rows[-1].state[6]),
        "fuel_used_kg": float(fuel_used),
        "covered": len(visited),
        "covered_ids": sorted(visited),
        "J": cost,
        "elapsed_seconds": time.time() - started,
        "output": str(output_path),
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--initial-mass", type=float, default=1200.0)
    parser.add_argument("--max-targets", type=int, default=12)
    parser.add_argument(
        "--output", type=Path, default=Path("output/CTOC14_Result_TeamID.txt")
    )
    parser.add_argument(
        "--checkpoint", type=Path, default=Path("output/route_checkpoint.json")
    )
    args = parser.parse_args()
    if not (MDRY_KG <= args.initial_mass <= 2000.0):
        parser.error("initial mass must be in [600, 2000] kg")
    summary = build_solution(
        args.initial_mass, args.max_targets, args.output, args.checkpoint
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
