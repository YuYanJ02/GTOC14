#!/usr/bin/env python3
"""Replace repeated Earth-to-174 prefixes with optimized direct departures.

For each selected spacecraft, the new first leg targets the complete state at
its first non-174 encounter.  Matching position, velocity, and mass lets every
later row remain unchanged, preserving the route and its full target coverage.
The remaining control freedom is used to minimize first-leg propellant.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares, minimize

from ctoc14_core import (
    AU_KM,
    DAY_S,
    MDRY_KG,
    M0_MAX_KG,
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
from extend_ctoc14_direct import (
    VINF_LIMIT_KMS,
    optimize_launch_candidate,
    refine_launch_candidate_fuel,
)
from extend_ctoc14_fleet import load_rows
from solve_ctoc14 import (
    CONTROL_LIMIT,
    Candidate,
    _linear_thrust_fuel_kg,
)
from validate_ctoc14 import check


@dataclass
class PrefixTask:
    sc_id: int
    old_rows: list[EventRow]
    target_row: EventRow


@dataclass
class PrefixResult:
    sc_id: int
    rows: list[EventRow] | None
    old_initial_mass: float
    new_initial_mass: float | None
    target_id: int
    position_error_km: float | None
    velocity_error_kms: float | None
    mass_error_kg: float | None
    message: str


PREFIX_VINF_LIMIT_KMS = 3.999999


def parse_spacecraft_spec(value: str) -> set[int]:
    selected: set[int] = set()
    for token in value.split(","):
        token = token.strip()
        if not token:
            continue
        if "-" in token:
            first, last = token.split("-", 1)
            selected.update(range(int(first), int(last) + 1))
        else:
            selected.add(int(token))
    return selected


def make_tasks(rows: list[EventRow], selected: set[int]) -> list[PrefixTask]:
    grouped: dict[int, list[EventRow]] = {}
    for row in rows:
        grouped.setdefault(row.sc_id, []).append(row)
    tasks = []
    for sc_id in sorted(selected):
        sc_rows = grouped[sc_id]
        target = next(
            row
            for row in sc_rows
            if row.event == 3 and row.asteroid_id != 174
        )
        tasks.append(PrefixTask(sc_id, sc_rows, target))
    return tasks


def optimize_prefix(task: PrefixTask, *, max_nfev: int, fuel_iterations: int) -> PrefixResult:
    asteroids = load_asteroids()
    launch_day = float(task.old_rows[0].time_days)
    encounter_day = float(task.target_row.time_days)
    target_id = task.target_row.asteroid_id
    target_state = np.array(task.target_row.state, copy=True)
    target_mass = float(target_state[6])
    candidate = Candidate(target_id, encounter_day, 0.0, 0.0)

    # A position-only direct solution supplies a much better seed than the
    # repeated-174 trajectory for this different control topology.
    position_seed = optimize_launch_candidate(
        launch_day,
        M0_MAX_KG,
        candidate,
        asteroids,
        max_nfev=140,
    )
    if position_seed is None:
        return PrefixResult(
            task.sc_id,
            None,
            float(task.old_rows[0].state[6]),
            None,
            target_id,
            None,
            None,
            None,
            "position-only direct seed failed",
        )
    position_seed = refine_launch_candidate_fuel(
        launch_day,
        M0_MAX_KG,
        candidate,
        asteroids,
        position_seed,
        maxiter=min(25, fuel_iterations),
    )
    seed_vinf = (
        position_seed.launch_state[3:6] - earth_state(launch_day)[3:6]
    ) * AU_KM / DAY_S
    raw0 = np.r_[
        inverse_bounded_vector(seed_vinf, VINF_LIMIT_KMS),
        inverse_bounded_vector(position_seed.leg.force_start, CONTROL_LIMIT),
        inverse_bounded_vector(position_seed.leg.force_end, CONTROL_LIMIT),
    ]
    thrust_days = encounter_day - launch_day - 0.1 - 1.0 / DAY_S

    def unpack(raw: np.ndarray):
        vinf = bounded_vector(raw[:3], VINF_LIMIT_KMS)
        f0 = bounded_vector(raw[3:6], CONTROL_LIMIT)
        f1 = bounded_vector(raw[6:9], CONTROL_LIMIT)
        fuel = _linear_thrust_fuel_kg(f0, f1, thrust_days)
        launch_mass = target_mass + fuel
        launch_state = np.r_[earth_state(launch_day), launch_mass]
        launch_state[3:6] += vinf * DAY_S / AU_KM
        return launch_state, vinf, f0, f1, fuel

    def propagate(raw: np.ndarray, rtol: float):
        launch_state, _vinf, f0, f1, _fuel = unpack(raw)
        return propagate_leg(
            launch_state,
            launch_day,
            target_id,
            encounter_day,
            f0,
            f1,
            rtol=rtol,
        )

    def state_residual(raw: np.ndarray, rtol: float) -> np.ndarray:
        leg = propagate(raw, rtol)
        position = (leg.state_encounter[:3] - target_state[:3]) * AU_KM / 10.0
        velocity = (
            (leg.state_encounter[3:6] - target_state[3:6])
            * AU_KM
            / DAY_S
            / 0.01
        )
        launch_mass = unpack(raw)[0][6]
        mass_limit = max(0.0, launch_mass - M0_MAX_KG)
        return np.r_[position, velocity, mass_limit]

    matched = least_squares(
        lambda raw: state_residual(raw, 3e-9),
        raw0,
        method="trf",
        max_nfev=max_nfev,
        xtol=2e-11,
        ftol=2e-11,
        gtol=2e-11,
    )

    # Once a full-state match exists, use its three null-space degrees of
    # freedom to reduce integral thrust while retaining position and velocity.
    def equality(raw: np.ndarray) -> np.ndarray:
        leg = propagate(raw, 3e-10)
        return np.r_[
            (leg.state_encounter[:3] - target_state[:3]) * AU_KM / 1000.0,
            (leg.state_encounter[3:6] - target_state[3:6])
            * AU_KM
            / DAY_S,
        ]

    def objective(raw: np.ndarray) -> float:
        return unpack(raw)[4] / 1000.0

    def mass_margin(raw: np.ndarray) -> float:
        return M0_MAX_KG - unpack(raw)[0][6]

    optimized = minimize(
        objective,
        matched.x,
        method="SLSQP",
        constraints=(
            {"type": "eq", "fun": equality},
            {"type": "ineq", "fun": mass_margin},
        ),
        options={"maxiter": fuel_iterations, "ftol": 2e-10, "disp": False},
    )
    corrected = least_squares(
        lambda raw: state_residual(raw, 2e-12),
        optimized.x,
        method="trf",
        max_nfev=80,
        xtol=2e-12,
        ftol=2e-12,
        gtol=2e-12,
    )
    launch_state, vinf, f0, f1, _fuel = unpack(corrected.x)
    leg = propagate(corrected.x, 8e-13)
    position_error = float(
        np.linalg.norm(leg.state_encounter[:3] - target_state[:3]) * AU_KM
    )
    velocity_error = float(
        np.linalg.norm(leg.state_encounter[3:6] - target_state[3:6])
        * AU_KM
        / DAY_S
    )
    mass_error = float(abs(leg.state_encounter[6] - target_mass))
    old_initial_mass = float(task.old_rows[0].state[6])
    if (
        position_error > 0.25
        or velocity_error > 2e-5
        or mass_error > 0.002
        or launch_state[6] > M0_MAX_KG + 1e-8
        or np.linalg.norm(vinf) > 4.0
    ):
        return optimize_coast_prefix(
            task,
            fuel_iterations=fuel_iterations,
            prefix_message=(
                "pure direct full-state match failed "
                f"(pos={position_error:.3g} km, vel={velocity_error:.3g} km/s); "
            ),
        )

    target_index = task.old_rows.index(task.target_row)
    rows = [
        EventRow(task.sc_id, 0, launch_day, launch_state, np.zeros(3), 0),
        EventRow(
            task.sc_id,
            1,
            leg.sample_start,
            leg.state_sample_start,
            f0,
            0,
        ),
        EventRow(
            task.sc_id,
            1,
            leg.sample_end,
            leg.state_sample_end,
            f1,
            0,
        ),
        EventRow(
            task.sc_id,
            3,
            encounter_day,
            leg.state_encounter,
            np.zeros(3),
            target_id,
        ),
    ]
    rows.extend(task.old_rows[target_index + 1 :])
    return PrefixResult(
        task.sc_id,
        rows,
        old_initial_mass,
        float(launch_state[6]),
        target_id,
        position_error,
        velocity_error,
        mass_error,
        f"vinf={np.linalg.norm(vinf):.6f} km/s",
    )


def optimize_coast_prefix(
    task: PrefixTask,
    *,
    fuel_iterations: int,
    prefix_message: str = "",
) -> PrefixResult:
    """Optimize a feasible coast-plus-thrust prefix without declaring 174."""
    target_index = task.old_rows.index(task.target_row)
    separator_index = max(
        index
        for index, row in enumerate(task.old_rows[:target_index])
        if row.event == 2
    )
    separator = task.old_rows[separator_index]
    controls = [
        row
        for row in task.old_rows[separator_index + 1 : target_index]
        if row.event == 1
    ]
    if len(controls) != 2:
        raise ValueError(
            f"SC {task.sc_id} expected two first-leg controls, got {len(controls)}"
        )

    launch_day = float(task.old_rows[0].time_days)
    encounter_day = float(task.target_row.time_days)
    target_id = task.target_row.asteroid_id
    target_state = np.array(task.target_row.state, copy=True)
    target_mass = float(target_state[6])
    thrust_days = encounter_day - separator.time_days - 0.1 - 1.0 / DAY_S
    old_launch = np.array(task.old_rows[0].state, copy=True)
    seed_vinf = (
        old_launch[3:6] - earth_state(launch_day)[3:6]
    ) * AU_KM / DAY_S
    seed_norm = float(np.linalg.norm(seed_vinf))
    if seed_norm >= PREFIX_VINF_LIMIT_KMS:
        seed_vinf *= (PREFIX_VINF_LIMIT_KMS - 1e-7) / seed_norm
    raw0 = np.r_[
        inverse_bounded_vector(seed_vinf, PREFIX_VINF_LIMIT_KMS),
        inverse_bounded_vector(controls[0].force, CONTROL_LIMIT),
        inverse_bounded_vector(controls[1].force, CONTROL_LIMIT),
    ]

    def unpack(raw: np.ndarray):
        vinf = bounded_vector(raw[:3], PREFIX_VINF_LIMIT_KMS)
        f0 = bounded_vector(raw[3:6], CONTROL_LIMIT)
        f1 = bounded_vector(raw[6:9], CONTROL_LIMIT)
        fuel = _linear_thrust_fuel_kg(f0, f1, thrust_days)
        launch_mass = target_mass + fuel
        launch_state = np.r_[earth_state(launch_day), launch_mass]
        launch_state[3:6] += vinf * DAY_S / AU_KM
        return launch_state, vinf, f0, f1, fuel

    def propagate(raw: np.ndarray, rtol: float):
        launch_state, _vinf, f0, f1, _fuel = unpack(raw)
        separator_state = coast(
            launch_state, launch_day, separator.time_days, rtol=rtol
        )
        leg = propagate_leg(
            separator_state,
            separator.time_days,
            target_id,
            encounter_day,
            f0,
            f1,
            rtol=rtol,
        )
        return separator_state, leg

    def equality(raw: np.ndarray, rtol: float = 3e-10) -> np.ndarray:
        _separator_state, leg = propagate(raw, rtol)
        return np.r_[
            (leg.state_encounter[:3] - target_state[:3]) * AU_KM / 1000.0,
            (leg.state_encounter[3:6] - target_state[3:6])
            * AU_KM
            / DAY_S,
        ]

    def objective(raw: np.ndarray) -> float:
        return unpack(raw)[4] / 1000.0

    def mass_margin(raw: np.ndarray) -> float:
        return M0_MAX_KG - unpack(raw)[0][6]

    optimized = minimize(
        objective,
        raw0,
        method="SLSQP",
        constraints=(
            {"type": "eq", "fun": equality},
            {"type": "ineq", "fun": mass_margin},
        ),
        options={"maxiter": fuel_iterations, "ftol": 2e-11, "disp": False},
    )

    def correction_residual(raw: np.ndarray) -> np.ndarray:
        residual = equality(raw, 2e-12)
        return np.r_[residual[:3], residual[3:] / 0.001]

    corrected = least_squares(
        correction_residual,
        optimized.x,
        method="trf",
        max_nfev=80,
        xtol=2e-12,
        ftol=2e-12,
        gtol=2e-12,
    )
    launch_state, vinf, f0, f1, _fuel = unpack(corrected.x)
    separator_state, leg = propagate(corrected.x, 8e-13)
    position_error = float(
        np.linalg.norm(leg.state_encounter[:3] - target_state[:3]) * AU_KM
    )
    velocity_error = float(
        np.linalg.norm(leg.state_encounter[3:6] - target_state[3:6])
        * AU_KM
        / DAY_S
    )
    mass_error = float(abs(leg.state_encounter[6] - target_mass))
    old_initial_mass = float(task.old_rows[0].state[6])
    if (
        position_error > 0.25
        or velocity_error > 2e-5
        or mass_error > 0.002
        or launch_state[6] > M0_MAX_KG + 1e-8
        or np.linalg.norm(vinf) > 4.0
    ):
        return PrefixResult(
            task.sc_id,
            None,
            old_initial_mass,
            float(launch_state[6]),
            target_id,
            position_error,
            velocity_error,
            mass_error,
            prefix_message + "coast-prefix optimization failed strict tolerances",
        )

    rows = [
        EventRow(task.sc_id, 0, launch_day, launch_state, np.zeros(3), 0),
        EventRow(
            task.sc_id,
            2,
            separator.time_days,
            separator_state,
            np.zeros(3),
            0,
        ),
        EventRow(
            task.sc_id,
            1,
            leg.sample_start,
            leg.state_sample_start,
            f0,
            0,
        ),
        EventRow(
            task.sc_id,
            1,
            leg.sample_end,
            leg.state_sample_end,
            f1,
            0,
        ),
        EventRow(
            task.sc_id,
            3,
            encounter_day,
            leg.state_encounter,
            np.zeros(3),
            target_id,
        ),
    ]
    rows.extend(task.old_rows[target_index + 1 :])
    return PrefixResult(
        task.sc_id,
        rows,
        old_initial_mass,
        float(launch_state[6]),
        target_id,
        position_error,
        velocity_error,
        mass_error,
        prefix_message
        + f"coast prefix optimized; vinf={np.linalg.norm(vinf):.6f} km/s",
    )


def run_prefix_task(
    task: PrefixTask,
    *,
    mode: str,
    max_nfev: int,
    fuel_iterations: int,
) -> PrefixResult:
    if mode == "coast":
        return optimize_coast_prefix(
            task,
            fuel_iterations=fuel_iterations,
        )
    return optimize_prefix(
        task,
        max_nfev=max_nfev,
        fuel_iterations=fuel_iterations,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("base", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--spacecraft", default="2-20")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--max-nfev", type=int, default=300)
    parser.add_argument("--fuel-iterations", type=int, default=45)
    parser.add_argument(
        "--mode",
        choices=("coast", "pure-direct"),
        default="coast",
        help="optimize the proven coast prefix or first attempt a pure direct leg",
    )
    args = parser.parse_args()

    base_result = check(args.base)
    if not base_result["valid"]:
        raise SystemExit("base submission is not valid")
    rows = load_rows(args.base)
    selected = parse_spacecraft_spec(args.spacecraft)
    tasks = make_tasks(rows, selected)
    results: dict[int, PrefixResult] = {}

    if args.workers == 1:
        for task in tasks:
            result = run_prefix_task(
                task,
                mode=args.mode,
                max_nfev=args.max_nfev,
                fuel_iterations=args.fuel_iterations,
            )
            results[result.sc_id] = result
            print(
                f"SC {result.sc_id}: {result.message}; m0={result.new_initial_mass} "
                f"pos={result.position_error_km} vel={result.velocity_error_kms} "
                f"mass={result.mass_error_kg}",
                flush=True,
            )
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(
                    run_prefix_task,
                    task,
                    mode=args.mode,
                    max_nfev=args.max_nfev,
                    fuel_iterations=args.fuel_iterations,
                ): task.sc_id
                for task in tasks
            }
            for future in as_completed(futures):
                result = future.result()
                results[result.sc_id] = result
                print(
                    f"SC {result.sc_id}: {result.message}; "
                    f"m0={result.new_initial_mass} pos={result.position_error_km} "
                    f"vel={result.velocity_error_kms} mass={result.mass_error_kg}",
                    flush=True,
                )

    successful = {
        sc_id: result for sc_id, result in results.items() if result.rows is not None
    }
    if not successful:
        raise SystemExit("no direct prefix reached the required terminal state")
    combined = []
    for sc_id in sorted({row.sc_id for row in rows}):
        if sc_id in successful:
            combined.extend(successful[sc_id].rows or [])
        else:
            combined.extend(row for row in rows if row.sc_id == sc_id)
    write_submission(
        args.output,
        combined,
        comments=[
            "CTOC14 direct prefixes with full terminal-state matching",
            f"replaced_spacecraft={','.join(map(str, sorted(successful)))}",
        ],
    )
    validation = check(args.output)
    report = {
        "base_J": base_result["J"],
        "candidate_J": validation["J"],
        "improvement": base_result["J"] - validation["J"],
        "successful": sorted(successful),
        "failed": sorted(selected - set(successful)),
        "prefixes": {
            str(sc_id): {
                "target": result.target_id,
                "old_initial_mass": result.old_initial_mass,
                "new_initial_mass": result.new_initial_mass,
                "position_error_km": result.position_error_km,
                "velocity_error_kms": result.velocity_error_kms,
                "mass_error_kg": result.mass_error_kg,
                "message": result.message,
            }
            for sc_id, result in sorted(results.items())
        },
        "validation": validation,
    }
    print(json.dumps(report, indent=2), flush=True)
    raise SystemExit(0 if validation["valid"] else 1)


if __name__ == "__main__":
    main()
