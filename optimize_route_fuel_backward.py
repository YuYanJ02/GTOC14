#!/usr/bin/env python3
"""Backward fuel optimization for fixed CTOC14 encounter routes.

Each powered leg is replaced by two linearly controlled thrust arcs separated
by an ordinary Event=2 coast boundary.  The optimizer holds the original
terminal position and velocity while minimizing the propellant required to
reach a prescribed terminal mass.  Processing legs backward propagates every
mass saving all the way to launch without changing the encounter sequence.
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
    M0_MAX_KG,
    EventRow,
    bounded_vector,
    inverse_bounded_vector,
    propagate_leg,
    write_submission,
)
from extend_ctoc14_fleet import load_rows
from solve_ctoc14 import CONTROL_LIMIT, _linear_thrust_fuel_kg
from validate_ctoc14 import check


@dataclass
class Segment:
    start: EventRow
    interior: list[EventRow]
    end: EventRow

    @property
    def controls(self) -> list[EventRow]:
        return [row for row in self.interior if row.event == 1]

    @property
    def encounter(self) -> EventRow | None:
        encounters = [row for row in self.interior if row.event == 3]
        return encounters[-1] if encounters else None


@dataclass
class LegOptimization:
    start_mass: float
    interior: list[EventRow]
    terminal_mass: float
    saving_kg: float
    optimized: bool
    message: str


@dataclass
class SpacecraftOptimization:
    sc_id: int
    rows: list[EventRow]
    old_initial_mass: float
    new_initial_mass: float
    optimized_legs: int
    total_powered_legs: int
    messages: list[str]


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


def split_segments(rows: list[EventRow]) -> list[Segment]:
    boundary_indices = [
        index for index, row in enumerate(rows) if row.event in (0, 2, 4)
    ]
    segments = []
    for left, right in zip(boundary_indices, boundary_indices[1:]):
        segments.append(Segment(rows[left], rows[left + 1 : right], rows[right]))
    return segments


def scaled_fallback(segment: Segment, terminal_mass: float) -> LegOptimization:
    encounter = segment.encounter
    if encounter is None:
        # A coast-only segment has constant mass.
        interior = [
            EventRow(
                row.sc_id,
                row.event,
                row.time_days,
                np.r_[row.state[:6], terminal_mass],
                np.zeros(3),
                row.asteroid_id,
            )
            for row in segment.interior
        ]
        return LegOptimization(
            terminal_mass,
            interior,
            terminal_mass,
            0.0,
            False,
            "coast segment",
        )

    original_terminal = float(encounter.state[6])
    scale = terminal_mass / original_terminal
    interior = []
    for row in segment.interior:
        state = np.array(row.state, copy=True)
        state[6] *= scale
        force = np.array(row.force, copy=True) * scale
        interior.append(
            EventRow(
                row.sc_id,
                row.event,
                row.time_days,
                state,
                force,
                row.asteroid_id,
            )
        )
    start_mass = float(segment.start.state[6]) * scale
    return LegOptimization(
        start_mass,
        interior,
        terminal_mass,
        0.0,
        False,
        "trajectory-preserving fallback",
    )


def optimize_powered_segment(
    segment: Segment,
    terminal_mass: float,
    *,
    max_nfev: int,
    fuel_iterations: int,
) -> LegOptimization:
    controls = segment.controls
    encounter = segment.encounter
    if len(controls) != 2 or encounter is None:
        return scaled_fallback(segment, terminal_mass)
    start_time = float(segment.start.time_days)
    encounter_time = float(encounter.time_days)
    duration = encounter_time - start_time
    if duration < 1.0:
        return scaled_fallback(segment, terminal_mass)
    midpoint = start_time + 0.5 * duration
    second_start = midpoint + 1.0 / DAY_S
    first_thrust_days = midpoint - start_time - 0.1 - 1.0 / DAY_S
    second_thrust_days = encounter_time - second_start - 0.1 - 1.0 / DAY_S
    if min(first_thrust_days, second_thrust_days) <= 0.1:
        return scaled_fallback(segment, terminal_mass)

    fallback = scaled_fallback(segment, terminal_mass)
    original_terminal = float(encounter.state[6])
    scale = terminal_mass / original_terminal
    original_t0 = controls[0].time_days
    original_t1 = controls[1].time_days

    def original_force(time_days: float) -> np.ndarray:
        fraction = (time_days - original_t0) / (original_t1 - original_t0)
        fraction = min(1.0, max(0.0, fraction))
        return scale * (
            (1.0 - fraction) * controls[0].force
            + fraction * controls[1].force
        )

    sample_times = (
        start_time + 0.1,
        midpoint - 1.0 / DAY_S,
        second_start + 0.1,
        encounter_time - 1.0 / DAY_S,
    )
    raw0 = np.concatenate(
        [inverse_bounded_vector(original_force(t), CONTROL_LIMIT) for t in sample_times]
    )
    start_position_velocity = np.array(segment.start.state[:6], copy=True)
    target_position_velocity = np.array(encounter.state[:6], copy=True)

    def unpack(raw: np.ndarray):
        forces = [
            bounded_vector(raw[index : index + 3], CONTROL_LIMIT)
            for index in range(0, 12, 3)
        ]
        fuel1 = _linear_thrust_fuel_kg(
            forces[0], forces[1], first_thrust_days
        )
        fuel2 = _linear_thrust_fuel_kg(
            forces[2], forces[3], second_thrust_days
        )
        start_mass = terminal_mass + fuel1 + fuel2
        start_state = np.r_[start_position_velocity, start_mass]
        return forces, start_state, fuel1 + fuel2

    def propagate(raw: np.ndarray, rtol: float):
        forces, start_state, _fuel = unpack(raw)
        first = propagate_leg(
            start_state,
            start_time,
            0,
            midpoint,
            forces[0],
            forces[1],
            rtol=rtol,
        )
        second = propagate_leg(
            first.state_separator,
            first.separator_time,
            encounter.asteroid_id,
            encounter_time,
            forces[2],
            forces[3],
            rtol=rtol,
        )
        return first, second

    def match_residual(raw: np.ndarray, rtol: float) -> np.ndarray:
        _first, second = propagate(raw, rtol)
        position = (
            second.state_encounter[:3] - target_position_velocity[:3]
        ) * AU_KM / 10.0
        velocity = (
            second.state_encounter[3:6] - target_position_velocity[3:6]
        ) * AU_KM / DAY_S / 0.01
        mass_limit = max(0.0, unpack(raw)[1][6] - M0_MAX_KG)
        return np.r_[position, velocity, mass_limit]

    matched = least_squares(
        lambda raw: match_residual(raw, 3e-9),
        raw0,
        method="trf",
        max_nfev=max_nfev,
        xtol=3e-10,
        ftol=3e-10,
        gtol=3e-10,
    )

    def equality(raw: np.ndarray) -> np.ndarray:
        _first, second = propagate(raw, 3e-10)
        return np.r_[
            (second.state_encounter[:3] - target_position_velocity[:3])
            * AU_KM
            / 1000.0,
            (second.state_encounter[3:6] - target_position_velocity[3:6])
            * AU_KM
            / DAY_S,
        ]

    optimized = minimize(
        lambda raw: unpack(raw)[2] / 1000.0,
        matched.x,
        method="SLSQP",
        constraints=(
            {"type": "eq", "fun": equality},
            {"type": "ineq", "fun": lambda raw: M0_MAX_KG - unpack(raw)[1][6]},
        ),
        options={"maxiter": fuel_iterations, "ftol": 3e-10, "disp": False},
    )
    def correction_residual(raw: np.ndarray) -> np.ndarray:
        residual = equality(raw)
        return np.r_[residual[:3], residual[3:] / 0.001]

    corrected = least_squares(
        correction_residual,
        optimized.x,
        method="trf",
        max_nfev=100,
        xtol=2e-12,
        ftol=2e-12,
        gtol=2e-12,
    )
    forces, start_state, _fuel = unpack(corrected.x)
    first, second = propagate(corrected.x, 8e-13)
    position_error = float(
        np.linalg.norm(second.state_encounter[:3] - target_position_velocity[:3])
        * AU_KM
    )
    velocity_error = float(
        np.linalg.norm(second.state_encounter[3:6] - target_position_velocity[3:6])
        * AU_KM
        / DAY_S
    )
    mass_error = float(abs(second.state_encounter[6] - terminal_mass))
    saving = fallback.start_mass - float(start_state[6])
    if (
        position_error > 0.70
        or velocity_error > 2e-5
        or mass_error > 0.002
        or start_state[6] > M0_MAX_KG + 1e-8
        or saving <= 1e-5
    ):
        fallback.message = (
            f"fallback: proposed saving={saving:.4f} kg "
            f"pos={position_error:.3g} km vel={velocity_error:.3g} km/s "
            f"mass={mass_error:.3g} kg"
        )
        return fallback

    sc_id = segment.start.sc_id
    interior = [
        EventRow(sc_id, 1, first.sample_start, first.state_sample_start, forces[0], 0),
        EventRow(sc_id, 1, first.sample_end, first.state_sample_end, forces[1], 0),
        EventRow(
            sc_id,
            2,
            first.separator_time,
            first.state_separator,
            np.zeros(3),
            0,
        ),
        EventRow(sc_id, 1, second.sample_start, second.state_sample_start, forces[2], 0),
        EventRow(sc_id, 1, second.sample_end, second.state_sample_end, forces[3], 0),
        EventRow(
            sc_id,
            3,
            encounter_time,
            second.state_encounter,
            np.zeros(3),
            encounter.asteroid_id,
        ),
    ]
    return LegOptimization(
        float(start_state[6]),
        interior,
        terminal_mass,
        saving,
        True,
        f"saved={saving:.4f} kg pos={position_error:.3g} km",
    )


def optimize_spacecraft(
    rows: list[EventRow],
    *,
    max_nfev: int,
    fuel_iterations: int,
) -> SpacecraftOptimization:
    segments = split_segments(rows)
    terminal_mass = float(rows[-1].state[6])
    optimized_by_index: dict[int, LegOptimization] = {}
    messages = []
    optimized_count = 0

    for index in range(len(segments) - 1, -1, -1):
        segment = segments[index]
        result = optimize_powered_segment(
            segment,
            terminal_mass,
            max_nfev=max_nfev,
            fuel_iterations=fuel_iterations,
        )
        optimized_by_index[index] = result
        terminal_mass = result.start_mass
        if segment.controls:
            messages.append(
                f"leg {index + 1} target="
                f"{segment.encounter.asteroid_id if segment.encounter else 0}: "
                f"{result.message}"
            )
            optimized_count += int(result.optimized)

    rebuilt = []
    for index, segment in enumerate(segments):
        result = optimized_by_index[index]
        if index == 0:
            start_state = np.array(segment.start.state, copy=True)
            start_state[6] = result.start_mass
            rebuilt.append(
                EventRow(
                    segment.start.sc_id,
                    segment.start.event,
                    segment.start.time_days,
                    start_state,
                    np.array(segment.start.force, copy=True),
                    segment.start.asteroid_id,
                )
            )
        rebuilt.extend(result.interior)
        end_state = np.array(segment.end.state, copy=True)
        end_state[6] = result.terminal_mass
        rebuilt.append(
            EventRow(
                segment.end.sc_id,
                segment.end.event,
                segment.end.time_days,
                end_state,
                np.array(segment.end.force, copy=True),
                segment.end.asteroid_id,
            )
        )
    return SpacecraftOptimization(
        rows[0].sc_id,
        rebuilt,
        float(rows[0].state[6]),
        float(rebuilt[0].state[6]),
        optimized_count,
        sum(bool(segment.controls) for segment in segments),
        list(reversed(messages)),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("base", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--spacecraft", default="1-29")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--max-nfev", type=int, default=100)
    parser.add_argument("--fuel-iterations", type=int, default=20)
    args = parser.parse_args()

    base_result = check(args.base)
    if not base_result["valid"]:
        raise SystemExit("base submission is not valid")
    all_rows = load_rows(args.base)
    grouped: dict[int, list[EventRow]] = {}
    for row in all_rows:
        grouped.setdefault(row.sc_id, []).append(row)
    selected = parse_spacecraft_spec(args.spacecraft)
    results: dict[int, SpacecraftOptimization] = {}

    if args.workers == 1:
        for sc_id in sorted(selected):
            result = optimize_spacecraft(
                grouped[sc_id],
                max_nfev=args.max_nfev,
                fuel_iterations=args.fuel_iterations,
            )
            results[sc_id] = result
            print(
                f"SC {sc_id}: m0 {result.old_initial_mass:.6f} -> "
                f"{result.new_initial_mass:.6f}; optimized "
                f"{result.optimized_legs}/{result.total_powered_legs}",
                flush=True,
            )
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(
                    optimize_spacecraft,
                    grouped[sc_id],
                    max_nfev=args.max_nfev,
                    fuel_iterations=args.fuel_iterations,
                ): sc_id
                for sc_id in sorted(selected)
            }
            for future in as_completed(futures):
                sc_id = futures[future]
                result = future.result()
                results[sc_id] = result
                print(
                    f"SC {sc_id}: m0 {result.old_initial_mass:.6f} -> "
                    f"{result.new_initial_mass:.6f}; optimized "
                    f"{result.optimized_legs}/{result.total_powered_legs}",
                    flush=True,
                )

    combined = []
    for sc_id in sorted(grouped):
        combined.extend(results[sc_id].rows if sc_id in results else grouped[sc_id])
    write_submission(
        args.output,
        combined,
        comments=[
            "CTOC14 backward fixed-route fuel optimization",
            f"optimized_spacecraft={','.join(map(str, sorted(results)))}",
        ],
    )
    validation = check(args.output)
    report = {
        "base_J": base_result["J"],
        "candidate_J": validation["J"],
        "improvement": base_result["J"] - validation["J"],
        "spacecraft": {
            str(sc_id): {
                "old_initial_mass": result.old_initial_mass,
                "new_initial_mass": result.new_initial_mass,
                "optimized_legs": result.optimized_legs,
                "total_powered_legs": result.total_powered_legs,
                "messages": result.messages,
            }
            for sc_id, result in sorted(results.items())
        },
        "validation": validation,
    }
    print(json.dumps(report, indent=2), flush=True)
    raise SystemExit(0 if validation["valid"] else 1)


if __name__ == "__main__":
    main()
