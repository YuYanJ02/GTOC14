#!/usr/bin/env python3
"""Global low-mass launch-window and asteroid-sequence search for CTOC14.

The original constructive solver is deliberately feasibility-first: it fixes a
route greedily and only then reduces fuel.  This module reverses that order.
It builds a large pool of launch/sequence candidates with a fast impulsive
Lambert model, rejects candidates whose *estimated* initial mass exceeds a
strict cap, realizes the survivors with the official continuous-thrust model,
and finally solves the fleet selection problem as a binary set cover.

The existing validated submission is always included in the route pool, so a
candidate file can only replace it after the independent checker reports a
strictly smaller objective value.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
import json
import math
from pathlib import Path
import pickle
import time
from typing import Iterable, Sequence

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import Bounds, LinearConstraint, milp

from ctoc14_core import (
    AU_KM,
    DAY_S,
    EARTH_EPOCH_MJD,
    EARTH_ELEMENTS,
    ISP,
    MDRY_KG,
    MISSION_DAYS,
    MU,
    NEA_EPOCH_MJD,
    T0_MJD,
    EventRow,
    asteroid_state,
    coast,
    coast_rhs,
    earth_state,
    load_asteroids,
    write_submission,
)
from extend_ctoc14_fleet import load_rows
from solve_ctoc14 import Candidate, append_leg_rows, optimize_candidate, refine_candidate_fuel
from validate_ctoc14 import check


VE_KMS = ISP * 9.80665 / 1000.0
FINAL_MASS_KG = MDRY_KG + 0.1
VINF_SAFETY_KMS = 3.995


@dataclass(frozen=True)
class LambertTransfer:
    departure_velocity: tuple[float, float, float]
    arrival_velocity: tuple[float, float, float]


@dataclass(frozen=True)
class ImpulsiveRoute:
    launch_day: float
    launch_velocity: tuple[float, float, float]
    asteroid_ids: tuple[int, ...]
    encounter_days: tuple[float, ...]
    arrival_velocity: tuple[float, float, float]
    maneuver_dv_kms: float

    @property
    def last_day(self) -> float:
        return self.encounter_days[-1]

    @property
    def last_id(self) -> int:
        return self.asteroid_ids[-1]

    @property
    def estimated_initial_mass(self) -> float:
        return FINAL_MASS_KG * math.exp(self.maneuver_dv_kms / VE_KMS)

    @property
    def estimated_cost(self) -> float:
        x = (self.estimated_initial_mass - MDRY_KG) / 1400.0
        return 1.0 + x + x * x


@dataclass
class RouteOption:
    rows: list[EventRow]
    covered: frozenset[int]
    cost: float
    source: str


class VectorEphemeris:
    """Vectorized two-body positions for all 300 asteroids."""

    def __init__(self, asteroids: dict[int, np.ndarray]):
        self.ids = np.array(sorted(asteroids), dtype=int)
        elements = np.vstack([asteroids[int(i)] for i in self.ids])
        self.a = elements[:, 0]
        self.e = elements[:, 1]
        inc, node, argp = np.deg2rad(elements[:, 2:5]).T
        self.mean_at_zero = np.deg2rad(elements[:, 5]) + np.sqrt(
            MU / self.a**3
        ) * (T0_MJD - NEA_EPOCH_MJD)
        self.mean_motion = np.sqrt(MU / self.a**3)

        co, so = np.cos(node), np.sin(node)
        cw, sw = np.cos(argp), np.sin(argp)
        ci, si = np.cos(inc), np.sin(inc)
        self.rotation = np.empty((len(self.ids), 3, 3))
        self.rotation[:, 0, 0] = co * cw - so * sw * ci
        self.rotation[:, 0, 1] = -co * sw - so * cw * ci
        self.rotation[:, 0, 2] = so * si
        self.rotation[:, 1, 0] = so * cw + co * sw * ci
        self.rotation[:, 1, 1] = -so * sw + co * cw * ci
        self.rotation[:, 1, 2] = -co * si
        self.rotation[:, 2, 0] = sw * si
        self.rotation[:, 2, 1] = cw * si
        self.rotation[:, 2, 2] = ci

    def positions(self, times: Sequence[float]) -> np.ndarray:
        times_array = np.atleast_1d(np.asarray(times, dtype=float))
        mean = self.mean_at_zero[:, None] + self.mean_motion[:, None] * times_array
        mean = np.remainder(mean + np.pi, 2.0 * np.pi) - np.pi
        eccentric = np.array(mean, copy=True)
        for _ in range(12):
            step = (
                eccentric - self.e[:, None] * np.sin(eccentric) - mean
            ) / (1.0 - self.e[:, None] * np.cos(eccentric))
            eccentric -= step
            if float(np.max(np.abs(step))) < 3e-14:
                break
        ce, se = np.cos(eccentric), np.sin(eccentric)
        perifocal = np.zeros((len(self.ids), len(times_array), 3))
        perifocal[:, :, 0] = self.a[:, None] * (ce - self.e[:, None])
        perifocal[:, :, 1] = (
            self.a[:, None]
            * np.sqrt(1.0 - self.e * self.e)[:, None]
            * se
        )
        inertial = np.einsum("aij,atj->ati", self.rotation, perifocal)
        return np.transpose(inertial, (1, 0, 2))


def _stumpff(z: float) -> tuple[float, float]:
    if z > 1e-7:
        root = math.sqrt(z)
        return (1.0 - math.cos(root)) / z, (root - math.sin(root)) / root**3
    if z < -1e-7:
        root = math.sqrt(-z)
        if root > 40.0:
            raise OverflowError("Lambert hyperbolic parameter overflow")
        return (math.cosh(root) - 1.0) / (-z), (
            math.sinh(root) - root
        ) / root**3
    # Series avoid cancellation around the parabolic case.
    return (
        0.5 - z / 24.0 + z * z / 720.0 - z**3 / 40320.0,
        1.0 / 6.0 - z / 120.0 + z * z / 5040.0 - z**3 / 362880.0,
    )


def lambert_universal(
    r1: np.ndarray,
    r2: np.ndarray,
    duration_days: float,
    *,
    long_way: bool = False,
) -> LambertTransfer | None:
    """Return a zero-revolution universal-variable Lambert solution."""
    if duration_days <= 0.0:
        return None
    r1n = float(np.linalg.norm(r1))
    r2n = float(np.linalg.norm(r2))
    cosine = float(np.clip(np.dot(r1, r2) / (r1n * r2n), -1.0, 1.0))
    sine_abs = math.sqrt(max(0.0, 1.0 - cosine * cosine))
    sine = -sine_abs if long_way else sine_abs
    denominator = 1.0 - cosine
    if denominator < 1e-12 or abs(sine) < 1e-12:
        return None
    a_term = sine * math.sqrt(r1n * r2n / denominator)
    if abs(a_term) < 1e-12:
        return None
    target = math.sqrt(MU) * duration_days

    def residual(z: float) -> float | None:
        try:
            c, s = _stumpff(z)
        except (OverflowError, ValueError):
            return None
        if c <= 0.0:
            return None
        y = r1n + r2n + a_term * (z * s - 1.0) / math.sqrt(c)
        if y <= 0.0:
            return None
        value = (y / c) ** 1.5 * s + a_term * math.sqrt(y) - target
        return value if math.isfinite(value) else None

    f0 = residual(0.0)
    if f0 is None:
        return None
    if f0 <= 0.0:
        lower, upper = 0.0, 0.5
        f_lower = f0
        f_upper = residual(upper)
        while (f_upper is None or f_upper < 0.0) and upper < 39.0:
            upper = min(39.0, upper * 1.7 + 0.25)
            f_upper = residual(upper)
        if f_upper is None or f_upper < 0.0:
            return None
    else:
        upper, lower = 0.0, -0.5
        f_upper = f0
        f_lower = residual(lower)
        while (f_lower is None or f_lower > 0.0) and lower > -160.0:
            lower = lower * 1.7 - 0.25
            f_lower = residual(lower)
        if f_lower is None:
            # Invalid y lies below the admissible interval and is equivalent
            # to a time of flight below the requested value.
            f_lower = -math.inf
        elif f_lower > 0.0:
            return None

    for _ in range(58):
        middle = 0.5 * (lower + upper)
        f_middle = residual(middle)
        if f_middle is None or f_middle < 0.0:
            lower = middle
        else:
            upper = middle
    z = 0.5 * (lower + upper)
    c, s = _stumpff(z)
    y = r1n + r2n + a_term * (z * s - 1.0) / math.sqrt(c)
    if y <= 0.0:
        return None
    f_lagrange = 1.0 - y / r1n
    g_lagrange = a_term * math.sqrt(y / MU)
    gdot_lagrange = 1.0 - y / r2n
    if abs(g_lagrange) < 1e-13:
        return None
    departure = (r2 - f_lagrange * r1) / g_lagrange
    arrival = (gdot_lagrange * r2 - r1) / g_lagrange
    if not np.all(np.isfinite(departure)) or not np.all(np.isfinite(arrival)):
        return None
    return LambertTransfer(tuple(departure), tuple(arrival))


def best_lambert(
    r1: np.ndarray,
    r2: np.ndarray,
    duration_days: float,
    reference_velocity: np.ndarray,
) -> tuple[LambertTransfer, float] | None:
    choices = []
    for long_way in (False, True):
        result = lambert_universal(r1, r2, duration_days, long_way=long_way)
        if result is None:
            continue
        delta_v = float(
            np.linalg.norm(np.asarray(result.departure_velocity) - reference_velocity)
            * AU_KM
            / DAY_S
        )
        choices.append((result, delta_v))
    return min(choices, key=lambda item: item[1]) if choices else None


def lambert_self_test(asteroids: dict[int, np.ndarray]) -> dict:
    tests = []
    for launch_day, asteroid_id, duration in (
        (0.0, 174, 115.838878),
        (400.0, 127, 300.0),
        (1200.0, 219, 500.0),
    ):
        initial = earth_state(launch_day)
        target = asteroid_state(asteroids, asteroid_id, launch_day + duration)
        choice = best_lambert(
            initial[:3], target[:3], duration, initial[3:6]
        )
        if choice is None:
            raise RuntimeError(f"Lambert self-test {asteroid_id} has no solution")
        transfer, delta_v = choice
        propagated = coast(
            np.r_[initial[:3], np.asarray(transfer.departure_velocity)],
            launch_day,
            launch_day + duration,
            rtol=2e-12,
        )
        miss_km = float(np.linalg.norm(propagated[:3] - target[:3]) * AU_KM)
        tests.append(
            {
                "asteroid": asteroid_id,
                "duration_days": duration,
                "departure_delta_v_kms": delta_v,
                "propagation_miss_km": miss_km,
            }
        )
    if max(item["propagation_miss_km"] for item in tests) > 0.2:
        raise RuntimeError(f"Lambert self-test failed: {tests}")
    vector = VectorEphemeris(asteroids)
    ephemeris_errors = []
    for day in (0.0, 1234.5, 5000.0):
        positions = vector.positions([day])[0]
        error_km = max(
            float(
                np.linalg.norm(
                    positions[asteroid_id - 1]
                    - asteroid_state(asteroids, asteroid_id, day)[:3]
                )
                * AU_KM
            )
            for asteroid_id in (1, 57, 174, 300)
        )
        ephemeris_errors.append({"day": day, "maximum_error_km": error_km})
    if max(item["maximum_error_km"] for item in ephemeris_errors) > 1e-3:
        raise RuntimeError(f"vector ephemeris self-test failed: {ephemeris_errors}")
    return {"tests": tests, "vector_ephemeris": ephemeris_errors}


def scan_ballistic_launches(
    asteroids: dict[int, np.ndarray],
    ephemeris: VectorEphemeris,
    *,
    launch_start_day: float,
    launch_max_day: float,
    launch_step_days: float,
    minimum_duration_days: float,
    maximum_duration_days: float,
    duration_step_days: float,
    max_seeds: int,
) -> list[ImpulsiveRoute]:
    """Scan the full launch window for zero-propellant first encounters."""
    del asteroids  # The vector ephemeris supplies all target positions here.
    best: dict[tuple[int, int, int, bool], tuple[float, ImpulsiveRoute]] = {}
    launch_days = np.arange(
        launch_start_day, launch_max_day + 0.1, launch_step_days
    )
    durations = np.arange(
        minimum_duration_days,
        maximum_duration_days + 0.1,
        duration_step_days,
    )
    for launch_index, launch_day in enumerate(launch_days, 1):
        earth = earth_state(float(launch_day))
        encounter_days = launch_day + durations
        mask = encounter_days < MISSION_DAYS - 30.0
        encounter_days = encounter_days[mask]
        positions = ephemeris.positions(encounter_days)
        for time_index, encounter_day in enumerate(encounter_days):
            duration = float(encounter_day - launch_day)
            for asteroid_index, asteroid_id in enumerate(ephemeris.ids):
                target = positions[time_index, asteroid_index]
                for long_way in (False, True):
                    transfer = lambert_universal(
                        earth[:3], target, duration, long_way=long_way
                    )
                    if transfer is None:
                        continue
                    departure = np.asarray(transfer.departure_velocity)
                    vinf = float(
                        np.linalg.norm(departure - earth[3:6]) * AU_KM / DAY_S
                    )
                    if vinf > VINF_SAFETY_KMS:
                        continue
                    route = ImpulsiveRoute(
                        float(launch_day),
                        transfer.departure_velocity,
                        (int(asteroid_id),),
                        (float(encounter_day),),
                        transfer.arrival_velocity,
                        0.0,
                    )
                    key = (
                        int(asteroid_id),
                        int(encounter_day // 120.0),
                        int(launch_day // 240.0),
                        long_way,
                    )
                    old = best.get(key)
                    # v-infinity is free, but retaining margin improves final
                    # numerical robustness and leaves room for local polishing.
                    metric = vinf + 2e-5 * encounter_day
                    if old is None or metric < old[0]:
                        best[key] = (metric, route)
        print(
            f"launch scan {launch_index}/{len(launch_days)} day={launch_day:.1f} "
            f"zero-fuel seeds={len(best)}",
            flush=True,
        )

    seeds = [item[1] for item in best.values()]
    # Round-robin selection avoids filling the beam with many epochs of the
    # same easy first target.
    by_target: dict[int, list[ImpulsiveRoute]] = defaultdict(list)
    for route in seeds:
        by_target[route.last_id].append(route)
    for routes in by_target.values():
        routes.sort(key=lambda route: (route.last_day, route.launch_day))
    selected = []
    depth = 0
    while len(selected) < max_seeds:
        added = False
        for asteroid_id in sorted(by_target):
            routes = by_target[asteroid_id]
            if depth < len(routes):
                selected.append(routes[depth])
                added = True
                if len(selected) >= max_seeds:
                    break
        if not added:
            break
        depth += 1
    return selected


def coast_samples(
    position: np.ndarray,
    velocity: np.ndarray,
    start_day: float,
    sample_days: np.ndarray,
) -> np.ndarray:
    state = np.r_[position, velocity]
    result = solve_ivp(
        coast_rhs,
        (float(start_day), float(sample_days[-1])),
        state,
        t_eval=sample_days,
        method="DOP853",
        rtol=2e-9,
        atol=1e-11,
    )
    if not result.success:
        raise RuntimeError(result.message)
    return result.y[:3].T


def expand_impulsive_route(
    route: ImpulsiveRoute,
    ephemeris: VectorEphemeris,
    *,
    minimum_leg_days: float,
    maximum_leg_days: float,
    leg_step_days: float,
    neighbor_per_time: int,
    candidates_per_route: int,
    maximum_leg_dv_kms: float,
    mass_cap_kg: float,
    target_priority: np.ndarray,
) -> list[ImpulsiveRoute]:
    last_position = ephemeris.positions([route.last_day])[0, route.last_id - 1]
    last_velocity = np.asarray(route.arrival_velocity)
    final_day = min(route.last_day + maximum_leg_days, MISSION_DAYS - 2.0)
    sample_days = np.arange(
        route.last_day + minimum_leg_days,
        final_day + 0.1,
        leg_step_days,
    )
    if not len(sample_days):
        return []
    ballistic_positions = coast_samples(
        last_position, last_velocity, route.last_day, sample_days
    )
    target_positions = ephemeris.positions(sample_days)
    distances = np.linalg.norm(
        target_positions - ballistic_positions[:, None, :], axis=2
    )
    visited_indices = np.asarray(route.asteroid_ids, dtype=int) - 1
    distances[:, visited_indices] = np.inf

    # Do not let geometric nearest-neighbour filtering permanently hide the
    # difficult asteroids.  Repeated global passes assign the largest priority
    # to targets that are absent or rare in the realized route pool.  A target
    # still has to be reasonably close to the ballistic coast, but a rare one
    # may displace a common near-duplicate before the Lambert solves are made.
    priority_scale = 1.0 + 0.8 * target_priority[1:]
    ranked_distances = distances / priority_scale[None, :]
    prefiltered: list[tuple[float, int, int]] = []
    count = min(neighbor_per_time, distances.shape[1])
    for time_index in range(len(sample_days)):
        nearest = np.argpartition(ranked_distances[time_index], count - 1)[:count]
        duration = sample_days[time_index] - route.last_day
        for asteroid_index in nearest:
            proxy = ranked_distances[time_index, asteroid_index] / duration
            prefiltered.append((float(proxy), time_index, int(asteroid_index)))
    prefiltered.sort()

    children = []
    used_pairs: set[tuple[int, int]] = set()
    for _proxy, time_index, asteroid_index in prefiltered:
        asteroid_id = asteroid_index + 1
        bucket = int(sample_days[time_index] // leg_step_days)
        key = (asteroid_id, bucket)
        if key in used_pairs:
            continue
        used_pairs.add(key)
        encounter_day = float(sample_days[time_index])
        duration = encounter_day - route.last_day
        choice = best_lambert(
            last_position,
            target_positions[time_index, asteroid_index],
            duration,
            last_velocity,
        )
        if choice is None:
            continue
        transfer, leg_dv = choice
        total_dv = route.maneuver_dv_kms + leg_dv
        estimated_mass = FINAL_MASS_KG * math.exp(total_dv / VE_KMS)
        if leg_dv > maximum_leg_dv_kms or estimated_mass > mass_cap_kg:
            continue
        children.append(
            ImpulsiveRoute(
                route.launch_day,
                route.launch_velocity,
                route.asteroid_ids + (asteroid_id,),
                route.encounter_days + (encounter_day,),
                transfer.arrival_velocity,
                total_dv,
            )
        )
        if len(children) >= candidates_per_route:
            break
    return children


def select_diverse_beam(
    routes: Iterable[ImpulsiveRoute],
    beam_width: int,
    target_priority: np.ndarray,
) -> list[ImpulsiveRoute]:
    best_by_signature: dict[tuple[int, int, int, int, int], ImpulsiveRoute] = {}
    for route in routes:
        previous_id = route.asteroid_ids[-2] if len(route.asteroid_ids) > 1 else 0
        signature = (
            route.asteroid_ids[0],
            previous_id,
            route.last_id,
            int(route.last_day // 45.0),
            int(route.maneuver_dv_kms // 0.75),
        )
        old = best_by_signature.get(signature)
        if old is None or (
            route.maneuver_dv_kms,
            route.last_day,
        ) < (old.maneuver_dv_kms, old.last_day):
            best_by_signature[signature] = route
    by_first: dict[int, list[ImpulsiveRoute]] = defaultdict(list)
    for route in best_by_signature.values():
        by_first[route.asteroid_ids[0]].append(route)

    def metric(route: ImpulsiveRoute) -> tuple[float, float]:
        # A seeded priority perturbation lets repeated passes explore different
        # launch families without accepting expensive individual routes.
        priority_bonus = 0.35 * float(
            np.mean(target_priority[np.asarray(route.asteroid_ids, dtype=int)])
        )
        return (
            route.maneuver_dv_kms
            + 1.5e-4 * (route.last_day - route.launch_day)
            - priority_bonus,
            route.last_day,
        )

    for group in by_first.values():
        group.sort(key=metric)
    group_order = sorted(
        by_first,
        key=lambda first_id: (
            metric(by_first[first_id][0])[0] - target_priority[first_id],
            first_id,
        ),
    )
    selected = []
    rank = 0
    while len(selected) < beam_width:
        added = False
        for first_id in group_order:
            group = by_first[first_id]
            if rank < len(group):
                selected.append(group[rank])
                added = True
                if len(selected) >= beam_width:
                    break
        if not added:
            break
        rank += 1
    return selected


def impulsive_beam_search(
    seeds: list[ImpulsiveRoute],
    ephemeris: VectorEphemeris,
    *,
    beam_width: int,
    maximum_targets: int,
    minimum_leg_days: float,
    maximum_leg_days: float,
    leg_step_days: float,
    neighbor_per_time: int,
    candidates_per_route: int,
    maximum_leg_dv_kms: float,
    mass_cap_kg: float,
    diversity_seed: int,
    existing_frequency: dict[int, int],
) -> list[ImpulsiveRoute]:
    random = np.random.default_rng(diversity_seed)
    rarity = np.asarray(
        [1.0 / (1.0 + existing_frequency.get(i, 0)) for i in range(1, 301)]
    )
    target_priority = np.r_[0.0, random.random(300) + 3.0 * rarity]
    beam = select_diverse_beam(seeds, beam_width, target_priority)
    archive = list(beam)
    for depth in range(2, maximum_targets + 1):
        started = time.time()
        children = []
        for index, route in enumerate(beam, 1):
            try:
                children.extend(
                    expand_impulsive_route(
                        route,
                        ephemeris,
                        minimum_leg_days=minimum_leg_days,
                        maximum_leg_days=maximum_leg_days,
                        leg_step_days=leg_step_days,
                        neighbor_per_time=neighbor_per_time,
                        candidates_per_route=candidates_per_route,
                        maximum_leg_dv_kms=maximum_leg_dv_kms,
                        mass_cap_kg=mass_cap_kg,
                        target_priority=target_priority,
                    )
                )
            except (RuntimeError, ValueError, FloatingPointError):
                continue
            if index % 25 == 0:
                print(
                    f"beam depth={depth} expanded={index}/{len(beam)} "
                    f"children={len(children)}",
                    flush=True,
                )
        beam = select_diverse_beam(children, beam_width, target_priority)
        archive.extend(beam)
        print(
            f"beam depth={depth} kept={len(beam)} raw={len(children)} "
            f"best_dv={min((r.maneuver_dv_kms for r in beam), default=float('nan')):.3f} "
            f"elapsed={time.time()-started:.1f}s",
            flush=True,
        )
        if not beam:
            break
    return archive


def select_routes_to_realize(
    archive: Iterable[ImpulsiveRoute],
    count: int,
    minimum_targets: int,
    existing_frequency: dict[int, int],
) -> list[ImpulsiveRoute]:
    candidates = [
        route for route in archive if len(route.asteroid_ids) >= minimum_targets
    ]
    frequency = defaultdict(int)
    for route in candidates:
        for asteroid_id in set(route.asteroid_ids):
            frequency[asteroid_id] += 1
    selected: list[ImpulsiveRoute] = []
    selected_union: set[int] = set()
    remaining = {route.asteroid_ids: route for route in candidates}
    while remaining and len(selected) < count:
        def selection_score(route: ImpulsiveRoute) -> tuple[float, float, float]:
            coverage = set(route.asteroid_ids)
            new_targets = coverage - selected_union
            rarity = sum(1.0 / frequency[target] for target in new_targets)
            cache_rarity = sum(
                1.0 / (1.0 + existing_frequency.get(target, 0))
                for target in new_targets
            )
            # Marginal coverage dominates once route quality is already
            # bounded by mass_cap.  Rarity breaks ties toward hard targets.
            return (
                8.0 * len(new_targets)
                + 3.0 * rarity
                + 10.0 * cache_rarity
                + len(coverage)
                - route.estimated_cost,
                -route.maneuver_dv_kms,
                -route.last_day,
            )

        chosen = max(remaining.values(), key=selection_score)
        selected.append(chosen)
        selected_union.update(chosen.asteroid_ids)
        remaining.pop(chosen.asteroid_ids)
    return selected


def _copy_row(row: EventRow, *, event: int | None = None) -> EventRow:
    return EventRow(
        row.sc_id,
        row.event if event is None else event,
        float(row.time_days),
        np.array(row.state, copy=True),
        np.array(row.force, copy=True),
        row.asteroid_id,
    )


def scale_route(rows: Sequence[EventRow]) -> list[EventRow]:
    final_mass = float(rows[-1].state[6])
    if final_mass < MDRY_KG - 1e-7:
        raise ValueError("route ends below dry mass")
    scale = min(1.0, FINAL_MASS_KG / final_mass)
    result = []
    for row in rows:
        copied = _copy_row(row)
        copied.state[6] *= scale
        copied.force *= scale
        result.append(copied)
    return result


def route_cost(rows: Sequence[EventRow]) -> float:
    initial_mass = float(rows[0].state[6])
    x = (initial_mass - MDRY_KG) / 1400.0
    return 1.0 + x + x * x


def route_prefix_options(
    rows: Sequence[EventRow], source: str, minimum_targets: int = 2
) -> list[RouteOption]:
    encounter_indices = [index for index, row in enumerate(rows) if row.event == 3]
    options = []
    for encounter_count in range(minimum_targets, len(encounter_indices) + 1):
        encounter_index = encounter_indices[encounter_count - 1]
        if encounter_index + 1 >= len(rows):
            continue
        prefix = [_copy_row(row) for row in rows[: encounter_index + 2]]
        prefix[-1].event = 4
        prefix[-1].asteroid_id = 0
        prefix[-1].force[:] = 0.0
        scaled = scale_route(prefix)
        covered = frozenset(
            row.asteroid_id for row in scaled if row.event == 3
        )
        options.append(
            RouteOption(
                scaled,
                covered,
                route_cost(scaled),
                f"{source}:prefix{encounter_count}",
            )
        )
    return options


def realize_impulsive_route(
    route: ImpulsiveRoute,
    *,
    initial_mass_kg: float,
    time_offsets_days: tuple[float, ...],
    time_candidate_count: int,
    max_nfev: int,
    fuel_iterations: int,
) -> tuple[list[RouteOption], dict]:
    asteroids = load_asteroids()
    launch_state = np.r_[
        earth_state(route.launch_day)[:3],
        np.asarray(route.launch_velocity),
        initial_mass_kg,
    ]
    first_day = route.encounter_days[0]
    first_state = coast(launch_state, route.launch_day, first_day, rtol=8e-13)
    first_target = asteroid_state(asteroids, route.asteroid_ids[0], first_day)
    first_miss = float(np.linalg.norm(first_state[:3] - first_target[:3]) * AU_KM)
    if first_miss > 0.5:
        return [], {"status": "first_miss", "miss_km": first_miss}
    separator_day = first_day + 1.0 / DAY_S
    separator_state = coast(first_state, first_day, separator_day, rtol=8e-13)
    prefix = [
        EventRow(1, 0, route.launch_day, launch_state, np.zeros(3), 0),
        EventRow(
            1,
            3,
            first_day,
            first_state,
            np.zeros(3),
            route.asteroid_ids[0],
        ),
        EventRow(1, 2, separator_day, separator_state, np.zeros(3), 0),
    ]
    state = separator_state
    current_day = separator_day
    optimized_legs = []
    realized_ids = [route.asteroid_ids[0]]
    leg_messages = []
    for leg_index, asteroid_id in enumerate(route.asteroid_ids[1:], 1):
        planned_day = route.encounter_days[leg_index]
        ranked_times = []
        for offset in time_offsets_days:
            encounter_day = planned_day + offset
            if encounter_day <= current_day + 5.0 or encounter_day >= MISSION_DAYS - 1.0:
                continue
            try:
                ballistic = coast(
                    state,
                    current_day,
                    encounter_day,
                    rtol=3e-8,
                )
                target = asteroid_state(asteroids, asteroid_id, encounter_day)
            except (RuntimeError, ValueError, FloatingPointError):
                continue
            duration = encounter_day - current_day
            acceleration_proxy = float(
                np.linalg.norm(target[:3] - ballistic[:3]) / duration**2
            )
            ranked_times.append((acceleration_proxy, encounter_day))
        ranked_times.sort()
        feasible = []
        for _proxy, encounter_day in ranked_times[:time_candidate_count]:
            candidate = Candidate(asteroid_id, encounter_day, 0.0, 0.0)
            try:
                result = optimize_candidate(
                    state,
                    current_day,
                    candidate,
                    asteroids,
                    max_nfev=max_nfev,
                )
                if result is not None:
                    feasible.append((result, candidate))
            except (RuntimeError, ValueError, FloatingPointError):
                continue
        if not feasible:
            leg_messages.append(f"target {asteroid_id}: no continuous-thrust solution")
            # A failed impulsive edge need not invalidate the rest of the
            # globally searched sequence.  Keep coasting from the last valid
            # encounter and try later planned targets at their absolute epochs.
            continue
        chosen, chosen_candidate = min(
            feasible, key=lambda item: item[0].fuel_kg
        )
        # Route-pool generation values breadth more than squeezing the last
        # fraction of a kilogram from every rejected candidate.  A zero value
        # deliberately skips SLSQP while retaining the high-accuracy shooting
        # and final independent validation performed by optimize_candidate.
        if fuel_iterations > 0:
            chosen = refine_candidate_fuel(
                state,
                current_day,
                chosen_candidate,
                asteroids,
                chosen,
                maxiter=fuel_iterations,
            )
        if chosen.leg.state_separator[6] < MDRY_KG + 0.2:
            leg_messages.append(f"target {asteroid_id}: mass cap exhausted")
            break
        optimized_legs.append(chosen)
        realized_ids.append(asteroid_id)
        state = chosen.leg.state_separator
        current_day = chosen.leg.separator_time
        leg_messages.append(
            f"target {asteroid_id}: fuel={chosen.fuel_kg:.3f}kg day={chosen.leg.encounter_time:.3f}"
        )

    rows = prefix[:2]
    if optimized_legs:
        rows.append(prefix[2])
        for index, optimized in enumerate(optimized_legs):
            append_leg_rows(
                rows,
                optimized.leg,
                is_last=index == len(optimized_legs) - 1,
            )
    else:
        final = prefix[2]
        final.event = 4
        rows.append(final)
    options = route_prefix_options(rows, "global", minimum_targets=2)
    return options, {
        "status": "ok",
        "planned_ids": list(route.asteroid_ids),
        "planned_days": list(route.encounter_days),
        "planned_targets": len(route.asteroid_ids),
        "realized_targets": len(realized_ids),
        "realized_ids": realized_ids,
        "first_miss_km": first_miss,
        "impulsive_dv_kms": route.maneuver_dv_kms,
        "messages": leg_messages,
    }


def _realize_worker(payload):
    (
        route,
        initial_mass_kg,
        time_offsets_days,
        time_candidate_count,
        max_nfev,
        fuel_iterations,
    ) = payload
    return realize_impulsive_route(
        route,
        initial_mass_kg=initial_mass_kg,
        time_offsets_days=time_offsets_days,
        time_candidate_count=time_candidate_count,
        max_nfev=max_nfev,
        fuel_iterations=fuel_iterations,
    )


def base_route_options(path: Path) -> list[RouteOption]:
    grouped: dict[int, list[EventRow]] = defaultdict(list)
    for row in load_rows(path):
        grouped[row.sc_id].append(row)
    options = []
    for sc_id, rows in sorted(grouped.items()):
        copied = [_copy_row(row) for row in rows]
        options.extend(
            route_prefix_options(copied, f"base:{sc_id}", minimum_targets=1)
        )
    return options


def base_full_route_options(path: Path) -> list[RouteOption]:
    """Return exactly one complete, validated option per base spacecraft."""
    grouped: dict[int, list[EventRow]] = defaultdict(list)
    for row in load_rows(path):
        grouped[row.sc_id].append(row)
    result = []
    for sc_id, rows in sorted(grouped.items()):
        options = route_prefix_options(
            [_copy_row(row) for row in rows],
            f"base:{sc_id}",
            minimum_targets=1,
        )
        if not options:
            continue
        result.append(max(options, key=lambda option: len(option.rows)))
    return result


def deduplicate_options(options: Iterable[RouteOption]) -> list[RouteOption]:
    best: dict[frozenset[int], RouteOption] = {}
    for option in options:
        if not option.covered:
            continue
        old = best.get(option.covered)
        if old is None or option.cost < old.cost - 1e-12:
            best[option.covered] = option
    return list(best.values())


def remove_dominated_options(options: Iterable[RouteOption]) -> list[RouteOption]:
    """Discard a route if a no-costlier route covers all of its targets."""
    unique = deduplicate_options(options)
    entries = [
        (
            option.cost,
            -len(option.covered),
            sum(1 << (asteroid_id - 1) for asteroid_id in option.covered),
            option,
        )
        for option in unique
        if option.cost < len(option.covered) - 1e-12
    ]
    entries.sort(key=lambda item: (item[0], item[1]))
    kept: list[tuple[int, RouteOption]] = []
    for _cost, _negative_size, mask, option in entries:
        if any(mask & kept_mask == mask for kept_mask, _kept_option in kept):
            continue
        kept.append((mask, option))
    return [option for _mask, option in kept]


def load_option_cache(path: Path) -> list[RouteOption]:
    if not path.exists():
        return []
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    options = []
    for item in payload:
        options.append(
            RouteOption(
                item["rows"],
                frozenset(item["covered"]),
                float(item["cost"]),
                str(item["source"]),
            )
        )
    return options


def save_option_cache(path: Path, options: Sequence[RouteOption]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = [
        {
            "rows": option.rows,
            "covered": sorted(option.covered),
            "cost": option.cost,
            "source": option.source,
        }
        for option in options
    ]
    with path.open("wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)


def promising_fleet_options(
    base_options: Sequence[RouteOption],
    cached_options: Sequence[RouteOption],
    *,
    cache_limit: int = 500,
    per_target: int = 4,
) -> list[RouteOption]:
    """Build a compact, high-value column set for repeated fleet solves."""
    chosen: dict[tuple[frozenset[int], float], RouteOption] = {}

    def add(option: RouteOption) -> None:
        chosen[(option.covered, round(option.cost, 12))] = option

    for option in base_options:
        add(option)
    ranked = sorted(
        cached_options,
        key=lambda option: (option.cost / len(option.covered), option.cost),
    )
    for option in ranked[:cache_limit]:
        add(option)
    if per_target > 0:
        by_target: dict[int, list[RouteOption]] = defaultdict(list)
        for option in cached_options:
            for asteroid_id in option.covered:
                by_target[asteroid_id].append(option)
        for asteroid_id in range(1, 301):
            target_options = sorted(
                by_target.get(asteroid_id, ()),
                key=lambda option: (
                    option.cost / len(option.covered),
                    option.cost,
                ),
            )
            for option in target_options[:per_target]:
                add(option)
    return list(chosen.values())


def select_fleet(
    options: list[RouteOption],
    *,
    time_limit: float = 120.0,
    mip_rel_gap: float = 0.0,
) -> tuple[list[RouteOption], set[int], float]:
    options = remove_dominated_options(options)
    route_count = len(options)
    variable_count = route_count + 300
    objective = np.r_[
        np.asarray([option.cost for option in options]),
        np.ones(300),
    ]
    coverage = np.zeros((300, variable_count))
    for route_index, option in enumerate(options):
        for asteroid_id in option.covered:
            coverage[asteroid_id - 1, route_index] = 1.0
    coverage[:, route_count:] = np.eye(300)
    result = milp(
        objective,
        integrality=np.ones(variable_count),
        bounds=Bounds(np.zeros(variable_count), np.ones(variable_count)),
        constraints=LinearConstraint(coverage, np.ones(300), np.full(300, np.inf)),
        options={"time_limit": time_limit, "mip_rel_gap": mip_rel_gap},
    )
    if result.x is None:
        raise RuntimeError(f"fleet MILP failed: {result.message}")
    selected = [
        option for index, option in enumerate(options) if result.x[index] > 0.5
    ]
    covered = set().union(*(option.covered for option in selected)) if selected else set()
    cost = sum(option.cost for option in selected) + 300 - len(covered)
    return selected, covered, float(cost)


def renumber_rows(options: Sequence[RouteOption]) -> list[EventRow]:
    result = []
    for sc_id, option in enumerate(options, 1):
        for row in option.rows:
            copied = _copy_row(row)
            copied.sc_id = sc_id
            result.append(copied)
    return result


def run_search(args: argparse.Namespace) -> dict:
    baseline = check(args.base)
    if not baseline["valid"]:
        raise RuntimeError("baseline submission is invalid")
    asteroids = load_asteroids()
    ephemeris = VectorEphemeris(asteroids)
    cached_before = deduplicate_options(load_option_cache(args.route_cache))
    existing_frequency: dict[int, int] = defaultdict(int)
    for option in cached_before:
        for asteroid_id in option.covered:
            existing_frequency[asteroid_id] += 1
    self_test = lambert_self_test(asteroids)
    print(json.dumps({"lambert_self_test": self_test}, indent=2), flush=True)

    seeds = scan_ballistic_launches(
        asteroids,
        ephemeris,
        launch_start_day=args.launch_start_day,
        launch_max_day=args.launch_max_day,
        launch_step_days=args.launch_step,
        minimum_duration_days=args.first_min_days,
        maximum_duration_days=args.first_max_days,
        duration_step_days=args.first_step,
        max_seeds=args.seed_count,
    )
    print(f"selected {len(seeds)} diverse zero-fuel launch seeds", flush=True)
    archive = impulsive_beam_search(
        seeds,
        ephemeris,
        beam_width=args.beam_width,
        maximum_targets=args.maximum_targets,
        minimum_leg_days=args.leg_min_days,
        maximum_leg_days=args.leg_max_days,
        leg_step_days=args.leg_step,
        neighbor_per_time=args.neighbors_per_time,
        candidates_per_route=args.children_per_route,
        maximum_leg_dv_kms=args.maximum_leg_dv,
        mass_cap_kg=args.mass_cap,
        diversity_seed=args.diversity_seed,
        existing_frequency=existing_frequency,
    )
    to_realize = select_routes_to_realize(
        archive,
        args.realize_count,
        args.minimum_realize_targets,
        existing_frequency,
    )
    planned_union = set().union(
        *(set(route.asteroid_ids) for route in to_realize)
    ) if to_realize else set()
    print(
        f"realizing {len(to_realize)} globally searched sequences with "
        f"m0={args.realize_mass:.1f}kg; planned union={len(planned_union)} targets",
        flush=True,
    )

    new_options: list[RouteOption] = []
    realization_log = []
    payloads = [
        (
            route,
            args.realize_mass,
            tuple(args.time_offsets),
            args.time_candidate_count,
            args.max_nfev,
            args.fuel_iterations,
        )
        for route in to_realize
    ]
    if args.workers == 1:
        iterator = enumerate(map(_realize_worker, payloads), 1)
        for index, (options, diagnostics) in iterator:
            for option in options:
                option.source = option.source.replace(
                    "global:",
                    f"global{args.diversity_seed}:route{index}:",
                    1,
                )
            new_options.extend(options)
            realization_log.append(diagnostics)
            if index % 5 == 0:
                save_option_cache(
                    args.route_cache,
                    deduplicate_options(cached_before + new_options),
                )
            print(
                f"realized {index}/{len(payloads)}: "
                f"{diagnostics['status']} "
                f"targets={diagnostics.get('realized_targets', 0)} "
                f"options={len(options)}",
                flush=True,
            )
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(_realize_worker, payload): index
                for index, payload in enumerate(payloads, 1)
            }
            completed = 0
            for future in as_completed(futures):
                completed += 1
                route_index = futures[future]
                try:
                    options, diagnostics = future.result()
                except Exception as exc:  # pragma: no cover - process boundary
                    options, diagnostics = [], {"status": "worker_error", "error": str(exc)}
                for option in options:
                    option.source = option.source.replace(
                        "global:",
                        f"global{args.diversity_seed}:route{route_index}:",
                        1,
                    )
                new_options.extend(options)
                realization_log.append(diagnostics)
                if completed % 5 == 0:
                    save_option_cache(
                        args.route_cache,
                        deduplicate_options(cached_before + new_options),
                    )
                print(
                    f"realized {completed}/{len(payloads)}: "
                    f"{diagnostics['status']} "
                    f"targets={diagnostics.get('realized_targets', 0)} "
                    f"options={len(options)}",
                    flush=True,
                )
    cached_options = deduplicate_options(
        cached_before + new_options
    )
    save_option_cache(args.route_cache, cached_options)
    options = deduplicate_options(base_route_options(args.base) + cached_options)
    selected, covered, estimated_j = select_fleet(options)
    if estimated_j > baseline["J"] + 1e-9:
        selected = base_full_route_options(args.base)
        covered = set().union(*(option.covered for option in selected))
        estimated_j = sum(option.cost for option in selected) + 300 - len(covered)
    rows = renumber_rows(selected)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_submission(
        args.output,
        rows,
        comments=[
            "CTOC14 global low-mass launch-window and route search",
            f"routes_in_pool={len(options)} selected={len(selected)} covered={len(covered)}",
            f"set_cover_objective={estimated_j:.12f}",
        ],
    )
    validation = check(args.output)
    sources = defaultdict(int)
    for option in selected:
        sources[option.source.split(":", 1)[0]] += 1
    summary = {
        "baseline_J": baseline["J"],
        "candidate_J": validation["J"],
        "improved": bool(
            validation["valid"] and validation["J"] < baseline["J"] - 1e-9
        ),
        "valid": validation["valid"],
        "spacecraft": validation["spacecraft"],
        "covered": validation["covered"],
        "missing": sorted(set(range(1, 301)) - set(validation["covered_ids"])),
        "pool_routes": len(options),
        "new_route_options": len(new_options),
        "cached_route_options": len(cached_options),
        "selected_sources": dict(sources),
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
        "best_new_routes": [
            {
                "source": option.source,
                "targets": len(option.covered),
                "cost": option.cost,
                "initial_mass_kg": float(option.rows[0].state[6]),
                "benefit": len(option.covered) - option.cost,
                "covered": sorted(option.covered),
            }
            for option in sorted(
                new_options,
                key=lambda item: (-(len(item.covered) - item.cost), item.cost),
            )[:30]
        ],
        "realizations": realization_log,
        "worst": validation["worst"],
        "errors": validation["errors"],
        "output": str(args.output),
        "realization_status_counts": dict(
            (status, sum(item.get("status") == status for item in realization_log))
            for status in sorted({item.get("status") for item in realization_log})
        ),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base", type=Path, default=Path("output/CTOC14_Result_TeamID.txt")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("output/candidate_global_low_mass.txt")
    )
    parser.add_argument(
        "--report", type=Path, default=Path("tmp/global_low_mass_report.json")
    )
    parser.add_argument(
        "--route-cache", type=Path, default=Path("tmp/global_route_pool.pkl")
    )
    parser.add_argument("--launch-start-day", type=float, default=0.0)
    parser.add_argument("--launch-max-day", type=float, default=3000.0)
    parser.add_argument("--launch-step", type=float, default=120.0)
    parser.add_argument("--first-min-days", type=float, default=80.0)
    parser.add_argument("--first-max-days", type=float, default=920.0)
    parser.add_argument("--first-step", type=float, default=60.0)
    parser.add_argument("--seed-count", type=int, default=450)
    parser.add_argument("--beam-width", type=int, default=90)
    parser.add_argument("--maximum-targets", type=int, default=12)
    parser.add_argument("--leg-min-days", type=float, default=60.0)
    parser.add_argument("--leg-max-days", type=float, default=900.0)
    parser.add_argument("--leg-step", type=float, default=30.0)
    parser.add_argument("--neighbors-per-time", type=int, default=3)
    parser.add_argument("--children-per-route", type=int, default=8)
    parser.add_argument("--maximum-leg-dv", type=float, default=4.5)
    parser.add_argument("--mass-cap", type=float, default=950.0)
    parser.add_argument("--minimum-realize-targets", type=int, default=5)
    parser.add_argument("--realize-count", type=int, default=36)
    parser.add_argument("--realize-mass", type=float, default=950.0)
    parser.add_argument(
        "--time-offsets", type=float, nargs="+", default=(-20.0, 0.0, 20.0)
    )
    parser.add_argument(
        "--time-candidate-count",
        type=int,
        default=2,
        help="number of cheaply prefiltered encounter epochs to shoot per target",
    )
    parser.add_argument("--max-nfev", type=int, default=45)
    parser.add_argument("--fuel-iterations", type=int, default=18)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--diversity-seed", type=int, default=0)
    parser.add_argument("--self-test", action="store_true")
    return parser


def main() -> None:
    args = make_parser().parse_args()
    asteroids = load_asteroids()
    if args.self_test:
        print(json.dumps(lambert_self_test(asteroids), indent=2))
        return
    if args.workers < 1:
        raise SystemExit("--workers must be positive")
    if args.time_candidate_count < 1:
        raise SystemExit("--time-candidate-count must be positive")
    if not (FINAL_MASS_KG < args.mass_cap <= 2000.0):
        raise SystemExit("--mass-cap must be in (600.1, 2000]")
    if not (FINAL_MASS_KG < args.realize_mass <= args.mass_cap + 1e-9):
        raise SystemExit("--realize-mass must be in (600.1, mass-cap]")
    summary = run_search(args)
    raise SystemExit(0 if summary["valid"] else 1)


if __name__ == "__main__":
    main()
