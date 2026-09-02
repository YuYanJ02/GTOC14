#!/usr/bin/env python3
"""Global multi-start search whose route rollouts use the true low-thrust state.

Lambert transfers are used only to scan the complete Earth-launch window for a
zero-propellant first encounter.  Every later target is chosen and shot again
from the spacecraft's *actual* continuous-thrust arrival state.  A collection
of randomized, rarity-aware rollouts is accumulated and a binary set-cover
problem chooses the final fleet together with prefixes of the validated base
routes.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path
from typing import Sequence

import numpy as np

from ctoc14_core import (
    AU_KM,
    DAY_S,
    MDRY_KG,
    MISSION_DAYS,
    EventRow,
    asteroid_state,
    coast,
    earth_state,
    load_asteroids,
    write_submission,
)
from search_low_mass_global import (
    FINAL_MASS_KG,
    ImpulsiveRoute,
    VectorEphemeris,
    base_full_route_options,
    base_route_options,
    coast_samples,
    deduplicate_options,
    lambert_self_test,
    load_option_cache,
    promising_fleet_options,
    renumber_rows,
    route_prefix_options,
    save_option_cache,
    scan_ballistic_launches,
    select_fleet,
)
from solve_ctoc14 import Candidate, append_leg_rows, optimize_candidate
from validate_ctoc14 import check


def actual_state_candidates(
    state: np.ndarray,
    current_day: float,
    ephemeris: VectorEphemeris,
    visited: set[int],
    target_priority: np.ndarray,
    *,
    forced_target: int | None = None,
    minimum_days: float,
    maximum_days: float,
    step_days: float,
    neighbors_per_time: int,
    keep: int,
) -> list[Candidate]:
    """Rank endpoints using a coast from the actual continuous-thrust state."""
    final_day = min(current_day + maximum_days, MISSION_DAYS - 2.0)
    sample_days = np.arange(
        current_day + minimum_days,
        final_day + 0.1,
        step_days,
    )
    if not len(sample_days):
        return []
    ballistic_positions = coast_samples(
        state[:3], state[3:6], current_day, sample_days
    )
    target_positions = ephemeris.positions(sample_days)
    distances = np.linalg.norm(
        target_positions - ballistic_positions[:, None, :], axis=2
    )
    if visited:
        distances[:, np.asarray(sorted(visited), dtype=int) - 1] = np.inf

    durations = sample_days - current_day
    # distance / duration tracks the propellant needed for a constant-accel
    # correction.  The small time term preserves room for later encounters.
    base_scores = distances / durations[:, None] + 2.0e-7 * durations[:, None]
    priority_scale = 1.0 + 0.28 * target_priority[1:]
    ranked_scores = base_scores / priority_scale[None, :]

    # First identify promising asteroid/time cells on the cheap grid.  The
    # shooting epoch must not remain locked to that grid: at typical relative
    # velocities even a few days of timing error turns a near-ballistic flyby
    # into an unnecessarily expensive powered correction.  A three-point
    # parabola of squared separation gives a cheap closest-approach epoch,
    # after which one vectorized coast recomputes the exact proxy distance.
    grid_pairs: list[tuple[float, int, int, int | None]] = []
    count = min(neighbors_per_time, 300 - len(visited))
    if count <= 0:
        return []
    for time_index, encounter_day in enumerate(sample_days):
        nearest = np.argpartition(ranked_scores[time_index], count - 1)[:count]
        for asteroid_index in nearest:
            distance = float(distances[time_index, asteroid_index])
            if not np.isfinite(distance):
                continue
            grid_pairs.append(
                (
                    float(ranked_scores[time_index, asteroid_index]),
                    time_index,
                    int(asteroid_index),
                    None,
                )
            )

    if forced_target is not None and forced_target not in visited:
        asteroid_index = forced_target - 1
        target_scores = ranked_scores[:, asteroid_index]
        finite_indices = np.flatnonzero(np.isfinite(target_scores))
        best_indices = sorted(
            finite_indices,
            key=lambda time_index: float(target_scores[time_index]),
        )[:2]
        for rank, time_index in enumerate(best_indices):
            grid_pairs.append(
                (
                    -1.0e6 + rank,
                    int(time_index),
                    asteroid_index,
                    rank,
                )
            )

    best_grid_pair: dict[
        tuple[int, int], tuple[float, int, int, int | None]
    ] = {}
    for item in grid_pairs:
        ranked_score, time_index, asteroid_index, forced_rank = item
        key = (time_index, asteroid_index)
        old = best_grid_pair.get(key)
        if old is None or ranked_score < old[0]:
            best_grid_pair[key] = item

    refined_specs: list[tuple[float, int, float, int | None]] = []
    for ranked_score, time_index, asteroid_index, forced_rank in best_grid_pair.values():
        encounter_day = float(sample_days[time_index])
        if 0 < time_index < len(sample_days) - 1:
            ym = float(distances[time_index - 1, asteroid_index] ** 2)
            y0 = float(distances[time_index, asteroid_index] ** 2)
            yp = float(distances[time_index + 1, asteroid_index] ** 2)
            denominator = ym - 2.0 * y0 + yp
            if denominator > 1e-24:
                half_width = 0.5 * float(
                    sample_days[time_index + 1] - sample_days[time_index - 1]
                )
                offset = 0.5 * (ym - yp) / denominator * half_width
                encounter_day += float(
                    np.clip(offset, -0.95 * half_width, 0.95 * half_width)
                )
        refined_specs.append(
            (ranked_score, asteroid_index, encounter_day, forced_rank)
        )

    refined_days = np.asarray(
        sorted({spec[2] for spec in refined_specs}), dtype=float
    )
    refined_ballistic = coast_samples(
        state[:3], state[3:6], current_day, refined_days
    )
    refined_targets = ephemeris.positions(refined_days)
    day_index = {float(day): index for index, day in enumerate(refined_days)}

    best_by_pair: dict[tuple[int, int], tuple[float, Candidate]] = {}
    for _grid_score, asteroid_index, encounter_day, forced_rank in refined_specs:
        index = day_index[float(encounter_day)]
        distance = float(
            np.linalg.norm(
                refined_targets[index, asteroid_index] - refined_ballistic[index]
            )
        )
        duration = encounter_day - current_day
        base_score = distance / duration + 2.0e-7 * duration
        ranked_score = base_score / priority_scale[asteroid_index]
        if forced_rank is not None:
            # Explicit bottleneck attempts are evaluated before the ordinary
            # nearest-neighbour list.  They still must pass the full shooting
            # solve and thrust/mass constraints.
            ranked_score = -1.0e6 + forced_rank
        candidate = Candidate(
            asteroid_index + 1,
            encounter_day,
            float(base_score),
            distance,
        )
        key = (candidate.asteroid_id, int(round(candidate.encounter_time * 10.0)))
        old = best_by_pair.get(key)
        if old is None or ranked_score < old[0]:
            best_by_pair[key] = (ranked_score, candidate)
    return [
        item[1]
        for item in sorted(best_by_pair.values(), key=lambda item: item[0])[:keep]
    ]


def select_rollout_seeds(
    seeds: Sequence[ImpulsiveRoute],
    count: int,
    existing_frequency: dict[int, int],
    diversity_seed: int,
    priority_targets: Sequence[int] = (),
) -> list[ImpulsiveRoute]:
    """Round-robin over rare first targets and distinct launch epochs."""
    random = np.random.default_rng(diversity_seed)
    by_target: dict[int, list[ImpulsiveRoute]] = defaultdict(list)
    for route in seeds:
        by_target[route.last_id].append(route)
    target_jitter = {asteroid_id: float(random.random()) for asteroid_id in by_target}
    priority_set = set(priority_targets)
    targets = sorted(
        by_target,
        key=lambda asteroid_id: (
            asteroid_id not in priority_set,
            -(1.0 / (1.0 + existing_frequency.get(asteroid_id, 0))),
            -target_jitter[asteroid_id],
            asteroid_id,
        ),
    )
    for asteroid_id, routes in by_target.items():
        # Earlier first encounters leave more mission time, while the random
        # perturbation makes repeated passes sample different launch families.
        route_jitter = random.random(len(routes))
        decorated = list(zip(routes, route_jitter))
        decorated.sort(
            key=lambda item: (
                item[0].last_day + 240.0 * item[1],
                item[0].launch_day,
            )
        )
        by_target[asteroid_id] = [item[0] for item in decorated]

    selected: list[ImpulsiveRoute] = []
    rank = 0
    while len(selected) < count:
        added = False
        for asteroid_id in targets:
            routes = by_target[asteroid_id]
            if rank < len(routes):
                selected.append(routes[rank])
                added = True
                if len(selected) >= count:
                    break
        if not added:
            break
        rank += 1
    return selected


def continuous_rollout(payload):
    (
        seed,
        route_index,
        initial_mass_kg,
        maximum_targets,
        stop_reserve_kg,
        minimum_leg_days,
        maximum_leg_days,
        leg_step_days,
        neighbors_per_time,
        candidate_keep,
        candidate_attempts,
        feasible_per_leg,
        max_nfev,
        existing_frequency,
        diversity_seed,
        priority_targets,
        priority_groups,
        priority_boost,
        rarity_reward_kg,
    ) = payload
    asteroids = load_asteroids()
    ephemeris = VectorEphemeris(asteroids)
    random = np.random.default_rng(diversity_seed * 100_003 + route_index)
    rarity = np.asarray(
        [0.0]
        + [1.0 / (1.0 + existing_frequency.get(i, 0)) for i in range(1, 301)]
    )
    target_priority = random.random(301) + 3.0 * rarity
    target_priority[0] = 0.0
    assigned_group: tuple[int, ...] = ()
    if priority_groups:
        matching_groups = [group for group in priority_groups if seed.last_id in group]
        raw_group = (
            matching_groups[0]
            if matching_groups
            else priority_groups[(route_index - 1) % len(priority_groups)]
        )
        remaining = [asteroid_id for asteroid_id in raw_group if asteroid_id != seed.last_id]
        random.shuffle(remaining)
        assigned_group = tuple(
            ([seed.last_id] if seed.last_id in raw_group else []) + remaining
        )
    elif priority_targets:
        # Cycle the strongest reward so a batch attacks several bottlenecks
        # instead of making every rollout chase the same rare asteroid.  When
        # the ballistic seed already starts at a requested bottleneck, regard
        # that target as satisfied.  This aligns the zero-fuel first encounter
        # with fleet-level needs instead of needlessly forcing a different
        # bottleneck later in the same route.
        if seed.last_id in priority_targets:
            assigned_group = (seed.last_id,)
        else:
            assigned_group = (
                priority_targets[(route_index - 1) % len(priority_targets)],
            )
    if priority_targets:
        for asteroid_id in priority_targets:
            target_priority[asteroid_id] += 0.25 * priority_boost
        for asteroid_id in assigned_group:
            target_priority[asteroid_id] += priority_boost

    launch_state = np.r_[
        earth_state(seed.launch_day)[:3],
        np.asarray(seed.launch_velocity),
        initial_mass_kg,
    ]
    first_day = seed.encounter_days[0]
    first_state = coast(launch_state, seed.launch_day, first_day, rtol=8e-13)
    first_target = asteroid_state(asteroids, seed.last_id, first_day)
    first_miss_km = float(
        np.linalg.norm(first_state[:3] - first_target[:3]) * AU_KM
    )
    if first_miss_km > 0.5:
        return [], {
            "status": "first_miss",
            "route_index": route_index,
            "miss_km": first_miss_km,
        }
    separator_day = first_day + 1.0 / DAY_S
    separator_state = coast(first_state, first_day, separator_day, rtol=8e-13)
    rows = [
        EventRow(1, 0, seed.launch_day, launch_state, np.zeros(3), 0),
        EventRow(1, 3, first_day, first_state, np.zeros(3), seed.last_id),
        EventRow(1, 2, separator_day, separator_state, np.zeros(3), 0),
    ]
    state = separator_state
    current_day = separator_day
    visited = {seed.last_id}
    leg_log = []

    while len(visited) < maximum_targets and state[6] > MDRY_KG + stop_reserve_kg:
        forced_target = next(
            (asteroid_id for asteroid_id in assigned_group if asteroid_id not in visited),
            None,
        )
        candidates = actual_state_candidates(
            state,
            current_day,
            ephemeris,
            visited,
            target_priority,
            forced_target=forced_target,
            minimum_days=minimum_leg_days,
            maximum_days=maximum_leg_days,
            step_days=leg_step_days,
            neighbors_per_time=neighbors_per_time,
            keep=candidate_keep,
        )
        feasible = []
        for candidate in candidates[:candidate_attempts]:
            try:
                optimized = optimize_candidate(
                    state,
                    current_day,
                    candidate,
                    asteroids,
                    max_nfev=max_nfev,
                )
            except (RuntimeError, ValueError, FloatingPointError):
                optimized = None
            if optimized is not None:
                feasible.append((optimized, candidate))
                if len(feasible) >= feasible_per_leg:
                    break
        if not feasible:
            # A wider recovery horizon is important for unusual arrival
            # velocities, but is attempted only after the normal cheap list.
            candidates = actual_state_candidates(
                state,
                current_day,
                ephemeris,
                visited,
                target_priority,
                forced_target=forced_target,
                minimum_days=max(240.0, minimum_leg_days),
                maximum_days=min(1000.0, maximum_leg_days * 1.8),
                step_days=max(45.0, leg_step_days),
                neighbors_per_time=neighbors_per_time,
                keep=candidate_keep,
            )
            for candidate in candidates[:candidate_attempts]:
                try:
                    optimized = optimize_candidate(
                        state,
                        current_day,
                        candidate,
                        asteroids,
                        max_nfev=max_nfev,
                    )
                except (RuntimeError, ValueError, FloatingPointError):
                    optimized = None
                if optimized is not None:
                    feasible.append((optimized, candidate))
                    if len(feasible) >= feasible_per_leg:
                        break
        if not feasible:
            leg_log.append("no feasible continuation")
            break

        def choice_key(item):
            optimized, candidate = item
            if forced_target is not None and candidate.asteroid_id == forced_target:
                return -1.0e6 + optimized.fuel_kg
            duration = candidate.encounter_time - current_day
            reward = rarity_reward_kg * target_priority[candidate.asteroid_id]
            return optimized.fuel_kg + 0.018 * duration - reward

        optimized, candidate = min(feasible, key=choice_key)
        if optimized.leg.state_separator[6] < MDRY_KG + stop_reserve_kg:
            leg_log.append(
                f"target {candidate.asteroid_id}: rejected at reserve boundary"
            )
            break
        append_leg_rows(rows, optimized.leg, is_last=False)
        visited.add(candidate.asteroid_id)
        state = optimized.leg.state_separator
        current_day = optimized.leg.separator_time
        leg_log.append(
            f"target {candidate.asteroid_id}: fuel={optimized.fuel_kg:.3f}kg "
            f"day={optimized.leg.encounter_time:.3f}"
        )

    rows[-1].event = 4
    rows[-1].asteroid_id = 0
    rows[-1].force[:] = 0.0
    source = f"continuous{diversity_seed}:route{route_index}"
    options = route_prefix_options(rows, source, minimum_targets=2)
    return options, {
        "status": "ok",
        "route_index": route_index,
        "launch_day": seed.launch_day,
        "first_target": seed.last_id,
        "first_day": first_day,
        "first_miss_km": first_miss_km,
        "assigned_group": list(assigned_group),
        "targets": len(visited),
        "target_ids": sorted(visited),
        "final_day": current_day,
        "unscaled_final_mass_kg": float(state[6]),
        "messages": leg_log,
    }


def run(args: argparse.Namespace) -> dict:
    baseline = check(args.base)
    if not baseline["valid"]:
        raise RuntimeError("baseline submission is invalid")
    priority_targets = list(dict.fromkeys(int(i) for i in args.priority_targets))
    priority_groups: list[tuple[int, ...]] = []
    if args.priority_weak_routes > 0:
        base_routes = base_full_route_options(args.base)
        base_frequency = Counter(
            asteroid_id for option in base_routes for asteroid_id in option.covered
        )
        weakness = []
        for option in base_routes:
            unique = sorted(
                asteroid_id
                for asteroid_id in option.covered
                if base_frequency[asteroid_id] == 1
            )
            weakness.append((len(unique), len(option.covered), option.cost, unique))
        for _unique_count, _target_count, _cost, unique in sorted(weakness)[
            : args.priority_weak_routes
        ]:
            priority_targets.extend(unique)
            if unique:
                priority_groups.append(tuple(unique))
    if args.priority_missing:
        priority_targets.extend(
            sorted(set(range(1, 301)) - set(baseline["covered_ids"]))
        )
    priority_targets = list(dict.fromkeys(priority_targets))
    invalid_priority = [i for i in priority_targets if not 1 <= i <= 300]
    if invalid_priority:
        raise ValueError(f"invalid priority targets: {invalid_priority}")
    if priority_targets:
        print(
            f"fleet-adaptive priority targets ({len(priority_targets)}): "
            + ",".join(map(str, priority_targets)),
            flush=True,
        )
    asteroids = load_asteroids()
    ephemeris = VectorEphemeris(asteroids)
    print(json.dumps({"lambert_self_test": lambert_self_test(asteroids)}, indent=2))
    cached_before = deduplicate_options(load_option_cache(args.route_cache))
    existing_frequency: dict[int, int] = defaultdict(int)
    for option in cached_before:
        for asteroid_id in option.covered:
            existing_frequency[asteroid_id] += 1

    launch_seeds = scan_ballistic_launches(
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
    rollout_seeds = select_rollout_seeds(
        launch_seeds,
        args.rollout_count,
        existing_frequency,
        args.diversity_seed,
        priority_targets,
    )
    print(
        f"realizing {len(rollout_seeds)} true-state rollouts from "
        f"{len(launch_seeds)} ballistic launch seeds",
        flush=True,
    )
    payloads = [
        (
            seed,
            index,
            args.initial_mass,
            args.maximum_targets,
            args.stop_reserve,
            args.leg_min_days,
            args.leg_max_days,
            args.leg_step,
            args.neighbors_per_time,
            args.candidate_keep,
            args.candidate_attempts,
            args.feasible_per_leg,
            args.max_nfev,
            existing_frequency,
            args.diversity_seed,
            tuple(priority_targets),
            tuple(priority_groups),
            args.priority_boost,
            args.rarity_reward_kg,
        )
        for index, seed in enumerate(rollout_seeds, 1)
    ]
    new_options = []
    diagnostics = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(continuous_rollout, payload): index
            for index, payload in enumerate(payloads, 1)
        }
        for completed, future in enumerate(as_completed(futures), 1):
            route_index = futures[future]
            try:
                options, diagnostic = future.result()
            except Exception as exc:  # pragma: no cover - process boundary
                options = []
                diagnostic = {
                    "status": "worker_error",
                    "route_index": route_index,
                    "error": str(exc),
                }
            new_options.extend(options)
            diagnostics.append(diagnostic)
            if completed % args.checkpoint_every == 0:
                save_option_cache(
                    args.route_cache,
                    deduplicate_options(cached_before + new_options),
                )
            print(
                f"rollout {completed}/{len(payloads)}: {diagnostic['status']} "
                f"targets={diagnostic.get('targets', 0)} options={len(options)}",
                flush=True,
            )

    cached = deduplicate_options(cached_before + new_options)
    save_option_cache(args.route_cache, cached)
    base_options = base_route_options(args.base)
    fleet_options = deduplicate_options(base_options + cached)
    solver_options = promising_fleet_options(
        base_options,
        cached,
        cache_limit=args.fleet_cache_limit,
        per_target=args.fleet_options_per_target,
    )
    selected, covered, estimated_j = select_fleet(
        solver_options,
        time_limit=args.fleet_time_limit,
        mip_rel_gap=args.fleet_mip_gap,
    )
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
            "CTOC14 true-state continuous global multi-start search",
            f"routes_in_pool={len(fleet_options)} selected={len(selected)} covered={len(covered)}",
            f"set_cover_objective={estimated_j:.12f}",
        ],
    )
    validation = check(args.output)
    source_counts: dict[str, int] = defaultdict(int)
    for option in selected:
        source_counts[option.source.split(":", 1)[0]] += 1
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
        "priority_targets": priority_targets,
        "priority_groups": [list(group) for group in priority_groups],
        "new_route_options": len(new_options),
        "cached_route_options": len(cached),
        "selected_sources": dict(source_counts),
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
        "rollouts": diagnostics,
        "worst": validation["worst"],
        "errors": validation["errors"],
        "output": str(args.output),
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
        "--output", type=Path, default=Path("output/candidate_continuous_global.txt")
    )
    parser.add_argument(
        "--report", type=Path, default=Path("tmp/continuous_global_report.json")
    )
    parser.add_argument(
        "--route-cache", type=Path, default=Path("tmp/global_route_pool.pkl")
    )
    parser.add_argument("--launch-start-day", type=float, default=0.0)
    parser.add_argument("--launch-max-day", type=float, default=2400.0)
    parser.add_argument("--launch-step", type=float, default=120.0)
    parser.add_argument("--first-min-days", type=float, default=80.0)
    parser.add_argument("--first-max-days", type=float, default=920.0)
    parser.add_argument("--first-step", type=float, default=60.0)
    parser.add_argument("--seed-count", type=int, default=500)
    parser.add_argument("--rollout-count", type=int, default=20)
    parser.add_argument("--initial-mass", type=float, default=900.0)
    parser.add_argument("--maximum-targets", type=int, default=15)
    parser.add_argument("--stop-reserve", type=float, default=8.0)
    parser.add_argument("--leg-min-days", type=float, default=70.0)
    parser.add_argument("--leg-max-days", type=float, default=520.0)
    parser.add_argument("--leg-step", type=float, default=30.0)
    parser.add_argument("--neighbors-per-time", type=int, default=4)
    parser.add_argument("--candidate-keep", type=int, default=18)
    parser.add_argument("--candidate-attempts", type=int, default=5)
    parser.add_argument("--feasible-per-leg", type=int, default=2)
    parser.add_argument("--max-nfev", type=int, default=30)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--checkpoint-every", type=int, default=2)
    parser.add_argument("--fleet-cache-limit", type=int, default=420)
    parser.add_argument("--fleet-options-per-target", type=int, default=4)
    parser.add_argument("--fleet-time-limit", type=float, default=30.0)
    parser.add_argument("--fleet-mip-gap", type=float, default=0.002)
    parser.add_argument("--diversity-seed", type=int, default=0)
    parser.add_argument("--priority-targets", type=int, nargs="*", default=())
    parser.add_argument(
        "--priority-weak-routes",
        type=int,
        default=0,
        help="prioritize unique targets of this many weakest base-fleet routes",
    )
    parser.add_argument(
        "--priority-missing",
        action="store_true",
        help="also prioritize targets currently absent from the base fleet",
    )
    parser.add_argument("--priority-boost", type=float, default=10.0)
    parser.add_argument(
        "--rarity-reward-kg",
        type=float,
        default=2.0,
        help="fuel-equivalent reward used when choosing rare feasible targets",
    )
    parser.add_argument("--self-test", action="store_true")
    return parser


def main() -> None:
    args = make_parser().parse_args()
    if args.self_test:
        print(json.dumps(lambert_self_test(load_asteroids()), indent=2))
        return
    if args.workers < 1 or args.rollout_count < 1:
        raise SystemExit("--workers and --rollout-count must be positive")
    if not (FINAL_MASS_KG < args.initial_mass <= 2000.0):
        raise SystemExit("--initial-mass must be in (600.1, 2000]")
    if args.candidate_attempts < args.feasible_per_leg:
        raise SystemExit("candidate attempts must be >= feasible-per-leg")
    if args.priority_weak_routes < 0:
        raise SystemExit("--priority-weak-routes must be nonnegative")
    if any(asteroid_id < 1 or asteroid_id > 300 for asteroid_id in args.priority_targets):
        raise SystemExit("priority target IDs must be in [1, 300]")
    summary = run(args)
    raise SystemExit(0 if summary["valid"] else 1)


if __name__ == "__main__":
    main()
