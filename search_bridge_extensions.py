"""Extend validated route skeletons to complete other fleet routes.

The global rollouts already produce long, low-cost trajectories, but many of
them touch only one or two targets from each incumbent spacecraft.  Such a
route looks excellent in isolation and still cannot remove a spacecraft from
the fleet.  This search works backwards from that fleet-level bottleneck:

* retain a validated base or cached route as a dynamically exact skeleton;
* scale its mass and thrust by the same factor, preserving its trajectory;
* continue from the true terminal state with additional propellant;
* explicitly target the few missing asteroids needed to subsume another
  incumbent route; and
* send every feasible prefix back to the shared route-pool set cover.

Only independently validated candidates are written.  The caller can run the
LNS fleet optimizer afterwards against the enlarged cache.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Sequence

import numpy as np

from ctoc14_core import (
    MDRY_KG,
    M0_MAX_KG,
    MISSION_DAYS,
    EventRow,
    load_asteroids,
    write_submission,
)
from search_continuous_global import actual_state_candidates
from search_low_mass_global import (
    RouteOption,
    VectorEphemeris,
    base_full_route_options,
    base_route_options,
    coast_samples,
    deduplicate_options,
    load_option_cache,
    promising_fleet_options,
    remove_dominated_options,
    renumber_rows,
    route_prefix_options,
    save_option_cache,
    select_fleet,
)
from solve_ctoc14 import Candidate, append_leg_rows, optimize_candidate
from validate_ctoc14 import check


FINAL_MASS_KG = MDRY_KG + 0.1
ALL_TARGETS = frozenset(range(1, 301))


@dataclass(frozen=True)
class SkeletonJob:
    option: RouteOption
    skeleton_index: int
    victim_groups: tuple[tuple[int, ...], ...]
    victim_route_ids: tuple[int, ...]
    desired_initial_mass_kg: float
    diversity_seed: int


def copy_row(row: EventRow) -> EventRow:
    return EventRow(
        row.sc_id,
        row.event,
        float(row.time_days),
        np.array(row.state, copy=True),
        np.array(row.force, copy=True),
        row.asteroid_id,
    )


def extension_scale_cap(option: RouteOption, thrust_limit_n: float = 0.49) -> float:
    initial_mass = float(option.rows[0].state[6])
    maximum_force = max(float(np.linalg.norm(row.force)) for row in option.rows)
    mass_cap = M0_MAX_KG / initial_mass
    force_cap = thrust_limit_n / maximum_force if maximum_force > 1e-14 else np.inf
    return float(min(mass_cap, force_cap))


def scaled_skeleton(
    option: RouteOption,
    desired_initial_mass_kg: float,
) -> tuple[list[EventRow], float]:
    initial_mass = float(option.rows[0].state[6])
    desired_scale = desired_initial_mass_kg / initial_mass
    scale = min(desired_scale, extension_scale_cap(option))
    if scale <= 1.000001:
        raise ValueError("skeleton has no safe mass/thrust scaling headroom")
    rows = []
    for row in option.rows:
        copied = copy_row(row)
        copied.state[6] *= scale
        copied.force *= scale
        rows.append(copied)
    rows[-1].event = 2
    rows[-1].asteroid_id = 0
    rows[-1].force[:] = 0.0
    return rows, float(scale)


def focused_state_candidates(
    state: np.ndarray,
    current_day: float,
    ephemeris: VectorEphemeris,
    focus_targets: Sequence[int],
    *,
    minimum_days: float,
    maximum_days: float,
    step_days: float,
    times_per_target: int,
) -> list[Candidate]:
    """Find refined closest-approach epochs for several requested targets."""
    requested = tuple(dict.fromkeys(int(target) for target in focus_targets))
    if not requested:
        return []
    final_day = min(current_day + maximum_days, MISSION_DAYS - 2.0)
    sample_days = np.arange(current_day + minimum_days, final_day + 0.1, step_days)
    if not len(sample_days):
        return []
    ballistic = coast_samples(state[:3], state[3:6], current_day, sample_days)
    target_positions = ephemeris.positions(sample_days)
    durations = sample_days - current_day

    specifications: list[tuple[int, int, float]] = []
    for asteroid_id in requested:
        asteroid_index = asteroid_id - 1
        distances = np.linalg.norm(
            target_positions[:, asteroid_index] - ballistic,
            axis=1,
        )
        proxy = distances / durations + 2.0e-7 * durations
        chosen_indices: list[int] = []
        for time_index in np.argsort(proxy):
            if all(abs(int(time_index) - old) >= 2 for old in chosen_indices):
                chosen_indices.append(int(time_index))
            if len(chosen_indices) >= times_per_target:
                break
        for time_index in chosen_indices:
            encounter_day = float(sample_days[time_index])
            if 0 < time_index < len(sample_days) - 1:
                ym = float(distances[time_index - 1] ** 2)
                y0 = float(distances[time_index] ** 2)
                yp = float(distances[time_index + 1] ** 2)
                denominator = ym - 2.0 * y0 + yp
                if denominator > 1e-24:
                    half_width = 0.5 * float(
                        sample_days[time_index + 1] - sample_days[time_index - 1]
                    )
                    offset = 0.5 * (ym - yp) / denominator * half_width
                    encounter_day += float(
                        np.clip(offset, -0.95 * half_width, 0.95 * half_width)
                    )
            specifications.append((asteroid_id, time_index, encounter_day))

    refined_days = np.asarray(
        sorted({item[2] for item in specifications}),
        dtype=float,
    )
    refined_ballistic = coast_samples(
        state[:3], state[3:6], current_day, refined_days
    )
    refined_targets = ephemeris.positions(refined_days)
    day_index = {float(day): index for index, day in enumerate(refined_days)}
    result = []
    for asteroid_id, _time_index, encounter_day in specifications:
        index = day_index[float(encounter_day)]
        distance = float(
            np.linalg.norm(
                refined_targets[index, asteroid_id - 1] - refined_ballistic[index]
            )
        )
        duration = encounter_day - current_day
        score = distance / duration + 2.0e-7 * duration
        result.append(Candidate(asteroid_id, encounter_day, score, distance))
    result.sort(key=lambda candidate: (candidate.score, candidate.encounter_time))
    return result


def remaining_group_targets(
    victim_groups: Sequence[Sequence[int]],
    visited: set[int],
) -> list[int]:
    """Order incomplete groups by how close they are to being fully covered."""
    incomplete = []
    for group_index, group in enumerate(victim_groups):
        remaining = [target for target in group if target not in visited]
        if remaining:
            incomplete.append((len(remaining), group_index, remaining))
    incomplete.sort()
    return [target for _count, _index, remaining in incomplete for target in remaining]


def diverse_candidate_prefix(
    candidates: Sequence[Candidate],
    limit: int,
) -> list[Candidate]:
    """Round-robin over target IDs before taking alternate encounter epochs."""
    by_target: dict[int, list[Candidate]] = defaultdict(list)
    for candidate in candidates:
        by_target[candidate.asteroid_id].append(candidate)
    target_order = sorted(
        by_target,
        key=lambda asteroid_id: (
            by_target[asteroid_id][0].score,
            asteroid_id,
        ),
    )
    selected = []
    rank = 0
    while len(selected) < limit:
        added = False
        for asteroid_id in target_order:
            group = by_target[asteroid_id]
            if rank < len(group):
                selected.append(group[rank])
                added = True
                if len(selected) >= limit:
                    break
        if not added:
            break
        rank += 1
    return selected


def bridge_rollout(payload) -> tuple[list[RouteOption], dict]:
    (
        job,
        maximum_additional_targets,
        stop_reserve_kg,
        minimum_leg_days,
        maximum_leg_days,
        leg_step_days,
        neighbors_per_time,
        ordinary_keep,
        focus_attempts,
        ordinary_attempts,
        feasible_per_leg,
        times_per_target,
        max_nfev,
        focus_reward_kg,
        completion_reward_kg,
        focus_diversity_kg,
        time_weight,
    ) = payload
    asteroids = load_asteroids()
    ephemeris = VectorEphemeris(asteroids)
    random = np.random.default_rng(job.diversity_seed * 100_003 + job.skeleton_index)
    try:
        rows, scale = scaled_skeleton(job.option, job.desired_initial_mass_kg)
    except ValueError as exc:
        return [], {"status": "no_headroom", "error": str(exc)}

    state = np.array(rows[-1].state, copy=True)
    current_day = float(rows[-1].time_days)
    visited = set(job.option.covered)
    original_target_count = len(visited)
    target_priority = random.random(301)
    target_priority[0] = 0.0
    for group in job.victim_groups:
        for asteroid_id in group:
            target_priority[asteroid_id] += 12.0
    focus_preference = {
        asteroid_id: float(random.uniform(-focus_diversity_kg, focus_diversity_kg))
        for group in job.victim_groups
        for asteroid_id in group
    }
    messages = []

    for _leg_index in range(maximum_additional_targets):
        if state[6] <= FINAL_MASS_KG + stop_reserve_kg:
            messages.append("reserve boundary")
            break
        focus_order = remaining_group_targets(job.victim_groups, visited)
        focused = focused_state_candidates(
            state,
            current_day,
            ephemeris,
            focus_order,
            minimum_days=minimum_leg_days,
            maximum_days=maximum_leg_days,
            step_days=leg_step_days,
            times_per_target=times_per_target,
        )
        ordinary = actual_state_candidates(
            state,
            current_day,
            ephemeris,
            visited,
            target_priority,
            minimum_days=minimum_leg_days,
            maximum_days=maximum_leg_days,
            step_days=leg_step_days,
            neighbors_per_time=neighbors_per_time,
            keep=ordinary_keep,
        )
        attempts: list[Candidate] = []
        seen: set[tuple[int, int]] = set()
        focus_prefix = diverse_candidate_prefix(focused, focus_attempts)
        for candidate in focus_prefix + ordinary[:ordinary_attempts]:
            if candidate.asteroid_id in visited:
                continue
            key = (
                candidate.asteroid_id,
                int(round(candidate.encounter_time * 10.0)),
            )
            if key not in seen:
                attempts.append(candidate)
                seen.add(key)

        feasible = []
        for candidate in attempts:
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
            if (
                optimized is not None
                and optimized.leg.state_separator[6]
                >= FINAL_MASS_KG + stop_reserve_kg
            ):
                feasible.append((optimized, candidate))
                if len(feasible) >= feasible_per_leg:
                    break
        if not feasible:
            messages.append("no feasible continuation")
            break

        def choice_key(item) -> float:
            optimized, candidate = item
            completions = sum(
                candidate.asteroid_id in group
                and all(
                    target in visited or target == candidate.asteroid_id
                    for target in group
                )
                for group in job.victim_groups
            )
            is_focus = any(
                candidate.asteroid_id in group for group in job.victim_groups
            )
            reward = focus_reward_kg if is_focus else 0.0
            reward += focus_preference.get(candidate.asteroid_id, 0.0)
            reward += completion_reward_kg * completions
            duration = candidate.encounter_time - current_day
            return optimized.fuel_kg + time_weight * duration - reward

        optimized, candidate = min(feasible, key=choice_key)
        if optimized.leg.state_separator[6] < FINAL_MASS_KG + stop_reserve_kg:
            messages.append(
                f"target {candidate.asteroid_id}: rejected at reserve boundary"
            )
            break
        append_leg_rows(rows, optimized.leg, is_last=False)
        visited.add(candidate.asteroid_id)
        state = optimized.leg.state_separator
        current_day = optimized.leg.separator_time
        completed = [
            route_id
            for route_id, group in zip(job.victim_route_ids, job.victim_groups)
            if set(group) <= visited
        ]
        messages.append(
            f"target {candidate.asteroid_id}: fuel={optimized.fuel_kg:.3f}kg "
            f"day={optimized.leg.encounter_time:.3f} completed={completed}"
        )

    rows[-1].event = 4
    rows[-1].asteroid_id = 0
    rows[-1].force[:] = 0.0
    source = f"bridge{job.diversity_seed}:skeleton{job.skeleton_index}"
    options = route_prefix_options(
        rows,
        source,
        minimum_targets=original_target_count + 1,
    )
    completed_routes = [
        route_id
        for route_id, group in zip(job.victim_route_ids, job.victim_groups)
        if set(group) <= visited
    ]
    return options, {
        "status": "ok",
        "skeleton_index": job.skeleton_index,
        "skeleton_source": job.option.source,
        "skeleton_targets": original_target_count,
        "skeleton_end_day": float(job.option.rows[-1].time_days),
        "scale": scale,
        "scaled_initial_mass_kg": float(rows[0].state[6]),
        "victim_route_ids": list(job.victim_route_ids),
        "victim_groups": [list(group) for group in job.victim_groups],
        "completed_victim_routes": completed_routes,
        "targets": len(visited),
        "added_targets": len(visited) - original_target_count,
        "target_ids": sorted(visited),
        "final_day": current_day,
        "unscaled_final_mass_kg": float(state[6]),
        "messages": messages,
    }


def option_family(option: RouteOption) -> str:
    return option.source.rsplit(":prefix", 1)[0]


def choose_skeletons(
    base_routes: Sequence[RouteOption],
    cached: Sequence[RouteOption],
    *,
    base_count: int,
    cache_count: int,
    maximum_end_day: float,
    minimum_scale_cap: float,
) -> list[RouteOption]:
    eligible_base = [
        option
        for option in base_routes
        if option.rows[-1].time_days <= maximum_end_day
        and extension_scale_cap(option) >= minimum_scale_cap
    ]
    eligible_base.sort(
        key=lambda option: (
            option.rows[-1].time_days + 55.0 * len(option.covered),
            option.cost,
        )
    )

    base_sets = [option.covered for option in base_routes]
    best_family: dict[str, tuple[float, RouteOption]] = {}
    for option in cached:
        end_day = float(option.rows[-1].time_days)
        scale_cap = extension_scale_cap(option)
        if (
            len(option.covered) < 5
            or end_day > maximum_end_day
            or scale_cap < minimum_scale_cap
        ):
            continue
        completed = sum(group <= option.covered for group in base_sets)
        nearly_completed = sum(
            0 < len(group - option.covered) <= 2 for group in base_sets
        )
        utility = (
            len(option.covered)
            + 4.0 * completed
            + 0.7 * nearly_completed
            - 0.0022 * max(0.0, end_day - 2600.0)
            - 0.8 * option.cost
            + 0.15 * min(scale_cap, 3.0)
        )
        family = option_family(option)
        old = best_family.get(family)
        if old is None or utility > old[0]:
            best_family[family] = (utility, option)
    cache_ranked = [
        item[1]
        for item in sorted(best_family.values(), key=lambda item: -item[0])
    ]
    # Equal coverage does not mean equal usefulness as an extension skeleton:
    # launch/arrival epochs and terminal velocity can be completely different.
    # Preserve those dynamic alternatives and remove only exact source repeats.
    result = []
    seen_sources = set()
    for option in eligible_base[:base_count] + cache_ranked[:cache_count]:
        key = (option.source, tuple(sorted(option.covered)))
        if key not in seen_sources:
            result.append(option)
            seen_sources.add(key)
    return result


def choose_victims(
    skeleton: RouteOption,
    base_routes: Sequence[RouteOption],
    *,
    repeat_index: int,
    diversity_seed: int,
    victim_routes_per_job: int,
    maximum_focus_targets: int,
) -> tuple[tuple[int, ...], tuple[tuple[int, ...], ...]]:
    random = np.random.default_rng(
        diversity_seed * 1_000_003 + repeat_index * 10_007 + len(skeleton.covered)
    )
    ranked = []
    for route_index, route in enumerate(base_routes, 1):
        remaining = tuple(sorted(route.covered - skeleton.covered))
        if not remaining or len(remaining) > maximum_focus_targets:
            continue
        overlap = len(route.covered) - len(remaining)
        jitter = float(random.random())
        score = (
            len(remaining),
            -overlap,
            len(route.covered),
            -route.cost,
            jitter,
        )
        ranked.append((score, route_index, remaining))
    ranked.sort(key=lambda item: item[0])
    if not ranked:
        return (), ()

    # Repeated jobs rotate through the best completion opportunities instead
    # of making every skeleton chase the same two hard asteroids.
    window = min(len(ranked), max(4, 3 * victim_routes_per_job))
    # Stagger equally sized victim routes across skeletons.  Without this
    # offset every early skeleton attacks the same shortest/highest-cost route,
    # even though a different terminal velocity may be the key to each target
    # family.
    start = (repeat_index + diversity_seed) % window
    selected = []
    focus_count = 0
    for offset in range(window):
        _score, route_index, remaining = ranked[(start + offset) % window]
        if focus_count + len(remaining) > maximum_focus_targets:
            continue
        selected.append((route_index, remaining))
        focus_count += len(remaining)
        if len(selected) >= victim_routes_per_job:
            break
    return (
        tuple(item[0] for item in selected),
        tuple(item[1] for item in selected),
    )


def run(args: argparse.Namespace) -> dict:
    baseline = check(args.base)
    if not baseline["valid"]:
        raise RuntimeError("baseline submission is invalid")
    base_full = base_full_route_options(args.base)
    priority_victim_ids = tuple(dict.fromkeys(args.priority_victim_routes))
    invalid_victims = [
        route_id
        for route_id in priority_victim_ids
        if route_id < 1 or route_id > len(base_full)
    ]
    if invalid_victims:
        raise ValueError(f"invalid priority victim route IDs: {invalid_victims}")
    cached_before = deduplicate_options(load_option_cache(args.route_cache))
    cached_skeletons = remove_dominated_options(cached_before)
    skeletons = choose_skeletons(
        base_full,
        cached_skeletons,
        base_count=args.base_skeletons,
        cache_count=args.cache_skeletons,
        maximum_end_day=args.skeleton_max_day,
        minimum_scale_cap=args.minimum_scale_cap,
    )
    if priority_victim_ids and args.priority_skeletons > 0:
        target_groups = [base_full[route_id - 1].covered for route_id in priority_victim_ids]
        best_by_family: dict[str, tuple[tuple, RouteOption]] = {}
        for option in cached_skeletons:
            if (
                option.rows[-1].time_days > args.skeleton_max_day
                or extension_scale_cap(option) < args.minimum_scale_cap
            ):
                continue
            missing_counts = [len(group - option.covered) for group in target_groups]
            positive = [count for count in missing_counts if 0 < count <= args.maximum_focus_targets]
            if not positive:
                continue
            rank = (
                min(positive),
                float(option.rows[-1].time_days),
                option.cost,
                -len(option.covered),
            )
            family = option_family(option)
            old = best_by_family.get(family)
            if old is None or rank < old[0]:
                best_by_family[family] = (rank, option)
        targeted = [
            item[1]
            for item in sorted(best_by_family.values(), key=lambda item: item[0])
        ][: args.priority_skeletons]
        existing = {(option.source, tuple(sorted(option.covered))) for option in skeletons}
        for option in reversed(targeted):
            key = (option.source, tuple(sorted(option.covered)))
            if key not in existing:
                skeletons.insert(0, option)
                existing.add(key)
    jobs = []
    for skeleton_index, skeleton in enumerate(skeletons, 1):
        for repeat_index in range(args.repeats_per_skeleton):
            if priority_victim_ids:
                selected = []
                for offset in range(len(priority_victim_ids)):
                    route_id = priority_victim_ids[
                        (repeat_index + offset) % len(priority_victim_ids)
                    ]
                    remaining = tuple(
                        sorted(base_full[route_id - 1].covered - skeleton.covered)
                    )
                    if remaining and len(remaining) <= args.maximum_focus_targets:
                        selected.append((route_id, remaining))
                    if len(selected) >= args.victim_routes_per_job:
                        break
                victim_ids = tuple(item[0] for item in selected)
                victim_groups = tuple(item[1] for item in selected)
            else:
                victim_ids, victim_groups = choose_victims(
                    skeleton,
                    base_full,
                    repeat_index=repeat_index,
                    diversity_seed=args.diversity_seed + skeleton_index,
                    victim_routes_per_job=args.victim_routes_per_job,
                    maximum_focus_targets=args.maximum_focus_targets,
                )
            if not victim_groups:
                continue
            jobs.append(
                SkeletonJob(
                    skeleton,
                    skeleton_index,
                    victim_groups,
                    victim_ids,
                    args.extension_initial_mass,
                    args.diversity_seed + repeat_index,
                )
            )
    if args.rollout_limit > 0:
        jobs = jobs[: args.rollout_limit]
    print(
        f"bridge skeletons={len(skeletons)} jobs={len(jobs)} "
        f"cached_before={len(cached_before)}",
        flush=True,
    )

    payloads = [
        (
            job,
            args.maximum_additional_targets,
            args.stop_reserve,
            args.leg_min_days,
            args.leg_max_days,
            args.leg_step,
            args.neighbors_per_time,
            args.ordinary_keep,
            args.focus_attempts,
            args.ordinary_attempts,
            args.feasible_per_leg,
            args.times_per_target,
            args.max_nfev,
            args.focus_reward_kg,
            args.completion_reward_kg,
            args.focus_diversity_kg,
            args.time_weight,
        )
        for job in jobs
    ]
    new_options: list[RouteOption] = []
    diagnostics = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(bridge_rollout, payload): index
            for index, payload in enumerate(payloads, 1)
        }
        for completed, future in enumerate(as_completed(futures), 1):
            job_index = futures[future]
            try:
                options, diagnostic = future.result()
            except Exception as exc:  # pragma: no cover - process boundary
                options = []
                diagnostic = {
                    "status": "worker_error",
                    "job_index": job_index,
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
                f"bridge {completed}/{len(payloads)}: {diagnostic['status']} "
                f"skeleton={diagnostic.get('skeleton_targets', 0)} "
                f"added={diagnostic.get('added_targets', 0)} "
                f"completed={diagnostic.get('completed_victim_routes', [])} "
                f"options={len(options)}",
                flush=True,
            )

    cached = deduplicate_options(cached_before + new_options)
    save_option_cache(args.route_cache, cached)
    base_options = base_route_options(args.base)
    compact = promising_fleet_options(
        base_options,
        cached,
        cache_limit=args.fleet_cache_limit,
        per_target=args.fleet_options_per_target,
    )
    solver_options = deduplicate_options(base_options + new_options + compact)
    selected, covered, estimated_j = select_fleet(
        solver_options,
        time_limit=args.fleet_time_limit,
        mip_rel_gap=args.fleet_mip_gap,
    )
    if estimated_j > baseline["J"] + 1e-9:
        selected = base_full
        covered = set().union(*(option.covered for option in selected))
        estimated_j = sum(option.cost for option in selected) + 300 - len(covered)
    rows = renumber_rows(selected)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_submission(
        args.output,
        rows,
        comments=[
            "CTOC14 targeted multi-route bridge-extension search",
            f"selected={len(selected)} covered={len(covered)} objective={estimated_j:.12f}",
        ],
    )
    validation = check(args.output)
    summary = {
        "baseline_J": baseline["J"],
        "candidate_J": validation["J"],
        "improved": bool(validation["valid"] and validation["J"] < baseline["J"] - 1e-9),
        "valid": validation["valid"],
        "spacecraft": validation["spacecraft"],
        "covered": validation["covered"],
        "missing": sorted(ALL_TARGETS - set(validation["covered_ids"])),
        "skeletons": len(skeletons),
        "jobs": len(jobs),
        "new_route_options": len(new_options),
        "cached_route_options": len(cached),
        "completed_victim_jobs": sum(
            bool(item.get("completed_victim_routes")) for item in diagnostics
        ),
        "selected_routes": [
            {
                "source": option.source,
                "targets": len(option.covered),
                "cost": option.cost,
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
    print(
        json.dumps(
            {key: value for key, value in summary.items() if key not in {"rollouts", "selected_routes"}},
            indent=2,
        ),
        flush=True,
    )
    return summary


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base",
        type=Path,
        default=Path("output/CTOC14_Result_TeamID.txt"),
    )
    parser.add_argument(
        "--route-cache",
        type=Path,
        default=Path("tmp/global_route_pool.pkl"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output/candidate_bridge_extensions.txt"),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("tmp/bridge_extensions_report.json"),
    )
    parser.add_argument("--base-skeletons", type=int, default=10)
    parser.add_argument("--cache-skeletons", type=int, default=14)
    parser.add_argument("--repeats-per-skeleton", type=int, default=2)
    parser.add_argument("--rollout-limit", type=int, default=0)
    parser.add_argument("--victim-routes-per-job", type=int, default=2)
    parser.add_argument(
        "--priority-victim-routes",
        type=int,
        nargs="*",
        default=(),
        help="1-based base-fleet route IDs to complete before adaptive victims",
    )
    parser.add_argument(
        "--priority-skeletons",
        type=int,
        default=18,
        help="extra early/near-complete cache skeletons for priority victims",
    )
    parser.add_argument("--maximum-focus-targets", type=int, default=8)
    parser.add_argument("--extension-initial-mass", type=float, default=1800.0)
    parser.add_argument("--minimum-scale-cap", type=float, default=1.12)
    parser.add_argument("--skeleton-max-day", type=float, default=4450.0)
    parser.add_argument("--maximum-additional-targets", type=int, default=10)
    parser.add_argument("--stop-reserve", type=float, default=8.0)
    parser.add_argument("--leg-min-days", type=float, default=55.0)
    parser.add_argument("--leg-max-days", type=float, default=900.0)
    parser.add_argument("--leg-step", type=float, default=35.0)
    parser.add_argument("--neighbors-per-time", type=int, default=4)
    parser.add_argument("--ordinary-keep", type=int, default=14)
    parser.add_argument("--focus-attempts", type=int, default=6)
    parser.add_argument("--ordinary-attempts", type=int, default=3)
    parser.add_argument("--feasible-per-leg", type=int, default=3)
    parser.add_argument("--times-per-target", type=int, default=2)
    parser.add_argument("--max-nfev", type=int, default=32)
    parser.add_argument("--focus-reward-kg", type=float, default=110.0)
    parser.add_argument("--completion-reward-kg", type=float, default=360.0)
    parser.add_argument("--focus-diversity-kg", type=float, default=55.0)
    parser.add_argument("--time-weight", type=float, default=0.014)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--checkpoint-every", type=int, default=2)
    parser.add_argument("--fleet-cache-limit", type=int, default=700)
    parser.add_argument("--fleet-options-per-target", type=int, default=7)
    parser.add_argument("--fleet-time-limit", type=float, default=90.0)
    parser.add_argument("--fleet-mip-gap", type=float, default=0.001)
    parser.add_argument("--diversity-seed", type=int, default=34)
    return parser


def main() -> None:
    args = make_parser().parse_args()
    if args.workers < 1 or args.repeats_per_skeleton < 1:
        raise SystemExit("worker and repeat counts must be positive")
    if not MDRY_KG < args.extension_initial_mass <= M0_MAX_KG:
        raise SystemExit("extension initial mass must be in (600, 2000]")
    summary = run(args)
    raise SystemExit(0 if summary["valid"] else 1)


if __name__ == "__main__":
    main()
