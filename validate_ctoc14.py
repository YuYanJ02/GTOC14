#!/usr/bin/env python3
"""Independent submission checker for the trajectory subset used by the solver."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import argparse
import math

import numpy as np

from ctoc14_core import (
    AU_KM,
    DAY_S,
    MDRY_KG,
    M0_MAX_KG,
    MISSION_DAYS,
    TMAX_N,
    asteroid_state,
    coast,
    earth_state,
    load_asteroids,
    _integrate,
    thrust_rhs,
)


def parse(path: Path) -> list[dict]:
    rows = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        f = line.split()
        if len(f) != 15:
            raise ValueError(f"expected 15 columns, got {len(f)}: {line}")
        rows.append(
            {
                "line": int(f[0]),
                "sc": int(f[1]),
                "event": int(f[2]),
                "time": float(f[3]) / DAY_S,
                "state": np.r_[
                    np.asarray(f[4:7], float) / AU_KM,
                    np.asarray(f[7:10], float) * DAY_S / AU_KM,
                    float(f[10]),
                ],
                "force": np.asarray(f[11:14], float),
                "asteroid": int(f[14]),
            }
        )
    return rows


def check(path: Path) -> dict:
    asteroids = load_asteroids()
    rows = parse(path)
    errors: list[str] = []
    if [row["line"] for row in rows] != list(range(1, len(rows) + 1)):
        errors.append("Line column is not consecutive")
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["sc"]].append(row)
    if sorted(grouped) != list(range(1, len(grouped) + 1)):
        errors.append("SC_ID values are not consecutive from 1")

    covered: set[int] = set()
    initial_masses = []
    worst = {"position_km": 0.0, "velocity_kms": 0.0, "mass_kg": 0.0, "flyby_km": 0.0}
    for sc_id, sc_rows in sorted(grouped.items()):
        if sc_rows[0]["event"] != 0 or sc_rows[-1]["event"] != 4:
            errors.append(f"SC {sc_id}: first/last event is not 0/4")
        times = [row["time"] for row in sc_rows]
        if any(b <= a for a, b in zip(times, times[1:])):
            errors.append(f"SC {sc_id}: times are not strictly increasing")
        if times[0] < 0 or times[-1] > MISSION_DAYS:
            errors.append(f"SC {sc_id}: event outside mission window")

        launch = sc_rows[0]
        initial_masses.append(launch["state"][6])
        e = earth_state(launch["time"])
        launch_position_error = np.linalg.norm(launch["state"][:3] - e[:3]) * AU_KM
        vinf = np.linalg.norm(launch["state"][3:6] - e[3:6]) * AU_KM / DAY_S
        if launch_position_error > 1.0 or vinf > 4.01:
            errors.append(
                f"SC {sc_id}: launch error {launch_position_error:.6g} km, v_inf {vinf:.6g} km/s"
            )
        if not (MDRY_KG <= launch["state"][6] <= M0_MAX_KG):
            errors.append(f"SC {sc_id}: invalid initial mass")

        # Identify each thrust arc.  Event 3 does not split an arc; 0/2/4 does.
        arc_for_interval: dict[int, tuple[list[dict], float, float]] = {}
        pending: list[tuple[int, dict]] = []
        for index, row in enumerate(sc_rows):
            if row["event"] == 1:
                pending.append((index, row))
            elif row["event"] in (0, 2, 4):
                if pending:
                    samples = [item[1] for item in pending]
                    if len(samples) != 2:
                        errors.append(f"SC {sc_id}: checker expects exactly two samples per thrust arc")
                    if len(samples) >= 2 and samples[1]["time"] - samples[0]["time"] < 0.1 - 1e-12:
                        errors.append(f"SC {sc_id}: thrust sample spacing below 0.1 day")
                    first_i, last_i = pending[0][0], pending[-1][0]
                    for interval_i in range(first_i - 1, index):
                        if interval_i >= 0:
                            arc_for_interval[interval_i] = (samples, samples[0]["time"], samples[-1]["time"])
                    pending = []

        for row in sc_rows:
            force_norm = np.linalg.norm(row["force"])
            if row["event"] == 1:
                if force_norm > TMAX_N + 1e-12:
                    errors.append(f"SC {sc_id}: thrust sample exceeds 0.5 N")
            elif force_norm > 1e-14:
                errors.append(f"SC {sc_id}: nonzero force on Event={row['event']}")

        for index, (left, right) in enumerate(zip(sc_rows, sc_rows[1:])):
            predicted = np.array(left["state"], copy=True)
            arc = arc_for_interval.get(index)
            if arc is None:
                predicted = coast(predicted, left["time"], right["time"], rtol=2e-12)
            else:
                samples, t1, t2 = arc
                a = max(left["time"], t1)
                b = min(right["time"], t2)
                if b <= a:
                    predicted = coast(
                        predicted, left["time"], right["time"], rtol=2e-12
                    )
                else:
                    if a > left["time"]:
                        predicted = coast(predicted, left["time"], a, rtol=2e-12)
                    predicted = _integrate(
                        thrust_rhs,
                        predicted,
                        a,
                        b,
                        rtol=2e-12,
                        args=(t1, t2, samples[0]["force"], samples[1]["force"]),
                    )
                    if right["time"] > b:
                        predicted = coast(predicted, b, right["time"], rtol=2e-12)
            dr = np.linalg.norm(predicted[:3] - right["state"][:3]) * AU_KM
            dv = np.linalg.norm(predicted[3:6] - right["state"][3:6]) * AU_KM / DAY_S
            dm = abs(predicted[6] - right["state"][6])
            worst["position_km"] = max(worst["position_km"], dr)
            worst["velocity_kms"] = max(worst["velocity_kms"], dv)
            worst["mass_kg"] = max(worst["mass_kg"], dm)
            if dr > 1.0 or dv > 0.001 or dm > 0.01:
                errors.append(
                    f"SC {sc_id} lines {left['line']}-{right['line']}: "
                    f"dr={dr:.6g} km dv={dv:.6g} km/s dm={dm:.6g} kg"
                )
            if predicted[6] < MDRY_KG - 1e-8:
                errors.append(f"SC {sc_id}: mass below dry mass")

        for row in sc_rows:
            if row["event"] != 3:
                continue
            if row["asteroid"] not in asteroids:
                errors.append(f"SC {sc_id}: invalid asteroid ID")
                continue
            target = asteroid_state(asteroids, row["asteroid"], row["time"])
            distance = np.linalg.norm(row["state"][:3] - target[:3]) * AU_KM
            worst["flyby_km"] = max(worst["flyby_km"], distance)
            if distance > 1000.0:
                errors.append(
                    f"SC {sc_id} line {row['line']}: flyby distance {distance:.6g} km"
                )
            else:
                covered.add(row["asteroid"])

    cost = 300 - len(covered)
    for mass in initial_masses:
        x = (mass - MDRY_KG) / 1400.0
        cost += 1.0 + x + x * x
    return {
        "valid": not errors,
        "spacecraft": len(grouped),
        "covered": len(covered),
        "covered_ids": sorted(covered),
        "J": cost,
        "worst": worst,
        "errors": errors,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("submission", type=Path)
    args = parser.parse_args()
    result = check(args.submission)
    print(f"valid={result['valid']}")
    print(f"spacecraft={result['spacecraft']} covered={result['covered']} J={result['J']:.12g}")
    print("covered_ids=" + ",".join(map(str, result["covered_ids"])))
    for key, value in result["worst"].items():
        print(f"worst_{key}={value:.12g}")
    for error in result["errors"]:
        print("ERROR", error)
    raise SystemExit(0 if result["valid"] else 1)


if __name__ == "__main__":
    main()
