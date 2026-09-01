#!/usr/bin/env python3
"""Apply exact, trajectory-preserving mass scaling to a CTOC14 submission.

For one spacecraft, multiplying every recorded mass and every thrust vector by
the same positive factor leaves both translational acceleration ``T / m`` and
the mass-flow equation unchanged.  The position and velocity history therefore
remain identical, while a factor below one reduces the launch-cost term.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from ctoc14_core import MDRY_KG, EventRow, write_submission
from validate_ctoc14 import check, parse


def optimize_mass_scaling(
    input_path: Path,
    output_path: Path,
    *,
    dry_margin_kg: float = 0.1,
) -> dict:
    """Scale each spacecraft independently and write a validated submission."""
    if dry_margin_kg <= 0.0:
        raise ValueError("dry-mass margin must be positive")

    parsed = parse(input_path)
    grouped: dict[int, list[dict]] = defaultdict(list)
    for row in parsed:
        grouped[row["sc"]].append(row)

    target_minimum = MDRY_KG + dry_margin_kg
    scales: dict[int, float] = {}
    launch_masses: dict[int, float] = {}
    optimized_rows: list[EventRow] = []
    for sc_id, rows in sorted(grouped.items()):
        minimum_mass = min(float(row["state"][6]) for row in rows)
        if minimum_mass < MDRY_KG - 1e-8:
            raise ValueError(f"SC {sc_id} is already below dry mass")
        scale = min(1.0, target_minimum / minimum_mass)
        scales[sc_id] = scale

        for row in rows:
            state = np.array(row["state"], dtype=float, copy=True)
            force = np.array(row["force"], dtype=float, copy=True)
            state[6] *= scale
            force *= scale
            optimized_rows.append(
                EventRow(
                    sc_id=sc_id,
                    event=row["event"],
                    time_days=row["time"],
                    state=state,
                    force=force,
                    asteroid_id=row["asteroid"],
                )
            )
        launch_masses[sc_id] = float(rows[0]["state"][6]) * scale

    covered_ids = {
        row["asteroid"] for row in parsed if row["event"] == 3
    }
    cost = 300.0 - len(covered_ids)
    for mass in launch_masses.values():
        x = (mass - MDRY_KG) / 1400.0
        cost += 1.0 + x + x * x

    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_submission(
        output_path,
        optimized_rows,
        comments=[
            "CTOC14 trajectory-preserving mass-scaled submission",
            f"spacecraft={len(grouped)} covered={len(covered_ids)} J={cost:.12f}",
            f"dry_mass_margin_kg={dry_margin_kg:.12f}",
        ],
    )

    result = check(output_path)
    result["scales"] = scales
    result["launch_masses_kg"] = launch_masses
    if not result["valid"]:
        raise RuntimeError(json.dumps(result, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--dry-margin-kg", type=float, default=0.1)
    args = parser.parse_args()
    result = optimize_mass_scaling(
        args.input,
        args.output,
        dry_margin_kg=args.dry_margin_kg,
    )
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
