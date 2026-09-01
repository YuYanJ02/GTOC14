#!/usr/bin/env python3
"""Print compact route and objective statistics for a CTOC14 submission."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np

from ctoc14_core import AU_KM, DAY_S, MDRY_KG, earth_state
from validate_ctoc14 import check, parse


def summarize(path: Path) -> dict:
    rows = parse(path)
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["sc"]].append(row)

    seen: set[int] = set()
    spacecraft = []
    for sc_id, sc_rows in sorted(grouped.items()):
        route = [
            row["asteroid"] for row in sc_rows if row["event"] == 3
        ]
        new_targets = [target for target in route if target not in seen]
        seen.update(route)
        launch = sc_rows[0]
        initial_mass = float(launch["state"][6])
        x = (initial_mass - MDRY_KG) / 1400.0
        vinf = float(
            np.linalg.norm(
                launch["state"][3:6] - earth_state(launch["time"])[3:6]
            )
            * AU_KM
            / DAY_S
        )
        spacecraft.append(
            {
                "sc": sc_id,
                "launch_day": launch["time"],
                "end_day": sc_rows[-1]["time"],
                "encounters": len(route),
                "new_unique": len(new_targets),
                "route": route,
                "initial_mass_kg": initial_mass,
                "final_mass_kg": float(sc_rows[-1]["state"][6]),
                "launch_cost": 1.0 + x + x * x,
                "vinf_kms": vinf,
            }
        )

    validation = check(path)
    return {
        "file": str(path),
        "rows": len(rows),
        "valid": validation["valid"],
        "spacecraft": len(grouped),
        "covered": validation["covered"],
        "missing_ids": sorted(set(range(1, 301)) - set(validation["covered_ids"])),
        "J": validation["J"],
        "launch_cost_sum": validation["J"] - (300 - validation["covered"]),
        "latest_day": max(row["time"] for row in rows),
        "max_vinf_kms": max(item["vinf_kms"] for item in spacecraft),
        "minimum_final_mass_kg": min(
            item["final_mass_kg"] for item in spacecraft
        ),
        "worst": validation["worst"],
        "spacecraft_routes": spacecraft,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("submission", type=Path)
    args = parser.parse_args()
    print(json.dumps(summarize(args.submission), indent=2))


if __name__ == "__main__":
    main()
