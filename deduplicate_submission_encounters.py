#!/usr/bin/env python3
"""Remove repeated asteroid declarations without changing any trajectory arc.

Event 3 is an observation marker and does not split a thrust arc.  When an
asteroid is already declared elsewhere in the fleet, later Event-3 rows can be
omitted: the adjacent states remain on the same coast or interpolated-thrust
arc, unique coverage is unchanged, and the submission becomes unambiguous.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ctoc14_core import write_submission
from extend_ctoc14_fleet import load_rows
from validate_ctoc14 import check


def run(source: Path, output: Path) -> dict:
    before = check(source)
    if not before["valid"]:
        raise RuntimeError(f"invalid source submission: {before['errors']}")

    seen: set[int] = set()
    kept = []
    removed: list[tuple[int, int]] = []
    for row in load_rows(source):
        if row.event == 3:
            if row.asteroid_id in seen:
                removed.append((row.sc_id, row.asteroid_id))
                continue
            seen.add(row.asteroid_id)
        kept.append(row)

    output.parent.mkdir(parents=True, exist_ok=True)
    write_submission(
        output,
        kept,
        comments=[
            "CTOC14 globally searched fleet; redundant Event-3 declarations removed",
            f"removed_duplicate_encounters={len(removed)} unique_coverage={len(seen)}",
        ],
    )
    after = check(output)
    if not after["valid"]:
        raise RuntimeError(f"deduplicated submission is invalid: {after['errors']}")
    if set(before["covered_ids"]) != set(after["covered_ids"]):
        raise RuntimeError("unique coverage changed during deduplication")
    if abs(before["J"] - after["J"]) > 2e-7:
        raise RuntimeError("objective changed during deduplication")
    return {
        "source_J": before["J"],
        "candidate_J": after["J"],
        "valid": after["valid"],
        "spacecraft": after["spacecraft"],
        "covered": after["covered"],
        "removed": len(removed),
        "removed_174": sum(asteroid_id == 174 for _sc, asteroid_id in removed),
        "output": str(output),
        "worst": after["worst"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    summary = run(args.source, args.output)
    for key, value in summary.items():
        print(f"{key}={value}")


if __name__ == "__main__":
    main()
