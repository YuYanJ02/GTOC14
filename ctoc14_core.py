#!/usr/bin/env python3
"""Core dynamics and file-format utilities for CTOC14 problem A.

The internal units are AU, day, kg, and newton.  Text submissions are written
in the units required by the statement: km, km/s, second, kg, and newton.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import math
import sys
from typing import Iterable, Sequence

ROOT = Path(__file__).resolve().parent
DEPS = ROOT / ".deps"
if DEPS.exists():
    sys.path.insert(0, str(DEPS))

import numpy as np
from scipy.integrate import solve_ivp


MU_KM = 1.32712440018e11
AU_KM = 149597870.7
DAY_S = 86400.0
G0 = 9.80665
ISP = 4000.0
TMAX_N = 0.5
MDRY_KG = 600.0
M0_MAX_KG = 2000.0
T0_MJD = 62502.0
NEA_EPOCH_MJD = 61200.0
EARTH_EPOCH_MJD = 60676.0
MISSION_DAYS = 5478.75
MU = MU_KM * DAY_S**2 / AU_KM**3
THRUST_ACCEL = DAY_S**2 / (AU_KM * 1000.0)  # (N/kg) -> AU/day^2
MDOT = DAY_S / (ISP * G0)  # |T| [N] -> kg/day

EARTH_ELEMENTS = np.array(
    [
        1.0009175020,
        0.017566762041,
        0.002976847126,
        189.953211282428,
        273.196254000254,
        357.4135031077,
    ],
    dtype=float,
)


def load_asteroids(path: str | Path = ROOT / "MEA.txt") -> dict[int, np.ndarray]:
    data: dict[int, np.ndarray] = {}
    with Path(path).open("r", encoding="utf-8") as handle:
        for raw in handle:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            fields = line.split()
            if len(fields) != 7:
                raise ValueError(f"bad MEA row: {raw.rstrip()}")
            asteroid_id = int(fields[0])
            data[asteroid_id] = np.asarray(fields[1:], dtype=float)
    if sorted(data) != list(range(1, 301)):
        raise ValueError("MEA.txt must contain asteroid IDs 1..300 exactly once")
    return data


def solve_kepler(mean_anomaly: float, eccentricity: float) -> float:
    """Solve M = E - e sin(E), accepting an unwrapped M."""
    m = math.remainder(mean_anomaly, 2.0 * math.pi)
    e_anomaly = m if eccentricity < 0.8 else math.copysign(math.pi, m or 1.0)
    for _ in range(20):
        f = e_anomaly - eccentricity * math.sin(e_anomaly) - m
        fp = 1.0 - eccentricity * math.cos(e_anomaly)
        step = f / fp
        e_anomaly -= step
        if abs(step) < 2e-15:
            break
    return e_anomaly


def elements_state(elements: Sequence[float], mjd: float, epoch_mjd: float) -> np.ndarray:
    """Return [r(AU), v(AU/day)] for classical elliptic elements."""
    a, e, inc_deg, node_deg, argp_deg, m0_deg = map(float, elements)
    inc, node, argp = np.deg2rad([inc_deg, node_deg, argp_deg])
    n = math.sqrt(MU / a**3)
    mean_anomaly = math.radians(m0_deg) + n * (mjd - epoch_mjd)
    ecc_anomaly = solve_kepler(mean_anomaly, e)
    ce, se = math.cos(ecc_anomaly), math.sin(ecc_anomaly)
    beta = math.sqrt(1.0 - e * e)

    r_pf = np.array([a * (ce - e), a * beta * se, 0.0])
    denom = 1.0 - e * ce
    v_pf = np.array([-a * n * se / denom, a * n * beta * ce / denom, 0.0])

    co, so = math.cos(node), math.sin(node)
    cw, sw = math.cos(argp), math.sin(argp)
    ci, si = math.cos(inc), math.sin(inc)
    rotation = np.array(
        [
            [co * cw - so * sw * ci, -co * sw - so * cw * ci, so * si],
            [so * cw + co * sw * ci, -so * sw + co * cw * ci, -co * si],
            [sw * si, cw * si, ci],
        ]
    )
    return np.concatenate((rotation @ r_pf, rotation @ v_pf))


def earth_state(time_days: float) -> np.ndarray:
    return elements_state(EARTH_ELEMENTS, T0_MJD + time_days, EARTH_EPOCH_MJD)


def asteroid_state(asteroids: dict[int, np.ndarray], asteroid_id: int, time_days: float) -> np.ndarray:
    return elements_state(
        asteroids[asteroid_id], T0_MJD + time_days, NEA_EPOCH_MJD
    )


def coast_rhs(_time: float, state: np.ndarray) -> np.ndarray:
    r = state[:3]
    rnorm = np.linalg.norm(r)
    out = np.empty_like(state)
    out[:3] = state[3:6]
    out[3:6] = -MU * r / rnorm**3
    if len(state) > 6:
        out[6] = 0.0
    return out


def thrust_rhs(
    time: float,
    state: np.ndarray,
    start: float,
    end: float,
    force_start: np.ndarray,
    force_end: np.ndarray,
) -> np.ndarray:
    r = state[:3]
    mass = state[6]
    fraction = 0.0 if end == start else (time - start) / (end - start)
    force = (1.0 - fraction) * force_start + fraction * force_end
    out = np.empty_like(state)
    out[:3] = state[3:6]
    out[3:6] = -MU * r / np.linalg.norm(r) ** 3 + force / mass * THRUST_ACCEL
    out[6] = -np.linalg.norm(force) * MDOT
    return out


def _integrate(
    rhs,
    state: np.ndarray,
    start: float,
    end: float,
    *,
    rtol: float = 3e-11,
    atol: float = 1e-12,
    args: tuple = (),
) -> np.ndarray:
    if end <= start:
        return np.array(state, dtype=float, copy=True)
    result = solve_ivp(
        rhs,
        (start, end),
        state,
        method="DOP853",
        rtol=rtol,
        atol=atol,
        args=args,
    )
    if not result.success:
        raise RuntimeError(result.message)
    return result.y[:, -1]


def coast(state: np.ndarray, start: float, end: float, *, rtol: float = 3e-11) -> np.ndarray:
    return _integrate(coast_rhs, state, start, end, rtol=rtol)


@dataclass
class LegResult:
    asteroid_id: int
    start_time: float
    sample_start: float
    sample_end: float
    encounter_time: float
    separator_time: float
    force_start: np.ndarray
    force_end: np.ndarray
    state_sample_start: np.ndarray
    state_sample_end: np.ndarray
    state_encounter: np.ndarray
    state_separator: np.ndarray


def propagate_leg(
    state: np.ndarray,
    start_time: float,
    asteroid_id: int,
    encounter_time: float,
    force_start: Sequence[float],
    force_end: Sequence[float],
    *,
    gap_days: float = 0.1,
    tail_days: float = 1.0 / DAY_S,
    separator_days: float = 1.0 / DAY_S,
    rtol: float = 3e-11,
) -> LegResult:
    """Propagate a leg represented by two linearly interpolated thrust samples."""
    sample_start = start_time + gap_days
    sample_end = encounter_time - tail_days
    if sample_end - sample_start < 0.1:
        raise ValueError("leg is too short for two valid thrust samples")
    f0 = np.asarray(force_start, dtype=float)
    f1 = np.asarray(force_end, dtype=float)
    if np.linalg.norm(f0) > TMAX_N + 1e-12 or np.linalg.norm(f1) > TMAX_N + 1e-12:
        raise ValueError("sample thrust exceeds 0.5 N")

    y_start = coast(state, start_time, sample_start, rtol=rtol)
    y_end = _integrate(
        thrust_rhs,
        y_start,
        sample_start,
        sample_end,
        rtol=rtol,
        args=(sample_start, sample_end, f0, f1),
    )
    y_encounter = coast(y_end, sample_end, encounter_time, rtol=rtol)
    separator_time = encounter_time + separator_days
    y_separator = coast(y_encounter, encounter_time, separator_time, rtol=rtol)
    return LegResult(
        asteroid_id=asteroid_id,
        start_time=start_time,
        sample_start=sample_start,
        sample_end=sample_end,
        encounter_time=encounter_time,
        separator_time=separator_time,
        force_start=f0,
        force_end=f1,
        state_sample_start=y_start,
        state_sample_end=y_end,
        state_encounter=y_encounter,
        state_separator=y_separator,
    )


def bounded_vector(raw: Sequence[float], maximum: float) -> np.ndarray:
    raw_array = np.asarray(raw, dtype=float)
    return maximum * raw_array / math.sqrt(1.0 + float(raw_array @ raw_array))


def inverse_bounded_vector(vector: Sequence[float], maximum: float) -> np.ndarray:
    value = np.asarray(vector, dtype=float)
    ratio2 = float(value @ value) / maximum**2
    ratio2 = min(ratio2, 1.0 - 1e-12)
    return value / (maximum * math.sqrt(1.0 - ratio2))


@dataclass
class EventRow:
    sc_id: int
    event: int
    time_days: float
    state: np.ndarray
    force: np.ndarray
    asteroid_id: int = 0

    def values(self, line_number: int) -> list[float | int]:
        position_km = self.state[:3] * AU_KM
        velocity_kms = self.state[3:6] * AU_KM / DAY_S
        return [
            line_number,
            self.sc_id,
            self.event,
            self.time_days * DAY_S,
            *position_km,
            *velocity_kms,
            self.state[6],
            *self.force,
            self.asteroid_id,
        ]


def format_row(row: EventRow, line_number: int) -> str:
    values = row.values(line_number)
    first = f"{values[0]:d} {values[1]:d} {values[2]:d}"
    floats = " ".join(f"{float(value):.16e}" for value in values[3:14])
    return f"{first} {floats} {int(values[14])}"


def write_submission(path: str | Path, rows: Iterable[EventRow], comments: Sequence[str] = ()) -> None:
    output = []
    for comment in comments:
        output.append(f"# {comment}")
    for line_number, row in enumerate(rows, 1):
        output.append(format_row(row, line_number))
    Path(path).write_text("\n".join(output) + "\n", encoding="utf-8")

