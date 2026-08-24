"""Parse individual lander SD-card log lines (the ``gems_*.txt`` payload grammar).

Only R:, V: (new 3-field format), P: (6- or 7-field), and !: (README
detailed-status shape) are recognized -- everything else (the undocumented
PM: tag, the untimestamped TS,/TP,/PS,/VS,/etc. tags that never occur in real
data, and old-format V: lines) returns ``None`` and is skipped by the reader.
"""

from __future__ import annotations

from datetime import datetime

_NULL_TOKENS = {"NA", "N/A", "NAN", "NULL", ""}


def _parse_ts(s: str) -> datetime | None:
    s = s.strip()
    if s.endswith("Z"):
        s = s[:-1]
    try:
        return datetime.fromisoformat(s)
    except ValueError:
        return None


def _parse_float(s: str) -> float | None:
    s = s.strip()
    if s.upper() in _NULL_TOKENS:
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _parse_r(payload: str) -> dict | None:
    fields = payload[2:].split(",")
    if len(fields) != 3:
        return None
    ts_str, mass_str, current_str = fields
    ts = _parse_ts(ts_str)
    mass = _parse_float(mass_str)
    current = _parse_float(current_str)
    if ts is None or mass is None or current is None:
        return None
    return {"tag": "R", "ts": ts, "mass": int(mass), "current": current}


_VALID_CHAMBERS = {"C1", "C2"}
_VALID_FLUSH_STATES = {"Re", "Fl", "Unknown"}


def _parse_v(payload: str) -> dict | None:
    fields = payload[2:].split(",")
    if len(fields) != 3:
        # Old verbose 4-field state-machine format, or malformed -- out of scope.
        return None
    ts_str, chamber, flush_state = fields
    if chamber not in _VALID_CHAMBERS or flush_state not in _VALID_FLUSH_STATES:
        return None
    ts = _parse_ts(ts_str)
    if ts is None:
        return None
    return {"tag": "V", "ts": ts, "chamber": chamber, "flush_state": flush_state}


def _parse_p(payload: str) -> dict | None:
    fields = payload[2:].split(",")
    if len(fields) == 6:
        ts_rtc_str, ts_scalup_str, temp_str, sal_str, oxygen_str, ph_str = fields
        pressure_mbar = None
    elif len(fields) == 7:
        ts_rtc_str, ts_scalup_str, temp_str, sal_str, pressure_str, oxygen_str, ph_str = fields
        pressure_mbar = _parse_float(pressure_str)
    else:
        return None
    ts = _parse_ts(ts_rtc_str)
    ts_scalup = _parse_ts(ts_scalup_str)
    if ts is None:
        return None
    return {
        "tag": "P",
        "ts": ts,
        "ts_scalup": ts_scalup,
        "temp_degc": _parse_float(temp_str),
        "sal_psu": _parse_float(sal_str),
        "pressure_mbar": pressure_mbar,
        "oxygen_mgl": _parse_float(oxygen_str),
        "ph": _parse_float(ph_str),
    }


def _parse_status(payload: str) -> dict | None:
    fields = payload[2:].split(",")
    if not fields:
        return None
    ts_str, *data_fields = fields
    ts = _parse_ts(ts_str)
    if ts is None:
        return None
    base = {
        "tag": "!",
        "ts": ts,
        "turbo_error": None,
        "turbo_speed_hz": None,
        "turbo_power_w": None,
        "turbo_voltage": None,
        "turbo_etemp_c": None,
        "turbo_btemp_c": None,
        "turbo_mtemp_c": None,
        "rga_filament": None,
        "raw_total_pressure_current": None,
        "pump_rpm": None,
        "payload_raw": None,
    }
    if len(data_fields) == 10:
        (
            turbo_error,
            turbo_speed_hz,
            turbo_power_w,
            turbo_voltage,
            turbo_etemp_c,
            turbo_btemp_c,
            turbo_mtemp_c,
            rga_filament,
            raw_tp,
            pump_rpm,
        ) = data_fields
        base.update(
            {
                "turbo_error": _parse_float(turbo_error),
                "turbo_speed_hz": _parse_float(turbo_speed_hz),
                "turbo_power_w": _parse_float(turbo_power_w),
                "turbo_voltage": _parse_float(turbo_voltage),
                "turbo_etemp_c": _parse_float(turbo_etemp_c),
                "turbo_btemp_c": _parse_float(turbo_btemp_c),
                "turbo_mtemp_c": _parse_float(turbo_mtemp_c),
                "rga_filament": _parse_float(rga_filament),
                "raw_total_pressure_current": _parse_float(raw_tp),
                "pump_rpm": _parse_float(pump_rpm),
            }
        )
    else:
        base["payload_raw"] = ",".join(data_fields)
    return base


def parse_line(payload: str) -> dict | None:
    """Parse one raw log line's payload (no outer surface-log wrapper) into a record dict.

    Returns ``None`` for malformed or out-of-scope lines (old-format V:, the
    undocumented PM: tag, and untimestamped tags that never occur in real
    data) rather than raising, so one bad line never aborts a run.
    """
    if payload.startswith("R:"):
        return _parse_r(payload)
    if payload.startswith("V:"):
        return _parse_v(payload)
    if payload.startswith("P:"):
        return _parse_p(payload)
    if payload.startswith("!:"):
        return _parse_status(payload)
    return None
