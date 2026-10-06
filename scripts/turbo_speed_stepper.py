# /// script
# requires-python = ">=3.10"
# dependencies = ["pyserial"]
# ///
"""Step the turbo setpoint on a schedule for the overnight RGA performance test.

Sends SPD<speed>\\n down from --start to --stop and back up in --step increments
(the turnaround speed is sent once), one command every --interval-s. Every command
and response is logged with a UTC timestamp to stdout and to --log, in the
"<iso_ts> <direction> <payload>" form that turbo_speed_analysis.py --commands-log reads.

    uv run scripts/turbo_speed_stepper.py --port /dev/tty.usbserial-XXXX
"""

from __future__ import annotations

import argparse
import time
from datetime import datetime, timezone
from pathlib import Path


def build_schedule(start: int, stop: int, step: int) -> list[int]:
    step = abs(step) if stop < start else -abs(step)
    leg = list(range(start, stop - step, -step))
    if leg[-1] != stop:
        leg.append(stop)
    return leg + leg[-2::-1]


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _log(log_file, direction: str, payload: str) -> None:
    line = f"{_now()} {direction} {payload}"
    print(line, flush=True)
    log_file.write(line + "\n")
    log_file.flush()


def _read_response(ser, timeout_s: float) -> list[str]:
    lines = []
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        raw = ser.readline()
        if raw:
            text = raw.decode("utf-8", errors="replace").strip()
            if text:
                lines.append(text)
    return lines


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--port", required=True)
    parser.add_argument("--baud", type=int, default=9600)
    parser.add_argument("--interval-s", type=float, default=3600.0)
    parser.add_argument("--start", type=int, default=1200)
    parser.add_argument("--stop", type=int, default=600)
    parser.add_argument("--step", type=int, default=100)
    parser.add_argument("--response-s", type=float, default=3.0, help="seconds to read responses after each send")
    parser.add_argument("--log", type=Path, default=Path("turbo_speed_commands.log"))
    parser.add_argument("--dry-run", action="store_true", help="print commands on schedule without opening the port")
    args = parser.parse_args(argv)

    schedule = build_schedule(args.start, args.stop, args.step)
    print(f"schedule ({len(schedule)} commands, every {args.interval_s:g} s): {schedule}", flush=True)

    ser = None
    if not args.dry_run:
        import serial

        ser = serial.Serial(args.port, args.baud, timeout=0.5)
    t0 = time.monotonic()
    try:
        with args.log.open("a", encoding="utf-8") as log_file:
            _log(log_file, "INFO", f"start port={args.port} dry_run={args.dry_run} schedule={schedule}")
            for i, speed in enumerate(schedule):
                delay = t0 + i * args.interval_s - time.monotonic()
                if delay > 0:
                    time.sleep(delay)
                cmd = f"SPD{speed}"
                _log(log_file, "SENT", cmd)
                if ser is not None:
                    ser.write(f"{cmd}\n".encode("ascii"))
                    ser.flush()
                    for line in _read_response(ser, args.response_s):
                        _log(log_file, "RECV", line)
            _log(log_file, "INFO", "schedule complete")
    except KeyboardInterrupt:
        print("interrupted; exiting without changing speed", flush=True)
        with args.log.open("a", encoding="utf-8") as log_file:
            _log(log_file, "INFO", "interrupted")
        return 130
    finally:
        if ser is not None:
            ser.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
