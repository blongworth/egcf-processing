# /// script
# requires-python = ">=3.10"
# dependencies = ["pyserial", "rich"]
# ///
"""Step the turbo setpoint on a schedule for the overnight RGA performance test.

Sends SPD<speed>\\n down from --start to --stop and back up in --step increments
(the turnaround speed is sent once): the first command immediately, then one every
--interval-s, anchored to a monotonic start time. Serial input is printed as it
arrives, with a status line pinned at the bottom. Every send and received line is
appended to --log as "<iso_ts> <direction> <payload>", the form that
turbo_speed_analysis.py --commands-log reads.

    uv run scripts/turbo_speed_stepper.py --port /dev/tty.usbserial-XXXX
"""

from __future__ import annotations

import argparse
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

from rich.console import Console
from rich.live import Live
from rich.text import Text


def build_schedule(start: int, stop: int, step: int) -> list[int]:
    step = abs(step) if stop < start else -abs(step)
    leg = list(range(start, stop - step, -step))
    if leg[-1] != stop:
        leg.append(stop)
    return leg + leg[-2::-1]


def _iso(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


class Stepper:
    def __init__(self, schedule: list[int], interval_s: float, log_file, console: Console):
        self.schedule = schedule
        self.interval_s = interval_s
        self.log_file = log_file
        self.console = console
        self.t0 = time.monotonic()
        self.wall0 = datetime.now(timezone.utc)
        self.next_i = 0
        self.last: str = "none sent yet"

    def log(self, direction: str, payload: str) -> None:
        line = f"{_iso(datetime.now(timezone.utc))} {direction} {payload}"
        self.log_file.write(line + "\n")
        self.log_file.flush()

    def due(self) -> bool:
        return self.next_i < len(self.schedule) and time.monotonic() >= self.t0 + self.next_i * self.interval_s

    def done(self) -> bool:
        return self.next_i >= len(self.schedule)

    def status(self) -> Text:
        n = len(self.schedule)
        if self.done():
            nxt = "schedule complete (Ctrl-C to exit)"
        else:
            at = self.wall0 + timedelta(seconds=self.next_i * self.interval_s)
            nxt = f"next: SPD{self.schedule[self.next_i]} @ {at.strftime('%H:%M:%SZ')} (step {self.next_i + 1}/{n})"
        return Text(f"Last: {self.last} | {nxt}", style="bold reverse")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--port", required=True)
    parser.add_argument("--baud", type=int, default=9600)
    parser.add_argument("--interval-s", type=float, default=3600.0)
    parser.add_argument("--start", type=int, default=1200)
    parser.add_argument("--stop", type=int, default=600)
    parser.add_argument("--step", type=int, default=100)
    parser.add_argument("--log", type=Path, default=Path("turbo_speed_commands.log"))
    parser.add_argument("--dry-run", action="store_true", help="run the schedule without opening the port")
    parser.add_argument("--exit-when-done", action="store_true", help="quit after the last command instead of waiting for Ctrl-C")
    args = parser.parse_args(argv)

    schedule = build_schedule(args.start, args.stop, args.step)
    console = Console()
    console.print(f"schedule ({len(schedule)} commands, every {args.interval_s:g} s): {schedule}")

    ser = None
    if not args.dry_run:
        import serial

        ser = serial.Serial(args.port, args.baud, timeout=0.5)

    with args.log.open("a", encoding="utf-8") as log_file:
        stepper = Stepper(schedule, args.interval_s, log_file, console)
        stepper.log("INFO", f"start port={args.port} dry_run={args.dry_run} schedule={schedule}")
        try:
            with Live(stepper.status(), console=console, refresh_per_second=4, transient=False) as live:
                while True:
                    if stepper.due():
                        cmd = f"SPD{schedule[stepper.next_i]}"
                        if ser is not None:
                            ser.write(f"{cmd}\n".encode("ascii"))
                        stepper.log("SENT", cmd)
                        sent_at = _iso(datetime.now(timezone.utc))
                        live.console.print(f"[bold cyan]{sent_at} SENT {cmd}[/]")
                        stepper.last = f"{cmd} @ {sent_at}"
                        stepper.next_i += 1
                        if stepper.done():
                            stepper.log("INFO", "schedule complete")
                        live.update(stepper.status())
                    if stepper.done() and args.exit_when_done:
                        break
                    if ser is not None:
                        raw = ser.readline()
                        text = raw.decode("utf-8", errors="replace").rstrip("\r\n")
                        if text:
                            stepper.log("RECV", text)
                            live.console.print(Text(text))
                    else:
                        time.sleep(0.5 if args.interval_s >= 0.5 else 0.05)
                    live.update(stepper.status())
        except KeyboardInterrupt:
            stepper.log("INFO", "interrupted")
            console.print("interrupted; exiting without changing speed")
            return 130
        finally:
            if ser is not None:
                ser.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
