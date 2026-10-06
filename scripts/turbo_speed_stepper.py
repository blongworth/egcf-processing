# /// script
# requires-python = ">=3.10"
# dependencies = ["pyserial", "rich"]
# ///
"""Step the turbo setpoint on a schedule for the overnight RGA performance test.

Steps the setpoint down from --start to --stop and back up in --step increments
(the turnaround speed is sent once). The first step starts immediately; each later
step starts --interval-s after the previous step's AON was sent, so every speed gets
a full interval of acquisition. The speed can't be changed while acquiring, so each
step runs:

    AOFF -> (OK,AOFF) -> SPD#### -> (OK,SPD) -> poll S until TURBO=ready -> AON -> (OK,AON)

An unacknowledged AOFF/SPD/AON is resent every --ack-timeout-s. S is polled every
--poll-s, and TURBO=ready only counts once --min-spin-s has passed since the SPD ack,
so a stale "ready" from before the change isn't trusted. AON is never sent until the
turbo reports ready, and the next step waits for it, however long that takes.
Serial input is printed as it arrives, with a status line pinned
at the bottom. Every send and received line is appended to --log as
"<iso_ts> <direction> <payload>", the form that turbo_speed_analysis.py
--commands-log reads.

    uv run scripts/turbo_speed_stepper.py --port /dev/tty.usbserial-XXXX
"""

from __future__ import annotations

import argparse
import re
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

from rich.console import Console
from rich.live import Live
from rich.text import Text

_OK_RE = re.compile(r"\bOK,(AOFF|AON|SPD)")
_READY_RE = re.compile(r"\bS,.*\bTURBO=ready\b")


def build_schedule(start: int, stop: int, step: int) -> list[int]:
    step = abs(step) if stop < start else -abs(step)
    leg = list(range(start, stop - step, -step))
    if leg[-1] != stop:
        leg.append(stop)
    return leg + leg[-2::-1]


def _iso(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def _dry_run_reply(cmd: str) -> str:
    if cmd == "S":
        return "S,Off,SPD=0,TURBO=ready,RGA=off"
    return f"OK,{cmd}"


class Stepper:
    """One step's AOFF/SPD/S/AON sequence, driven by poll() and on_line() from the main loop."""

    def __init__(self, schedule: list[int], args: argparse.Namespace, send, log_file, console):
        self.schedule = schedule
        self.args = args
        self.send_raw = send
        self.log_file = log_file
        self.console = console
        self.next_due: float | None = time.monotonic()
        self.next_due_wall: datetime | None = datetime.now(timezone.utc)
        self.next_i = 0
        self.phase = "idle"
        self.setpoint: int | None = None
        self.phase_cmd = ""
        self.sent_at = 0.0
        self.spd_ok_at = 0.0
        self.next_poll = 0.0
        self.polls = 0
        self.last = "none sent yet"
        self.completed = 0

    def log(self, direction: str, payload: str) -> None:
        self.log_file.write(f"{_iso(datetime.now(timezone.utc))} {direction} {payload}\n")
        self.log_file.flush()

    def note(self, payload: str, style: str = "yellow") -> None:
        self.log("INFO", payload)
        self.console.print(f"[{style}]{_iso(datetime.now(timezone.utc))} {payload}[/]")

    def send(self, cmd: str) -> None:
        self.log("SENT", cmd)
        if cmd != "S":
            self.console.print(f"[bold cyan]{_iso(datetime.now(timezone.utc))} SENT {cmd}[/]")
        self.send_raw(cmd)

    def _command(self, cmd: str) -> None:
        self.phase_cmd = cmd
        self.sent_at = time.monotonic()
        self.send(cmd)

    def done(self) -> bool:
        return self.next_i >= len(self.schedule) and self.phase == "idle"

    def poll(self) -> None:
        now = time.monotonic()
        if self.next_i < len(self.schedule) and self.next_due is not None and now >= self.next_due:
            if self.phase != "idle":
                self.note(f"WARNING: step SPD{self.setpoint} still waiting for OK,AON when next step came due; moving on")
            self.next_due = self.next_due_wall = None
            self.setpoint = self.schedule[self.next_i]
            self.next_i += 1
            self.polls = 0
            self.phase = "aoff"
            self._command("AOFF")
        elif self.phase in ("aoff", "spd", "aon") and now - self.sent_at >= self.args.ack_timeout_s:
            self.note(f"WARNING: no OK for {self.phase_cmd} after {self.args.ack_timeout_s:g} s; resending")
            self._command(self.phase_cmd)
        elif self.phase == "ready" and now >= self.next_poll:
            self.next_poll = now + self.args.poll_s
            self.polls += 1
            self.send("S")

    def on_line(self, text: str) -> None:
        now = time.monotonic()
        ok = _OK_RE.search(text)
        if self.phase == "aoff" and ok and ok.group(1) == "AOFF":
            self.phase = "spd"
            self._command(f"SPD{self.setpoint}")
        elif self.phase == "spd" and ok and ok.group(1) == "SPD":
            self.phase = "ready"
            self.spd_ok_at = now
            self.next_poll = now + self.args.min_spin_s
        elif self.phase == "ready" and _READY_RE.search(text) and now - self.spd_ok_at >= self.args.min_spin_s:
            self.phase = "aon"
            self._command("AON")
            self.next_due = now + self.args.interval_s
            self.next_due_wall = datetime.now(timezone.utc) + timedelta(seconds=self.args.interval_s)
            self.last = f"SPD{self.setpoint}, AON @ {_iso(datetime.now(timezone.utc))}"
        elif self.phase == "aon" and ok and ok.group(1) == "AON":
            self.phase = "idle"
            self.completed += 1
            self.note(f"step SPD{self.setpoint} complete: turbo ready, acquisition on", style="green")
            if self.next_i >= len(self.schedule):
                self.note("schedule complete", style="green")

    def status(self) -> Text:
        n = len(self.schedule)
        now = time.monotonic()
        waiting = {
            "aoff": "waiting for OK,AOFF",
            "spd": f"waiting for OK,SPD{self.setpoint}",
            "ready": f"waiting for TURBO=ready ({self.polls} S polls, {now - self.spd_ok_at:.0f} s)",
            "aon": "waiting for OK,AON",
        }
        parts = [f"Last: {self.last}"]
        if self.phase != "idle":
            parts.append(f"step {self.next_i}/{n} SPD{self.setpoint}: {waiting[self.phase]}")
        if self.next_i < n:
            nxt = f"SPD{self.schedule[self.next_i]} (step {self.next_i + 1}/{n})"
            if self.next_due_wall is None:
                parts.append(f"next: {nxt} {self.args.interval_s:g} s after AON")
            else:
                parts.append(f"next: {nxt} @ {self.next_due_wall.strftime('%H:%M:%SZ')}")
        elif self.phase == "idle":
            parts.append("schedule complete (Ctrl-C to exit)")
        return Text(" | ".join(parts), style="bold reverse")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--port", required=True)
    parser.add_argument("--baud", type=int, default=9600)
    parser.add_argument("--interval-s", type=float, default=3600.0)
    parser.add_argument("--start", type=int, default=1200)
    parser.add_argument("--stop", type=int, default=600)
    parser.add_argument("--step", type=int, default=100)
    parser.add_argument("--ack-timeout-s", type=float, default=30.0, help="resend AOFF/SPD/AON if no OK within this (default 30)")
    parser.add_argument("--poll-s", type=float, default=10.0, help="S poll interval while waiting for TURBO=ready (default 10)")
    parser.add_argument("--min-spin-s", type=float, default=30.0, help="ignore TURBO=ready until this long after the SPD ack (default 30)")
    parser.add_argument("--log", type=Path, default=Path("turbo_speed_commands.log"))
    parser.add_argument("--dry-run", action="store_true", help="run the schedule without opening the port; replies are simulated")
    parser.add_argument("--exit-when-done", action="store_true", help="quit after the last step instead of waiting for Ctrl-C")
    args = parser.parse_args(argv)

    schedule = build_schedule(args.start, args.stop, args.step)
    console = Console()
    console.print(f"schedule ({len(schedule)} steps, each held {args.interval_s:g} s after AON): {schedule}")

    ser = None
    if not args.dry_run:
        import serial

        ser = serial.Serial(args.port, args.baud, timeout=0.5)
    pending: list[str] = []

    def send(cmd: str) -> None:
        if ser is not None:
            ser.write(f"{cmd}\n".encode("ascii"))
        else:
            pending.append(_dry_run_reply(cmd))

    with args.log.open("a", encoding="utf-8") as log_file:
        try:
            with Live(console=console, refresh_per_second=4, transient=False) as live:
                stepper = Stepper(schedule, args, send, log_file, live.console)
                stepper.log("INFO", f"start port={args.port} dry_run={args.dry_run} schedule={schedule}")
                while True:
                    stepper.poll()
                    live.update(stepper.status())
                    if stepper.done() and args.exit_when_done:
                        break
                    if ser is not None:
                        lines = [ser.readline().decode("utf-8", errors="replace").rstrip("\r\n")]
                    else:
                        time.sleep(0.5 if args.interval_s >= 0.5 else 0.05)
                        lines, pending[:] = list(pending), []
                    for text in filter(None, lines):
                        stepper.log("RECV", text)
                        live.console.print(Text(text))
                        stepper.on_line(text)
                    live.update(stepper.status())
        except KeyboardInterrupt:
            log_file.write(f"{_iso(datetime.now(timezone.utc))} INFO interrupted\n")
            console.print("interrupted; exiting without changing speed or acquisition state")
            return 130
        finally:
            if ser is not None:
                ser.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
