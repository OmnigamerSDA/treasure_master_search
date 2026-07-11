#!/usr/bin/env python3
"""Durable cert-tier queue/runner for intermittent CUDA key-clearing sweeps (BFS or raceway engine).

Operator surface:
  init           build the durable queue DB from shard files.
  operate        INTERACTIVE console: start+supervise a run AND watch the live dashboard in one view;
                 pause/resume/quit with single keys. Easiest way to drive a tier.
  run            lease + run bounded batches (one worker per device); --engine bfs|raceway.
  status         one-shot snapshot: progress %/ETA, in-flight runs, queue, alerts.
  monitor        live refreshing read-only dashboard (progress/ETA, in-flight runs, verifier hits).
  request-stop   PAUSE: stop leasing new batches; signal active workers to stop at the next key.
  clear-stop     un-pause (clear the stop request) so a subsequent `run` resumes.
  reset-running  recover stuck 'running' leases (after a hard crash) back to 'pending'.
  scan-alerts    backfill alerts from completed verified CSVs (bfs engine).

Interactive (one view):   operate --db ... --engine raceway --devices 1,0 --loops 0 [tuning...]
                          then press p=pause r=resume q=quit d=detach.
Two-shell:                run (one shell) + monitor (another); request-stop to pause, clear-stop + run
                          to resume. reset-running only after an uncontrolled crash.

Raceway run defaults: hit verification runs in the BACKGROUND (overlaps the next batch's GPU compute) and
batch reap is event-driven (instant relaunch, no polling); --sync-verify forces the old blocking path.
--batch-keys default is 8000 (raise it for fast/high-cert tiers to amortize per-batch overhead). --daemon
runs ONE resident worker per device (batches fed over its stdin, no per-batch CUDA restart) for the big
low-cert tiers; needs a daemon-capable --binary. Per-tier wave geometry is operator-driven via the
--raceway-* flags; you may plug in validated per-tier tuning through a cert_tier_geometry_override.py hook
(see _load_tier_geometry_override).
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import io
import json
import os
import queue
import select
import shlex
import shutil
import sqlite3
import subprocess
import threading
import sys
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, NamedTuple


SCHEMA_VERSION = 1


@dataclass(frozen=True)
class LeasedBatch:
    run_id: str
    device: str
    shard_index: int
    keys: list[tuple[int, int]]
    keyfile: Path
    summary_path: Path
    log_path: Path
    hit_path: Path | None
    verified_path: Path | None
    record_hit_path: Path | None
    record_verified_path: Path | None
    stop_file: Path
    command: list[str]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def parse_devices(value: str) -> list[str]:
    devices = [part.strip() for part in value.split(",") if part.strip()]
    if not devices:
        raise SystemExit("--devices must name at least one CUDA device")
    return devices


def shard_paths(shard_dir: Path, min_bits: int, devices: list[str]) -> list[Path]:
    paths = [shard_dir / f"keys_cert_ge{min_bits}_shard{i:02d}.tsv" for i in range(len(devices))]
    if all(path.exists() for path in paths):
        return paths
    paths = [shard_dir / f"keys_cert_ge{min_bits}_shard{i:02d}.bin" for i in range(len(devices))]
    if all(path.exists() for path in paths):
        return paths
    paths = [shard_dir / f"keys_cert_ge{min_bits}_shard{i:02d}.bin5" for i in range(len(devices))]
    if all(path.exists() for path in paths):
        return paths
    raise SystemExit(
        f"missing {len(devices)} shard files in {shard_dir}; run build_cert_tier_shards.py --shards {len(devices)}"
    )


def iter_keys(path: Path) -> Iterable[tuple[int, int]]:
    if path.suffix == ".bin5":
        with path.open("rb") as fh:
            while True:
                data = fh.read(5)
                if not data:
                    return
                if len(data) != 5:
                    raise ValueError(f"truncated bin32u8 shard: {path}")
                yield int.from_bytes(data[:4], "little"), data[4]
    elif path.suffix == ".bin":
        with path.open("rb") as fh:
            while True:
                data = fh.read(4)
                if not data:
                    return
                if len(data) != 4:
                    raise ValueError(f"truncated bin32 shard: {path}")
                yield int.from_bytes(data, "little"), 0
    else:
        with path.open("r", encoding="ascii") as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                cols = line.split()
                key = int(cols[0], 0)
                cert_bits = int(cols[1]) if len(cols) > 1 else 0
                yield key, cert_bits


def connect(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def ensure_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS meta (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS keys (
            key INTEGER PRIMARY KEY,
            cert_bits INTEGER NOT NULL,
            shard_index INTEGER NOT NULL,
            preferred_device TEXT NOT NULL,
            source_order INTEGER NOT NULL,
            status TEXT NOT NULL DEFAULT 'pending',
            attempts INTEGER NOT NULL DEFAULT 0,
            lease_id TEXT,
            last_run_id TEXT,
            updated_at TEXT NOT NULL,
            elapsed_s REAL,
            shed_bits INTEGER,
            logical_window INTEGER,
            f1 INTEGER,
            final_frontier INTEGER,
            flag_sum INTEGER,
            representative_hits INTEGER,
            verify_enqueued_hits INTEGER,
            map1_ms REAL,
            deep_ms REAL,
            total_ms REAL,
            host_setup_ms REAL,
            hit_write_ms REAL,
            verify_enqueue_ms REAL,
            verify_state_copy_ms REAL,
            hit_file TEXT,
            verified_file TEXT,
            error TEXT
        );

        CREATE INDEX IF NOT EXISTS idx_keys_status_device_order
            ON keys(status, preferred_device, source_order);
        CREATE INDEX IF NOT EXISTS idx_keys_lease
            ON keys(lease_id);

        CREATE TABLE IF NOT EXISTS runs (
            run_id TEXT PRIMARY KEY,
            device TEXT NOT NULL,
            shard_index INTEGER NOT NULL,
            status TEXT NOT NULL,
            started_at TEXT NOT NULL,
            ended_at TEXT,
            key_count INTEGER NOT NULL,
            returncode INTEGER,
            log_path TEXT NOT NULL,
            summary_path TEXT NOT NULL,
            keyfile_path TEXT NOT NULL,
            hit_path TEXT,
            verified_path TEXT,
            command_json TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS alerts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL,
            run_id TEXT NOT NULL,
            device TEXT NOT NULL,
            key INTEGER NOT NULL,
            data INTEGER NOT NULL,
            reason TEXT NOT NULL,
            flags INTEGER,
            screen_flag INTEGER,
            machine_flags INTEGER,
            final_rts INTEGER,
            entry_opcodes INTEGER,
            control_flow INTEGER,
            structural INTEGER,
            linear_clean INTEGER,
            linear_score INTEGER,
            b00 TEXT,
            b01 TEXT,
            b02 TEXT,
            b03 TEXT,
            b04 TEXT,
            b07 TEXT,
            b2e TEXT,
            b50 TEXT,
            source_verified_file TEXT NOT NULL,
            UNIQUE(run_id, key, data)
        );

        CREATE INDEX IF NOT EXISTS idx_alerts_created
            ON alerts(created_at);
        CREATE INDEX IF NOT EXISTS idx_alerts_reason
            ON alerts(reason);
        """
    )
    conn.execute(
        "INSERT OR REPLACE INTO meta(key, value) VALUES('schema_version', ?)",
        (str(SCHEMA_VERSION),),
    )
    conn.commit()


def get_meta(conn: sqlite3.Connection, key: str) -> str:
    row = conn.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
    return str(row["value"]) if row else ""


def set_meta(conn: sqlite3.Connection, key: str, value: str) -> None:
    conn.execute("INSERT OR REPLACE INTO meta(key, value) VALUES(?, ?)", (key, value))
    conn.commit()


def clear_meta(conn: sqlite3.Connection, key: str) -> None:
    conn.execute("DELETE FROM meta WHERE key=?", (key,))
    conn.commit()


def stop_requested(conn: sqlite3.Connection) -> bool:
    return bool(get_meta(conn, "stop_requested_at"))


def command_init(args: argparse.Namespace) -> int:
    devices = parse_devices(args.devices)
    shard_dir = Path(args.shard_dir)
    paths = shard_paths(shard_dir, args.min_bits, devices)
    conn = connect(Path(args.db))
    ensure_schema(conn)
    started = utc_now()
    conn.execute("INSERT OR REPLACE INTO meta(key, value) VALUES('shard_dir', ?)", (str(shard_dir),))
    conn.execute("INSERT OR REPLACE INTO meta(key, value) VALUES('min_bits', ?)", (str(args.min_bits),))
    conn.execute("INSERT OR REPLACE INTO meta(key, value) VALUES('devices', ?)", (",".join(devices),))
    conn.execute("INSERT OR REPLACE INTO meta(key, value) VALUES('initialized_at', ?)", (started,))

    total_inserted = 0
    for shard_index, (device, path) in enumerate(zip(devices, paths, strict=True)):
        batch: list[tuple[int, int, int, str, int, str]] = []
        shard_seen = 0
        shard_inserted = 0
        for source_order, (key, cert_bits) in enumerate(iter_keys(path)):
            if args.max_keys_per_shard and shard_seen >= args.max_keys_per_shard:
                break
            shard_seen += 1
            batch.append((key, cert_bits, shard_index, device, source_order, started))
            if len(batch) >= args.insert_batch:
                before = conn.total_changes
                conn.executemany(
                    """
                    INSERT OR IGNORE INTO keys
                        (key, cert_bits, shard_index, preferred_device, source_order, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    batch,
                )
                shard_inserted += conn.total_changes - before
                conn.commit()
                batch.clear()
        if batch:
            before = conn.total_changes
            conn.executemany(
                """
                INSERT OR IGNORE INTO keys
                    (key, cert_bits, shard_index, preferred_device, source_order, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                batch,
            )
            shard_inserted += conn.total_changes - before
            conn.commit()
        total_inserted += shard_inserted
        print(f"shard {shard_index:02d} device={device} source={path} queued={shard_inserted:,}", flush=True)

    total = conn.execute("SELECT COUNT(*) FROM keys").fetchone()[0]
    print(f"db={args.db} inserted={total_inserted:,} total_keys={total:,}")
    return 0


def print_table(rows: list[sqlite3.Row], columns: list[str]) -> None:
    if not rows:
        return
    widths = [len(col) for col in columns]
    data = []
    for row in rows:
        vals = [str(row[col]) for col in columns]
        data.append(vals)
        widths = [max(width, len(val)) for width, val in zip(widths, vals, strict=True)]
    print("  ".join(col.ljust(width) for col, width in zip(columns, widths, strict=True)))
    print("  ".join("-" * width for width in widths))
    for vals in data:
        print("  ".join(val.ljust(width) for val, width in zip(vals, widths, strict=True)))


def fmt_duration(seconds: float | None) -> str:
    if seconds is None:
        return "n/a"
    seconds = max(0.0, float(seconds))
    d, rem = divmod(int(seconds), 86400)
    h, rem = divmod(rem, 3600)
    m, s = divmod(rem, 60)
    if d:
        return f"{d}d{h:02d}h"
    if h:
        return f"{h}h{m:02d}m"
    if m:
        return f"{m}m{s:02d}s"
    return f"{s}s"


def progress_rows(conn: sqlite3.Connection) -> tuple[list[dict], dict]:
    """Per-device pending/running/done + steady keys/s + ETA, plus an aggregate summary dict."""
    counts: dict[str, dict[str, int]] = {}
    for r in conn.execute(
        "SELECT preferred_device AS dev, status, COUNT(*) AS n FROM keys GROUP BY 1, 2"
    ).fetchall():
        counts.setdefault(r["dev"], {})[r["status"]] = r["n"]
    rates: dict[str, float | None] = {}
    for r in conn.execute(
        "SELECT preferred_device AS dev, COUNT(*) AS c, SUM(elapsed_s) AS e "
        "FROM keys WHERE status='done' AND elapsed_s IS NOT NULL GROUP BY 1"
    ).fetchall():
        rates[r["dev"]] = (r["c"] / r["e"]) if r["e"] and r["e"] > 0 else None
    rows: list[dict] = []
    agg = {"total": 0, "done": 0, "pending": 0, "running": 0, "eta_s": None}
    worst_eta = 0.0
    eta_known = False
    for dev in sorted(counts):
        c = counts[dev]
        pending = c.get("pending", 0)
        running = c.get("running", 0)
        done = c.get("done", 0)
        failed = c.get("failed", 0)
        total = sum(c.values())
        rate = rates.get(dev)
        remaining = pending + running
        eta = (remaining / rate) if (rate and rate > 0) else None
        if eta is not None:
            worst_eta = max(worst_eta, eta)
            eta_known = True
        rows.append(
            {
                "device": dev,
                "done": done,
                "pending": pending,
                "running": running,
                "failed": failed,
                "pct": f"{(100.0 * done / total):.2f}" if total else "0.00",
                "keys_per_s": f"{rate:.2f}" if rate else "n/a",
                "eta": fmt_duration(eta),
            }
        )
        agg["total"] += total
        agg["done"] += done
        agg["pending"] += pending
        agg["running"] += running
    agg["eta_s"] = worst_eta if eta_known else None
    agg["pct"] = (100.0 * agg["done"] / agg["total"]) if agg["total"] else 0.0
    return rows, agg


def active_run_rows(conn: sqlite3.Connection) -> list[dict]:
    """In-flight runs with completed-so-far (from the live worker summary) and live rate."""
    now = datetime.now(timezone.utc)
    out: list[dict] = []
    for r in conn.execute(
        "SELECT run_id, device, status, key_count, summary_path, started_at FROM runs "
        "WHERE status IN ('leased', 'running') ORDER BY started_at"
    ).fetchall():
        completed = summary_data_row_count(Path(r["summary_path"]))
        try:
            started = datetime.fromisoformat(r["started_at"])
            elapsed = (now - started).total_seconds()
        except (ValueError, TypeError):
            elapsed = None
        rate = (completed / elapsed) if (elapsed and elapsed > 0 and completed) else None
        remaining = max(0, int(r["key_count"]) - completed)
        out.append(
            {
                "run_id": r["run_id"],
                "device": r["device"],
                "status": r["status"],
                "progress": f"{completed}/{r['key_count']}",
                "elapsed": fmt_duration(elapsed),
                "keys_per_s": f"{rate:.2f}" if rate else "n/a",
                "batch_eta": fmt_duration(remaining / rate) if rate else "n/a",
            }
        )
    return out


def command_status(args: argparse.Namespace) -> int:
    conn = connect(Path(args.db))
    ensure_schema(conn)
    stop_at = get_meta(conn, "stop_requested_at")
    if stop_at:
        print(f"PAUSED: stop requested at {stop_at}; active workers stop after their current key. "
              "Resume with: clear-stop, then run.\n")

    prog_rows, agg = progress_rows(conn)
    print(
        f"Progress: {agg['done']:,}/{agg['total']:,} done ({agg['pct']:.3f}%)  "
        f"pending={agg['pending']:,} running={agg['running']:,}  "
        f"ETA(parallel)={fmt_duration(agg['eta_s'])}"
    )
    print_table(prog_rows, ["device", "done", "pending", "running", "failed", "pct", "keys_per_s", "eta"])

    active = active_run_rows(conn)
    print("\nActive Runs (in-flight):")
    if active:
        print_table(active, ["run_id", "device", "status", "progress", "elapsed", "keys_per_s", "batch_eta"])
    else:
        print("  (none running)")

    rows = conn.execute(
        """
        SELECT preferred_device AS device, status, COUNT(*) AS keys
        FROM keys
        GROUP BY preferred_device, status
        ORDER BY preferred_device, status
        """
    ).fetchall()
    print("\nQueue:")
    print_table(rows, ["device", "status", "keys"])

    tier_rows = conn.execute(
        """
        SELECT cert_bits, status, COUNT(*) AS keys
        FROM keys
        GROUP BY cert_bits, status
        ORDER BY cert_bits, status
        """
    ).fetchall()
    print("\nTiers:")
    print_table(tier_rows, ["cert_bits", "status", "keys"])

    perf_rows = conn.execute(
        """
        SELECT preferred_device AS device,
               COUNT(*) AS done_keys,
               ROUND(SUM(elapsed_s), 3) AS elapsed_s,
               ROUND(CASE WHEN SUM(elapsed_s) > 0 THEN COUNT(*) / SUM(elapsed_s) ELSE 0 END, 3) AS keys_per_s,
               SUM(representative_hits) AS reps,
               SUM(verify_enqueued_hits) AS verify_hits,
               ROUND(SUM(verify_state_copy_ms), 3) AS verify_state_copy_ms
        FROM keys
        WHERE status='done'
        GROUP BY preferred_device
        ORDER BY preferred_device
        """
    ).fetchall()
    if perf_rows:
        print("\nDone Performance:")
        print_table(
            perf_rows,
            ["device", "done_keys", "elapsed_s", "keys_per_s", "reps", "verify_hits", "verify_state_copy_ms"],
        )

    alert_rows = conn.execute(
        """
        SELECT reason, COUNT(*) AS alerts
        FROM alerts
        GROUP BY reason
        ORDER BY alerts DESC, reason
        """
    ).fetchall()
    if alert_rows:
        print("\nAlerts:")
        print_table(alert_rows, ["reason", "alerts"])

    recent_alerts = conn.execute(
        """
        SELECT created_at, device, printf('0x%08x', key) AS key, data, reason,
               final_rts, structural, linear_clean, linear_score
        FROM alerts
        ORDER BY id DESC
        LIMIT ?
        """,
        (args.recent_alerts,),
    ).fetchall()
    if recent_alerts:
        print("\nRecent Alerts:")
        print_table(
            recent_alerts,
            ["created_at", "device", "key", "data", "reason", "final_rts", "structural", "linear_clean", "linear_score"],
        )

    run_rows = conn.execute(
        """
        SELECT run_id, device, status, key_count, returncode, started_at, ended_at
        FROM runs
        ORDER BY started_at DESC
        LIMIT ?
        """,
        (args.recent,),
    ).fetchall()
    if run_rows:
        print("\nRecent Runs:")
        print_table(run_rows, ["run_id", "device", "status", "key_count", "returncode", "started_at", "ended_at"])
    return 0


def render_monitor(conn: sqlite3.Connection, recent_alerts: int) -> None:
    """Compact one-screen operator dashboard: progress, in-flight runs, and notable verifier hits."""
    stop_at = get_meta(conn, "stop_requested_at")
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    print(f"cert-tier monitor  {now}")
    if stop_at:
        print(f"  ** PAUSED (stop requested {stop_at}) -- workers stop at next key boundary; "
              "clear-stop then run to resume **")
    prog_rows, agg = progress_rows(conn)
    print(
        f"  progress {agg['done']:,}/{agg['total']:,} ({agg['pct']:.3f}%)  "
        f"pending={agg['pending']:,} running={agg['running']:,}  ETA={fmt_duration(agg['eta_s'])}"
    )
    print()
    print_table(prog_rows, ["device", "done", "pending", "running", "failed", "pct", "keys_per_s", "eta"])
    active = active_run_rows(conn)
    print("\nActive (in-flight):")
    if active:
        print_table(active, ["run_id", "device", "status", "progress", "elapsed", "keys_per_s", "batch_eta"])
    else:
        print("  (none running)")
    # Verifier hits: COUNTS of the meaningful signal combinations, not a per-row list. Every recorded
    # alert already implies checksum-match + all-entries-valid, so we don't repeat those; the
    # differentiators are structural (strongest = looks like real code) > rts > linear. Only the rare
    # STRUCTURAL candidates are listed, since those are the ones worth a human look.
    def acount(where: str) -> int:
        return conn.execute(f"SELECT COUNT(*) FROM alerts WHERE {where}").fetchone()[0]

    total_alerts = acount("1")
    print(f"\nVerifier hits (checksum + all-entries-valid assumed): {total_alerts}")
    if total_alerts:
        print(
            f"  structural={acount('structural=1')}   rts={acount('final_rts=1')}   "
            f"linear_clean={acount('linear_clean=1')}   linear_score>=32={acount('linear_score>=32')}"
        )
        n_struct = acount("structural=1")
        if n_struct:
            rows = conn.execute(
                "SELECT printf('0x%08x', key) AS key, printf('0x%08x', data) AS data, device, reason "
                "FROM alerts WHERE structural=1 ORDER BY id DESC LIMIT ?",
                (recent_alerts,),
            ).fetchall()
            print(f"  *** STRONG structural candidate(s) -- review: showing {min(n_struct, recent_alerts)} of {n_struct} ***")
            print_table(rows, ["key", "data", "device", "reason"])


def capture_stdout(fn) -> str:
    """Run fn() and return everything it printed (so a frame can be redrawn in place)."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        fn()
    return buf.getvalue()


def draw_frame(body: str, first: bool) -> None:
    """Redraw a full-screen frame IN PLACE to avoid flicker and scroll-trail: home the cursor, rewrite
    each line clearing to end-of-line, then clear everything below. A full screen-clear happens only on
    the first frame (`first=True`)."""
    lines = body.rstrip("\n").split("\n")
    prefix = "\033[2J" if first else ""  # one-time full clear on entry only
    sys.stdout.write(prefix + "\033[H" + "\n".join(line + "\033[K" for line in lines) + "\033[J")
    sys.stdout.flush()


def command_monitor(args: argparse.Namespace) -> int:
    conn = connect(Path(args.db))
    ensure_schema(conn)
    if args.once:
        render_monitor(conn, args.recent_alerts)
        return 0
    first = True
    try:
        while True:
            if args.no_clear:
                render_monitor(conn, args.recent_alerts)
                print(f"\n(refresh {args.interval:.0f}s; Ctrl-C to exit)", flush=True)
            else:
                body = capture_stdout(lambda: render_monitor(conn, args.recent_alerts))
                body += f"\n(refresh {args.interval:.0f}s; Ctrl-C to exit)"
                draw_frame(body, first)
                first = False
            time.sleep(max(1.0, args.interval))
    except KeyboardInterrupt:
        print("\nmonitor stopped")
    return 0


def run_argv_passthrough() -> list[str]:
    """Tokens after the `operate` subcommand — valid `run` args (operate shares run's arg set)."""
    try:
        idx = sys.argv.index("operate")
    except ValueError:
        return []
    return sys.argv[idx + 1 :]


def command_operate(args: argparse.Namespace) -> int:
    """Interactive console: launches/supervises a `run` (logged to a file so the terminal is free for
    the dashboard), renders the live status view, and takes single-key controls."""
    if not Path(args.binary).exists():
        raise SystemExit(f"CUDA binary not found: {args.binary}")
    if args.engine == "raceway" and not Path(args.inspect_binary).exists():
        raise SystemExit(f"raceway clearing needs the CPU receiver: {args.inspect_binary} not found")

    conn = connect(Path(args.db))
    ensure_schema(conn)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    run_cmd = [sys.executable, os.path.abspath(__file__), "run", *run_argv_passthrough()]

    state = {"proc": None, "log": None, "log_path": None}

    def launch() -> None:
        clear_meta(conn, "stop_requested_at")  # ensure not paused so the run leases
        lp = out_dir / f"operate_run_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.log"
        fh = lp.open("w", encoding="utf-8")
        state["proc"] = subprocess.Popen(run_cmd, stdout=fh, stderr=subprocess.STDOUT, text=True)
        state["log"], state["log_path"] = fh, lp

    def alive() -> bool:
        return state["proc"] is not None and state["proc"].poll() is None

    launch()
    is_tty = sys.stdin.isatty()
    old_term = None
    if is_tty:
        import termios
        import tty

        old_term = termios.tcgetattr(sys.stdin)
        tty.setcbreak(sys.stdin.fileno())
    first = True
    try:
        while True:
            def frame() -> None:
                render_monitor(conn, 10)
                paused = bool(get_meta(conn, "stop_requested_at"))
                run_state = "PAUSED" if paused else ("RUNNING" if alive() else "STOPPED/idle")
                lp = state["log_path"]
                print(f"\nrun: {run_state}   engine={args.engine} devices={args.devices}   log: {lp.name if lp else '-'}")
                print("keys:  [p]ause   [r]esume   [q]uit(stop+exit)   [d]etach(leave run going)   [space]refresh")
                if not is_tty:
                    print("(stdin is not a TTY: display-only; use request-stop/clear-stop from another shell)")

            draw_frame(capture_stdout(frame), first)
            first = False

            key = None
            if is_tty:
                ready, _, _ = select.select([sys.stdin], [], [], 2.0)
                if ready:
                    key = sys.stdin.read(1)
            else:
                time.sleep(2.0)

            if key in ("q", "Q"):
                if alive():
                    set_meta(conn, "stop_requested_at", utc_now())
                    print("\nstopping run (workers finish current key)...", flush=True)
                    try:
                        state["proc"].wait(timeout=900)
                    except subprocess.TimeoutExpired:
                        state["proc"].terminate()
                print("\noperate: stopped and exiting.")
                break
            if key in ("d", "D"):
                print("\noperate: detaching; the run continues in the background. "
                      "Re-attach with `monitor`, or pause with `request-stop`.")
                break
            if key in ("p", "P"):
                set_meta(conn, "stop_requested_at", utc_now())
            elif key in ("r", "R"):
                clear_meta(conn, "stop_requested_at")
                if not alive():
                    launch()
    except KeyboardInterrupt:
        if alive():
            set_meta(conn, "stop_requested_at", utc_now())
        print("\noperate: interrupted; run will stop at the next key boundary.")
    finally:
        if is_tty and old_term is not None:
            import termios

            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_term)
        if state["log"] is not None:
            state["log"].close()
    return 0


def build_bfs_command(args: argparse.Namespace, batch: LeasedBatch) -> list[str]:
    cmd = [
        args.binary,
        "--device",
        batch.device,
        "--map1-frontier-key-file",
        str(batch.keyfile),
        "--workunit_size",
        str(args.workunit_size),
        "--precert",
        "--map1-frontier-wide",
        "--map1-frontier-stream-deep",
        "--map1-frontier-chunk",
        str(args.map1_frontier_chunk),
        "--map1-frontier-deep-k",
        str(args.map1_frontier_deep_k),
        "--map1-frontier-worker-summary",
        str(batch.summary_path),
        "--worker-stop-file",
        str(batch.stop_file),
    ]
    if args.map1_frontier_stream_cuts:
        cmd.extend(["--map1-frontier-stream-cuts", args.map1_frontier_stream_cuts])
    if args.map1_frontier_drain_cap_bits != "0":
        cmd.extend(["--map1-frontier-drain-cap-bits", args.map1_frontier_drain_cap_bits])
        cmd.extend(["--map1-frontier-drain-cap-ways", args.map1_frontier_drain_cap_ways])
    if args.map1_frontier_drain_cap_distinct:
        cmd.append("--map1-frontier-drain-cap-distinct")
    if args.map1_frontier_overlap_setup:
        cmd.append("--map1-frontier-overlap-setup")
    if batch.hit_path is not None:
        cmd.extend(["--map1-frontier-hit-file", str(batch.hit_path)])
    if batch.verified_path is not None:
        cmd.extend(["--map1-frontier-verified-file", str(batch.verified_path)])
        if args.map1_frontier_verify_final_rts:
            cmd.append("--map1-frontier-verify-final-rts")
    return cmd


def raceway_hit_dir(batch: LeasedBatch) -> Path:
    return batch.hit_path if batch.hit_path is not None else batch.summary_path.parent / "hits"


def _load_tier_geometry_override():
    """Optional operator extension point for per-tier wave geometry.

    A good wave geometry (single-batch window vs. tiled, drain boundaries, cap-ways) is highly
    problem- and hardware-specific, so none is baked in. To plug in your own validated per-tier
    tuning without editing this file, drop a ``cert_tier_geometry_override.py`` on the path
    (e.g. next to this script) exposing::

        def tier_geometry_override(min_cert, wave_batch, boundaries, cap_ways):
            # return a (wave_batch, boundaries, cap_ways) triple of strings
            ...

    Returns the callable, or None when no override module is present (the default: the operator's
    own --raceway-* flags and the generic cap auto-size are used as-is).
    """
    try:
        from cert_tier_geometry_override import tier_geometry_override
        return tier_geometry_override
    except ImportError:
        return None


def build_raceway_command(args: argparse.Namespace, batch: LeasedBatch) -> list[str]:
    # Optimized raceway screen: emits one representative per distinct final state (FN-safe state-dedup),
    # classified carnival(0x08)/other-world(0x09)/dual(0x0B), into a per-key hit-dir. The high-value
    # (other-world/dual) candidates are forwarded+verified on CPU after the worker exits.
    min_cert = min((cb for _key, cb in batch.keys), default=0)
    wave_batch = str(args.raceway_wave_continue_batch)
    boundaries = str(args.raceway_boundaries)
    cap_ways = str(args.raceway_cap_ways)
    # Per-tier wave geometry (single-batch window / drain boundaries / cap-ways) is problem- and
    # hardware-specific; by default the operator's --raceway-* flags below are used as given. An
    # operator may supply a validated per-tier override via cert_tier_geometry_override.py (see
    # _load_tier_geometry_override); when present it may adjust these three knobs for this batch's tier.
    _override = _load_tier_geometry_override()
    if _override is not None:
        wave_batch, boundaries, cap_ways = (str(v) for v in _override(min_cert, wave_batch, boundaries, cap_ways))
    cmd = [
        args.binary,
        "--device",
        batch.device,
        "--raceway-key-file",
        str(batch.keyfile),
        "--workunit_size",
        str(args.workunit_size),
        "--precert",
        "--raceway-direct-wave-continue-batch",
        wave_batch,
        "--raceway-direct-wave-state",
        "--raceway-direct-wave-boundaries",
        boundaries,
        "--raceway-direct-wave-span-ilp",
        str(args.raceway_span_ilp),
        "--raceway-worker-summary",
        str(batch.summary_path),
        "--raceway-hit-dir",
        str(raceway_hit_dir(batch)),
        "--worker-stop-file",
        str(batch.stop_file),
    ]
    cap_bits = str(args.raceway_cap_bits)
    if cap_bits == "0":
        # Auto-size the cap to ~1x the post-cert logical window (2^(32-cert_bits)) -> (30 - cert_bits)
        # buckets, using the batch's smallest cert (largest window) so the cap holds the heaviest
        # frontier, clamped to a VRAM-safe range. Override with --raceway-cap-bits to pin a value.
        cap_bits = str(max(12, min(26, 30 - min_cert)))
    cmd.extend(["--raceway-cap-bits", cap_bits, "--raceway-cap-ways", cap_ways])
    return cmd


def build_worker_command(args: argparse.Namespace, batch: LeasedBatch) -> list[str]:
    return build_raceway_command(args, batch) if args.engine == "raceway" else build_bfs_command(args, batch)


def write_keyfile(path: Path, keys: list[tuple[int, int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="ascii") as fh:
        fh.write("# key\tcert_bits\n")
        for key, cert_bits in keys:
            fh.write(f"0x{key:08x}\t{cert_bits}\n")


def lease_batch(conn: sqlite3.Connection, args: argparse.Namespace, device: str, out_dir: Path) -> LeasedBatch | None:
    now = utc_now()
    conn.execute("BEGIN IMMEDIATE")
    try:
        rows = conn.execute(
            """
            SELECT key, cert_bits, shard_index
            FROM keys
            WHERE status='pending' AND preferred_device=?
            ORDER BY source_order
            LIMIT ?
            """,
            (device, args.batch_keys),
        ).fetchall()
        if not rows:
            conn.execute("COMMIT")
            return None
        shard_index = int(rows[0]["shard_index"])
        keys = [(int(row["key"]), int(row["cert_bits"])) for row in rows]
        run_id = f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_dev{device}_{uuid.uuid4().hex[:8]}"
        run_dir = out_dir / "runs" / run_id
        keyfile = run_dir / f"dev{device}.keys.tsv"
        summary_path = run_dir / f"worker_summary_dev{device}.tsv"
        log_path = run_dir / f"dev{device}.log"
        if args.engine == "raceway":
            hit_path = run_dir / ("hits" if args.keep_raceway_provenance else ".raceway_hits.tmp")
            verified_path = None
            record_hit_path = hit_path if args.keep_raceway_provenance else None
            record_verified_path = None
        else:
            hit_path = (run_dir / f"dev{device}.hits.tsv") if args.buffered_hit_stream else None
            verified_path = (
                run_dir / f"dev{device}.verified.csv"
                if args.verified_stream
                else run_dir / f".dev{device}.verified.tmp.csv"
            )
            record_hit_path = hit_path
            record_verified_path = verified_path if args.verified_stream else None
        stop_file = run_dir / f"dev{device}.stop"
        placeholder = ",".join("?" for _ in keys)
        conn.execute(
            f"""
            UPDATE keys
            SET status='running', lease_id=?, last_run_id=?, attempts=attempts+1, updated_at=?
            WHERE key IN ({placeholder})
            """,
            [run_id, run_id, now, *[key for key, _ in keys]],
        )
        batch = LeasedBatch(
            run_id=run_id,
            device=device,
            shard_index=shard_index,
            keys=keys,
            keyfile=keyfile,
            summary_path=summary_path,
            log_path=log_path,
            hit_path=hit_path,
            verified_path=verified_path,
            record_hit_path=record_hit_path,
            record_verified_path=record_verified_path,
            stop_file=stop_file,
            command=[],
        )
        cmd = build_worker_command(args, batch)
        batch = LeasedBatch(
            run_id=batch.run_id,
            device=batch.device,
            shard_index=batch.shard_index,
            keys=batch.keys,
            keyfile=batch.keyfile,
            summary_path=batch.summary_path,
            log_path=batch.log_path,
            hit_path=batch.hit_path,
            verified_path=batch.verified_path,
            record_hit_path=batch.record_hit_path,
            record_verified_path=batch.record_verified_path,
            stop_file=batch.stop_file,
            command=cmd,
        )
        conn.execute(
            """
            INSERT INTO runs
                (run_id, device, shard_index, status, started_at, key_count, log_path,
                 summary_path, keyfile_path, hit_path, verified_path, command_json)
            VALUES (?, ?, ?, 'leased', ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                batch.run_id,
                batch.device,
                batch.shard_index,
                now,
                len(keys),
                str(batch.log_path),
                str(batch.summary_path),
                str(batch.keyfile),
                str(batch.record_hit_path) if batch.record_hit_path else None,
                str(batch.record_verified_path) if batch.record_verified_path else None,
                json.dumps(batch.command),
            ),
        )
        conn.execute("COMMIT")
        return batch
    except Exception:
        conn.execute("ROLLBACK")
        raise


def parse_int(value: str) -> int | None:
    if value == "":
        return None
    return int(value, 0)


def parse_float(value: str) -> float | None:
    if value == "":
        return None
    return float(value)


def parse_csv_int(row: dict[str, str], key: str, default: int = 0) -> int:
    value = row.get(key, "")
    if value == "":
        return default
    return int(value, 0)


def alert_log_path(args: argparse.Namespace) -> Path | None:
    if getattr(args, "no_alert_log", False):
        return None
    value = getattr(args, "alert_log", "")
    if value:
        return Path(value)
    return Path(args.out_dir) / "alerts.tsv"


def iter_verified_rows(path: Path) -> Iterable[dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return
    text = path.read_text(encoding="ascii", errors="replace")
    if not text:
        return
    if not text.endswith("\n"):
        cut = text.rfind("\n")
        if cut < 0:
            return
        text = text[: cut + 1]
    reader = csv.DictReader(io.StringIO(text))
    for row in reader:
        if row.get("key") and row.get("data"):
            yield row


def alert_reasons(row: dict[str, str], linear_threshold: int) -> list[str]:
    reasons: list[str] = []
    if parse_csv_int(row, "final_rts"):
        reasons.append("final_rts")
    if parse_csv_int(row, "entry_opcodes"):
        reasons.append("entry_opcodes")
    if parse_csv_int(row, "control_flow"):
        reasons.append("control_flow")
    if parse_csv_int(row, "structural"):
        reasons.append("structural")
    if parse_csv_int(row, "linear_clean"):
        reasons.append("linear_clean")
    linear_score = parse_csv_int(row, "linear_score")
    if linear_threshold > 0 and linear_score >= linear_threshold:
        reasons.append(f"linear_score_ge_{linear_threshold}")
    return reasons


def append_alert_log(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists() or path.stat().st_size == 0
    columns = [
        "created_at",
        "run_id",
        "device",
        "key",
        "data",
        "reason",
        "flags",
        "screen_flag",
        "machine_flags",
        "final_rts",
        "entry_opcodes",
        "control_flow",
        "structural",
        "linear_clean",
        "linear_score",
        "source_verified_file",
    ]
    with path.open("a", encoding="ascii", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns, delimiter="\t", extrasaction="ignore")
        if write_header:
            writer.writeheader()
        for row in rows:
            out = dict(row)
            out["key"] = f"0x{int(out['key']):08x}"
            writer.writerow(out)


def scan_verified_alerts(
    conn: sqlite3.Connection,
    run_id: str,
    device: str,
    verified_path: Path | None,
    log_path: Path | None,
    linear_threshold: int,
    source_verified_file: str | None = None,
) -> int:
    if verified_path is None or not verified_path.exists():
        return 0
    source_verified_file = source_verified_file or str(verified_path)
    created_at = utc_now()
    inserted_rows: list[dict[str, object]] = []
    conn.execute("BEGIN IMMEDIATE")
    try:
        for row in iter_verified_rows(verified_path):
            reasons = alert_reasons(row, linear_threshold)
            if not reasons:
                continue
            key_value = parse_csv_int(row, "key")
            data_value = parse_csv_int(row, "data")
            values = {
                "created_at": created_at,
                "run_id": run_id,
                "device": device,
                "key": key_value,
                "data": data_value,
                "reason": ",".join(reasons),
                "flags": parse_csv_int(row, "flags"),
                "screen_flag": parse_csv_int(row, "screen_flag"),
                "machine_flags": parse_csv_int(row, "machine_flags"),
                "final_rts": parse_csv_int(row, "final_rts"),
                "entry_opcodes": parse_csv_int(row, "entry_opcodes"),
                "control_flow": parse_csv_int(row, "control_flow"),
                "structural": parse_csv_int(row, "structural"),
                "linear_clean": parse_csv_int(row, "linear_clean"),
                "linear_score": parse_csv_int(row, "linear_score"),
                "b00": row.get("b00", ""),
                "b01": row.get("b01", ""),
                "b02": row.get("b02", ""),
                "b03": row.get("b03", ""),
                "b04": row.get("b04", ""),
                "b07": row.get("b07", ""),
                "b2e": row.get("b2e", ""),
                "b50": row.get("b50", ""),
                "source_verified_file": source_verified_file,
            }
            cur = conn.execute(
                """
                INSERT OR IGNORE INTO alerts
                    (created_at, run_id, device, key, data, reason, flags, screen_flag, machine_flags,
                     final_rts, entry_opcodes, control_flow, structural, linear_clean, linear_score,
                     b00, b01, b02, b03, b04, b07, b2e, b50, source_verified_file)
                VALUES
                    (:created_at, :run_id, :device, :key, :data, :reason, :flags, :screen_flag, :machine_flags,
                     :final_rts, :entry_opcodes, :control_flow, :structural, :linear_clean, :linear_score,
                     :b00, :b01, :b02, :b03, :b04, :b07, :b2e, :b50, :source_verified_file)
                """,
                values,
            )
            if cur.rowcount:
                inserted_rows.append(values)
        conn.execute("COMMIT")
    except Exception:
        conn.execute("ROLLBACK")
        raise
    append_alert_log(log_path, inserted_rows) if log_path is not None else None
    return len(inserted_rows)


def collect_raceway_candidates(hit_dir: Path) -> list[tuple[int, int, int]]:
    """Read per-key raceway hit files; return (key, data, flag) for high-value (other-world 0x09 /
    dual 0x0B) candidates. Carnival (0x08) hits target world A and are irrelevant to the bonus2 clear."""
    import re

    out: list[tuple[int, int, int]] = []
    if not hit_dir.exists():
        return out
    for f in sorted(hit_dir.glob("key_0x*.hits.tsv")):
        m = re.search(r"key_0x([0-9a-fA-F]+)\.hits", f.name)
        if not m:
            continue
        key = int(m.group(1), 16)
        for line in f.read_text(encoding="ascii", errors="replace").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                continue
            data = int(parts[0], 0)
            flag = int(parts[1], 0)
            if flag in (0x09, 0x0B):  # other-world / dual
                out.append((key, data, flag))
    return out


# Bonus2 clear: a forwarded other-world candidate is a strong-hit ALERT if its full-forward state
# shows a real other-world checksum match and/or valid machine code. ALL_ENTRIES_VALID = 0x04.
ALL_ENTRIES_VALID = 0x04


class PendingVerify(NamedTuple):
    """A raceway CPU-verify subprocess launched non-blocking so the GPU can relaunch the next
    batch immediately; harvested (verdict parsed + alerts inserted) once the subprocess exits."""
    proc: "subprocess.Popen[str]"
    candidate_csv: Path
    verdict_csv: Path
    hit_dir: Path
    flag_by_kd: dict
    run_id: str
    device: str
    linear_threshold: int
    log_path: "Path | None"
    batch_label: str
    keep_provenance: bool
    source_verified_file: str


def cleanup_raceway_provenance(pending: "PendingVerify") -> None:
    if pending.keep_provenance:
        return
    pending.candidate_csv.unlink(missing_ok=True)
    pending.verdict_csv.unlink(missing_ok=True)
    shutil.rmtree(pending.hit_dir, ignore_errors=True)


def start_raceway_verify(run_id, device, batch, linear_threshold, inspect_binary, log_path, keep_provenance):
    """Launch inspect_bonus2_survivors NON-blocking on this batch's other-world/dual candidates.
    Returns a PendingVerify (or None if there are no candidates). The ~3s CPU forward now overlaps
    the next batch's GPU compute instead of blocking the device relaunch."""
    hit_dir = raceway_hit_dir(batch)
    candidates = collect_raceway_candidates(hit_dir)
    if not candidates:
        if not keep_provenance:
            shutil.rmtree(hit_dir, ignore_errors=True)
        return None
    run_dir = batch.summary_path.parent
    if keep_provenance:
        cand_csv = run_dir / f"dev{device}.raceway_candidates.csv"
        verdict_csv = run_dir / f"dev{device}.raceway_verified.csv"
        source_verified_file = str(verdict_csv)
    else:
        cand_csv = run_dir / f".dev{device}.raceway_candidates.tmp.csv"
        verdict_csv = run_dir / f".dev{device}.raceway_verified.tmp.csv"
        source_verified_file = f"ephemeral:{run_id}:raceway_verified"
    with cand_csv.open("w", encoding="ascii") as fh:
        fh.write("key,data,flags\n")
        for key, data, _flag in candidates:
            # 0x05 = OTHER_WORLD|ALL_ENTRIES_VALID forces inspect_bonus2_survivors to forward+inspect.
            fh.write(f"{key},{data},5\n")
    if not keep_provenance:
        shutil.rmtree(hit_dir, ignore_errors=True)
    flag_by_kd = {(key, data): flag for key, data, flag in candidates}
    proc = subprocess.Popen([inspect_binary, str(cand_csv), str(verdict_csv)],
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    return PendingVerify(
        proc,
        cand_csv,
        verdict_csv,
        hit_dir,
        flag_by_kd,
        run_id,
        device,
        linear_threshold,
        log_path,
        batch.run_id,
        keep_provenance,
        source_verified_file,
    )


def harvest_raceway_verify(conn: sqlite3.Connection, pending: "PendingVerify") -> int:
    """Parse a completed verify subprocess's verdict and insert confirmed strong-hit alerts.
    Caller must ensure pending.proc has exited. DB writes stay on the main thread (no races)."""
    try:
        out = pending.proc.communicate()[0]
        if pending.proc.returncode != 0:
            raise RuntimeError(f"raceway CPU verifier failed rc={pending.proc.returncode}: {out}")
        run_id = pending.run_id
        device = pending.device
        verdict_csv = pending.verdict_csv
        flag_by_kd = pending.flag_by_kd
        linear_threshold = pending.linear_threshold
        log_path = pending.log_path
        created_at = utc_now()
        inserted_rows: list[dict[str, object]] = []
        conn.execute("BEGIN IMMEDIATE")
        try:
            for row in iter_verified_rows(verdict_csv):
                machine_flags = parse_csv_int(row, "machine_flags")
                linear_clean = parse_csv_int(row, "linear_clean")
                linear_score = parse_csv_int(row, "linear_score")
                if not (machine_flags & ALL_ENTRIES_VALID):
                    continue
                final_rts = parse_csv_int(row, "final_rts")
                entry_opcodes = parse_csv_int(row, "entry_opcodes")
                control_flow = parse_csv_int(row, "control_flow")
                structural = parse_csv_int(row, "structural")
                reasons = ["all_entries_valid"]
                if structural:
                    reasons.append("structural")
                if final_rts:
                    reasons.append("rts")
                if linear_clean:
                    reasons.append("linear_clean")
                if linear_threshold > 0 and linear_score >= linear_threshold:
                    reasons.append(f"linear_score_ge_{linear_threshold}")
                key_value = parse_csv_int(row, "key")
                data_value = parse_csv_int(row, "data")
                values = {
                    "created_at": created_at,
                    "run_id": run_id,
                    "device": device,
                    "key": key_value,
                    "data": data_value,
                    "reason": ",".join(reasons),
                    "flags": flag_by_kd.get((key_value, data_value), 0),
                    "screen_flag": flag_by_kd.get((key_value, data_value), 0),
                    "machine_flags": machine_flags,
                    "final_rts": final_rts,
                    "entry_opcodes": entry_opcodes,
                    "control_flow": control_flow,
                    "structural": structural,
                    "linear_clean": linear_clean,
                    "linear_score": linear_score,
                    "b00": row.get("b00", ""),
                    "b01": row.get("b01", ""),
                    "b02": row.get("b02", ""),
                    "b03": row.get("b03", ""),
                    "b04": row.get("b04", ""),
                    "b07": row.get("b07", ""),
                    "b2e": row.get("b2e", ""),
                    "b50": row.get("b50", ""),
                    "source_verified_file": pending.source_verified_file,
                }
                cur = conn.execute(
                    """
                    INSERT OR IGNORE INTO alerts
                        (created_at, run_id, device, key, data, reason, flags, screen_flag, machine_flags,
                         final_rts, entry_opcodes, control_flow, structural, linear_clean, linear_score,
                         b00, b01, b02, b03, b04, b07, b2e, b50, source_verified_file)
                    VALUES
                        (:created_at, :run_id, :device, :key, :data, :reason, :flags, :screen_flag, :machine_flags,
                         :final_rts, :entry_opcodes, :control_flow, :structural, :linear_clean, :linear_score,
                         :b00, :b01, :b02, :b03, :b04, :b07, :b2e, :b50, :source_verified_file)
                    """,
                    values,
                )
                if cur.rowcount:
                    inserted_rows.append(values)
            conn.execute("COMMIT")
        except Exception:
            conn.execute("ROLLBACK")
            raise
        if log_path is not None:
            append_alert_log(log_path, inserted_rows)
        return len(inserted_rows)
    finally:
        cleanup_raceway_provenance(pending)


def verify_raceway_alerts(
    conn: sqlite3.Connection,
    run_id: str,
    device: str,
    batch: LeasedBatch,
    log_path: Path | None,
    linear_threshold: int,
    inspect_binary: str,
    keep_provenance: bool,
) -> int:
    """Synchronous verify (start + wait + harvest); used by --sync-verify, periodic ingest, and shutdown."""
    pending = start_raceway_verify(run_id, device, batch, linear_threshold, inspect_binary, log_path, keep_provenance)
    if pending is None:
        return 0
    pending.proc.wait()
    return harvest_raceway_verify(conn, pending)


def cleanup_bfs_verified_stream(args: argparse.Namespace, batch: LeasedBatch) -> None:
    if args.engine == "bfs" and not args.verified_stream and batch.verified_path is not None:
        batch.verified_path.unlink(missing_ok=True)


def alerts_for_batch(
    conn: sqlite3.Connection,
    args: argparse.Namespace,
    batch: LeasedBatch,
    alerts_path: Path | None,
) -> int:
    """Engine-appropriate strong-hit alert scan for a finished/partial batch."""
    if args.engine == "raceway":
        return verify_raceway_alerts(
            conn,
            batch.run_id,
            batch.device,
            batch,
            alerts_path,
            args.alert_linear_score,
            args.inspect_binary,
            args.keep_raceway_provenance,
        )
    try:
        source = None if args.verified_stream else f"ephemeral:{batch.run_id}:bfs_verified"
        return scan_verified_alerts(
            conn, batch.run_id, batch.device, batch.verified_path, alerts_path, args.alert_linear_score, source
        )
    finally:
        cleanup_bfs_verified_stream(args, batch)


def load_summary_rows(path: Path) -> dict[int, dict[str, str]]:
    rows_by_key: dict[int, dict[str, str]] = {}
    if path.exists():
        with path.open("r", encoding="ascii", newline="") as fh:
            for row in csv.DictReader(fh, delimiter="\t"):
                if not row.get("key"):
                    continue
                rows_by_key[int(row["key"], 0)] = row
    return rows_by_key


def summary_data_row_count(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="ascii", errors="replace") as fh:
        count = sum(1 for line in fh if line.strip())
    return max(0, count - 1)


def ingest_summary(
    conn: sqlite3.Connection,
    batch: LeasedBatch,
    returncode: int | None = None,
    final: bool = True,
    missing_pending: bool = False,
) -> int:
    now = utc_now()
    rows_by_key = load_summary_rows(batch.summary_path)
    missing_rows = 0

    conn.execute("BEGIN IMMEDIATE")
    try:
        for key, _cert_bits in batch.keys:
            row = rows_by_key.get(key)
            if row is None:
                missing_rows += 1
                if not final:
                    continue
                if returncode == 0 and not missing_pending:
                    conn.execute(
                        """
                        UPDATE keys
                        SET status='failed', lease_id=NULL, updated_at=?, error='missing summary row'
                        WHERE key=? AND lease_id=?
                        """,
                        (now, key, batch.run_id),
                    )
                else:
                    conn.execute(
                        """
                        UPDATE keys
                        SET status='pending', lease_id=NULL, updated_at=?, error=?
                        WHERE key=? AND lease_id=?
                        """,
                        (
                            now,
                            "worker stopped before summary" if returncode == 0 else "worker interrupted before summary",
                            key,
                            batch.run_id,
                        ),
                    )
                continue

            row_rc = int(row.get("returncode") or "0")
            status = "done" if row_rc == 0 else "failed"
            conn.execute(
                """
                UPDATE keys
                SET status=?, lease_id=NULL, updated_at=?, error=?,
                    elapsed_s=?, shed_bits=?, logical_window=?, f1=?, final_frontier=?,
                    flag_sum=?, representative_hits=?, verify_enqueued_hits=?,
                    map1_ms=?, deep_ms=?, total_ms=?, host_setup_ms=?, hit_write_ms=?,
                    verify_enqueue_ms=?, verify_state_copy_ms=?,
                    hit_file=?, verified_file=?
                WHERE key=? AND last_run_id=?
                """,
                (
                    status,
                    now,
                    None if row_rc == 0 else f"returncode {row_rc}",
                    parse_float(row.get("elapsed_s", "")),
                    parse_int(row.get("shed_bits", "")),
                    parse_int(row.get("logical_window", "")),
                    parse_int(row.get("F1", "")),
                    parse_int(row.get("final_frontier", "")),
                    parse_int(row.get("flag_sum", "")),
                    parse_int(row.get("representative_hits", "")),
                    parse_int(row.get("verify_enqueued_hits", "")),
                    parse_float(row.get("map1_ms", "")),
                    parse_float(row.get("deep_ms", "")),
                    parse_float(row.get("total_ms", "")),
                    parse_float(row.get("host_setup_ms", "")),
                    parse_float(row.get("hit_write_ms", "")),
                    parse_float(row.get("verify_enqueue_ms", "")),
                    parse_float(row.get("verify_state_copy_ms", "")),
                    str(batch.record_hit_path) if batch.record_hit_path else None,
                    str(batch.record_verified_path) if batch.record_verified_path else None,
                    key,
                    batch.run_id,
                ),
            )

        if final:
            remaining = conn.execute(
                "SELECT COUNT(*) FROM keys WHERE lease_id=? AND status='running'",
                (batch.run_id,),
            ).fetchone()[0]
            if returncode == 0 and remaining == 0:
                status = "stopped" if missing_pending and missing_rows else "done"
            else:
                status = "failed"
            conn.execute(
                """
                UPDATE runs
                SET status=?, ended_at=?, returncode=?
                WHERE run_id=?
                """,
                (status, now, returncode, batch.run_id),
            )
        conn.execute("COMMIT")
        return len(rows_by_key)
    except Exception:
        conn.execute("ROLLBACK")
        raise


def prepare_batch_files(batch: LeasedBatch) -> None:
    """Materialize a batch's on-disk inputs/outputs (keyfile, dirs) before the worker runs it.
    Used by both launch_batch (spawn-per-batch) and the daemon path (which feeds batches over stdin)."""
    batch.log_path.parent.mkdir(parents=True, exist_ok=True)
    write_keyfile(batch.keyfile, batch.keys)
    batch.stop_file.unlink(missing_ok=True)
    if batch.hit_path:
        batch.hit_path.parent.mkdir(parents=True, exist_ok=True)
    if batch.verified_path:
        batch.verified_path.parent.mkdir(parents=True, exist_ok=True)
    if "--raceway-hit-dir" in batch.command:
        # The raceway worker opens per-key hit files in this dir but does not create it.
        raceway_hit_dir(batch).mkdir(parents=True, exist_ok=True)


def launch_batch(batch: LeasedBatch, dry_run: bool) -> subprocess.Popen[str] | None:
    prepare_batch_files(batch)
    if dry_run:
        print(f"# {batch.run_id} device={batch.device} keys={len(batch.keys)}")
        print(shlex.join(batch.command))
        return None
    log_fh = batch.log_path.open("w", encoding="utf-8")
    proc = subprocess.Popen(batch.command, stdout=log_fh, stderr=subprocess.STDOUT, text=True)
    proc._cert_log_fh = log_fh  # type: ignore[attr-defined]
    return proc


def request_worker_stop(batch: LeasedBatch) -> None:
    batch.stop_file.parent.mkdir(parents=True, exist_ok=True)
    batch.stop_file.write_text(f"stop_requested_at={utc_now()}\n", encoding="ascii")


def command_run(args: argparse.Namespace) -> int:
    if args.engine not in ("bfs", "raceway"):
        raise SystemExit("cert_tier_ops supports --engine bfs or raceway")
    if not Path(args.binary).exists():
        raise SystemExit(f"CUDA binary not found: {args.binary}")
    if args.engine == "raceway" and not Path(args.inspect_binary).exists():
        raise SystemExit(
            f"raceway clearing needs the CPU receiver/verifier: {args.inspect_binary} not found "
            "(build: make -C src/bruteforce inspect_bonus2_survivors)"
        )
    devices = parse_devices(args.devices)
    out_dir = Path(args.out_dir)
    conn = connect(Path(args.db))
    ensure_schema(conn)
    alerts_path = alert_log_path(args)

    if args.dry_run:
        for device in devices:
            rows = conn.execute(
                """
                SELECT key, cert_bits, shard_index
                FROM keys
                WHERE status='pending' AND preferred_device=?
                ORDER BY source_order
                LIMIT ?
                """,
                (device, args.batch_keys),
            ).fetchall()
            if not rows:
                print(f"# device={device} no pending work")
                continue
            keys = [(int(row["key"]), int(row["cert_bits"])) for row in rows]
            shard_index = int(rows[0]["shard_index"])
            run_id = f"dryrun_dev{device}"
            run_dir = out_dir / "runs" / run_id
            if args.engine == "raceway":
                hit_path = run_dir / ("hits" if args.keep_raceway_provenance else ".raceway_hits.tmp")
                verified_path = None
                record_hit_path = hit_path if args.keep_raceway_provenance else None
                record_verified_path = None
            else:
                hit_path = (run_dir / f"dev{device}.hits.tsv") if args.buffered_hit_stream else None
                verified_path = (
                    run_dir / f"dev{device}.verified.csv"
                    if args.verified_stream
                    else run_dir / f".dev{device}.verified.tmp.csv"
                )
                record_hit_path = hit_path
                record_verified_path = verified_path if args.verified_stream else None
            batch = LeasedBatch(
                run_id=run_id,
                device=device,
                shard_index=shard_index,
                keys=keys,
                keyfile=run_dir / f"dev{device}.keys.tsv",
                summary_path=run_dir / f"worker_summary_dev{device}.tsv",
                log_path=run_dir / f"dev{device}.log",
                hit_path=hit_path,
                verified_path=verified_path,
                record_hit_path=record_hit_path,
                record_verified_path=record_verified_path,
                stop_file=run_dir / f"dev{device}.stop",
                command=[],
            )
            cmd = build_worker_command(args, batch)
            print(f"# dry-run device={device} keys={len(keys)} first=0x{keys[0][0]:08x} last=0x{keys[-1][0]:08x}")
            print(shlex.join(cmd))
        return 0

    failures = 0
    start = time.monotonic()
    # Independent per-device pipelines: each device leases and runs its own batches back-to-back and
    # never idles waiting on the other device. A device that finishes relaunches in the SAME tick.
    dev_state: dict[str, dict] = {
        device: {"proc": None, "batch": None, "batches": 0, "drained": False, "last_ingested": 0}
        for device in devices
    }
    # Worker-completion events (device name) pushed by per-worker watcher threads (see launch_device),
    # so the loop reaps + relaunches the instant a worker exits instead of polling on a 0.5s sleep.
    completion_q: "queue.Queue[str]" = queue.Queue()
    stop_announced = False

    def launch_device(device: str) -> bool:
        batch = lease_batch(conn, args, device, out_dir)
        if batch is None:
            return False
        print(f"leased {batch.run_id} device={device} keys={len(batch.keys)}", flush=True)
        proc = launch_batch(batch, False)
        conn.execute("UPDATE runs SET status='running' WHERE run_id=?", (batch.run_id,))
        conn.commit()
        st = dev_state[device]
        st.update(proc=proc, batch=batch, batches=st["batches"] + 1, last_ingested=0)
        # Event-driven reap: a daemon watcher blocks on proc exit and signals the loop the instant the
        # worker finishes, so the device relaunches with ~0 detection latency (no 0.5s poll). The worker's
        # stdout is a log FILE (not a pipe), so wait() cannot deadlock on an unread pipe.
        threading.Thread(target=lambda p=proc, d=device: (p.wait(), completion_q.put(d)),
                         daemon=True).start()
        return True

    def finish_device(device: str, rc: int) -> None:
        nonlocal failures
        st = dev_state[device]
        batch = st["batch"]
        st["proc"]._cert_log_fh.close()  # type: ignore[attr-defined]
        ingest_summary(conn, batch, rc, missing_pending=batch.stop_file.exists())
        failures += int(rc != 0)
        if args.engine == "raceway" and not args.sync_verify:
            # Launch the ~3s CPU verify NON-blocking so this device relaunches immediately; the
            # verdict is harvested (alerts inserted) in a later loop tick, overlapping the next
            # batch's GPU compute. DB writes stay on the main thread -> no sqlite races.
            pv = start_raceway_verify(batch.run_id, device, batch, args.alert_linear_score,
                                      args.inspect_binary, alerts_path, args.keep_raceway_provenance)
            if pv is not None:
                pending_verifies.append(pv)
            print(f"{batch.run_id} device={device} rc={rc} verify=bg log={batch.log_path}", flush=True)
        else:
            alerts = alerts_for_batch(conn, args, batch, alerts_path)
            alert_msg = f" alerts={alerts}" if alerts else ""
            print(f"{batch.run_id} device={device} rc={rc}{alert_msg} log={batch.log_path}", flush=True)
        st.update(proc=None, batch=None)

    pending_verifies: list[PendingVerify] = []

    def harvest_pending_verifies() -> None:
        if not pending_verifies:
            return
        still: list[PendingVerify] = []
        for pv in pending_verifies:
            if pv.proc.poll() is None:
                still.append(pv)
                continue
            try:
                n = harvest_raceway_verify(conn, pv)
                if n:
                    print(f"{pv.batch_label} device={pv.device} alerts={n} (bg-verified)", flush=True)
            except Exception as exc:  # never let a verify failure kill the run loop
                print(f"WARNING: bg-verify harvest failed for {pv.batch_label}: {exc}", file=sys.stderr, flush=True)
        pending_verifies[:] = still

    def drain_pending_verifies() -> None:
        for pv in pending_verifies:
            try:
                pv.proc.wait()
                n = harvest_raceway_verify(conn, pv)
                if n:
                    print(f"{pv.batch_label} device={pv.device} alerts={n} (bg-verified, drained)", flush=True)
            except Exception as exc:
                print(f"WARNING: bg-verify drain failed for {pv.batch_label}: {exc}", file=sys.stderr, flush=True)
        pending_verifies.clear()

    if args.daemon:
        # ── Persistent-worker daemon lifecycle ───────────────────────────────────────────────────
        # One resident worker per device; batches are fed over its stdin (keyfile\tsummary\thitdir) and
        # completion is signalled by a "DAEMON_DONE" line on its stdout. Eliminates the ~0.5s/batch CUDA
        # startup. Reuses the same lease / ingest / async-verify helpers as the spawn-per-batch path.
        for _d in devices:
            dev_state[_d]["daemon"] = None
            dev_state[_d]["daemon_log"] = None

        def feed_batch(device: str, batch: "LeasedBatch") -> None:
            st = dev_state[device]
            st["daemon"].stdin.write(f"{batch.keyfile}\t{batch.summary_path}\t{raceway_hit_dir(batch)}\n")
            st["daemon"].stdin.flush()

        def send_stop(device: str) -> None:
            st = dev_state[device]
            try:
                st["daemon"].stdin.write("STOP\n")
                st["daemon"].stdin.flush()
            except (BrokenPipeError, ValueError, OSError):
                pass
            st["drained"] = True

        def launch_daemon(device: str) -> bool:
            batch = lease_batch(conn, args, device, out_dir)
            if batch is None:
                return False
            prepare_batch_files(batch)
            cmd = build_worker_command(args, batch) + ["--raceway-daemon"]
            log_fh = open(out_dir / f"daemon_dev{device}.log", "a")
            proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                    stderr=log_fh, text=True, bufsize=1)
            conn.execute("UPDATE runs SET status='running' WHERE run_id=?", (batch.run_id,))
            conn.commit()
            st = dev_state[device]
            st.update(daemon=proc, batch=batch, batches=st["batches"] + 1, daemon_log=log_fh, last_ingested=0)

            def _reader(p=proc, dev=device):
                try:
                    for line in p.stdout:
                        if line.strip() == "DAEMON_DONE":
                            completion_q.put((dev, "done"))
                finally:
                    completion_q.put((dev, "exit"))
            threading.Thread(target=_reader, daemon=True).start()
            print(f"daemon-leased {batch.run_id} device={device} keys={len(batch.keys)}", flush=True)
            return True

        for _d in devices:
            if args.loops and dev_state[_d]["batches"] >= args.loops:
                dev_state[_d]["drained"] = True
            elif not launch_daemon(_d):
                dev_state[_d]["drained"] = True
                print(f"device={_d} no pending work", flush=True)

        daemon_stop_signalled = False
        while True:
            stop = stop_requested(conn)
            time_up = bool(args.stop_after_min) and (time.monotonic() - start) >= args.stop_after_min * 60.0
            # Mid-batch stop: a resident daemon only reads a stdin STOP *between* batches, which can be many
            # minutes into an 8000-key batch. Writing its per-device stop-file (stable across batches) makes
            # the engine break at its next key boundary (checked per key), so q / request-stop / stop-after-min
            # take effect in ~1 key instead of a whole batch. The batch-boundary STOP below still covers a
            # daemon caught between batches.
            if (stop or time_up) and not daemon_stop_signalled:
                daemon_stop_signalled = True
                for _d in devices:
                    st_d = dev_state[_d]
                    if st_d.get("daemon") is not None and st_d.get("batch") is not None and not st_d.get("drained"):
                        request_worker_stop(st_d["batch"])
                print(f"{'stop requested' if stop else 'stop-after-min reached'}; "
                      "daemon workers stop after their current key", flush=True)
            harvest_pending_verifies()
            if not any(dev_state[_d].get("daemon") is not None for _d in devices):
                drain_pending_verifies()
                break
            try:
                dev, ev = completion_q.get(timeout=1.0)
            except queue.Empty:
                continue
            st = dev_state[dev]
            if ev == "done":
                batch = st["batch"]
                stopping = bool(stop or time_up or (args.loops and st["batches"] >= args.loops))
                # On a mid-batch stop the summary is PARTIAL; return the unprocessed leased keys to 'pending'
                # (missing_pending) rather than failing them, so a resumed run re-leases them.
                ingest_summary(conn, batch, 0, missing_pending=stopping)
                if args.engine == "raceway" and not args.sync_verify:
                    pv = start_raceway_verify(batch.run_id, dev, batch, args.alert_linear_score,
                                              args.inspect_binary, alerts_path, args.keep_raceway_provenance)
                    if pv is not None:
                        pending_verifies.append(pv)
                    print(f"{batch.run_id} device={dev} batch-done verify=bg", flush=True)
                else:
                    alerts = alerts_for_batch(conn, args, batch, alerts_path)
                    print(f"{batch.run_id} device={dev} batch-done alerts={alerts}", flush=True)
                if stopping:
                    send_stop(dev)
                    continue
                nb = lease_batch(conn, args, dev, out_dir)
                if nb is None:
                    send_stop(dev)
                    print(f"device={dev} no pending work", flush=True)
                    continue
                prepare_batch_files(nb)
                conn.execute("UPDATE runs SET status='running' WHERE run_id=?", (nb.run_id,))
                conn.commit()
                st["batch"] = nb
                st["batches"] += 1
                feed_batch(dev, nb)
            elif ev == "exit":
                rc = st["daemon"].poll() if st["daemon"] is not None else None
                if st.get("daemon_log"):
                    st["daemon_log"].close()
                if not st.get("drained"):
                    failures += 1
                    print(f"WARNING: daemon device={dev} exited unexpectedly rc={rc}; in-flight batch "
                          f"{st['batch'].run_id} keys remain 'running' (run reset-running to recover)",
                          file=sys.stderr, flush=True)
                st["daemon"] = None
        return failures

    last_poll = 0.0
    try:
        while True:
            now = time.monotonic()
            stop = stop_requested(conn)
            time_up = bool(args.stop_after_min) and (now - start) >= args.stop_after_min * 60.0
            if (stop or time_up) and not stop_announced:
                stop_announced = True
                for st in dev_state.values():
                    if st["proc"] is not None:
                        request_worker_stop(st["batch"])
                why = "stop requested" if stop else "stop-after-min reached"
                print(f"{why}; active workers exit after their current key; no new batches will lease", flush=True)

            # 1) reap completed workers (frees the device's slot for an immediate relaunch below)
            for device, st in dev_state.items():
                if st["proc"] is None:
                    continue
                rc = st["proc"].poll()
                if rc is not None:
                    finish_device(device, rc)

            # 2) launch any idle device that still has work (independent of the other device)
            if not (stop or time_up):
                for device in devices:
                    st = dev_state[device]
                    if st["proc"] is not None or st["drained"]:
                        continue
                    if args.loops and st["batches"] >= args.loops:
                        continue
                    if not launch_device(device):
                        st["drained"] = True
                        print(f"device={device} no pending work", flush=True)

            # 2b) harvest any completed background verifies (overlapped the just-relaunched batch)
            harvest_pending_verifies()

            # 3) periodic PROGRESS ingest for active devices (feeds the live monitor + durable checkpoint).
            # Verification is intentionally NOT run here: the old mid-batch alerts_for_batch re-verified the
            # whole CUMULATIVE candidate set on every tick (collect_raceway_candidates reads the entire growing
            # hit-dir), so an N-key batch cost ~sum(chunks) ≈ several x the keys in redundant CPU forwards, and
            # each call blocked the loop (delaying the OTHER device's reap on dual-GPU). All hits are verified
            # exactly once at batch end via the background verify (finish_device) and, for killed batches, in
            # the interrupt handler — so alerts are still complete, just surfaced at batch end instead of mid-batch.
            if args.ingest_every_keys and now - last_poll >= args.ingest_poll_s:
                for device, st in dev_state.items():
                    if st["proc"] is None:
                        continue
                    row_count = summary_data_row_count(st["batch"].summary_path)
                    if row_count - st["last_ingested"] >= args.ingest_every_keys:
                        ingested = ingest_summary(conn, st["batch"], final=False)
                        st["last_ingested"] = row_count
                        print(
                            f"{st['batch'].run_id} device={device} progress_ingested={ingested}/{len(st['batch'].keys)}",
                            flush=True,
                        )
                last_poll = now

            # 4) termination: nothing active and nothing left to launch (or stopping)
            any_active = any(st["proc"] is not None for st in dev_state.values())
            if not any_active:
                drain_pending_verifies()  # harvest final batches' bg verifies before exiting
                if stop or time_up:
                    break
                can_launch = any(
                    not st["drained"] and not (args.loops and st["batches"] >= args.loops)
                    for st in dev_state.values()
                )
                if not can_launch:
                    print("no pending work for requested devices; exiting", flush=True)
                    break

            # Event-driven wait: block until a worker exits (watcher thread signals completion_q) so reap +
            # relaunch happen with ~0 latency; the timeout is only a fallback tick for the time-based periodic
            # progress-ingest and bg-verify harvest. Drain coalesced events (the reap loop handles all devices).
            if any_active:
                try:
                    completion_q.get(timeout=1.0)
                    try:
                        while True:
                            completion_q.get_nowait()
                    except queue.Empty:
                        pass
                except queue.Empty:
                    pass
            else:
                time.sleep(0.05)
    except KeyboardInterrupt:
        print("interrupt received; terminating active workers", file=sys.stderr)
        for st in dev_state.values():
            if st["proc"] is not None and st["proc"].poll() is None:
                st["proc"].terminate()
        for device, st in dev_state.items():
            if st["proc"] is None:
                continue
            try:
                rc = st["proc"].wait(timeout=15)
            except subprocess.TimeoutExpired:
                st["proc"].kill()
                rc = st["proc"].wait()
            st["proc"]._cert_log_fh.close()  # type: ignore[attr-defined]
            if st["batch"].summary_path.exists():
                ingest_summary(conn, st["batch"], rc, missing_pending=st["batch"].stop_file.exists())
                alerts_for_batch(conn, args, st["batch"], alerts_path)
            print(
                f"{st['batch'].run_id} device={device} interrupted rc={rc}; "
                "completed summary rows were ingested; run reset-running before retrying if keys remain running",
                flush=True,
            )
        drain_pending_verifies()
        raise
    return failures


def command_reset_running(args: argparse.Namespace) -> int:
    conn = connect(Path(args.db))
    ensure_schema(conn)
    rows = conn.execute(
        """
        SELECT lease_id, COUNT(*) AS keys
        FROM keys
        WHERE status='running'
        GROUP BY lease_id
        ORDER BY lease_id
        """
    ).fetchall()
    if not rows:
        print("no running leases")
        return 0
    print_table(rows, ["lease_id", "keys"])
    if args.dry_run:
        return 0
    now = utc_now()
    conn.execute("BEGIN IMMEDIATE")
    conn.execute(
        """
        UPDATE keys
        SET status='pending', lease_id=NULL, updated_at=?, error='reset-running'
        WHERE status='running'
        """,
        (now,),
    )
    conn.execute(
        """
        UPDATE runs
        SET status='reset', ended_at=COALESCE(ended_at, ?)
        WHERE status IN ('leased', 'running')
        """,
        (now,),
    )
    conn.commit()
    print("reset running keys to pending")
    return 0


def command_scan_alerts(args: argparse.Namespace) -> int:
    conn = connect(Path(args.db))
    ensure_schema(conn)
    rows = conn.execute(
        """
        SELECT run_id, device, verified_path
        FROM runs
        WHERE verified_path IS NOT NULL
        ORDER BY started_at
        """
    ).fetchall()
    log_path = alert_log_path(args)
    total = 0
    for row in rows:
        inserted = scan_verified_alerts(
            conn,
            row["run_id"],
            row["device"],
            Path(row["verified_path"]),
            log_path,
            args.alert_linear_score,
        )
        if inserted:
            print(f"{row['run_id']} device={row['device']} alerts={inserted}")
        total += inserted
    print(f"new_alerts={total}")
    return 0


def command_request_stop(args: argparse.Namespace) -> int:
    conn = connect(Path(args.db))
    ensure_schema(conn)
    requested_at = utc_now()
    set_meta(conn, "stop_requested_at", requested_at)
    print(f"stop_requested_at={requested_at}")
    return 0


def command_clear_stop(args: argparse.Namespace) -> int:
    conn = connect(Path(args.db))
    ensure_schema(conn)
    old_value = get_meta(conn, "stop_requested_at")
    clear_meta(conn, "stop_requested_at")
    if old_value:
        print(f"cleared stop_requested_at={old_value}")
    else:
        print("no stop request was set")
    return 0


def add_common_run_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--db", default="common/results/tier_ops/queue.sqlite")
    parser.add_argument("--out-dir", default="common/results/tier_ops")
    parser.add_argument("--binary", default="src/bruteforce/test_cuda/test_cuda")
    parser.add_argument("--devices", default="1,0")


def add_run_args(run: argparse.ArgumentParser) -> None:
    """Args shared by `run` and the interactive `operate` console (which supervises a `run`)."""
    add_common_run_args(run)
    run.add_argument("--engine", choices=["bfs", "raceway"], default="raceway",
                     help="Clearing engine. raceway (default) is faster on every measured tier; "
                          "bfs is the exact-dedup reference, kept as a fallback.")
    run.add_argument(
        "--batch-keys",
        type=int,
        default=8000,
        help="Keys leased per worker per batch. With verify overlapped (background) and event-driven reap, "
        "the only per-batch fixed cost is worker startup (~1s), so larger batches amortize it away. Periodic "
        "ingest keeps done-keys durable (reset-running recovers a crash), so large batches are safe. Raise it "
        "for fast/high-cert tiers where per-key compute is small.",
    )
    # Raceway engine (--engine raceway): optimized screen + CPU-receiver verification of high-value hits.
    run.add_argument("--raceway-wave-continue-batch", default="auto", help="Raceway bounded-wave size; per-tier tunable (auto = size to the post-cert window).")
    run.add_argument("--raceway-boundaries", default="2,6,16", help="Raceway completed-map drain cadence (comma-separated map indices).")
    run.add_argument("--raceway-span-ilp", default="4", help="Raceway cap-span ILP (4 is a good default).")
    run.add_argument("--raceway-cap-bits", default="0", help="Raceway cap 2^B buckets; 0 = auto-size per tier from the batch's cert (30 - cert_bits, VRAM-clamped). Override to pin a value.")
    run.add_argument("--raceway-cap-ways", default="4", help="Raceway cap ways per bucket.")
    run.add_argument(
        "--inspect-binary",
        default="src/bruteforce/inspect_bonus2_survivors",
        help="CPU receiver that forwards+verifies raceway other-world/dual hit candidates.",
    )
    run.add_argument("--loops", type=int, default=1, help="Batch rounds to run; 0 means until no pending work.")
    run.add_argument("--stop-after-min", type=float, default=0.0, help="Do not lease a new batch after this many minutes.")
    run.add_argument("--daemon", action="store_true",
                     help="Persistent-worker mode: launch ONE resident worker per device and feed successive "
                     "batches over its stdin (keyfile\\tsummary\\thitdir), eliminating the ~0.5s per-batch CUDA "
                     "startup. Requires a daemon-capable --binary (built with --raceway-daemon support). Assumes a "
                     "uniform tier per device (worker flags are fixed at launch).")
    run.add_argument("--sync-verify", action="store_true",
                     help="Run the CPU hit-verifier synchronously (blocks device relaunch). Default: verify runs in the "
                     "background, overlapping the next batch's GPU compute.")
    run.add_argument(
        "--keep-raceway-provenance",
        action="store_true",
        help="Persist raceway per-key hit files plus full candidate/verified CSVs. Default keeps only alerts, "
        "queue state, logs, keys, and worker summaries; temporary non-alert verifier streams are removed after harvest.",
    )
    run.add_argument(
        "--ingest-every-keys",
        type=int,
        default=250,
        help="Commit completed summary rows to SQLite every N new keys per active worker; 0 disables "
        "progress ingest. Bounds rework on a hard crash (at slow low-cert rates a smaller N commits more "
        "often). Also feeds the live monitor's in-flight progress.",
    )
    run.add_argument(
        "--ingest-poll-s",
        type=float,
        default=5.0,
        help="Minimum seconds between active-worker summary scans for progress ingest.",
    )
    run.add_argument("--workunit-size", default=str(1 << 32))
    run.add_argument("--map1-frontier-chunk", default=str(1 << 20))
    run.add_argument("--map1-frontier-deep-k", default="4")
    run.add_argument("--map1-frontier-stream-cuts", default="")
    run.add_argument("--map1-frontier-drain-cap-bits", default="0")
    run.add_argument("--map1-frontier-drain-cap-ways", default="4")
    run.add_argument("--map1-frontier-drain-cap-distinct", action="store_true")
    run.add_argument("--map1-frontier-overlap-setup", action="store_true")
    run.add_argument(
        "--buffered-hit-stream",
        action="store_true",
        default=False,
        help="BFS engine only: persist the full buffered representative hit stream. Default is off.",
    )
    run.add_argument("--no-buffered-hit-stream", action="store_false", dest="buffered_hit_stream")
    run.add_argument(
        "--verified-stream",
        action="store_true",
        default=False,
        help="BFS engine only: persist the full verified CSV. Default uses a temporary verified CSV for alert "
        "ingestion and removes it after scanning.",
    )
    run.add_argument("--no-verified-stream", action="store_false", dest="verified_stream")
    run.add_argument("--map1-frontier-verify-final-rts", action="store_true")
    run.add_argument(
        "--alert-linear-score",
        type=int,
        default=32,
        help="Alert when verifier linear_score is >= N; 0 disables score-based alerts.",
    )
    run.add_argument(
        "--alert-log",
        default="",
        help="Append-only strong-hit alert TSV; default is OUT_DIR/alerts.tsv.",
    )
    run.add_argument("--no-alert-log", action="store_true", help="Disable append-only alert TSV; DB alerts remain enabled.")
    run.add_argument("--dry-run", action="store_true")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="command", required=True)

    init = sub.add_parser("init", help="Import shard keys into the durable queue DB.")
    add_common_run_args(init)
    init.add_argument("--shard-dir", default="common/results/tier_sweep")
    init.add_argument("--min-bits", type=int, default=12)
    init.add_argument("--max-keys-per-shard", type=int, default=0)
    init.add_argument("--insert-batch", type=int, default=100000)

    status = sub.add_parser("status", help="Show queue/run progress, ETA, in-flight runs, and alerts.")
    status.add_argument("--db", default="common/results/tier_ops/queue.sqlite")
    status.add_argument("--recent", type=int, default=8)
    status.add_argument("--recent-alerts", type=int, default=8)

    monitor = sub.add_parser("monitor", help="Live operator dashboard: progress/ETA, in-flight runs, verifier hits.")
    monitor.add_argument("--db", default="common/results/tier_ops/queue.sqlite")
    monitor.add_argument("--interval", type=float, default=10.0, help="Refresh seconds (default 10).")
    monitor.add_argument("--once", action="store_true", help="Print one snapshot and exit.")
    monitor.add_argument("--no-clear", action="store_true", help="Do not clear the screen between refreshes.")
    monitor.add_argument("--recent-alerts", type=int, default=10)

    run = sub.add_parser("run", help="Lease and run bounded batches, one worker per requested device.")
    add_run_args(run)

    operate = sub.add_parser(
        "operate",
        help="Interactive console: supervise a run AND show the live dashboard; pause/resume/quit with keys.",
    )
    add_run_args(operate)

    reset = sub.add_parser("reset-running", help="Return running leases to pending after an interrupted run.")
    reset.add_argument("--db", default="common/results/tier_ops/queue.sqlite")
    reset.add_argument("--dry-run", action="store_true")

    scan = sub.add_parser("scan-alerts", help="Backfill/refresh alerts from completed verified CSV files.")
    scan.add_argument("--db", default="common/results/tier_ops/queue.sqlite")
    scan.add_argument("--out-dir", default="common/results/tier_ops")
    scan.add_argument("--alert-linear-score", type=int, default=32)
    scan.add_argument("--alert-log", default="")
    scan.add_argument("--no-alert-log", action="store_true")

    stop = sub.add_parser("request-stop", help="PAUSE: stop leasing new batches and signal active workers to stop after their current key.")
    stop.add_argument("--db", default="common/results/tier_ops/queue.sqlite")

    clear_stop = sub.add_parser("clear-stop", help="RESUME: clear a pending pause/stop request so a subsequent run continues.")
    clear_stop.add_argument("--db", default="common/results/tier_ops/queue.sqlite")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.command == "init":
        return command_init(args)
    if args.command == "status":
        return command_status(args)
    if args.command == "monitor":
        return command_monitor(args)
    if args.command == "operate":
        return command_operate(args)
    if args.command == "run":
        return command_run(args)
    if args.command == "reset-running":
        return command_reset_running(args)
    if args.command == "scan-alerts":
        return command_scan_alerts(args)
    if args.command == "request-stop":
        return command_request_stop(args)
    if args.command == "clear-stop":
        return command_clear_stop(args)
    raise SystemExit(f"unknown command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
