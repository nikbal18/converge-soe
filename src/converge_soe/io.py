"""Streaming result writers and checkpoint/resume.

Results hit disk as they are produced, never at the end: a crash at hour 40
of a year-long run loses at most ``flush_every`` timesteps, and ``--resume``
continues from the last checkpoint.

Per substation per scenario:
    doe.parquet bus.parquet branch.parquet viol.parquet thermal.parquet
    run.log failures.csv _checkpoint.json

* Parquet via pyarrow.parquet.ParquetWriter, one row group per flush
  (default every 200 timesteps). ~10x smaller than CSV and far faster to
  read back for analysis. ``csv_mirror=True`` additionally writes CSV with
  the same schema for eyeballing.
* The checkpoint is written atomically (tmp file + os.replace) after every
  flush and carries the thermal state, the last completed timestamp, and a
  SHA-256 fingerprint of the inputs. Resume refuses if the fingerprint
  differs — a silently stale resume is worse than no resume.
* Failed timesteps are ALWAYS recorded to failures.csv, at every verbosity.
"""

import csv
import json
import logging
import os
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

logger = logging.getLogger(__name__)

CHECKPOINT_SCHEMA_VERSION = 1
TABLE_NAMES = ("doe", "bus", "branch", "viol", "thermal")


class _StreamingTable:
    """Buffers row dicts; flushes row groups through one ParquetWriter.

    Parquet files cannot be appended in place, so on resume the existing
    file's row groups are re-ingested into the fresh writer before new rows
    are written (one read at startup — bounded by the results so far).
    """

    def __init__(self, path, csv_mirror=False, resume_existing=False):
        self.path = Path(path)
        self.csv_path = self.path.with_suffix(".csv") if csv_mirror else None
        self._writer = None
        self._csv_writer = None
        self._csv_file = None
        self._rows = []
        self._schema_names = None
        self._resume_table = None
        if resume_existing and self.path.exists():
            try:
                self._resume_table = pq.read_table(self.path)
                self._schema_names = self._resume_table.schema.names
            except Exception as e:            # unreadable partial file
                logger.warning("could not re-ingest %s for resume: %s",
                               self.path, e)

    def append(self, row):
        self._rows.append(row)

    def extend(self, rows):
        self._rows.extend(rows)

    def flush(self):
        if not self._rows:
            return
        if self._schema_names is None:
            self._schema_names = list(self._rows[0].keys())
        # tolerate missing keys in later rows
        rows = [{k: r.get(k) for k in self._schema_names} for r in self._rows]
        table = pa.Table.from_pylist(rows)
        if self._writer is None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            if self._resume_table is not None:
                self._schema = self._resume_table.schema
                self._writer = pq.ParquetWriter(self.path, self._schema)
                self._writer.write_table(self._resume_table)
                self._resume_table = None
                table = table.cast(self._schema)
            else:
                self._writer = pq.ParquetWriter(self.path, table.schema)
                self._schema = table.schema
        else:
            table = table.cast(self._schema)
        self._writer.write_table(table)

        if self.csv_path is not None:
            new = self._csv_writer is None
            if new:
                self._csv_file = open(self.csv_path, "w", newline="")
                self._csv_writer = csv.DictWriter(self._csv_file,
                                                  fieldnames=self._schema_names)
                self._csv_writer.writeheader()
            self._csv_writer.writerows(rows)
            self._csv_file.flush()
        self._rows = []

    def close(self):
        self.flush()
        if self._writer is None and self._resume_table is not None:
            pass  # nothing new was written; the existing file stands
        if self._writer is not None:
            self._writer.close()
        if self._csv_file is not None:
            self._csv_file.close()


class SubstationWriter:
    """All output streams for one (scenario, substation) run."""

    def __init__(self, outdir, substation, scenario, inputs_fingerprint="",
                 code_version="unknown", flush_every=200, csv_mirror=False,
                 resume_existing=False):
        self.outdir = Path(outdir)
        self.outdir.mkdir(parents=True, exist_ok=True)
        self.substation = substation
        self.scenario = scenario
        self.inputs_fingerprint = inputs_fingerprint
        self.code_version = code_version
        self.flush_every = int(flush_every)

        self.tables = {n: _StreamingTable(self.outdir / f"{n}.parquet",
                                          csv_mirror=csv_mirror,
                                          resume_existing=resume_existing)
                       for n in TABLE_NAMES}

        self._failures_path = self.outdir / "failures.csv"
        self._failures_file = None

        self.n_completed = 0
        self.n_failed = 0
        self.last_completed_timestamp = None
        self._since_flush = 0

    # -- appending ---------------------------------------------------------
    def append(self, table, rows):
        self.tables[table].extend(rows)

    def complete_timestep(self, timestamp, thermal_state):
        """Call once per successfully solved timestep."""
        self.n_completed += 1
        self.last_completed_timestamp = str(timestamp)
        self._thermal_state = thermal_state
        self._since_flush += 1
        if self._since_flush >= self.flush_every:
            self.flush()

    def record_failure(self, timestamp, reason, ipopt_status="",
                       ipopt_message=""):
        """Failed timesteps are always recorded, at every verbosity."""
        self.n_failed += 1
        new = self._failures_file is None
        if new:
            self._failures_file = open(self._failures_path, "a", newline="")
            self._failures_csv = csv.writer(self._failures_file)
            if self._failures_path.stat().st_size == 0:
                self._failures_csv.writerow(
                    ["timestamp", "reason", "ipopt_status", "ipopt_message"])
        self._failures_csv.writerow([timestamp, reason, ipopt_status,
                                     ipopt_message])
        self._failures_file.flush()

    # -- checkpointing -----------------------------------------------------
    def flush(self):
        for t in self.tables.values():
            t.flush()
        self._write_checkpoint()
        self._since_flush = 0

    def _write_checkpoint(self):
        ck = {
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "substation": self.substation,
            "scenario": self.scenario,
            "last_completed_timestamp": self.last_completed_timestamp,
            "n_timesteps_completed": self.n_completed,
            "n_failed": self.n_failed,
            "thermal_state": getattr(self, "_thermal_state", {}),
            "inputs_fingerprint": self.inputs_fingerprint,
            "code_version": self.code_version,
        }
        tmp = self.outdir / "_checkpoint.json.tmp"
        tmp.write_text(json.dumps(ck, indent=1, default=str))
        os.replace(tmp, self.outdir / "_checkpoint.json")

    def close(self):
        self.flush()
        for t in self.tables.values():
            t.close()
        if self._failures_file is not None:
            self._failures_file.close()


# ---------------------------------------------------------------------------
# Resume
# ---------------------------------------------------------------------------
def read_checkpoint(outdir):
    p = Path(outdir) / "_checkpoint.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except (json.JSONDecodeError, OSError) as e:
        logger.warning("unreadable checkpoint %s: %s", p, e)
        return None


class StaleResumeError(RuntimeError):
    pass


def resume_state(outdir, inputs_fingerprint, restart=False):
    """Decide how to start a substation run.

    Returns (start_after_timestamp, thermal_state, n_completed, n_failed).
    start_after_timestamp is None for a fresh start. Raises StaleResumeError
    if a checkpoint exists but its fingerprint differs and restart is False —
    a silently stale resume is worse than no resume.
    """
    ck = read_checkpoint(outdir)
    if ck is None or restart:
        return None, {}, 0, 0
    if ck.get("inputs_fingerprint") != inputs_fingerprint:
        raise StaleResumeError(
            f"{outdir}: checkpoint exists but the inputs have changed "
            f"(checkpoint fingerprint {ck.get('inputs_fingerprint', '?')[:24]}…, "
            f"current {inputs_fingerprint[:24]}…). The partial results were "
            "produced from different inputs. Re-run with --restart to discard "
            "them and start over.")
    return (ck.get("last_completed_timestamp"), ck.get("thermal_state", {}),
            ck.get("n_timesteps_completed", 0), ck.get("n_failed", 0))


def fresh_output_dir(outdir):
    """Delete a substation's partial outputs before a restart."""
    outdir = Path(outdir)
    for n in TABLE_NAMES:
        for suffix in (".parquet", ".csv"):
            p = outdir / f"{n}{suffix}"
            if p.exists():
                p.unlink()
    for name in ("_checkpoint.json", "failures.csv", "run.log"):
        p = outdir / name
        if p.exists():
            p.unlink()


# ---------------------------------------------------------------------------
# Run-level manifest
# ---------------------------------------------------------------------------
def update_run_manifest(run_dir, substation, scenario, status, **extra):
    """Track per-substation status so a feeder-level --resume knows what is
    already complete. Atomic read-modify-write."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    p = run_dir / "_manifest.json"
    manifest = {}
    if p.exists():
        try:
            manifest = json.loads(p.read_text())
        except json.JSONDecodeError:
            manifest = {}
    manifest.setdefault(scenario, {})[substation] = {"status": status, **extra}
    tmp = run_dir / "_manifest.json.tmp"
    tmp.write_text(json.dumps(manifest, indent=1, default=str))
    os.replace(tmp, p)
    return manifest


def git_code_version(repo_root):
    """Current git sha, with '-dirty' when the tree has local changes."""
    import subprocess
    try:
        sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                             cwd=repo_root, capture_output=True, text=True,
                             timeout=10).stdout.strip()
        if not sha:
            return "unknown"
        dirty = subprocess.run(["git", "status", "--porcelain"],
                               cwd=repo_root, capture_output=True, text=True,
                               timeout=10).stdout.strip()
        return sha + ("-dirty" if dirty else "")
    except (OSError, subprocess.TimeoutExpired):
        return "unknown"
