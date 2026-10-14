"""Validated report inputs, derived-data caching, and atomic publication."""

import csv
import hashlib
import json
import logging
import math
import os
import tempfile
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path

logger = logging.getLogger(__name__)
_diagnostics = ContextVar("report_diagnostics", default=None)
_cache_directory = ContextVar("report_cache_directory", default=None)
_pending_pages = ContextVar("report_pending_pages", default=None)


def report_warning(message, diagnostics=None):
    logger.warning("%s", message)
    target = diagnostics if diagnostics is not None else _diagnostics.get()
    if target is not None and message not in target:
        target.append(message)


@contextmanager
def report_context(cache_directory):
    diagnostics = []
    pages = {}
    tokens = (
        _diagnostics.set(diagnostics),
        _cache_directory.set(Path(cache_directory)),
        _pending_pages.set(pages),
    )
    try:
        yield diagnostics, pages
    finally:
        _pending_pages.reset(tokens[2])
        _cache_directory.reset(tokens[1])
        _diagnostics.reset(tokens[0])


def atomic_write_text(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent,
            prefix="." + path.name + "-", suffix=".tmp", delete=False,
        ) as stream:
            temporary_path = Path(stream.name)
            stream.write(text)
            stream.flush()
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def stage_page(path, text):
    pages = _pending_pages.get()
    if pages is None:
        atomic_write_text(path, text)
    else:
        pages[Path(path)] = text


def _eligible_record(epoch, cycle, run_started_at, start_cycle):
    return not (
        start_cycle > 0 and run_started_at is not None
        and isinstance(cycle, int) and cycle >= start_cycle
        and epoch < run_started_at
    )


def _read_csv(path, required, parse, run_started_at, start_cycle, diagnostics):
    path = Path(path)
    if not path.exists():
        return []
    records = []
    invalid_count = 0
    try:
        with path.open(newline="", encoding="utf-8", errors="replace") as stream:
            reader = csv.DictReader(stream)
            if not reader.fieldnames or not set(required).issubset(reader.fieldnames):
                report_warning("Missing required CSV columns in {}.".format(path), diagnostics)
                return []
            for row in reader:
                try:
                    if any(row.get(key) is None for key in required) or None in row:
                        raise ValueError("incomplete or oversized row")
                    record = parse(row)
                except (KeyError, TypeError, ValueError, OverflowError):
                    invalid_count += 1
                    continue
                if _eligible_record(
                    record["epoch"], record["cycle"], run_started_at, start_cycle
                ):
                    records.append(record)
    except (OSError, csv.Error) as exc:
        report_warning("Could not read {}: {}".format(path, exc), diagnostics)
    if invalid_count:
        report_warning(
            "Skipped {} invalid or incomplete CSV row(s) in {}.".format(invalid_count, path),
            diagnostics,
        )
    return sorted(records, key=lambda record: record["epoch"])


def _finite_float(value):
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("non-finite value")
    return result


def _cycle_value(value):
    value = value.strip()
    if value.casefold() in {"", "none"}:
        return None
    try:
        return int(value)
    except ValueError:
        return value


def read_resource_samples(path, run_started_at=None, start_cycle=0, diagnostics=None):
    def parse(row):
        record = {
            "timestamp": row.get("timestamp") or "",
            "epoch": _finite_float(row["epoch"]),
            "cycle": _cycle_value(row["cycle"]),
            "tree_cpu_pct": _finite_float(row["tree_cpu_pct"]),
            "tree_rss_gib": _finite_float(row["tree_rss_gib"]),
        }
        if record["tree_cpu_pct"] < 0 or record["tree_rss_gib"] < 0:
            raise ValueError("negative resource usage")
        for key in ("sys_cpu_pct", "sys_ram_used_gib", "sys_ram_total_gib"):
            value = row.get(key)
            record[key] = None if value in (None, "") else _finite_float(value)
            if record[key] is not None and record[key] < 0:
                raise ValueError("negative system usage")
        return record
    return _read_csv(
        path, ("epoch", "cycle", "tree_cpu_pct", "tree_rss_gib"), parse,
        run_started_at, start_cycle, diagnostics,
    )


def read_phase_events(path, run_started_at=None, start_cycle=0, diagnostics=None):
    from .resource_chart import RESOURCE_PHASE_STYLES

    def parse(row):
        phase = row["phase"].strip().lower()
        event = row["event"].strip().lower()
        if phase not in RESOURCE_PHASE_STYLES or event not in {"start", "end"}:
            raise ValueError("unrecognized phase event")
        return {
            "timestamp": row.get("timestamp") or "",
            "epoch": _finite_float(row["epoch"]),
            "cycle": _cycle_value(row["cycle"]),
            "phase": phase,
            "event": event,
        }
    return _read_csv(
        path, ("epoch", "cycle", "phase", "event"), parse,
        run_started_at, start_cycle, diagnostics,
    )


class DerivedDataCache:
    def __init__(self, path, computation):
        directory = _cache_directory.get()
        self.path = Path(path)
        self.cache_path = None
        self.identity = None
        if directory is None:
            return
        try:
            stat = self.path.stat()
        except OSError as exc:
            report_warning("Could not stat cache input {}: {}".format(path, exc))
            return
        self.identity = [
            computation, str(self.path.resolve()), stat.st_dev, stat.st_ino,
            stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns,
        ]
        key = hashlib.sha256(
            (computation + "\0" + str(self.path.resolve())).encode("utf-8")
        ).hexdigest()
        self.cache_path = directory / (key + ".json")

    def load(self):
        if self.cache_path is None or not self.cache_path.exists():
            return None
        try:
            saved = json.loads(self.cache_path.read_text(encoding="utf-8"))
            if not isinstance(saved, dict):
                raise ValueError("invalid cache object")
            if saved.get("identity") == self.identity:
                payload = saved.get("data")
                if not isinstance(payload, dict):
                    raise ValueError("invalid cache payload")
                return payload
        except (OSError, ValueError) as exc:
            report_warning("Ignoring unreadable derived-data cache {}: {}".format(self.cache_path, exc))
        return None

    def save(self, data):
        if self.cache_path is None:
            return
        try:
            stat = self.path.stat()
            if [
                stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns
            ] != self.identity[2:]:
                report_warning("Input changed during report extraction: {}.".format(self.path))
                return
            atomic_write_text(
                self.cache_path,
                json.dumps({"identity": self.identity, "data": data}, allow_nan=False),
            )
        except (OSError, ValueError) as exc:
            report_warning("Could not cache report data for {}: {}".format(self.path, exc))
