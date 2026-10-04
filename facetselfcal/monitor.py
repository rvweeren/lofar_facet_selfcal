"""Background resource monitor for process-tree CPU and memory usage."""

import csv
import html
import logging
import os
import tempfile
import threading
import time
from collections import deque
from datetime import datetime
from pathlib import Path

from .resource_chart import (
    RESOURCE_PHASE_STYLES,
    generate_resource_svg,
    phase_intervals_from_events,
)

logger = logging.getLogger(__name__)

try:
    import psutil
    _PSUTIL_AVAILABLE = True
except ImportError:
    psutil = None
    _PSUTIL_AVAILABLE = False


class ResourceMonitor:
    """Monitors CPU and RAM usage of the current process and all its subprocesses.

    Samples metrics periodically in a daemon thread and writes them to a CSV file
    under the specified log directory.

    Parameters
    ----------
    log_dir : str or pathlib.Path, optional
        Directory where resource usage logs are saved (default 'logs').
    filename : str, optional
        CSV filename (default 'resource_usage.csv').
    interval : float, optional
        Sampling interval in seconds (default 15.0).
    start_cycle : int, optional
        Starting self-calibration cycle. If 0, an existing log is archived;
        if > 0, new records are appended.
    """

    CSV_HEADER = [
        "timestamp",
        "epoch",
        "cycle",
        "tree_cpu_pct",
        "tree_rss_gib",
        "sys_cpu_pct",
        "sys_ram_used_gib",
        "sys_ram_total_gib",
    ]
    PHASE_CSV_HEADER = ["timestamp", "epoch", "cycle", "phase", "event"]

    def __init__(self, log_dir="logs", filename="resource_usage.csv", interval=15.0, start_cycle=0):
        self.log_dir = Path(log_dir)
        self.filename = filename
        self.interval = max(1.0, float(interval))
        self.start_cycle = int(start_cycle)

        self._cycle = start_cycle
        self._cycle_started = False
        self._cycle_lock = threading.Lock()
        self._phase = None
        self._phase_cycle = None
        self._phase_log_path = self.log_dir / "resource_phases.csv"
        self._phase_file = None
        self._phase_csv_writer = None
        self._phase_events = deque(maxlen=1000)
        self._stop_event = threading.Event()
        self._thread = None
        self._file = None
        self._csv_writer = None
        self._last_tree_cpu_time = None
        self._last_sample_time = None
        self._parent_process = None
        self._latest_sample = None
        self._resource_samples = deque(maxlen=300)
        self._live_page_path = self.log_dir.parent / "html_overview" / "resource-live.html"
        self._live_chart_path = self.log_dir.parent / "html_overview" / "resource-chart-live.html"
        self._live_page_started = False
        self._live_message = None

    def _archive_existing_log(self, log_path):
        """Archive existing log when start_cycle == 0, matching selfcal.log behavior."""
        if not log_path.exists():
            return
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        archive_path = log_path.with_name(f"{log_path.stem}_{timestamp}{log_path.suffix}")
        archive_index = 1
        while archive_path.exists():
            archive_path = log_path.with_name(
                f"{log_path.stem}_{timestamp}_{archive_index}{log_path.suffix}"
            )
            archive_index += 1
        try:
            log_path.rename(archive_path)
            logger.info("Archived previous resource log to %s", archive_path)
        except OSError as exc:
            logger.warning("Could not archive previous resource log %s: %s", log_path, exc)

    def _setup_file(self):
        """Prepare the CSV log file and writer."""
        self.log_dir.mkdir(parents=True, exist_ok=True)
        log_path = self.log_dir / self.filename

        if self.start_cycle == 0 and log_path.exists():
            self._archive_existing_log(log_path)

        file_exists = log_path.is_file() and log_path.stat().st_size > 0
        mode = "a" if (file_exists and self.start_cycle > 0) else "w"

        self._file = open(log_path, mode=mode, newline="", encoding="utf-8")
        self._csv_writer = csv.writer(self._file)
        if mode == "w" or not file_exists:
            self._csv_writer.writerow(self.CSV_HEADER)
            self._file.flush()

    def _setup_phase_file(self):
        """Prepare the phase-event log, archiving it for a fresh run."""
        self.log_dir.mkdir(parents=True, exist_ok=True)
        if self.start_cycle == 0 and self._phase_log_path.exists():
            self._archive_existing_log(self._phase_log_path)

        file_exists = self._phase_log_path.is_file() and self._phase_log_path.stat().st_size > 0
        mode = "a" if (file_exists and self.start_cycle > 0) else "w"
        self._phase_file = open(
            self._phase_log_path, mode=mode, newline="", encoding="utf-8"
        )
        self._phase_csv_writer = csv.writer(self._phase_file)
        if mode == "w" or not file_exists:
            self._phase_csv_writer.writerow(self.PHASE_CSV_HEADER)
            self._phase_file.flush()

    def set_cycle(self, cycle):
        """Update the active self-calibration cycle tag.

        Parameters
        ----------
        cycle : int or str
            Active cycle index or label.
        """
        with self._cycle_lock:
            self._cycle = cycle
            self._cycle_started = True

    def _record_phase_event(self, epoch, cycle, phase, event):
        timestamp = datetime.fromtimestamp(epoch).strftime("%Y-%m-%d %H:%M:%S")
        record = {
            "timestamp": timestamp,
            "epoch": epoch,
            "cycle": cycle,
            "phase": phase,
            "event": event,
        }
        self._phase_events.append(record)
        if self._phase_csv_writer is not None:
            try:
                self._phase_csv_writer.writerow(
                    [
                        timestamp,
                        f"{epoch:.2f}",
                        "" if cycle is None else str(cycle),
                        phase,
                        event,
                    ]
                )
                self._phase_file.flush()
            except Exception as exc:
                logger.warning("Failed writing resource phase event: %s", exc)

    def record_phase_intervals(self, phases, start_epoch, end_epoch):
        """Record overlapping phases that share a command's start and end times."""
        normalized_phases = list(
            dict.fromkeys(str(phase).strip().lower() for phase in phases)
        )
        invalid_phases = [
            phase for phase in normalized_phases if phase not in RESOURCE_PHASE_STYLES
        ]
        if invalid_phases:
            raise ValueError(
                "phases must be recognized resource phases; got {}".format(
                    ", ".join(invalid_phases)
                )
            )
        if not normalized_phases:
            return

        start_epoch = float(start_epoch)
        end_epoch = float(end_epoch)
        if end_epoch < start_epoch:
            raise ValueError("end_epoch must not precede start_epoch")

        with self._cycle_lock:
            cycle = self._cycle if self._cycle_started else None
            for phase in normalized_phases:
                self._record_phase_event(start_epoch, cycle, phase, "start")
            for phase in normalized_phases:
                self._record_phase_event(end_epoch, cycle, phase, "end")

        self._write_live_page()

    def set_phase(self, phase):
        """Record a workflow phase transition for the active cycle."""
        if phase is not None:
            phase = str(phase).strip().lower()
            if phase not in RESOURCE_PHASE_STYLES:
                raise ValueError(
                    "phase must be a recognized resource phase or None"
                )

        now = time.time()
        with self._cycle_lock:
            previous_phase = self._phase
            if phase == self._phase:
                return previous_phase

            if self._phase is not None:
                self._record_phase_event(
                    now, self._phase_cycle, self._phase, "end"
                )

            self._phase = phase
            if phase is None:
                self._phase_cycle = None
            else:
                self._phase_cycle = self._cycle
                self._record_phase_event(
                    now, self._phase_cycle, phase, "start"
                )

        self._write_live_page()
        return previous_phase

    def _write_live_page(self, refresh=True, message=None):
        if message is not None:
            self._live_message = message

        with self._cycle_lock:
            latest_sample = self._latest_sample
            resource_samples = list(self._resource_samples)
            phase_events = list(self._phase_events)
            current_phase = self._phase

        refresh_meta = ""
        if refresh:
            refresh_seconds = max(1, int(round(self.interval)))
            refresh_meta = '<meta http-equiv="refresh" content="{}">'.format(
                refresh_seconds
            )

        if latest_sample is None:
            live_content = '<p class="resource-live-message">{}</p>'.format(
                html.escape(self._live_message or "Waiting for the first resource sample.")
            )
        else:
            sample = latest_sample
            cards = (
                ("Process CPU", "{:.0f}%".format(sample["tree_cpu_pct"])),
                ("Process RAM", "{:.2f} GiB".format(sample["tree_rss_gib"])),
                ("System CPU", "{:.0f}%".format(sample["sys_cpu_pct"])),
                (
                    "System RAM used / total",
                    "{:.1f} / {:.1f} GiB".format(
                        sample["sys_ram_used_gib"], sample["sys_ram_total_gib"]
                    ),
                ),
            )
            live_content = '<div class="resource-live-grid">{}</div>'.format(
                "".join(
                    '<div class="resource-live-card"><span>{}</span><strong>{}</strong></div>'.format(
                        html.escape(label), html.escape(value)
                    )
                    for label, value in cards
                )
            )
            phase_label = RESOURCE_PHASE_STYLES.get(current_phase, (None, None))[0]
            phase_text = " - {}".format(html.escape(phase_label)) if phase_label else ""
            live_content += '<p class="resource-live-meta">Cycle {}{} - Updated {}</p>'.format(
                html.escape(str(sample["cycle"])),
                phase_text,
                html.escape(sample["timestamp"]),
            )
            self._live_message = None

        if latest_sample is None:
            chart_content = '<p class="resource-live-message">{}</p>'.format(
                html.escape(self._live_message or "Waiting for the first resource sample.")
            )
        else:
            sample = latest_sample
            phase_intervals = phase_intervals_from_events(
                phase_events,
                end_epoch=max(sample["epoch"], time.time()),
            )
            chart_legend = (
                '<div class="resource-live-chart-legend">'
                '<span class="resource-live-legend-item"><span class="resource-live-swatch" style="background:#0d9488;"></span>Process Tree CPU (% of one core)</span>'
                '<span class="resource-live-legend-item"><span class="resource-live-swatch" style="background:#d97706;"></span>Process Tree RAM (GiB)</span>'
                '<span class="resource-live-legend-item"><span class="resource-live-swatch resource-live-swatch-dashed"></span>Cycle transition</span>'
                + '</div>'
            )
            chart_content = (
                '<div class="resource-live-chart">{}<div class="resource-live-chart-plot">{}</div></div>'.format(
                    chart_legend,
                    generate_resource_svg(resource_samples, phase_intervals),
                )
            )
            phase_label = RESOURCE_PHASE_STYLES.get(current_phase, (None, None))[0]
            phase_text = " - {}".format(html.escape(phase_label)) if phase_label else ""
            chart_content += '<p class="resource-live-meta">Cycle {}{} - Updated {}</p>'.format(
                html.escape(str(sample["cycle"])),
                phase_text,
                html.escape(sample["timestamp"]),
            )

        refresh_meta = ""
        if refresh:
            refresh_seconds = max(1, int(round(self.interval)))
            refresh_meta = '<meta http-equiv="refresh" content="{}">'.format(
                refresh_seconds
            )

        def make_page(title, content, fit_to_parent=False):
            fit_script = ""
            if fit_to_parent:
                fit_script = """<script>
(function () {
    var reportHeight = function () {
        var main = document.querySelector("main");
        if (!main) return;
        var height = Math.ceil(main.getBoundingClientRect().height + 24);
        window.parent.postMessage(
            { type: "facetselfcal-resource-chart-size", height: height },
            "*"
        );
    };
    window.addEventListener("message", function (event) {
        if (
            event.source === window.parent &&
            event.data &&
            event.data.type === "facetselfcal-resource-chart-size-request"
        ) {
            reportHeight();
        }
    });
    window.addEventListener("load", reportHeight);
    window.addEventListener("resize", reportHeight);
    if (document.fonts && document.fonts.ready) {
        document.fonts.ready.then(reportHeight);
    }
    window.requestAnimationFrame(reportHeight);
})();
</script>"""
            return "\n".join(
                (
                "<!doctype html>",
                '<html lang="en">',
                "<head>",
                '<meta charset="utf-8">',
                '<meta name="viewport" content="width=device-width, initial-scale=1">',
                refresh_meta,
                "<title>{}</title>".format(html.escape(title)),
                "<style>",
                ":root { --line: #e2e8f0; --teal-dark: #0f766e; --amber: #b45309; --muted: #64748b; --ink-secondary: #334155; }",
                "* { box-sizing: border-box; }",
                "body { margin: 0; padding: 10px; color: #0f172a; background: #f8fafc; font: 13px/1.4 system-ui, sans-serif; }",
                ".resource-live-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(145px, 1fr)); gap: 8px; }",
                ".resource-live-card { min-width: 0; padding: 8px 10px; border: 1px solid #e2e8f0; border-top: 2px solid #0d9488; border-radius: 4px; background: #fff; }",
                ".resource-live-card span { display: block; color: #64748b; font-size: 10px; font-weight: 700; text-transform: uppercase; }",
                ".resource-live-card strong { display: block; margin-top: 3px; color: #0f172a; font-size: 18px; line-height: 1.2; overflow-wrap: anywhere; }",
                ".resource-live-chart { width: 100%; }",
                ".resource-live-chart-legend { display: flex; flex-wrap: wrap; gap: 12px; margin: 0 0 8px; color: #334155; font-size: 12px; font-weight: 600; }",
                ".resource-live-legend-item { display: inline-flex; align-items: center; gap: 5px; white-space: nowrap; }",
                ".resource-live-swatch { display: inline-block; width: 12px; height: 3px; border-radius: 2px; }",
                ".resource-live-swatch-dashed { height: 0; border-top: 2px dashed #94a3b8; border-radius: 0; }",
                ".resource-live-chart-plot { width: 100%; overflow: hidden; }",
                ".resource-live-chart-plot svg { display: block; max-width: 100%; }",
                ".resource-live-meta, .resource-live-message { margin: 8px 0 0; color: #64748b; font-size: 11px; }",
                ".resource-live-message { padding: 12px; border: 1px dashed #cbd5e1; border-radius: 4px; background: #fff; }",
                "</style>",
                "</head>",
                "<body>",
                "<main>{}</main>".format(content),
                fit_script,
                "</body>",
                "</html>",
            )
            )

        live_page = make_page("Live resource usage", live_content)
        chart_page = make_page(
            "Live resource usage chart", chart_content, fit_to_parent=True
        )
        live_written = self._write_live_snapshot(self._live_page_path, live_page)
        chart_written = self._write_live_snapshot(self._live_chart_path, chart_page)
        return live_written and chart_written

    def _write_live_snapshot(self, page_path, page):
        temporary_path = None
        try:
            page_path.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=page_path.parent,
                prefix=".resource-live-",
                suffix=".tmp",
                delete=False,
            ) as stream:
                stream.write(page)
                temporary_path = Path(stream.name)
            os.replace(temporary_path, page_path)
            self._live_page_started = True
            return True
        except OSError as exc:
            logger.warning("Could not update live resource page %s: %s", page_path, exc)
            return False
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)

    def _get_tree_cpu_time_and_rss(self):
        """Calculate total cumulative CPU time and resident memory of the process tree."""
        if not _PSUTIL_AVAILABLE or self._parent_process is None:
            return 0.0, 0.0

        total_cpu_time = 0.0
        total_rss = 0.0

        # Parent process accounting
        try:
            pt = self._parent_process.cpu_times()
            total_cpu_time += pt.user + pt.system + getattr(pt, "children_user", 0.0) + getattr(pt, "children_system", 0.0)
            total_rss += self._parent_process.memory_info().rss
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            return 0.0, 0.0

        # Active children accounting
        try:
            children = self._parent_process.children(recursive=True)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            children = []

        for child in children:
            try:
                ct = child.cpu_times()
                total_cpu_time += ct.user + ct.system
                total_rss += child.memory_info().rss
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass

        return total_cpu_time, total_rss

    def _sample(self):
        """Take a single resource measurement and append to CSV."""
        if not _PSUTIL_AVAILABLE:
            return

        now = time.time()
        timestamp_str = datetime.fromtimestamp(now).strftime("%Y-%m-%d %H:%M:%S")

        with self._cycle_lock:
            current_cycle = self._cycle
            current_phase = self._phase

        total_cpu_time, total_rss = self._get_tree_cpu_time_and_rss()

        tree_cpu_pct = 0.0
        if self._last_sample_time is not None and self._last_tree_cpu_time is not None:
            delta_time = now - self._last_sample_time
            delta_cpu = total_cpu_time - self._last_tree_cpu_time
            if delta_time > 0:
                tree_cpu_pct = max(0.0, (delta_cpu / delta_time) * 100.0)

        self._last_sample_time = now
        self._last_tree_cpu_time = total_cpu_time

        tree_rss_gib = total_rss / (1024.0 ** 3)

        # System-level metrics
        sys_cpu_pct = 0.0
        sys_ram_used_gib = 0.0
        sys_ram_total_gib = 0.0
        try:
            sys_cpu_pct = psutil.cpu_percent(interval=None)
            mem = psutil.virtual_memory()
            sys_ram_used_gib = (mem.total - mem.available) / (1024.0 ** 3)
            sys_ram_total_gib = mem.total / (1024.0 ** 3)
        except Exception:
            pass

        row = [
            timestamp_str,
            f"{now:.2f}",
            str(current_cycle),
            f"{tree_cpu_pct:.1f}",
            f"{tree_rss_gib:.2f}",
            f"{sys_cpu_pct:.1f}",
            f"{sys_ram_used_gib:.2f}",
            f"{sys_ram_total_gib:.2f}",
        ]

        if self._csv_writer is not None:
            try:
                self._csv_writer.writerow(row)
                self._file.flush()
            except Exception as exc:
                logger.warning("Failed writing resource sample: %s", exc)

        latest_sample = {
            "timestamp": timestamp_str,
            "epoch": now,
            "cycle": current_cycle,
            "phase": current_phase,
            "tree_cpu_pct": tree_cpu_pct,
            "tree_rss_gib": tree_rss_gib,
            "sys_cpu_pct": sys_cpu_pct,
            "sys_ram_used_gib": sys_ram_used_gib,
            "sys_ram_total_gib": sys_ram_total_gib,
        }
        with self._cycle_lock:
            self._latest_sample = latest_sample
            self._resource_samples.append(latest_sample)
        self._write_live_page()

    def _run(self):
        """Worker loop executed in background thread."""
        try:
            self._parent_process = psutil.Process()
            if _PSUTIL_AVAILABLE:
                # Prime system cpu_percent
                try:
                    psutil.cpu_percent(interval=None)
                except Exception:
                    pass
        except Exception as exc:
            logger.warning("ResourceMonitor initialization error: %s", exc)
            return

        try:
            self._sample()
        except Exception as exc:
            logger.warning("ResourceMonitor initial sample error: %s", exc)

        while not self._stop_event.is_set():
            if self._stop_event.wait(self.interval):
                break
            try:
                self._sample()
            except Exception as exc:
                logger.warning("ResourceMonitor sample error: %s", exc)

    def start(self):
        """Start the background monitoring thread."""
        if not _PSUTIL_AVAILABLE:
            logger.warning("psutil is not available; resource monitoring disabled.")
            self._write_live_page(refresh=False, message="Resource monitoring is unavailable.")
            return

        try:
            self._setup_file()
        except Exception as exc:
            logger.warning("Could not setup resource usage log: %s", exc)
            self._write_live_page(refresh=False, message="Resource monitoring could not start.")
            return

        try:
            self._setup_phase_file()
        except Exception as exc:
            logger.warning("Could not setup resource phase log: %s", exc)

        self._write_live_page(refresh=True)
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, name="ResourceMonitor", daemon=True)
        self._thread.start()
        logger.info("ResourceMonitor started (interval=%.1fs, log=%s)", self.interval, self.log_dir / self.filename)

    def stop(self):
        """Stop background monitoring and flush remaining data."""
        self.set_phase(None)
        if self._thread is None or not self._thread.is_alive():
            if self._file is not None and not self._file.closed:
                try:
                    self._file.close()
                except Exception:
                    pass
            if self._phase_file is not None and not self._phase_file.closed:
                try:
                    self._phase_file.close()
                except Exception:
                    pass
            if self._live_page_started:
                self._write_live_page(refresh=False, message=self._live_message)
            return

        self._stop_event.set()
        self._thread.join(timeout=5.0)

        # Take one final sample on shutdown
        try:
            self._sample()
        except Exception:
            pass

        if self._file is not None and not self._file.closed:
            try:
                self._file.close()
            except Exception:
                pass
        if self._phase_file is not None and not self._phase_file.closed:
            try:
                self._phase_file.close()
            except Exception:
                pass

        self._write_live_page(refresh=False, message=self._live_message)

        logger.info("ResourceMonitor stopped.")
