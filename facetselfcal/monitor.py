"""Background resource monitor for process-tree CPU and memory usage."""

import csv
import html
import logging
import threading
import time
from collections import deque
from datetime import datetime
from pathlib import Path

from .report_support import atomic_write_text, read_phase_events, read_resource_samples
from .resource_chart import (
    DEFAULT_RESOURCE_SAMPLE_COUNT,
    RESOURCE_PHASE_STYLES,
    RESOURCE_RAM_COLOR,
    clip_phase_intervals_to_window,
    generate_resource_svg,
    phase_intervals_from_events,
    resource_chart_browser_js,
    resource_chart_data_json,
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
        self._run_started_at = time.time()
        self._lifecycle_lock = threading.RLock()
        self._publication_lock = threading.RLock()
        self._state = "idle"
        self._stopped = False
        self._files_initialized = False
        self._chart_published = False

        self._cycle = start_cycle
        self._cycle_started = False
        self._cycle_lock = threading.Lock()
        self._phase = None
        self._phase_cycle = None
        self._phase_log_path = self.log_dir / "resource_phases.csv"
        self._phase_file = None
        self._phase_csv_writer = None
        self._phase_events = deque()
        self._stop_event = threading.Event()
        self._thread = None
        self._file = None
        self._csv_writer = None
        self._last_tree_cpu_time = None
        self._last_sample_time = None
        self._parent_process = None
        self._latest_sample = None
        self._resource_samples = deque()
        self._live_page_path = self.log_dir.parent / "html_overview" / "resource-live.html"
        self._live_chart_path = self.log_dir.parent / "html_overview" / "resource-chart-live.html"
        self._live_page_started = False
        self._live_message = None
        diagnostics = []
        log_path = self.log_dir / self.filename
        if self.start_cycle > 0 and log_path.exists():
            self._resource_samples.extend(read_resource_samples(
                log_path, run_started_at=self._run_started_at,
                start_cycle=self.start_cycle, diagnostics=diagnostics,
            ))
        if self.start_cycle > 0 and self._phase_log_path.exists():
            self._phase_events.extend(read_phase_events(
                self._phase_log_path, run_started_at=self._run_started_at,
                start_cycle=self.start_cycle, diagnostics=diagnostics,
            ))
        if self._resource_samples:
            self._latest_sample = self._resource_samples[-1]
        if diagnostics:
            self._live_message = " ".join(diagnostics)

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
            raise

    @staticmethod
    def _terminate_csv_for_append(log_path):
        """Keep an unterminated existing row separate from appended records."""
        with log_path.open("r+b") as stream:
            stream.seek(0, 2)
            if stream.tell() == 0:
                return
            stream.seek(-1, 2)
            if stream.read(1) not in (b"\n", b"\r"):
                stream.write(b"\n")

    def _setup_file(self):
        """Prepare the CSV log file and writer."""
        self.log_dir.mkdir(parents=True, exist_ok=True)
        log_path = self.log_dir / self.filename

        if self.start_cycle == 0 and not self._files_initialized and log_path.exists():
            self._archive_existing_log(log_path)

        file_exists = log_path.is_file() and log_path.stat().st_size > 0
        mode = "a" if (file_exists and (self.start_cycle > 0 or self._files_initialized)) else "w"

        if mode == "a":
            self._terminate_csv_for_append(log_path)
        self._file = open(log_path, mode=mode, newline="", encoding="utf-8")
        self._csv_writer = csv.writer(self._file)
        if mode == "w" or not file_exists:
            self._csv_writer.writerow(self.CSV_HEADER)
            self._file.flush()

    def _setup_phase_file(self):
        """Prepare the phase-event log, archiving it for a fresh run."""
        self.log_dir.mkdir(parents=True, exist_ok=True)
        if self.start_cycle == 0 and not self._files_initialized and self._phase_log_path.exists():
            self._archive_existing_log(self._phase_log_path)

        file_exists = self._phase_log_path.is_file() and self._phase_log_path.stat().st_size > 0
        mode = "a" if (file_exists and (self.start_cycle > 0 or self._files_initialized)) else "w"
        if mode == "a":
            self._terminate_csv_for_append(self._phase_log_path)
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
                self._live_message = "Failed writing resource phase event: {}".format(exc)

    @staticmethod
    def _normalized_phase_list(phases):
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
        return normalized_phases

    def begin_phase_intervals(self, phases, start_epoch=None):
        """Open overlapping phases that share a command's start time.

        The start events are written immediately, so the chart shows the
        phases while the command runs. Pass the returned token to
        :meth:`end_phase_intervals` when the command finishes.

        Returns
        -------
        dict or None
            Token describing the opened phases, or None if nothing was opened.
        """
        normalized_phases = self._normalized_phase_list(phases)
        if not normalized_phases:
            return None
        start_epoch = time.time() if start_epoch is None else float(start_epoch)
        with self._cycle_lock:
            if self._stopped:
                return None
            cycle = self._cycle if self._cycle_started else None
            for phase in normalized_phases:
                self._record_phase_event(start_epoch, cycle, phase, "start")
        self._write_live_page(phase_only=True)
        return {"phases": normalized_phases, "cycle": cycle, "start_epoch": start_epoch}

    def end_phase_intervals(self, token, end_epoch=None):
        """Close phases opened by :meth:`begin_phase_intervals`."""
        if not token:
            return
        end_epoch = time.time() if end_epoch is None else float(end_epoch)
        end_epoch = max(end_epoch, token["start_epoch"])
        with self._cycle_lock:
            if self._stopped:
                return
            for phase in token["phases"]:
                self._record_phase_event(end_epoch, token["cycle"], phase, "end")
        self._write_live_page(phase_only=True)

    def record_phase_intervals(self, phases, start_epoch, end_epoch):
        """Record overlapping phases that share a command's start and end times."""
        if float(end_epoch) < float(start_epoch):
            raise ValueError("end_epoch must not precede start_epoch")
        token = self.begin_phase_intervals(phases, start_epoch)
        self.end_phase_intervals(token, end_epoch)

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
            if self._stopped and phase is not None:
                return previous_phase
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

        self._write_live_page(phase_only=True)
        return previous_phase

    def _write_live_page(self, refresh=True, message=None, phase_only=False):
        # Serialize snapshot capture, rendering and both replacements, not just writes.
        with self._publication_lock:
            if message is not None:
                self._live_message = message
            if self._stopped:
                refresh = False
            try:
                return self._render_live_page(refresh, phase_only)
            except Exception as exc:
                logger.warning("Could not render live resource page: %s", exc)
                self._live_message = "Could not render live resource page: {}".format(exc)
                page = '<!doctype html><html><body><p>{}</p></body></html>'.format(
                    html.escape(self._live_message)
                )
                live_written = self._write_live_snapshot(self._live_page_path, page)
                chart_written = self._write_live_snapshot(self._live_chart_path, page)
                self._chart_published = False
                return False

    def _render_live_page(self, refresh, phase_only):
        update_chart = not phase_only or not self._chart_published or not refresh
        with self._cycle_lock:
            latest_sample = self._latest_sample
            resource_samples = list(self._resource_samples) if update_chart else []
            phase_events = list(self._phase_events) if update_chart else []
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
                ("System CPU", "{:.0f}%".format(sample["sys_cpu_pct"])
                 if sample["sys_cpu_pct"] is not None else "Unavailable"),
                (
                    "System RAM used / total",
                    "{:.1f} / {:.1f} GiB".format(
                        sample["sys_ram_used_gib"], sample["sys_ram_total_gib"]
                    ) if sample["sys_ram_used_gib"] is not None
                    and sample["sys_ram_total_gib"] is not None else "Unavailable",
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
            if self._live_message:
                live_content += '<p class="resource-live-message">{}</p>'.format(
                    html.escape(self._live_message)
                )

        chart_content = ""
        if not update_chart:
            pass
        elif latest_sample is None:
            chart_content = '<p class="resource-live-message">{}</p>'.format(
                html.escape(self._live_message or "Waiting for the first resource sample.")
            )
        else:
            sample = latest_sample
            # Bars and curves come from the same snapshot: open phases end at
            # the latest sample, never past it, so both advance together.
            all_phase_intervals = clip_phase_intervals_to_window(
                phase_intervals_from_events(phase_events, end_epoch=sample["epoch"]),
                resource_samples[0]["epoch"],
                sample["epoch"],
            )
            recent_samples = resource_samples[-DEFAULT_RESOURCE_SAMPLE_COUNT:]
            phase_intervals = clip_phase_intervals_to_window(
                all_phase_intervals,
                recent_samples[0]["epoch"],
                recent_samples[-1]["epoch"],
            )
            chart_data = resource_chart_data_json(
                resource_samples, all_phase_intervals
            )
            chart_legend = (
                '<div class="resource-live-chart-legend">'
                '<span class="resource-live-legend-item"><span class="resource-live-swatch" style="background:#0d9488;"></span>Process Tree CPU (% of one core)</span>'
                f'<span class="resource-live-legend-item"><span class="resource-live-swatch" style="background:{RESOURCE_RAM_COLOR};"></span>Process Tree RAM (GiB)</span>'
                '<span class="resource-live-legend-item"><span class="resource-live-swatch resource-live-swatch-dashed"></span>Cycle transition</span>'
                + '</div>'
            )
            chart_content = (
                '<div class="resource-live-chart">{}'
                '<div class="resource-live-chart-plot" data-resource-chart-live>'
                '<div data-resource-chart-target>{}</div>'
                '<script type="application/json" data-resource-chart-data>{}</script>'
                '</div></div>'.format(
                    chart_legend,
                    generate_resource_svg(recent_samples, phase_intervals),
                    chart_data,
                )
            )
            phase_label = RESOURCE_PHASE_STYLES.get(current_phase, (None, None))[0]
            phase_text = " - {}".format(html.escape(phase_label)) if phase_label else ""
            chart_content += '<p class="resource-live-meta">Cycle {}{} - Updated {}</p>'.format(
                html.escape(str(sample["cycle"])),
                phase_text,
                html.escape(sample["timestamp"]),
            )
            if self._live_message:
                chart_content += '<p class="resource-live-message">{}</p>'.format(
                    html.escape(self._live_message)
                )

        refresh_meta = ""
        if refresh:
            refresh_seconds = max(1, int(round(self.interval)))
            refresh_meta = '<meta http-equiv="refresh" content="{}">'.format(
                refresh_seconds
            )

        def make_page(
            title, content, fit_to_parent=False, include_chart_script=False
        ):
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
            chart_script = ""
            if include_chart_script:
                chart_script = "<script>{}</script>".format(
                    resource_chart_browser_js()
                )
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
                chart_script,
                fit_script,
                "</body>",
                "</html>",
            )
            )

        live_page = make_page("Live resource usage", live_content)
        live_written = self._write_live_snapshot(self._live_page_path, live_page)
        chart_written = True
        if update_chart:
            chart_page = make_page(
                "Live resource usage chart",
                chart_content,
                fit_to_parent=True,
                include_chart_script=True,
            )
            chart_written = self._write_live_snapshot(self._live_chart_path, chart_page)
            self._chart_published = chart_written
        return live_written and chart_written

    def _write_live_snapshot(self, page_path, page):
        try:
            atomic_write_text(page_path, page)
            self._live_page_started = True
            return True
        except OSError as exc:
            logger.warning("Could not update live resource page %s: %s", page_path, exc)
            self._live_message = "Could not update live resource page: {}".format(exc)
            return False

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
        sys_cpu_pct = None
        sys_ram_used_gib = None
        sys_ram_total_gib = None
        system_errors = []
        try:
            sys_cpu_pct = psutil.cpu_percent(interval=None)
        except Exception as exc:
            logger.warning("System CPU metrics unavailable: %s", exc)
            system_errors.append("System CPU metrics unavailable: {}".format(exc))
        try:
            mem = psutil.virtual_memory()
            sys_ram_used_gib = (mem.total - mem.available) / (1024.0 ** 3)
            sys_ram_total_gib = mem.total / (1024.0 ** 3)
        except Exception as exc:
            logger.warning("System RAM metrics unavailable: %s", exc)
            system_errors.append("System RAM metrics unavailable: {}".format(exc))
        if system_errors:
            self._live_message = " ".join(system_errors)

        row = [
            timestamp_str,
            f"{now:.2f}",
            str(current_cycle),
            f"{tree_cpu_pct:.1f}",
            f"{tree_rss_gib:.2f}",
            "" if sys_cpu_pct is None else f"{sys_cpu_pct:.1f}",
            "" if sys_ram_used_gib is None else f"{sys_ram_used_gib:.2f}",
            "" if sys_ram_total_gib is None else f"{sys_ram_total_gib:.2f}",
        ]

        if self._csv_writer is not None:
            try:
                self._csv_writer.writerow(row)
                self._file.flush()
            except Exception as exc:
                logger.warning("Failed writing resource sample: %s", exc)
                self._live_message = "Failed writing resource sample: {}".format(exc)

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
                except Exception as exc:
                    logger.warning("System CPU metrics initialization unavailable: %s", exc)
                    self._live_message = "System CPU metrics initialization unavailable: {}".format(exc)
        except Exception as exc:
            self._report_error("ResourceMonitor initialization error", exc)
            return

        try:
            self._sample()
        except Exception as exc:
            self._report_error("ResourceMonitor initial sample error", exc)

        while not self._stop_event.is_set():
            if self._stop_event.wait(self.interval):
                break
            try:
                self._sample()
            except Exception as exc:
                self._report_error("ResourceMonitor sample error", exc)

    def _report_error(self, context, exc):
        logger.warning("%s: %s", context, exc)
        self._write_live_page(message="{}: {}".format(context, exc))

    def _close_files(self):
        with self._cycle_lock:
            for attribute, writer_attribute in (
                ("_file", "_csv_writer"), ("_phase_file", "_phase_csv_writer")
            ):
                stream = getattr(self, attribute)
                if stream is not None and not stream.closed:
                    try:
                        stream.close()
                    except OSError as exc:
                        logger.warning("Could not close resource log: %s", exc)
                        self._live_message = "Could not close resource log: {}".format(exc)
                setattr(self, writer_attribute, None)

    def _finish_stop(self):
        with self._lifecycle_lock:
            self._stopped = True
            with self._cycle_lock:
                if self._phase is not None:
                    self._record_phase_event(
                        time.time(), self._phase_cycle, self._phase, "end"
                    )
                    self._phase = None
                    self._phase_cycle = None
            self._close_files()
            self._state = "stopped"
            message = "Resource monitoring stopped."
            if self._live_message:
                message += " " + self._live_message
            self._write_live_page(refresh=False, message=message)

    def _worker_main(self):
        try:
            self._run()
        except Exception as exc:
            self._report_error("ResourceMonitor worker error", exc)
        finally:
            if self._parent_process is not None and self._stop_event.is_set():
                try:
                    self._sample()
                except Exception as exc:
                    self._report_error("ResourceMonitor final sample error", exc)
            self._finish_stop()

    def start(self):
        """Start once; a timed-out worker must finish before another can start."""
        with self._lifecycle_lock:
            if self._thread is not None and self._thread.is_alive():
                if self._state == "stopping":
                    logger.warning("ResourceMonitor is still stopping; cannot restart.")
                return
            if not _PSUTIL_AVAILABLE:
                logger.warning("psutil is not available; resource monitoring disabled.")
                self._stopped = True
                self._write_live_page(refresh=False, message="Resource monitoring is unavailable.")
                return

            self._stopped = False
            if self._state == "stopped":
                self._live_message = None
            self._last_sample_time = None
            self._last_tree_cpu_time = None
            self._stop_event.clear()
            try:
                self._setup_file()
                self._setup_phase_file()
                self._files_initialized = True
                self._write_live_page(refresh=True)
                self._thread = threading.Thread(
                    target=self._worker_main, name="ResourceMonitor", daemon=True
                )
                self._state = "running"
                self._thread.start()
            except Exception as exc:
                logger.warning("Resource monitoring could not start: %s", exc)
                self._stop_event.set()
                self._stopped = True
                self._state = "stopped"
                self._close_files()
                self._write_live_page(
                    refresh=False, message="Resource monitoring could not start: {}".format(exc)
                )
                return
            logger.info("ResourceMonitor started (interval=%.1fs, log=%s)", self.interval, self.log_dir / self.filename)

    def stop(self):
        """Request shutdown, leaving file cleanup to a worker that is still alive."""
        with self._lifecycle_lock:
            self._stop_event.set()
            self._stopped = True
            thread = self._thread
            if thread is None or not thread.is_alive():
                if self._state != "stopped":
                    self._finish_stop()
                return
            self._state = "stopping"
        if thread is threading.current_thread():
            return
        thread.join(timeout=5.0)
        if thread.is_alive():
            message = "Resource monitoring shutdown timed out; waiting for the worker to finish."
            logger.warning(message)
            self._live_message = message
            # Do not wait indefinitely for a render already in progress.
            if self._publication_lock.acquire(blocking=False):
                try:
                    self._write_live_page(refresh=False)
                finally:
                    self._publication_lock.release()
            return
        with self._lifecycle_lock:
            if self._thread is thread and self._state != "stopped":
                self._finish_stop()
        logger.info("ResourceMonitor stopped.")
