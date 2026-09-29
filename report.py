"""Generate a browsable, offline HTML report from a facetselfcal run directory."""

import argparse
import ast
import html
import json
import os
import re
from collections import defaultdict, deque
from datetime import datetime
from pathlib import Path
from urllib.parse import quote


_PAGE_NAMES = {
    "index.html": "Overview",
    "imaging.html": "Imaging",
    "calibration.html": "Calibration",
    "datasets.html": "Measurement sets",
    "run-details.html": "Run details",
}

_SUMMARY_KEYS = (
    "telescope",
    "DDE",
    "imager",
    "imagename",
    "start",
    "stop",
    "imsize",
    "pixelscale",
    "niter",
    "soltype_list",
    "solint_list",
    "nchan_list",
    "soltypecycles_list",
    "facetdirections",
    "forwidefield",
)

_MS_METADATA_FIELDS = (
    ("Telescope", ("Telescope",)),
    ("VLA configuration", ("VLA configuration",)),
    ("Integration time (s)", ("Integration time [s]",)),
    ("Observation duration (hr)", ("Observation duration [hr]",)),
    ("Number of channels", ("Number of channels",)),
    ("Bandwidth (MHz)", ("Bandwidth [MHz]",)),
    ("Channel width (kHz)", ("Channel width [kHz]",)),
    ("Start frequency (MHz)", ("Start frequnecy [MHz]", "Start frequency [MHz]")),
    ("End frequency (MHz)", ("End frequency [MHz]",)),
)

_CSS = r"""
:root {
  color-scheme: light;
  --paper: #f3f6f4;
  --surface: #ffffff;
  --ink: #192a28;
  --muted: #60716d;
  --line: #d5dfdb;
  --green: #176b60;
  --green-dark: #123e3a;
  --green-pale: #e1f1ed;
  --amber: #8d4f14;
  --amber-pale: #fff1dc;
  --red: #982f32;
  --red-pale: #fbe8e6;
  --code: #eff3f1;
}
* { box-sizing: border-box; }
body {
  margin: 0;
  color: var(--ink);
  background: var(--paper);
  font: 15px/1.55 Verdana, "DejaVu Sans", sans-serif;
}
a { color: var(--green); text-underline-offset: 3px; }
a:hover { color: var(--green-dark); }
.site-header { background: var(--green-dark); color: #f5fbf8; }
.masthead, nav, main, footer { width: min(1280px, calc(100% - 40px)); margin: 0 auto; }
.masthead { padding: 24px 0 21px; }
.eyebrow { margin: 0 0 8px; color: #a9d1c7; font-size: 12px; font-weight: 700; text-transform: uppercase; }
h1, h2, h3 { line-height: 1.2; }
h1 { margin: 0; font: 36px/1.12 Georgia, "DejaVu Serif", serif; }
.header-subtitle { max-width: 860px; margin: 10px 0 0; color: #d0e2dc; overflow-wrap: anywhere; }
nav { display: flex; gap: 4px; overflow-x: auto; border-top: 1px solid #43635e; }
nav a { flex: 0 0 auto; padding: 12px 14px; color: #e1efea; text-decoration: none; font-size: 13px; }
nav a[aria-current="page"] { color: #ffffff; background: #24574f; box-shadow: inset 0 -3px #9bd2c3; }
nav a:hover { color: white; background: #214b45; }
main { padding: 28px 0 58px; }
.page-intro { margin: 0 0 22px; color: var(--muted); max-width: 940px; }
.page-intro strong { color: var(--ink); }
h2 { margin: 0 0 14px; font: 25px/1.2 Georgia, "DejaVu Serif", serif; }
h3 { margin: 0 0 9px; font-size: 16px; }
section { margin: 26px 0 0; }
.section-heading { display: flex; align-items: baseline; justify-content: space-between; gap: 16px; border-bottom: 1px solid var(--line); padding-bottom: 9px; margin-bottom: 14px; }
.section-heading p { margin: 0; color: var(--muted); font-size: 13px; }
.statusline { display: flex; flex-wrap: wrap; align-items: center; gap: 12px; margin-top: 16px; }
.status { display: inline-block; border-radius: 3px; padding: 5px 10px; font-size: 12px; font-weight: 700; text-transform: uppercase; letter-spacing: .04em; }
.status-completed { color: #14594e; background: #cde9df; }
.status-running { color: #174c61; background: #d8edf2; }
.status-failed { color: #81272b; background: #f4d3d1; }
.status-interrupted, .status-stopped, .status-unknown { color: #77420e; background: #fae4bd; }
.status-detail { color: #d0e2dc; font-size: 13px; }
.metrics { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 1px; border: 1px solid var(--line); background: var(--line); margin: 22px 0 30px; }
.metric { min-width: 0; padding: 15px 17px; background: var(--surface); }
.metric-value { display: block; font: 28px/1.1 Georgia, "DejaVu Serif", serif; color: var(--green-dark); overflow-wrap: anywhere; }
.metric-label { display: block; margin-top: 6px; color: var(--muted); font-size: 12px; }
.notice { margin: 18px 0; border-left: 4px solid var(--amber); background: var(--amber-pale); padding: 12px 15px; color: #56360f; }
.notice.error { border-color: var(--red); background: var(--red-pale); color: #672326; }
.notice p { margin: 0; }
.data-table { width: 100%; border-collapse: collapse; background: var(--surface); }
.table-scroll { max-width: 100%; overflow-x: auto; }
.data-table th, .data-table td { padding: 9px 11px; border-bottom: 1px solid var(--line); text-align: left; vertical-align: top; }
.data-table th { width: 220px; color: var(--muted); font-size: 12px; font-weight: 700; }
.data-table td { overflow-wrap: anywhere; }
code, pre, .mono { font-family: "DejaVu Sans Mono", "Courier New", monospace; }
code { font-size: .92em; }
pre { max-width: 100%; margin: 0; padding: 13px 15px; overflow: auto; background: var(--code); border: 1px solid var(--line); white-space: pre-wrap; overflow-wrap: anywhere; font-size: 12px; }
.file-list { margin: 0; padding: 0; list-style: none; }
.file-list li { display: flex; justify-content: space-between; align-items: baseline; gap: 14px; padding: 8px 10px; border-bottom: 1px solid var(--line); background: var(--surface); }
.file-list li:nth-child(even) { background: #f8faf9; }
.file-size { flex: 0 0 auto; color: var(--muted); font-size: 12px; }
.search-row { display: flex; align-items: center; gap: 12px; margin: 13px 0; }
.search-row label { color: var(--muted); font-size: 13px; }
.search-row input { width: min(480px, 100%); min-height: 40px; border: 1px solid #9aada6; border-radius: 3px; padding: 8px 11px; color: var(--ink); background: white; font: inherit; }
.search-row input:focus { outline: 3px solid #a7d7ca; outline-offset: 1px; }
.gallery { display: grid; grid-template-columns: repeat(auto-fill, minmax(230px, 1fr)); gap: 12px; }
figure { min-width: 0; margin: 0; border: 1px solid var(--line); background: var(--surface); }
figure a { display: block; background: #e9efec; }
figure img { display: block; width: 100%; height: 175px; object-fit: contain; }
figcaption { padding: 9px 11px; font-size: 12px; overflow-wrap: anywhere; }
figcaption .caption-detail { display: block; color: var(--muted); margin-top: 3px; }
.blink-controls { display: flex; flex-wrap: wrap; align-items: end; gap: 10px; padding: 13px; }
.blink-controls label { display: grid; gap: 4px; min-width: 0; color: var(--muted); font-size: 12px; }
.blink-rate { min-width: 190px; }
.blink-rate input { width: 150px; vertical-align: middle; }
.blink-rate output { margin-left: 5px; color: var(--ink); }
.blink-controls button { min-height: 40px; border: 1px solid var(--green); border-radius: 3px; padding: 7px 12px; color: white; background: var(--green); font: inherit; font-weight: 700; cursor: pointer; }
.blink-controls button:hover { background: var(--green-dark); }
.blink-controls button:focus-visible, .blink-rate input:focus-visible { outline: 3px solid #a7d7ca; outline-offset: 1px; }
.blink-preview { margin: 0 13px 13px; }
.blink-preview img { height: min(70vh, 720px); object-fit: contain; }
details { margin: 11px 0; border: 1px solid var(--line); background: var(--surface); }
details > summary { cursor: pointer; padding: 11px 13px; color: var(--green-dark); font-weight: 700; }
details[open] > summary { border-bottom: 1px solid var(--line); }
details > :not(summary) { margin-left: 13px; margin-right: 13px; }
details > .data-table, details > .file-list { margin-bottom: 13px; }
.cycle-links { display: flex; flex-wrap: wrap; gap: 6px; padding: 0 13px 13px; }
.cycle-links a { display: inline-block; padding: 5px 8px; border: 1px solid var(--line); border-radius: 3px; background: #f6faf8; text-decoration: none; font-size: 12px; }
.plot-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(205px, 1fr)); gap: 10px; padding: 12px; }
.log-events { display: grid; gap: 8px; }
.log-event { border-left: 3px solid var(--amber); padding: 8px 11px; background: var(--amber-pale); font-size: 13px; overflow-wrap: anywhere; }
.log-event.error { border-color: var(--red); background: var(--red-pale); }
.log-event small { display: block; color: var(--muted); margin-bottom: 3px; }
.page-links { display: flex; flex-wrap: wrap; gap: 9px; margin: 14px 0; }
.page-links a { padding: 8px 11px; border: 1px solid var(--line); border-radius: 3px; background: white; text-decoration: none; font-size: 13px; }
.empty { padding: 16px; color: var(--muted); background: var(--surface); border: 1px dashed #aebdb7; }
[hidden] { display: none !important; }
footer { border-top: 1px solid var(--line); padding: 18px 0 26px; color: var(--muted); font-size: 12px; }
@media (max-width: 850px) { .metrics { grid-template-columns: repeat(3, minmax(0, 1fr)); } }
@media (max-width: 600px) {
  .masthead, nav, main, footer { width: min(100% - 24px, 1280px); }
  .masthead { padding-top: 18px; }
  h1 { font-size: 30px; }
  main { padding-top: 20px; }
  .metrics { grid-template-columns: repeat(2, minmax(0, 1fr)); }
  .metric { padding: 12px; }
  .metric-value { font-size: 24px; }
  .data-table th, .data-table td { padding: 7px; }
  .data-table th { width: 125px; }
  .section-heading { display: block; }
  .file-list li { display: block; }
  .file-size { display: block; margin-top: 2px; }
}
"""

_JS = r"""
document.addEventListener("DOMContentLoaded", function () {
    document.querySelectorAll("[data-blink-tool]").forEach(function (tool) {
        var frames = JSON.parse(tool.getAttribute("data-blink-images") || "[]");
        var interval = tool.querySelector("[data-blink-interval]");
        var intervalOutput = tool.querySelector("[data-blink-interval-output]");
        var toggle = tool.querySelector("[data-blink-toggle]");
        var image = tool.querySelector("[data-blink-image]");
        var caption = tool.querySelector("[data-blink-caption]");
        var timer = null;
        var frameIndex = 0;
        var render = function () {
            var frame = frames[frameIndex];
            if (!frame) return;
            image.src = frame.src;
            image.alt = frame.label;
            caption.textContent = frame.label;
        };
        var tick = function () {
            frameIndex = (frameIndex + 1) % frames.length;
            render();
        };
        var stop = function () {
            if (timer !== null) {
                window.clearInterval(timer);
                timer = null;
            }
            toggle.textContent = "Start blinking";
            toggle.setAttribute("aria-pressed", "false");
        };
        var start = function () {
            frameIndex = 0;
            render();
            timer = window.setInterval(tick, Number(interval.value));
            toggle.textContent = "Stop blinking";
            toggle.setAttribute("aria-pressed", "true");
        };

        if (frames.length < 2) return;

        interval.addEventListener("input", function () {
            intervalOutput.value = interval.value + " ms";
            intervalOutput.textContent = intervalOutput.value;
            if (timer !== null) {
                window.clearInterval(timer);
                timer = window.setInterval(tick, Number(interval.value));
            }
        });
        toggle.addEventListener("click", function () {
            if (timer === null) start();
            else stop();
        });
        render();
    });

  document.querySelectorAll("[data-filter-target]").forEach(function (input) {
    var selector = input.getAttribute("data-filter-target");
    var items = Array.from(document.querySelectorAll(selector));
    var groups = Array.from(document.querySelectorAll("details[data-filter-group]"));
    var update = function () {
      var query = input.value.trim().toLowerCase();
      items.forEach(function (item) {
        var text = item.getAttribute("data-search") || item.textContent || "";
        item.hidden = query.length > 0 && !text.toLowerCase().includes(query);
      });
      groups.forEach(function (group) {
        var visible = Array.from(group.querySelectorAll(selector)).some(function (item) {
          return !item.hidden;
        });
        group.hidden = !visible;
        if (query && visible) group.open = true;
      });
    };
    input.addEventListener("input", update);
  });
});
"""


def _escape(value):
    return html.escape(str(value), quote=True)


def _parse_config(path):
    config = {}
    if not path.is_file():
        return config
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return config
    for line in lines:
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            continue
        try:
            config[key] = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            config[key] = value
    return config


def _as_list(value):
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value]
    if isinstance(value, str) and value.strip():
        return [value]
    return []


def _display_value(value):
    if isinstance(value, (list, tuple)):
        return ", ".join(str(item) for item in value)
    if value is None:
        return "Not set"
    if isinstance(value, bool):
        return "Yes" if value else "No"
    return str(value)


def _format_size(size):
    value = float(size)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024 or unit == "TB":
            return "{:.1f} {}".format(value, unit)
        value /= 1024
    return "{:.1f} TB".format(value)


def _relative_url(path, base_dir):
    relative = os.path.relpath(str(path), str(base_dir))
    return quote(Path(relative).as_posix(), safe="/._-()[]")


def _page_shell(title, active_page, body, nested=False, subtitle="Offline processing report"):
    prefix = "../" if nested else ""
    nav = []
    for page, label in _PAGE_NAMES.items():
        current = ' aria-current="page"' if page == active_page else ""
        nav.append(
            '<a href="{}{}"{}>{}</a>'.format(
                prefix, _escape(page), current, _escape(label)
            )
        )
    return """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta name="color-scheme" content="light">
  <title>{title} | facetselfcal report</title>
  <link rel="stylesheet" href="{prefix}assets/report.css">
  <script src="{prefix}assets/report.js" defer></script>
</head>
<body>
<header class="site-header">
  <div class="masthead">
    <p class="eyebrow">facetselfcal / processing report</p>
    <h1>{title}</h1>
    <p class="header-subtitle">{subtitle}</p>
  </div>
  <nav aria-label="Report pages">{nav}</nav>
</header>
<main>{body}</main>
<footer>This report uses local HTML, CSS, JavaScript, and run products only. Keep it alongside the run directory to browse it offline.</footer>
</body>
</html>
""".format(
        title=_escape(title),
        prefix=prefix,
        subtitle=_escape(subtitle),
        nav="\n    ".join(nav),
        body=body,
    )


def _section(title, content, note=None):
    note_html = "<p>{}</p>".format(_escape(note)) if note else ""
    return (
        '<section><div class="section-heading"><h2>{}</h2>{}</div>{}</section>'.format(
            _escape(title), note_html, content
        )
    )


def _filter_input(target, label="Filter"):
    return (
        '<div class="search-row"><label for="report-filter">{}</label>'
        '<input id="report-filter" type="search" data-filter-target="{}" '
        'placeholder="Type to filter" autocomplete="off"></div>'
    ).format(_escape(label), _escape(target))


def _config_value_html(value):
    if isinstance(value, (list, tuple)) and len(value) > 8:
        items = "".join("<li><code>{}</code></li>".format(_escape(item)) for item in value)
        return "{} entries <details><summary>Show values</summary><ol>{}</ol></details>".format(
            len(value), items
        )
    return "<code>{}</code>".format(_escape(_display_value(value)))


def _config_table(config, keys=None, filterable=False):
    if keys is None:
        keys = sorted(config)
    rows = []
    for key in keys:
        if key not in config:
            continue
        value = config[key]
        search_attr = ""
        row_class = ""
        if filterable:
            row_class = ' class="config-row"'
            search_attr = ' data-search="{}"'.format(_escape(key + " " + _display_value(value)))
        rows.append(
            "<tr{}{}><th scope=\"row\">{}</th><td>{}</td></tr>".format(
                row_class, search_attr, _escape(key), _config_value_html(value)
            )
        )
    if not rows:
        return '<p class="empty">No saved configuration was found.</p>'
    return '<table class="data-table"><tbody>{}</tbody></table>'.format("".join(rows))


def _artifact_is_current(path, run_started_at):
    if run_started_at is None:
        return True
    try:
        return path.stat().st_mtime >= run_started_at
    except OSError:
        return True


def _scan_artifacts(run_root, run_started_at=None, current_run_cycles=None):
    overview_dir = run_root / "plots"
    overview_plots = sorted(
        (
            path for path in overview_dir.glob("*.png")
            if path.is_file() and _artifact_is_current(path, run_started_at)
        ),
        key=lambda path: path.name.lower(),
    ) if overview_dir.is_dir() else []

    calibration_sets = []
    cycle_names = set(current_run_cycles or ())
    for directory in sorted(run_root.glob("solution_plots_*"), key=lambda path: path.name.lower()):
        if not directory.is_dir():
            continue
        cycles = defaultdict(list)
        for path in directory.rglob("*"):
            if (
                not path.is_file()
                or path.suffix.lower() != ".png"
                or not _artifact_is_current(path, run_started_at)
            ):
                continue
            match = re.search(r"selfcalcycle(\d+)", path.name, re.IGNORECASE)
            cycle = match.group(1) if match else "other"
            if (
                cycle != "other"
                and current_run_cycles is not None
                and cycle not in current_run_cycles
            ):
                continue
            cycles[cycle].append(path)
            if cycle != "other":
                cycle_names.add(cycle)
        for paths in cycles.values():
            paths.sort(key=lambda path: path.name.lower())
        if cycles:
            calibration_sets.append((directory, dict(cycles)))

    fits_dir = run_root / "fits_images"
    fits_files = sorted(
        (
            path for path in fits_dir.rglob("*")
            if path.is_file()
            and (path.name.lower().endswith(".fits") or path.name.lower().endswith(".fits.gz"))
            and _artifact_is_current(path, run_started_at)
        ),
        key=lambda path: path.name.lower(),
    ) if fits_dir.is_dir() else []

    solutions_dir = run_root / "h5_solutions"
    solution_files = []
    if solutions_dir.is_dir():
        for path in solutions_dir.rglob("*.h5"):
            if not path.is_file() or not _artifact_is_current(path, run_started_at):
                continue
            match = re.search(r"selfcalcycle(\d+)", path.name, re.IGNORECASE)
            if match:
                cycle = match.group(1)
                if current_run_cycles is not None and cycle not in current_run_cycles:
                    continue
                cycle_names.add(cycle)
            solution_files.append(path)
    solution_files.sort(key=lambda path: path.name.lower())

    ms_directories = sorted(
        (
            path for path in run_root.iterdir()
            if path.is_dir()
            and path.name.lower().endswith(".ms")
            and not path.name.lower().startswith("solution_plots_")
        ),
        key=lambda path: path.name.lower(),
    )

    return {
        "overview_plots": overview_plots,
        "calibration_sets": calibration_sets,
        "fits_files": fits_files,
        "solution_files": solution_files,
        "ms_directories": ms_directories,
        "cycles": sorted(cycle_names, key=lambda value: int(value) if value.isdigit() else 10**9),
    }


def _parse_log_line(line):
    match = re.match(
        r"^(DEBUG|INFO|WARNING|ERROR|CRITICAL):"
        r"(?:(\d{2}/\d{2}/\d{4} \d{2}:\d{2}:\d{2})\s+----\s*)?(.*)$",
        line,
    )
    if match:
        return match.group(1), match.group(2) or "", match.group(3)
    match = re.match(
        r"^(\d{4}-\d{2}-\d{2}[^ ]*(?: [^ ]+)?)\s+-\s+.*?\s+-\s+"
        r"(DEBUG|INFO|WARNING|ERROR|CRITICAL)\s+-\s+(.*)$",
        line,
    )
    if match:
        return match.group(2), match.group(1), match.group(3)
    return "", "", line


def _log_timestamp_epoch(timestamp):
    if not timestamp:
        return None
    try:
        parsed = datetime.strptime(timestamp, "%m/%d/%Y %H:%M:%S")
    except ValueError:
        try:
            parsed = datetime.fromisoformat(timestamp)
        except ValueError:
            return None
    return parsed.timestamp()


def _current_run_context(run_root):
    path = run_root / "logs" / "selfcal.log"
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return None, None

    version_markers = []
    invocation_markers = []
    timestamp_by_line = {}
    invocation_pattern = re.compile(
        r"facetselfcal(?:\.py)?\b.*(?:\s-i(?:\s|=)|--config(?:\s|=))",
        re.IGNORECASE,
    )
    for index, line in enumerate(lines):
        _, timestamp, message = _parse_log_line(line)
        timestamp_by_line[index] = _log_timestamp_epoch(timestamp)
        if message.startswith("VERSION:"):
            version_markers.append(index)
        if invocation_pattern.search(message):
            invocation_markers.append(index)

    latest_version = version_markers[-1] if version_markers else None
    latest_invocation = invocation_markers[-1] if invocation_markers else None
    if latest_invocation is None:
        start_index = latest_version
    else:
        previous_invocation = invocation_markers[-2] if len(invocation_markers) > 1 else -1
        if (
            latest_version is not None
            and previous_invocation < latest_version <= latest_invocation
        ):
            start_index = latest_version
        else:
            start_index = latest_invocation
    if start_index is None:
        return None, None

    current_cycles = set()
    cycle_pattern = re.compile(r"selfcalcycle(\d+)", re.IGNORECASE)
    cycle_start_pattern = re.compile(
        r"Starting self-calibration cycle\s+(\d+)", re.IGNORECASE
    )
    for line in lines[start_index:]:
        cycle_ids = [match.group(1) for match in cycle_pattern.finditer(line)]
        cycle_ids.extend(
            match.group(1) for match in cycle_start_pattern.finditer(line)
        )
        current_cycles.update(str(int(cycle)).zfill(3) for cycle in cycle_ids)
    return timestamp_by_line.get(start_index), current_cycles


def _scan_logs(run_root):
    candidates = [run_root / "logs" / "selfcal.log", run_root / "h5plot.log"]
    logs = []
    warnings = deque(maxlen=20)
    errors = deque(maxlen=20)
    warning_count = 0
    error_count = 0
    invocations = 0
    timestamps = []

    for path in candidates:
        if not path.is_file():
            continue
        tail = deque(maxlen=80)
        line_count = 0
        try:
            with path.open("r", encoding="utf-8", errors="replace") as stream:
                for line_count, raw_line in enumerate(stream, start=1):
                    line = raw_line.rstrip("\r\n")
                    tail.append(line)
                    level, timestamp, message = _parse_log_line(line)
                    if timestamp:
                        timestamps.append(timestamp)
                    if level == "WARNING":
                        warning_count += 1
                        warnings.append((path, line_count, timestamp, message))
                    elif level in ("ERROR", "CRITICAL"):
                        error_count += 1
                        errors.append((path, line_count, timestamp, message))
                    if re.search(r"facetselfcal(?:\.py)?\b.*(?:\s-i(?:\s|=)|--config(?:\s|=))", line, re.IGNORECASE):
                        invocations += 1
        except OSError:
            continue
        logs.append({"path": path, "lines": line_count, "tail": list(tail)})

    return {
        "files": logs,
        "warnings": list(warnings),
        "errors": list(errors),
        "warning_count": warning_count,
        "error_count": error_count,
        "invocations": invocations,
        "timestamps": timestamps,
    }


def _measurement_set_metadata(run_root):
    path = run_root / "logs" / "selfcal.log"
    if not path.is_file():
        return {}

    metadata = {}
    current_metadata = None
    try:
        with path.open("r", encoding="utf-8", errors="replace") as stream:
            for raw_line in stream:
                _, _, message = _parse_log_line(raw_line.rstrip("\r\n"))
                normalized_message = message.strip()
                if normalized_message == "================":
                    current_metadata = None
                    continue
                if normalized_message.startswith("===") and normalized_message.endswith("==="):
                    ms_path = normalized_message[3:-3].strip()
                    if ms_path:
                        current_metadata = {}
                        metadata[_canonical_ms_path(ms_path, run_root)] = current_metadata
                    else:
                        current_metadata = None
                    continue
                if current_metadata is None:
                    continue

                label, separator, value = message.partition(":")
                if not separator:
                    continue
                for field, log_labels in _MS_METADATA_FIELDS:
                    if label.strip() in log_labels:
                        current_metadata[field] = value.strip()
                        break
    except OSError:
        return metadata
    return metadata


def _canonical_ms_path(path, run_root):
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = run_root / candidate
    normalized_path = os.path.normpath(str(candidate))
    return re.sub(r"(?:\.(?:copy|avg))+$", "", normalized_path, flags=re.IGNORECASE)


def _link(path, base_dir, label=None):
    href = _relative_url(path, base_dir)
    text = label if label is not None else path.name
    return '<a href="{}">{}</a>'.format(_escape(href), _escape(text))


def _figure(path, base_dir, caption, detail=None, extra_search="", item_class="image-entry"):
    href = _relative_url(path, base_dir)
    detail_html = (
        '<span class="caption-detail">{}</span>'.format(_escape(detail)) if detail else ""
    )
    search_text = "{} {} {}".format(caption, detail or "", extra_search)
    return (
        '<figure class="{}" data-search="{}">'
        '<a href="{}" title="Open full-size image">'
        '<img src="{}" alt="{}" loading="lazy" decoding="async"></a>'
        '<figcaption><code>{}</code>{}</figcaption></figure>'
    ).format(
        _escape(item_class),
        _escape(search_text),
        _escape(href),
        _escape(href),
        _escape(caption),
        _escape(caption),
        detail_html,
    )


def _blink_controls(paths, base_dir):
    if len(paths) < 2:
        return ""

    frames = [
        {"src": _relative_url(path, base_dir), "label": path.stem}
        for path in paths
    ]
    first_href = _relative_url(paths[0], base_dir)
    return (
        '<div class="blink-tool" data-blink-tool data-blink-images="{}">'
        '<div class="blink-controls">'
        '<label class="blink-rate">Blink interval'
        '<span><input type="range" min="100" max="2000" step="100" value="500" '
        'data-blink-interval aria-label="Blink interval in milliseconds">'
        '<output data-blink-interval-output>500 ms</output></span></label>'
        '<button type="button" data-blink-toggle aria-pressed="false">Start blinking</button>'
        '</div>'
        '<figure class="blink-preview"><img data-blink-image src="{}" alt="{}">'
        '<figcaption><code data-blink-caption>{}</code></figcaption></figure>'
        '</div>'
    ).format(
        _escape(json.dumps(frames, separators=(",", ":"))),
        _escape(first_href),
        _escape(paths[0].stem),
        _escape(paths[0].stem),
    )


def _status(status):
    value = (status or "unknown").lower()
    if value == "running":
        return "Running", "running"
    if value == "completed":
        return "Completed", "completed"
    if value == "failed":
        return "Failed", "failed"
    if value == "interrupted":
        return "Interrupted", "interrupted"
    if value == "stopped":
        return "Stopped early", "stopped"
    return "Status not recorded", "unknown"


def _render_log_events(events, total_count, is_error=False):
    if not events:
        return '<p class="empty">No {} records were found in the saved logs.</p>'.format(
            "error" if is_error else "warning"
        )
    entries = []
    for path, line_number, timestamp, message in reversed(events):
        meta = "{}:{}".format(path.name, line_number)
        if timestamp:
            meta = "{} | {}".format(meta, timestamp)
        entries.append(
            '<div class="log-event{}"><small>{}</small>{}</div>'.format(
                " error" if is_error else "", _escape(meta), _escape(message)
            )
        )
    omitted = max(0, total_count - len(events))
    note = '<p class="page-intro">Showing the latest {} of {} records.</p>'.format(
        len(events), total_count
    )
    if omitted:
        note = '<p class="page-intro">Showing the latest {} of {}; {} earlier records are omitted here but remain in the full log.</p>'.format(
            len(events), total_count, omitted
        )
    return note + '<div class="log-events">{}</div>'.format("".join(entries))


def _root_file_link(path, run_root, site_dir, label=None):
    return _link(path, site_dir, label or path.name)


def _write_page(path, title, active, body, nested=False, subtitle="Offline processing report"):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        _page_shell(title, active, body, nested=nested, subtitle=subtitle),
        encoding="utf-8",
    )


def _overview_page(site_dir, run_root, config, artifacts, logs, status, error):
    title = str(config.get("imagename") or run_root.name)
    status_text, status_class = _status(status)
    ms_inputs = _as_list(config.get("ms"))
    metrics = [
        (len(ms_inputs), "Configured measurement sets"),
        (len(artifacts["cycles"]), "Solution cycles found"),
        (len(artifacts["solution_files"]), "Solution H5 files"),
        (logs["warning_count"] + logs["error_count"], "Warnings and errors"),
    ]
    metric_html = "".join(
        '<div class="metric"><span class="metric-value">{}</span><span class="metric-label">{}</span></div>'.format(
            _escape(value), _escape(label)
        )
        for value, label in metrics
    )
    status_detail = ""
    if status in (None, "unknown"):
        status_detail = "This report was generated from existing files; the original process exit status was not recorded."
    elif status == "running":
        status_detail = "facetselfcal is running. This report is refreshed after each completed self-calibration cycle."
    elif status == "completed":
        status_detail = "facetselfcal returned normally."
    elif status == "failed":
        status_detail = "facetselfcal exited with an error."
    elif status == "interrupted":
        status_detail = "The run was interrupted."
    elif status == "stopped":
        status_detail = "The workflow stopped early by request."

    body_parts = [
        '<p class="page-intro">Summary of the products and logs currently present in this run directory. The report uses local files and does not require a network connection.</p>',
        '<div class="statusline"><span class="status status-{}">{}</span><span class="status-detail">{}</span></div>'.format(
            status_class, _escape(status_text), _escape(status_detail)
        ),
        '<div class="metrics">{}</div>'.format(metric_html),
    ]
    if error:
        body_parts.append(
            '<div class="notice error"><strong>Run error</strong><p>{}</p></div>'.format(
                _escape(error)
            )
        )
    if logs["invocations"] > 1:
        body_parts.append(
            '<div class="notice"><strong>Combined run history</strong><p>The saved self-calibration log contains {} command records. This report summarizes the current output directory; some products may come from different run segments.</p></div>'.format(
                logs["invocations"]
            )
        )

    summary_html = _config_table(config, _SUMMARY_KEYS)
    body_parts.append(_section("Run configuration", summary_html, "Selected values from full_config.txt."))

    if artifacts["overview_plots"]:
        preview_paths = artifacts["overview_plots"][:]
        selected = [p for p in preview_paths if p.name.lower().startswith("im_")]
        if len(selected) > 4:
            selected = selected[:2] + selected[-2:]
        if not selected:
            selected = preview_paths[:4]
        preview_html = '<div class="gallery">{}</div>'.format(
            "".join(
                _figure(
                    path,
                    site_dir,
                    path.stem,
                    detail="Overview plot",
                    item_class="image-entry",
                )
                for path in selected
            )
        )
        body_parts.append(
            _section(
                "Image progression",
                preview_html,
                "{} overview plots are shown here; browse all imaging previews and FITS products on the Imaging page.".format(
                    len(artifacts["overview_plots"])
                ),
            )
        )

    issues_content = _render_log_events(
        logs["warnings"], logs["warning_count"], is_error=False
    )
    body_parts.append(_section("Recent warnings", issues_content))
    errors_content = _render_log_events(logs["errors"], logs["error_count"], is_error=True)
    body_parts.append(_section("Recent errors", errors_content))
    body_parts.append(
        '<div class="page-links"><a href="calibration.html">Calibration plots: {} files</a><a href="datasets.html">Measurement-set details</a><a href="run-details.html">Full configuration and logs</a></div>'.format(
            sum(len(paths) for _, cycles in artifacts["calibration_sets"] for paths in cycles.values())
        )
    )
    _write_page(
        site_dir / "index.html",
        title,
        "index.html",
        "".join(body_parts),
        subtitle="Run overview / {}".format(run_root.name),
    )


def _imaging_page(site_dir, run_root, config, artifacts):
    title = str(config.get("imagename") or run_root.name)
    plots = artifacts["overview_plots"]
    numbered_series = defaultdict(list)
    for path in plots:
        match = re.fullmatch(r"(.+)_([0-9]+)", path.stem)
        if match:
            numbered_series[match.group(1).casefold()].append((int(match.group(2)), path))
    sequence_order = {}
    sequence_paths = set()
    for prefix, frames in numbered_series.items():
        if len(frames) < 2:
            continue
        for frame_number, path in sorted(frames, key=lambda frame: (frame[0], frame[1].name.casefold())):
            sequence_paths.add(path)
            sequence_order[path] = (prefix, frame_number)

    progression = [
        path for path in plots
        if path.name.lower().startswith("im_") or path in sequence_paths
    ]
    progression.sort(key=lambda path: sequence_order.get(path, (path.stem.casefold(), -1)))
    diagnostics = [path for path in plots if path not in progression]
    sections = []
    if plots:
        sections.append(_filter_input(".image-entry", "Filter plot names"))
        if progression:
            figures = "".join(
                _figure(path, site_dir, path.stem, detail="Imaging overview")
                for path in progression
            )
            blink_controls = _blink_controls(progression, site_dir)
            sections.append(
                '<details data-filter-group open><summary>Image progression plots ({})</summary>{}<div class="gallery">{}</div></details>'.format(
                    len(progression), blink_controls, figures
                )
            )
        if diagnostics:
            figures = "".join(
                _figure(path, site_dir, path.stem, detail="Diagnostic plot")
                for path in diagnostics
            )
            sections.append(
                '<details data-filter-group><summary>Other imaging diagnostics ({})</summary><div class="gallery">{}</div></details>'.format(
                    len(diagnostics), figures
                )
            )
    else:
        sections.append('<p class="empty">No PNG previews were found in plots/.</p>')

    product_categories = (
        ("Images", re.compile(r"-image(?:-|\.|$)", re.IGNORECASE)),
        ("Models", re.compile(r"-model(?:-|\.|$)", re.IGNORECASE)),
        ("Residuals", re.compile(r"-residual(?:-|\.|$)", re.IGNORECASE)),
        ("Dirty images", re.compile(r"-dirty(?:-|\.|$)", re.IGNORECASE)),
        ("PSFs", re.compile(r"-psf(?:-|\.|$)", re.IGNORECASE)),
        ("Beams", re.compile(r"-beam(?:-|\.|$)", re.IGNORECASE)),
        ("Mask products", re.compile(r"(?:^|[.-])mask(?:[.-]|$)", re.IGNORECASE)),
    )
    fits_groups = {category: [] for category, _ in product_categories}
    fits_groups["Other FITS products"] = []
    for path in artifacts["fits_files"]:
        for category, pattern in product_categories:
            if pattern.search(path.name):
                fits_groups[category].append(path)
                break
        else:
            fits_groups["Other FITS products"].append(path)

    fit_sections = []
    if artifacts["fits_files"]:
        fit_sections.append(_filter_input(".fits-entry", "Filter FITS filenames"))
        category_order = [category for category, _ in product_categories]
        category_order.append("Other FITS products")
        for category in category_order:
            paths = fits_groups.get(category, [])
            if not paths:
                continue
            entries = []
            for path in paths:
                entries.append(
                    '<li class="fits-entry" data-search="{}">{}<span class="file-size">{}</span></li>'.format(
                        _escape(path.name),
                        _link(path, site_dir),
                        _escape(_format_size(path.stat().st_size)),
                    )
                )
            fit_sections.append(
                '<details data-filter-group{}><summary>{} ({})</summary><ul class="file-list">{}</ul></details>'.format(
                    " open" if not fit_sections else "",
                    _escape(category),
                    len(paths),
                    "".join(entries),
                )
            )
    else:
        fit_sections.append('<p class="empty">No FITS products were found in fits_images/.</p>')

    body = (
        '<p class="page-intro">PNG previews open locally at full size. FITS files are linked as products; use an astronomy FITS viewer to inspect their pixel data.</p>'
        + _section("PNG previews", "".join(sections))
        + _section("FITS products", "".join(fit_sections), "{} files found.".format(len(artifacts["fits_files"])))
    )
    _write_page(site_dir / "imaging.html", title, "imaging.html", body)


def _cycle_sort_key(value):
    if value.isdigit():
        return (0, int(value))
    return (1, value)


def _plot_page(site_dir, run_root, title, dataset_name, dataset_slug, cycle, paths):
    calibration_dir = site_dir / "calibration"
    page_path = calibration_dir / "{}.html".format(dataset_slug)
    cycle_title = "Cycle {}".format(cycle) if cycle != "other" else "Other plots"
    figures = []
    for path in paths:
        name = path.name
        tags = []
        lower = name.lower()
        for token, label in (("amp", "amplitude"), ("phase", "phase"), ("poldiff", "polarization difference")):
            if token in lower:
                tags.append(label)
        direction = re.search(r"(?:dir|dil)(\d+)", name, re.IGNORECASE)
        if direction:
            tags.append("direction {}".format(direction.group(1)))
        polarization = re.search(r"pol([a-z0-9]+)", name, re.IGNORECASE)
        if polarization:
            tags.append("polarization {}".format(polarization.group(1)))
        figures.append(
            _figure(
                path,
                calibration_dir,
                name,
                detail=" / ".join(tags) if tags else cycle_title,
                extra_search="{} {} {}".format(dataset_name, cycle_title, " ".join(tags)),
                item_class="plot-item",
            )
        )
    body = (
        '<p class="page-intro"><a href="../calibration.html">Calibration index</a> / {} / {}. '
        'This page contains {} plots and loads thumbnails lazily.</p>'.format(
            _escape(dataset_name), _escape(cycle_title), len(paths)
        )
        + _filter_input(".plot-item", "Filter plot filenames")
        + '<div class="gallery">{}</div>'.format("".join(figures))
    )
    _write_page(
        page_path,
        title,
        "calibration.html",
        body,
        nested=True,
        subtitle="Calibration plots / {} / {}".format(dataset_name, cycle_title),
    )
    return page_path


def _calibration_page(site_dir, run_root, config, artifacts):
    title = str(config.get("imagename") or run_root.name)
    calibration_sets = artifacts["calibration_sets"]
    all_cycles = sorted(
        {cycle for _, cycles in calibration_sets for cycle in cycles},
        key=_cycle_sort_key,
    )
    if not calibration_sets:
        body = '<p class="empty">No calibration plots from the current run were found. Products from earlier runs are not shown.</p>'
        _write_page(site_dir / "calibration.html", title, "calibration.html", body)
        return

    rows = []
    cycle_headers = "".join("<th scope=\"col\">{}</th>".format(_escape(cycle)) for cycle in all_cycles)
    for index, (directory, cycles) in enumerate(calibration_sets, start=1):
        dataset_slug = "ms-{:02d}".format(index)
        dataset_name = directory.name.removeprefix("solution_plots_")
        cells = []
        for cycle in all_cycles:
            paths = cycles.get(cycle, [])
            if not paths:
                cells.append('<td class="muted">-</td>')
                continue
            cycle_slug = "cycle-{}".format(cycle)
            page_slug = "{}-{}.html".format(dataset_slug, cycle_slug)
            _plot_page(
                site_dir,
                run_root,
                title,
                dataset_name,
                page_slug[:-5],
                cycle,
                paths,
            )
            cells.append(
                '<td><a href="calibration/{}">{} plots</a></td>'.format(
                    _escape(page_slug), len(paths)
                )
            )
        rows.append(
            '<tr class="dataset-row" data-search="{}"><th scope="row">{}</th>{}</tr>'.format(
                _escape(dataset_name), _escape(dataset_name), "".join(cells)
            )
        )

    body = (
        '<p class="page-intro">Calibration plots are split into one local page per measurement set and solution cycle. Open a cycle page to load its thumbnails lazily.</p>'
        + _filter_input(".dataset-row", "Filter measurement sets")
        + '<div class="table-scroll"><table class="data-table"><thead><tr><th scope="col">Measurement set</th>{}</tr></thead><tbody>{}</tbody></table></div>'.format(
            cycle_headers, "".join(rows)
        )
    )
    _write_page(site_dir / "calibration.html", title, "calibration.html", body)


def _datasets_page(site_dir, run_root, config, artifacts):
    title = str(config.get("imagename") or run_root.name)
    inputs = _as_list(config.get("ms"))
    metadata_by_ms = _measurement_set_metadata(run_root)
    column_count = len(_MS_METADATA_FIELDS) + 1
    input_rows = []
    for path in inputs:
        metadata = dict(metadata_by_ms.get(_canonical_ms_path(path, run_root), {}))
        telescope = metadata.get("Telescope", "").strip().upper()
        if telescope and telescope not in {"VLA", "EVLA"}:
            metadata.setdefault("VLA configuration", "Not applicable")
        search_text = " ".join([path] + list(metadata.values()))
        metadata_cells = "".join(
            "<td>{}</td>".format(_escape(metadata.get(field, "Not recorded")))
            for field, _ in _MS_METADATA_FIELDS
        )
        input_rows.append(
            '<tr class="dataset-row" data-search="{}"><th scope="row"><code>{}</code></th>{}</tr>'.format(
                _escape(search_text), _escape(path), metadata_cells
            )
        )
    if not input_rows:
        input_rows.append(
            '<tr><td colspan="{}">No input MS list was found in full_config.txt.</td></tr>'.format(
                column_count
            )
        )

    body = (
        '<p class="page-intro">Input paths come from full_config.txt. Telescope and observation metadata is read from matching entries in logs/selfcal.log; fields not recorded there are shown as Not recorded.</p>'
        + _filter_input(".dataset-row", "Filter configured inputs")
        + '<div class="table-scroll"><table class="data-table"><thead><tr><th scope="col">Measurement Set</th>{}</tr></thead><tbody>{}</tbody></table></div>'.format(
            "".join(
                '<th scope="col">{}</th>'.format(_escape(field))
                for field, _ in _MS_METADATA_FIELDS
            ),
            "".join(input_rows),
        )
    )
    _write_page(site_dir / "datasets.html", title, "datasets.html", body)


def _run_details_page(site_dir, run_root, config, artifacts, logs, command_text, status, error):
    title = str(config.get("imagename") or run_root.name)
    file_links = []
    for relative in ("full_config.txt", "facetselfcal.txt", "logs/selfcal.log", "h5plot.log"):
        path = run_root / relative
        if path.is_file():
            file_links.append(
                '<li>{}<span class="file-size">{}</span></li>'.format(
                    _link(path, site_dir), _escape(_format_size(path.stat().st_size))
                )
            )
    links_html = '<ul class="file-list">{}</ul>'.format("".join(file_links)) if file_links else '<p class="empty">No configuration or log files were found.</p>'

    tail_sections = []
    for log in logs["files"]:
        tail = "\n".join(log["tail"])
        tail_sections.append(
            '<details><summary>{} / last {} of {} lines</summary><pre>{}</pre></details>'.format(
                _escape(log["path"].name), min(80, log["lines"]), log["lines"], _escape(tail)
            )
        )
    log_tails = "".join(tail_sections) if tail_sections else '<p class="empty">No logs were found.</p>'

    config_filter = _filter_input(".config-row", "Filter configuration options")
    config_table = _config_table(config, filterable=True)
    command_section = ""
    if command_text:
        command_section = '<details><summary>Recorded command line</summary><pre>{}</pre></details>'.format(
            _escape(command_text)
        )

    inventory_rows = [
        ("Calibration PNG plots", sum(len(paths) for _, cycles in artifacts["calibration_sets"] for paths in cycles.values())),
        ("FITS images", len(artifacts["fits_files"])),
        ("Solution H5 files", len(artifacts["solution_files"])),
        ("Top-level MS directories", len(artifacts["ms_directories"])),
    ]
    inventory = '<table class="data-table"><tbody>{}</tbody></table>'.format(
        "".join(
            '<tr><th scope="row">{}</th><td>{}</td></tr>'.format(_escape(label), count)
            for label, count in inventory_rows
        )
    )
    status_text, status_class = _status(status)
    status_block = '<div class="statusline"><span class="status status-{}">{}</span></div>'.format(
        status_class, _escape(status_text)
    )
    if error:
        status_block += '<div class="notice error"><strong>Run error</strong><p>{}</p></div>'.format(_escape(error))
    if logs["invocations"] > 1:
        status_block += '<div class="notice"><strong>Combined log</strong><p>{} command-line invocation records were found. The log may span multiple processing segments.</p></div>'.format(logs["invocations"])

    body = (
        '<p class="page-intro">The report references the run configuration and logs in place. Values are shown as recorded, without interpreting instrument-specific settings.</p>'
        + status_block
        + _section("Configuration files and logs", links_html)
        + _section("Recorded command", command_section or '<p class="empty">No facetselfcal.txt command record was found.</p>')
        + _section("Saved configuration", config_filter + config_table)
        + _section("Artifact inventory", inventory)
        + _section("Warning log records", _render_log_events(logs["warnings"], logs["warning_count"]))
        + _section("Error log records", _render_log_events(logs["errors"], logs["error_count"], is_error=True))
        + _section("Recent log excerpts", log_tails, "At most the last 80 lines of each log are included here; the full logs are linked above.")
    )
    _write_page(site_dir / "run-details.html", title, "run-details.html", body)


def generate_html_overview(run_directory=".", status="unknown", error=None, output_directory=None):
    """Generate static HTML pages and return the overview index path.

    Parameters
    ----------
    run_directory : str or pathlib.Path
        Existing facetselfcal output directory.
    status : str
        Run status: running, completed, failed, interrupted, stopped, or unknown.
    error : str, optional
        Exception summary to include for failed runs.
    output_directory : str or pathlib.Path, optional
        Override the default ``<run_directory>/html_overview`` output location.
    """
    run_root = Path(run_directory).expanduser().resolve()
    if not run_root.is_dir():
        raise NotADirectoryError(str(run_root))
    site_dir = (
        Path(output_directory).expanduser().resolve()
        if output_directory is not None
        else run_root / "html_overview"
    )
    site_dir.mkdir(parents=True, exist_ok=True)
    asset_dir = site_dir / "assets"
    asset_dir.mkdir(parents=True, exist_ok=True)
    (asset_dir / "report.css").write_text(_CSS, encoding="utf-8")
    (asset_dir / "report.js").write_text(_JS, encoding="utf-8")

    config = _parse_config(run_root / "full_config.txt")
    command_path = run_root / "facetselfcal.txt"
    command_text = ""
    if command_path.is_file():
        try:
            command_text = command_path.read_text(encoding="utf-8", errors="replace").strip()
        except OSError:
            pass
    run_started_at, current_run_cycles = _current_run_context(run_root)
    artifacts = _scan_artifacts(run_root, run_started_at, current_run_cycles)
    logs = _scan_logs(run_root)

    _overview_page(site_dir, run_root, config, artifacts, logs, status, error)
    _imaging_page(site_dir, run_root, config, artifacts)
    _calibration_page(site_dir, run_root, config, artifacts)
    _datasets_page(site_dir, run_root, config, artifacts)
    _run_details_page(site_dir, run_root, config, artifacts, logs, command_text, status, error)
    return site_dir / "index.html"


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Generate an offline HTML overview for an existing facetselfcal run."
    )
    parser.add_argument(
        "run_directory",
        nargs="?",
        default=".",
        help="facetselfcal output directory (default: current directory)",
    )
    parser.add_argument(
        "--output",
        help="report destination (default: RUN_DIRECTORY/html_overview)",
    )
    parser.add_argument(
        "--status",
        choices=("unknown", "completed", "failed", "interrupted", "stopped"),
        default="unknown",
        help="optional known status for an existing run",
    )
    args = parser.parse_args(argv)
    index = generate_html_overview(
        args.run_directory,
        status=args.status,
        output_directory=args.output,
    )
    print("[facetselfcal] HTML report created:", index)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
