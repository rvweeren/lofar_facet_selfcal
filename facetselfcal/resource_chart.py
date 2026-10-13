"""Shared SVG rendering for resource-usage charts."""

import html
import json
import math
import re


RESOURCE_PHASE_STYLES = {
    "phaseup": ("Phaseup", "#E7B2B2"),
    "average": ("Average", "#DA6AC1"),
    "phaseshift": ("Phaseshift", "#7C3AED"),
    "filter": ("Filter", "#64748B"),
    "aoflagger": ("AOFlagger", "#BE123C"),
    "imaging": ("Imaging", "#2563EB"),
    "predict": ("Predict", "#A0AEF0"),
    "solve": ("Solve", "#0F766E"),
    "applycal": ("Applycal", "#9DDCCC"),
}

RESOURCE_RAM_COLOR = RESOURCE_PHASE_STYLES["imaging"][1]
DEFAULT_RESOURCE_SAMPLE_COUNT = 300

DP3_EXECUTION_START_MARKER = "FACETSELFCAL_DP3_STEP_START "
DP3_EXECUTION_END_MARKER = "FACETSELFCAL_DP3_STEP_END "


_DP3_PHASE_BY_TYPE = {
    "stationadder": "phaseup",
    "phaseup": "phaseup",
    "averager": "average",
    "phaseshifter": "phaseshift",
    "filter": "filter",
    "aoflag": "aoflagger",
    "aoflagger": "aoflagger",
}
_PHASE_ROW_SPACING = 20
_MIN_PHASE_BAR_WIDTH = 2.0


def _escape(value):
    return html.escape(str(value), quote=True)


def dp3_command_phases(command):
    """Return recognized DP3 steps in the order configured by ``steps=[...]``."""
    command_text = str(command)
    if not re.match(r"^\s*DP3(?:\s|$)", command_text, re.IGNORECASE):
        return []

    steps_match = re.search(
        r"(?:^|\s)steps=\[([^\]]*)\]", command_text, re.IGNORECASE
    )
    if steps_match is None:
        return []

    step_types = {
        name.casefold(): step_type.casefold()
        for name, step_type in re.findall(
            r"(?:^|\s)([A-Za-z0-9_]+)\.type=([^\s]+)",
            command_text,
            re.IGNORECASE,
        )
    }
    step_names = [
        step_name.strip().casefold()
        for step_name in steps_match.group(1).split(",")
    ]
    has_phaseup = any(
        _DP3_PHASE_BY_TYPE.get(step_types.get(step_name)) == "phaseup"
        for step_name in step_names
    )
    phases = []
    for step_name in step_names:
        phase = _DP3_PHASE_BY_TYPE.get(step_types.get(step_name))
        if phase == "filter" and has_phaseup:
            continue
        if phase is not None and phase not in phases:
            phases.append(phase)
    return phases


def dp3_workflow_step_kinds(command):
    """Return workflow step kinds represented by a DP3 command."""
    command_text = str(command)
    if not re.match(r"^\s*DP3(?:\s|$)", command_text, re.IGNORECASE):
        return []

    step_kinds = dp3_command_phases(command_text)
    if re.search(
        r"(?:^|\s)[A-Za-z0-9_]+\.type=ddecal(?:\s|$)",
        command_text,
        re.IGNORECASE,
    ):
        step_kinds.append("solve")
    if re.search(
        r"(?:^|\s)[A-Za-z0-9_]+\.type=applycal(?:\s|$)",
        command_text,
        re.IGNORECASE,
    ):
        step_kinds.append("apply")
    return list(dict.fromkeys(step_kinds))


def phase_intervals_from_events(events, end_epoch=None):
    """Pair phase start/end events into intervals for chart rendering."""
    intervals = []
    active = {}

    def close_interval(active_interval, end_epoch_value):
        end_epoch_value = max(active_interval["start_epoch"], end_epoch_value)
        intervals.append({
            "phase": active_interval["phase"],
            "cycle": active_interval.get("cycle"),
            "start_epoch": active_interval["start_epoch"],
            "end_epoch": end_epoch_value,
        })

    for event in sorted(events, key=lambda item: float(item["epoch"])):
        try:
            event_epoch = float(event["epoch"])
            phase = str(event["phase"]).lower()
            event_type = str(event["event"]).lower()
        except (KeyError, TypeError, ValueError):
            continue

        cycle = event.get("cycle")
        key = (phase, str(cycle))
        if event_type == "start":
            active.setdefault(key, []).append({
                "phase": phase,
                "cycle": cycle,
                "start_epoch": event_epoch,
            })
        elif event_type == "end" and active.get(key):
            active_interval = active[key].pop(0)
            close_interval(active_interval, event_epoch)
            if not active[key]:
                del active[key]

    if end_epoch is not None:
        for active_intervals in active.values():
            for active_interval in active_intervals:
                close_interval(active_interval, float(end_epoch))

    return intervals


def _normalized_phase_intervals(phase_intervals):
    normalized = []
    for interval in phase_intervals or []:
        try:
            phase = str(interval["phase"]).lower()
            start_epoch = float(interval["start_epoch"])
            end_epoch = float(interval["end_epoch"])
        except (AttributeError, KeyError, TypeError, ValueError, OverflowError):
            continue
        if (
            phase not in RESOURCE_PHASE_STYLES
            or not math.isfinite(start_epoch)
            or not math.isfinite(end_epoch)
            or end_epoch < start_epoch
        ):
            continue
        normalized.append((phase, start_epoch, end_epoch, interval))
    return normalized


def clip_phase_intervals_to_window(phase_intervals, start_epoch, end_epoch):
    """Clip valid phase intervals to a time window, dropping non-overlaps."""
    start_epoch = float(start_epoch)
    end_epoch = float(end_epoch)
    if end_epoch < start_epoch:
        raise ValueError("end_epoch must not precede start_epoch")

    clipped = []
    for _phase, interval_start, interval_end, interval in _normalized_phase_intervals(
        phase_intervals
    ):
        if interval_end < start_epoch or interval_start > end_epoch:
            continue
        clipped.append(
            {
                **interval,
                "start_epoch": max(start_epoch, interval_start),
                "end_epoch": min(end_epoch, interval_end),
            }
        )
    return clipped


def phase_names_from_intervals(phase_intervals):
    """Return recognized phases with intervals, in display order."""
    present_phases = {
        phase
        for phase, _start_epoch, _end_epoch, _interval in
        _normalized_phase_intervals(phase_intervals)
    }
    return [phase for phase in RESOURCE_PHASE_STYLES if phase in present_phases]


def downsample_resource_samples(samples, point_limit=300):
    """Reduce rendering work while retaining extrema and cycle boundaries."""
    if len(samples) <= point_limit:
        return samples
    indices = {0, len(samples) - 1}
    bucket_size = math.ceil(len(samples) / max(1, (point_limit - 2) // 4))
    for start in range(0, len(samples), bucket_size):
        bucket = range(start, min(start + bucket_size, len(samples)))
        for field in ("tree_cpu_pct", "tree_rss_gib"):
            indices.add(min(bucket, key=lambda index: samples[index][field]))
            indices.add(max(bucket, key=lambda index: samples[index][field]))
    for index in range(1, len(samples)):
        if samples[index].get("cycle") != samples[index - 1].get("cycle"):
            indices.update((index - 1, index))
    return [samples[index] for index in sorted(indices)]


def generate_resource_svg(samples, phase_intervals=None):
    """Render an inline SVG graph of process CPU and RAM usage over time."""
    if not samples:
        return ""
    width = 960
    phase_intervals = _normalized_phase_intervals(phase_intervals)
    present_phases = {
        phase for phase, _start_epoch, _end_epoch, _interval in phase_intervals
    }
    visible_phases = [
        phase for phase in RESOURCE_PHASE_STYLES if phase in present_phases
    ]
    show_activity = bool(visible_phases)
    phase_row_count = len(visible_phases)
    height = 294 + phase_row_count * _PHASE_ROW_SPACING if show_activity else 280
    pad_l = 100 if show_activity else 65
    pad_r = 110
    pad_t = 30
    pad_b = 64 + phase_row_count * _PHASE_ROW_SPACING if show_activity else 40
    plot_w = width - pad_l - pad_r
    plot_h = height - pad_t - pad_b

    plot_epochs = [float(sample["epoch"]) for sample in samples]
    for _phase, start_epoch, end_epoch, _interval in phase_intervals:
        plot_epochs.extend([start_epoch, end_epoch])

    t_min = min(plot_epochs)
    t_max = max(plot_epochs)
    if t_max <= t_min:
        t_max = t_min + 1.0

    max_cpu = max(s["tree_cpu_pct"] for s in samples)
    max_ram = max(s["tree_rss_gib"] for s in samples)

    cpu_y_max = max(100.0, max_cpu * 1.15)
    cpu_y_max = math.ceil(cpu_y_max / 50.0) * 50.0

    ram_y_max = max(1.0, max_ram * 1.15)
    ram_y_max = math.ceil(ram_y_max * 2.0) / 2.0 if ram_y_max <= 10.0 else math.ceil(ram_y_max / 5.0) * 5.0

    def mx(epoch):
        return pad_l + ((epoch - t_min) / (t_max - t_min)) * plot_w

    def my_cpu(val):
        return pad_t + plot_h - (max(0.0, val) / cpu_y_max) * plot_h

    def my_ram(val):
        return pad_t + plot_h - (max(0.0, val) / ram_y_max) * plot_h

    aria_label = "Process tree resource utilization"
    if show_activity:
        aria_label += " and workflow phases"
    aria_label += " over time"
    svg_parts = [
        f'<svg viewBox="0 0 {width} {height}" width="100%" height="auto" preserveAspectRatio="xMidYMid meet" role="img" aria-label="{aria_label}">'
    ]

    for i in range(5):
        ratio = i / 4.0
        y = pad_t + plot_h - ratio * plot_h
        cpu_val = ratio * cpu_y_max
        ram_val = ratio * ram_y_max
        dash = ' stroke-dasharray="3,3"' if i > 0 else ""
        svg_parts.append(
            f'<line x1="{pad_l}" y1="{y:.1f}" x2="{pad_l + plot_w}" y2="{y:.1f}" stroke="var(--line)"{dash} stroke-width="1" />'
        )
        svg_parts.append(
            f'<text x="{pad_l - 8}" y="{y + 4:.1f}" text-anchor="end" fill="var(--teal-dark)" font-size="11" font-family="system-ui, sans-serif">{cpu_val:.0f}%</text>'
        )
        svg_parts.append(
            f'<text x="{pad_l + plot_w + 8}" y="{y + 4:.1f}" text-anchor="start" fill="{RESOURCE_RAM_COLOR}" font-size="11" font-family="system-ui, sans-serif">{ram_val:.1f} GiB</text>'
        )

    svg_parts.append(
        f'<text transform="rotate(-90)" x="-{pad_t + plot_h / 2:.1f}" y="16" text-anchor="middle" fill="var(--teal-dark)" font-weight="600" font-size="11" font-family="system-ui, sans-serif">CPU (% of one core)</text>'
    )
    svg_parts.append(
        f'<text transform="rotate(90)" x="{pad_t + plot_h / 2:.1f}" y="-{width - 16}" text-anchor="middle" fill="{RESOURCE_RAM_COLOR}" font-weight="600" font-size="11" font-family="system-ui, sans-serif">RAM (GiB)</text>'
    )

    num_x_ticks = 5 if (t_max - t_min) >= 10 else 2
    for j in range(num_x_ticks):
        ratio = j / (num_x_ticks - 1)
        t_val = t_min + ratio * (t_max - t_min)
        x = pad_l + ratio * plot_w
        dt_sec = t_val - t_min
        if dt_sec < 3600:
            time_lbl = f"+{dt_sec / 60.0:.0f}m"
        else:
            time_lbl = f"+{dt_sec / 3600.0:.1f}h"
        svg_parts.append(
            f'<text x="{x:.1f}" y="{pad_t + plot_h + 18}" text-anchor="middle" fill="var(--muted)" font-size="11" font-family="system-ui, sans-serif">{time_lbl}</text>'
        )

    activity_top = pad_t + plot_h + 38
    activity_bottom = activity_top + phase_row_count * _PHASE_ROW_SPACING + 2
    if show_activity:
        phase_row_y = {
            phase: activity_top + index * _PHASE_ROW_SPACING
            for index, phase in enumerate(visible_phases)
        }
        for phase in visible_phases:
            label, color = RESOURCE_PHASE_STYLES[phase]
            row_y = phase_row_y[phase]
            svg_parts.append(
                f'<text x="8" y="{row_y + 10}" fill="{color}" font-size="11" font-weight="600" font-family="system-ui, sans-serif">{label}</text>'
            )

        for phase, start_epoch, end_epoch, interval in phase_intervals:
            if end_epoch < t_min or start_epoch > t_max:
                continue

            label, color = RESOURCE_PHASE_STYLES[phase]
            row_y = phase_row_y[phase]
            bar_x = mx(max(t_min, start_epoch))
            bar_right = mx(min(t_max, end_epoch))
            natural_width = bar_right - bar_x
            if natural_width < _MIN_PHASE_BAR_WIDTH:
                continue
            bar_width = min(plot_w - (bar_x - pad_l), natural_width)
            duration = max(0.0, end_epoch - start_epoch)
            if duration < 60:
                duration_label = f"{duration:.0f}s"
            elif duration < 3600:
                duration_label = f"{duration / 60.0:.1f}m"
            else:
                duration_label = f"{duration / 3600.0:.1f}h"
            cycle = interval.get("cycle")
            cycle_label = f"Cycle {cycle} - " if cycle not in (None, "") else ""
            tooltip = f"{cycle_label}{label} ({duration_label})"
            svg_parts.append(
                f'<rect x="{bar_x:.1f}" y="{row_y}" width="{bar_width:.1f}" height="7" rx="2" fill="{color}"><title>{_escape(tooltip)}</title></rect>'
            )

    prev_cycle = None
    for sample in samples:
        cycle = sample["cycle"]
        if prev_cycle is not None and cycle != prev_cycle:
            x_line = mx(sample["epoch"])
            separator_bottom = activity_bottom if show_activity else pad_t + plot_h
            svg_parts.append(
                f'<line x1="{x_line:.1f}" y1="{pad_t}" x2="{x_line:.1f}" y2="{separator_bottom}" stroke="#94a3b8" stroke-width="1.5" stroke-dasharray="4,4" />'
            )
            svg_parts.append(
                f'<text x="{x_line + 4:.1f}" y="{pad_t + 12}" fill="var(--ink-secondary)" font-size="10" font-weight="600" font-family="system-ui, sans-serif">Cycle {_escape(str(cycle))}</text>'
            )
        prev_cycle = cycle

    cpu_coords = [(mx(s["epoch"]), my_cpu(s["tree_cpu_pct"])) for s in samples]
    if cpu_coords:
        area_pts = [f"{cpu_coords[0][0]:.1f},{pad_t + plot_h}"] + [f"{x:.1f},{y:.1f}" for x, y in cpu_coords] + [f"{cpu_coords[-1][0]:.1f},{pad_t + plot_h}"]
        svg_parts.append(f'<polygon points="{" ".join(area_pts)}" fill="#0d9488" fill-opacity="0.12" />')
        line_pts = [f"{x:.1f},{y:.1f}" for x, y in cpu_coords]
        svg_parts.append(f'<polyline points="{" ".join(line_pts)}" fill="none" stroke="#0d9488" stroke-width="2" stroke-linejoin="round" />')

    ram_coords = [(mx(s["epoch"]), my_ram(s["tree_rss_gib"])) for s in samples]
    if ram_coords:
        line_pts = [f"{x:.1f},{y:.1f}" for x, y in ram_coords]
        svg_parts.append(f'<polyline points="{" ".join(line_pts)}" fill="none" stroke="{RESOURCE_RAM_COLOR}" stroke-width="2.5" stroke-linejoin="round" />')

    svg_parts.append("</svg>")
    return "".join(svg_parts)


_RESOURCE_CHART_BROWSER_JS = r"""
(function () {
    "use strict";

    var SVG_NS = "http://www.w3.org/2000/svg";
    var PHASE_STYLES = __RESOURCE_PHASE_STYLES__;
    var RAM_COLOR = PHASE_STYLES.imaging.color;
    var PHASE_ROW_SPACING = 20;
    var MIN_PHASE_BAR_WIDTH = 2;

    function addElement(parent, name, attributes, text) {
        var element = document.createElementNS(SVG_NS, name);
        Object.keys(attributes || {}).forEach(function (key) {
            element.setAttribute(key, String(attributes[key]));
        });
        if (text !== undefined && text !== null) {
            element.textContent = String(text);
        }
        parent.appendChild(element);
        return element;
    }

    function normalizeIntervals(intervals) {
        return (Array.isArray(intervals) ? intervals : []).reduce(
            function (normalized, interval) {
                if (!interval || typeof interval !== "object") return normalized;
                var phase = String(interval.phase || "").toLowerCase();
                var start = Number(interval.start_epoch);
                var end = Number(interval.end_epoch);
                if (
                    !PHASE_STYLES[phase] ||
                    !Number.isFinite(start) ||
                    !Number.isFinite(end) ||
                    end < start
                ) {
                    return normalized;
                }
                normalized.push({
                    phase: phase,
                    cycle: interval.cycle,
                    start_epoch: start,
                    end_epoch: end
                });
                return normalized;
            },
            []
        );
    }

    function renderSvg(target, inputSamples, inputIntervals) {
        var samples = (Array.isArray(inputSamples) ? inputSamples : []).filter(
            function (sample) {
                return (
                    sample &&
                    Number.isFinite(Number(sample.epoch)) &&
                    Number.isFinite(Number(sample.tree_cpu_pct)) &&
                    Number.isFinite(Number(sample.tree_rss_gib))
                );
            }
        );
        while (target.firstChild) target.removeChild(target.firstChild);
        if (!samples.length) {
            target.textContent = "No valid resource samples recorded.";
            return;
        }

        var intervals = normalizeIntervals(inputIntervals);
        var visiblePhases = Object.keys(PHASE_STYLES).filter(function (phase) {
            return intervals.some(function (interval) {
                return interval.phase === phase;
            });
        });
        var showActivity = visiblePhases.length > 0;
        var phaseRowCount = visiblePhases.length;
        var width = 960;
        var height = showActivity ? 294 + phaseRowCount * PHASE_ROW_SPACING : 280;
        var padLeft = showActivity ? 100 : 65;
        var padRight = 110;
        var padTop = 30;
        var padBottom = showActivity
            ? 64 + phaseRowCount * PHASE_ROW_SPACING
            : 40;
        var plotWidth = width - padLeft - padRight;
        var plotHeight = height - padTop - padBottom;
        var timeMin = Infinity;
        var timeMax = -Infinity;
        var maxCpu = -Infinity;
        var maxRam = -Infinity;
        samples.forEach(function (sample) {
            var epoch = Number(sample.epoch);
            timeMin = Math.min(timeMin, epoch);
            timeMax = Math.max(timeMax, epoch);
            maxCpu = Math.max(maxCpu, Number(sample.tree_cpu_pct));
            maxRam = Math.max(maxRam, Number(sample.tree_rss_gib));
        });
        intervals.forEach(function (interval) {
            timeMin = Math.min(
                timeMin,
                interval.start_epoch,
                interval.end_epoch
            );
            timeMax = Math.max(
                timeMax,
                interval.start_epoch,
                interval.end_epoch
            );
        });
        if (timeMax <= timeMin) timeMax = timeMin + 1;

        var cpuMax = Math.max(100, maxCpu * 1.15);
        cpuMax = Math.ceil(cpuMax / 50) * 50;
        var ramMax = Math.max(1, maxRam * 1.15);
        ramMax = ramMax <= 10
            ? Math.ceil(ramMax * 2) / 2
            : Math.ceil(ramMax / 5) * 5;
        var mapX = function (epoch) {
            return padLeft + ((epoch - timeMin) / (timeMax - timeMin)) * plotWidth;
        };
        var mapCpuY = function (value) {
            return padTop + plotHeight - (Math.max(0, value) / cpuMax) * plotHeight;
        };
        var mapRamY = function (value) {
            return padTop + plotHeight - (Math.max(0, value) / ramMax) * plotHeight;
        };
        var svg = document.createElementNS(SVG_NS, "svg");
        svg.setAttribute("viewBox", "0 0 " + width + " " + height);
        svg.setAttribute("width", "100%");
        svg.setAttribute("height", "auto");
        svg.setAttribute("preserveAspectRatio", "xMidYMid meet");
        svg.setAttribute("role", "img");
        svg.setAttribute(
            "aria-label",
            "Process tree resource utilization" +
                (showActivity ? " and workflow phases" : "") +
                " over time"
        );

        for (var i = 0; i < 5; i += 1) {
            var ratio = i / 4;
            var y = padTop + plotHeight - ratio * plotHeight;
            var dash = i > 0 ? "3,3" : null;
            var lineAttributes = {
                x1: padLeft,
                y1: y.toFixed(1),
                x2: padLeft + plotWidth,
                y2: y.toFixed(1),
                stroke: "var(--line)",
                "stroke-width": 1
            };
            if (dash) lineAttributes["stroke-dasharray"] = dash;
            addElement(svg, "line", lineAttributes);
            addElement(
                svg,
                "text",
                {
                    x: padLeft - 8,
                    y: (y + 4).toFixed(1),
                    "text-anchor": "end",
                    fill: "var(--teal-dark)",
                    "font-size": 11,
                    "font-family": "system-ui, sans-serif"
                },
                (ratio * cpuMax).toFixed(0) + "%"
            );
            addElement(
                svg,
                "text",
                {
                    x: padLeft + plotWidth + 8,
                    y: (y + 4).toFixed(1),
                    "text-anchor": "start",
                    fill: RAM_COLOR,
                    "font-size": 11,
                    "font-family": "system-ui, sans-serif"
                },
                (ratio * ramMax).toFixed(1) + " GiB"
            );
        }

        addElement(
            svg,
            "text",
            {
                transform: "rotate(-90)",
                x: -(padTop + plotHeight / 2),
                y: 16,
                "text-anchor": "middle",
                fill: "var(--teal-dark)",
                "font-weight": 600,
                "font-size": 11,
                "font-family": "system-ui, sans-serif"
            },
            "CPU (% of one core)"
        );
        addElement(
            svg,
            "text",
            {
                transform: "rotate(90)",
                x: padTop + plotHeight / 2,
                y: -(width - 16),
                "text-anchor": "middle",
                fill: RAM_COLOR,
                "font-weight": 600,
                "font-size": 11,
                "font-family": "system-ui, sans-serif"
            },
            "RAM (GiB)"
        );

        var tickCount = timeMax - timeMin >= 10 ? 5 : 2;
        for (var tick = 0; tick < tickCount; tick += 1) {
            var tickRatio = tick / (tickCount - 1);
            var tickEpoch = timeMin + tickRatio * (timeMax - timeMin);
            var deltaSeconds = tickEpoch - timeMin;
            var timeLabel = deltaSeconds < 3600
                ? "+" + Math.round(deltaSeconds / 60) + "m"
                : "+" + (deltaSeconds / 3600).toFixed(1) + "h";
            addElement(
                svg,
                "text",
                {
                    x: (padLeft + tickRatio * plotWidth).toFixed(1),
                    y: padTop + plotHeight + 18,
                    "text-anchor": "middle",
                    fill: "var(--muted)",
                    "font-size": 11,
                    "font-family": "system-ui, sans-serif"
                },
                timeLabel
            );
        }

        var activityTop = padTop + plotHeight + 38;
        var activityBottom =
            activityTop + phaseRowCount * PHASE_ROW_SPACING + 2;
        var phaseRowY = {};
        visiblePhases.forEach(function (phase, index) {
            var rowY = activityTop + index * PHASE_ROW_SPACING;
            phaseRowY[phase] = rowY;
            addElement(
                svg,
                "text",
                {
                    x: 8,
                    y: rowY + 10,
                    fill: PHASE_STYLES[phase].color,
                    "font-size": 11,
                    "font-weight": 600,
                    "font-family": "system-ui, sans-serif"
                },
                PHASE_STYLES[phase].label
            );
        });
        intervals.forEach(function (interval) {
            if (
                !Object.prototype.hasOwnProperty.call(phaseRowY, interval.phase) ||
                interval.end_epoch < timeMin ||
                interval.start_epoch > timeMax
            ) {
                return;
            }
            var barX = mapX(Math.max(timeMin, interval.start_epoch));
            var barRight = mapX(Math.min(timeMax, interval.end_epoch));
            var naturalWidth = barRight - barX;
            if (naturalWidth < MIN_PHASE_BAR_WIDTH) return;
            var barWidth = Math.min(
                plotWidth - (barX - padLeft),
                naturalWidth
            );
            var duration = Math.max(0, interval.end_epoch - interval.start_epoch);
            var durationLabel = duration < 60
                ? duration.toFixed(0) + "s"
                : duration < 3600
                    ? (duration / 60).toFixed(1) + "m"
                    : (duration / 3600).toFixed(1) + "h";
            var cycleLabel =
                interval.cycle === undefined ||
                interval.cycle === null ||
                interval.cycle === ""
                    ? ""
                    : "Cycle " + interval.cycle + " - ";
            var bar = addElement(
                svg,
                "rect",
                {
                    x: barX.toFixed(1),
                    y: phaseRowY[interval.phase],
                    width: barWidth.toFixed(1),
                    height: 7,
                    rx: 2,
                    fill: PHASE_STYLES[interval.phase].color
                }
            );
            addElement(
                bar,
                "title",
                {},
                cycleLabel +
                    PHASE_STYLES[interval.phase].label +
                    " (" +
                    durationLabel +
                    ")"
            );
        });

        var hasPreviousCycle = false;
        var previousCycle;
        samples.forEach(function (sample) {
            var cycle = sample.cycle;
            if (hasPreviousCycle && cycle !== previousCycle) {
                var separatorX = mapX(Number(sample.epoch));
                var separatorBottom = showActivity
                    ? activityBottom
                    : padTop + plotHeight;
                addElement(svg, "line", {
                    x1: separatorX.toFixed(1),
                    y1: padTop,
                    x2: separatorX.toFixed(1),
                    y2: separatorBottom,
                    stroke: "#94a3b8",
                    "stroke-width": 1.5,
                    "stroke-dasharray": "4,4"
                });
                addElement(
                    svg,
                    "text",
                    {
                        x: separatorX + 4,
                        y: padTop + 12,
                        fill: "var(--ink-secondary)",
                        "font-size": 10,
                        "font-weight": 600,
                        "font-family": "system-ui, sans-serif"
                    },
                    "Cycle " + cycle
                );
            }
            previousCycle = cycle;
            hasPreviousCycle = true;
        });

        var cpuCoordinates = samples.map(function (sample) {
            return [
                mapX(Number(sample.epoch)),
                mapCpuY(Number(sample.tree_cpu_pct))
            ];
        });
        var firstCpu = cpuCoordinates[0];
        var lastCpu = cpuCoordinates[cpuCoordinates.length - 1];
        var areaPoints = [
            firstCpu[0].toFixed(1) + "," + (padTop + plotHeight)
        ].concat(
            cpuCoordinates.map(function (point) {
                return point[0].toFixed(1) + "," + point[1].toFixed(1);
            }),
            [lastCpu[0].toFixed(1) + "," + (padTop + plotHeight)]
        );
        addElement(svg, "polygon", {
            points: areaPoints.join(" "),
            fill: "#0d9488",
            "fill-opacity": 0.12
        });
        addElement(svg, "polyline", {
            points: cpuCoordinates.map(function (point) {
                return point[0].toFixed(1) + "," + point[1].toFixed(1);
            }).join(" "),
            fill: "none",
            stroke: "#0d9488",
            "stroke-width": 2,
            "stroke-linejoin": "round"
        });
        addElement(svg, "polyline", {
            points: samples.map(function (sample) {
                return (
                    mapX(Number(sample.epoch)).toFixed(1) +
                    "," +
                    mapRamY(Number(sample.tree_rss_gib)).toFixed(1)
                );
            }).join(" "),
            fill: "none",
            stroke: RAM_COLOR,
            "stroke-width": 2.5,
            "stroke-linejoin": "round"
        });
        target.appendChild(svg);
    }

    function decodeResourceData(data) {
        if (!data || !Array.isArray(data.samples)) {
            throw new Error("Resource chart samples are missing.");
        }
        if (data.version === undefined) {
            if (!data.samples.every(function (sample) {
                return sample && ["epoch", "tree_cpu_pct", "tree_rss_gib"].every(
                    function (key) { return Number.isFinite(sample[key]); }
                );
            })) {
                throw new Error("Invalid resource chart sample.");
            }
            return data;
        }
        if (data.version !== 1) {
            throw new Error("Unsupported resource chart data version.");
        }
        if (!data.decoded) {
            data.decoded = {
                samples: data.samples.map(function (row) {
                    if (!Array.isArray(row) || row.length !== 4 ||
                        ![row[0], row[2], row[3]].every(Number.isFinite)) {
                        throw new Error("Invalid compact resource chart sample.");
                    }
                    return {epoch: row[0], cycle: row[1],
                        tree_cpu_pct: row[2], tree_rss_gib: row[3]};
                }),
                phase_intervals: data.phase_intervals || []
            };
        }
        return data.decoded;
    }

    function downsample(samples) {
        if (samples.length <= 300) return samples;
        var indices = new Set([0, samples.length - 1]);
        var bucketSize = Math.ceil(samples.length / 74);
        for (var start = 0; start < samples.length; start += bucketSize) {
            ["tree_cpu_pct", "tree_rss_gib"].forEach(function (field) {
                var minimum = start;
                var maximum = start;
                for (var index = start; index < Math.min(start + bucketSize, samples.length); index++) {
                    if (samples[index][field] < samples[minimum][field]) minimum = index;
                    if (samples[index][field] > samples[maximum][field]) maximum = index;
                }
                indices.add(minimum);
                indices.add(maximum);
            });
        }
        for (var index = 1; index < samples.length; index++) {
            if (samples[index].cycle !== samples[index - 1].cycle) {
                indices.add(index - 1);
                indices.add(index);
            }
        }
        return Array.from(indices).sort(function (a, b) { return a - b; })
            .map(function (index) { return samples[index]; });
    }

    function renderResourceWindow(target, data, requestedCount) {
        if (!Number.isSafeInteger(requestedCount) || requestedCount < 1) {
            target.textContent = "Enter a positive whole number of samples.";
            return 0;
        }
        data = decodeResourceData(data);
        var samples = data.samples;
        if (!samples.length) {
            target.textContent = "No valid resource samples recorded.";
            return 0;
        }
        var effectiveCount = Math.min(requestedCount, samples.length);
        var recentSamples = samples.slice(-effectiveCount);
        var firstEpoch = Number(recentSamples[0].epoch);
        var lastEpoch = Number(recentSamples[recentSamples.length - 1].epoch);
        var recentIntervals = normalizeIntervals(
            data.phase_intervals
        ).filter(function (interval) {
            return interval.end_epoch >= firstEpoch &&
                interval.start_epoch <= lastEpoch;
        }).map(function (interval) {
            return {
                phase: interval.phase,
                cycle: interval.cycle,
                start_epoch: Math.max(firstEpoch, interval.start_epoch),
                end_epoch: Math.min(lastEpoch, interval.end_epoch)
            };
        });
        renderSvg(target, downsample(recentSamples), recentIntervals);
        return effectiveCount;
    }

    window.facetselfcalRenderResourceWindow = renderResourceWindow;
    window.facetselfcalDecodeResourceData = decodeResourceData;

    document.addEventListener("DOMContentLoaded", function () {
        document.querySelectorAll("[data-resource-chart-live]").forEach(
            function (container) {
                var target = container.querySelector("[data-resource-chart-target]");
                var dataElement = container.querySelector("[data-resource-chart-data]");
                if (!target || !dataElement) return;
                var data;
                try {
                    data = decodeResourceData(JSON.parse(dataElement.textContent || "{}"));
                } catch (error) {
                    target.textContent = "Resource chart data could not be read.";
                    console.error("Could not parse live resource chart data.", error);
                    return;
                }
                var parameters = new URLSearchParams(window.location.search);
                var rawCount = parameters.get("sample_limit");
                var requestedCount = rawCount === null
                    ? __DEFAULT_RESOURCE_SAMPLE_COUNT__
                    : Number(rawCount);
                var displayedCount = renderResourceWindow(
                    target,
                    data,
                    requestedCount
                );
                if (window.parent !== window) {
                    window.parent.postMessage(
                        {
                            type: "facetselfcal-resource-chart-window",
                            requestedCount: requestedCount,
                            displayedCount: displayedCount,
                            availableCount:
                                data && Array.isArray(data.samples)
                                    ? data.samples.length
                                    : 0
                        },
                        "*"
                    );
                }
            }
        );
    });
})();
"""


def resource_chart_data_json(samples, phase_intervals=None):
    """Serialize chart data safely for embedding in an offline HTML page."""
    serialized = json.dumps(
        {
            "version": 1,
            "samples": [
                [sample["epoch"], sample.get("cycle"), sample["tree_cpu_pct"], sample["tree_rss_gib"]]
                for sample in samples
            ],
            "phase_intervals": phase_intervals or [],
        },
        separators=(",", ":"),
        allow_nan=False,
    )
    return serialized.replace("&", "\\u0026").replace("<", "\\u003c").replace(
        ">", "\\u003e"
    )


def resource_chart_browser_js():
    """Return the browser-side resource SVG renderer with current phase styles."""
    phase_styles = {
        phase: {"label": label, "color": color}
        for phase, (label, color) in RESOURCE_PHASE_STYLES.items()
    }
    browser_js = _RESOURCE_CHART_BROWSER_JS.replace(
        "__RESOURCE_PHASE_STYLES__",
        json.dumps(phase_styles, separators=(",", ":")),
    )
    return browser_js.replace(
        "__DEFAULT_RESOURCE_SAMPLE_COUNT__",
        str(DEFAULT_RESOURCE_SAMPLE_COUNT),
    )
