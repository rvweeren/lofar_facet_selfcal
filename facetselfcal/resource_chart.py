"""Shared SVG rendering for resource-usage charts."""

import html
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


def phase_names_from_intervals(phase_intervals):
    """Return recognized phases with intervals, in display order."""
    present_phases = {
        phase
        for phase, _start_epoch, _end_epoch, _interval in
        _normalized_phase_intervals(phase_intervals)
    }
    return [phase for phase in RESOURCE_PHASE_STYLES if phase in present_phases]


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
