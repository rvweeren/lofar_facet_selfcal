"""Shared SVG rendering for resource-usage charts."""

import html
import math


def _escape(value):
    return html.escape(str(value), quote=True)


def generate_resource_svg(samples):
    """Render an inline SVG graph of process CPU and RAM usage over time."""
    if not samples:
        return ""
    width = 960
    height = 280
    pad_l = 65
    pad_r = 65
    pad_t = 30
    pad_b = 40
    plot_w = width - pad_l - pad_r
    plot_h = height - pad_t - pad_b

    t_min = samples[0]["epoch"]
    t_max = samples[-1]["epoch"]
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

    svg_parts = [
        f'<svg viewBox="0 0 {width} {height}" width="100%" height="auto" preserveAspectRatio="xMidYMid meet" role="img" aria-label="Process tree resource utilization over time">'
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
            f'<text x="{pad_l + plot_w + 8}" y="{y + 4:.1f}" text-anchor="start" fill="var(--amber)" font-size="11" font-family="system-ui, sans-serif">{ram_val:.1f} GiB</text>'
        )

    svg_parts.append(
        f'<text transform="rotate(-90)" x="-{pad_t + plot_h / 2:.1f}" y="16" text-anchor="middle" fill="var(--teal-dark)" font-weight="600" font-size="11" font-family="system-ui, sans-serif">CPU (%)</text>'
    )
    svg_parts.append(
        f'<text transform="rotate(90)" x="{pad_t + plot_h / 2:.1f}" y="-{width - 16}" text-anchor="middle" fill="var(--amber)" font-weight="600" font-size="11" font-family="system-ui, sans-serif">RAM (GiB)</text>'
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

    prev_cycle = None
    for sample in samples:
        cycle = sample["cycle"]
        if prev_cycle is not None and cycle != prev_cycle:
            x_line = mx(sample["epoch"])
            svg_parts.append(
                f'<line x1="{x_line:.1f}" y1="{pad_t}" x2="{x_line:.1f}" y2="{pad_t + plot_h}" stroke="#94a3b8" stroke-width="1.5" stroke-dasharray="4,4" />'
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
        svg_parts.append(f'<polyline points="{" ".join(line_pts)}" fill="none" stroke="#d97706" stroke-width="2.5" stroke-linejoin="round" />')

    svg_parts.append("</svg>")
    return "".join(svg_parts)