"""Generate a browsable, offline HTML report from a facetselfcal run directory."""

import argparse
import ast
import html
import json
import math
import os
import re
import shlex
from collections import defaultdict, deque
from datetime import datetime
from pathlib import Path
from urllib.parse import quote

from .report_support import (
    DerivedDataCache,
    atomic_write_text,
    read_phase_events,
    read_resource_samples,
    report_context,
    report_warning,
    stage_page,
)
from .resource_chart import (
    COMMAND_MARKER,
    DEFAULT_RESOURCE_SAMPLE_COUNT,
    DP3_EXECUTION_END_MARKER,
    DP3_EXECUTION_START_MARKER,
    RESOURCE_PHASE_STYLES,
    RESOURCE_RAM_COLOR,
    clip_phase_intervals_to_window,
    dp3_command_phases,
    dp3_workflow_step_kinds,
    generate_resource_svg,
    phase_intervals_from_events,
    resource_chart_browser_js,
    resource_chart_data_json,
    downsample_resource_samples,
)

_FULL_RESOURCE_CHART_POINT_LIMIT = 300


# Navigation order. Pages in _OPTIONAL_PAGES are shown only when the run
# uses the matching mode (see _optional_pages).
_PAGE_NAMES = {
    "index.html": "Overview",
    "imaging.html": "Imaging",
    "extraction.html": "Extraction",
    "directions.html": "Directions",
    "calibration.html": "Calibration",
    "bandpass.html": "Bandpass",
    "datasets.html": "Measurement sets",
    "run-details.html": "Run details",
}
_OPTIONAL_PAGES = frozenset({"extraction.html", "directions.html", "bandpass.html"})

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
    "smoothnessconstraint_list",
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

_IMAGE_METRIC_FIELDS = (
    ("max_image", "Max image"),
    ("min_image", "Min image"),
    ("rms_noise", "RMS noise"),
    ("dynamic_range", "Dynamic range"),
)
_JY_IMAGE_METRICS = frozenset(("max_image", "min_image", "rms_noise"))
_CYCLE_IMAGE_METRIC_FIELDS = (
    ("rms_noise", "RMS noise"),
    ("dynamic_range", "Dynamic range"),
)

_CSS = r"""
:root {
  color-scheme: light;
  --paper: #f8fafc;
  --surface: #ffffff;
  --surface-alt: #f1f5f9;
  --ink: #0f172a;
  --ink-secondary: #334155;
  --muted: #64748b;
  --line: #e2e8f0;
  --line-light: #f1f5f9;
  --teal: #0d9488;
  --teal-dark: #0f766e;
  --teal-deep: #134e4a;
  --teal-light: #f0fdfa;
  --teal-border: #99f6e4;
  --blue: #0284c7;
  --blue-pale: #f0f9ff;
  --blue-border: #bae6fd;
  --amber: #b45309;
  --amber-pale: #fffbeb;
  --amber-border: #fde68a;
  --red: #b91c1c;
  --red-pale: #fef2f2;
  --red-border: #fecaca;
  --green: #15803d;
  --green-pale: #f0fdf4;
  --green-border: #bbf7d0;
  /* RESOURCE_PHASE_COLORS */
  --workflow-imaging-pale: #EFF6FF;
  --workflow-imaging-ink: #1D4ED8;
  --workflow-predict-pale: #F1F3FE;
  --workflow-predict-ink: #4655A0;
  --workflow-solve-pale: #F0FDFA;
  --workflow-solve-ink: #115E59;
  --workflow-applycal-pale: #F0FAF7;
  --workflow-applycal-ink: #276658;
  --workflow-average-pale: #FFF0FA;
  --workflow-average-ink: #8A2E72;
  --workflow-phaseup-pale: #FFF5F5;
  --workflow-phaseup-ink: #7A4242;
  --workflow-phaseshift-pale: #F5F3FF;
  --workflow-phaseshift-ink: #5B21B6;
  --workflow-filter-pale: #F1F5F9;
  --workflow-filter-ink: #334155;
  --workflow-aoflagger-pale: #FFF1F2;
  --workflow-aoflagger-ink: #9F1239;
  --code: #f1f5f9;
}
* { box-sizing: border-box; }
body {
  margin: 0;
  color: var(--ink);
  background: var(--paper);
  font: 14px/1.6 system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
}
a { color: var(--teal-dark); text-underline-offset: 3px; }
a:hover { color: var(--teal-deep); }
.site-header { background: var(--teal-deep); color: #f8fafc; border-bottom: 1px solid #115e59; }
.masthead, nav, main, footer { width: min(1320px, calc(100% - 40px)); margin: 0 auto; }
.masthead { padding: 22px 0 18px; }
.eyebrow { margin: 0 0 6px; color: #5eead4; font-size: 11px; font-weight: 700; text-transform: uppercase; letter-spacing: .08em; }
h1, h2, h3 { line-height: 1.25; color: var(--ink); }
h1 { margin: 0; font-size: 28px; font-weight: 700; color: #ffffff; }
.header-subtitle { max-width: 900px; margin: 8px 0 0; color: #ccfbf1; font-size: 13px; overflow-wrap: anywhere; }
nav { display: flex; gap: 4px; overflow-x: auto; border-top: 1px solid #115e59; }
nav a { flex: 0 0 auto; padding: 11px 16px; color: #ccfbf1; text-decoration: none; font-size: 13px; font-weight: 600; border-bottom: 3px solid transparent; }
nav a[aria-current="page"] { color: #ffffff; background: #115e59; border-bottom-color: #5eead4; }
nav a:hover { color: #ffffff; background: #166e66; }
main { padding: 26px 0 58px; }
.page-intro { margin: 0 0 20px; color: var(--muted); max-width: 960px; font-size: 14px; }
.page-intro strong { color: var(--ink); }
h2 { margin: 0; font-size: 20px; font-weight: 700; }
h3 { margin: 0 0 10px; font-size: 15px; font-weight: 600; color: var(--ink-secondary); }
section { margin: 26px 0 0; }
.section-heading { display: flex; align-items: baseline; justify-content: space-between; gap: 16px; border-bottom: 1px solid var(--line); padding-bottom: 8px; margin-bottom: 14px; }
.section-heading p { margin: 0; color: var(--muted); font-size: 13px; }
.statusline { display: flex; flex-wrap: wrap; align-items: center; gap: 12px; margin-top: 14px; }
.status { display: inline-block; border-radius: 4px; padding: 4px 10px; font-size: 12px; font-weight: 700; text-transform: uppercase; letter-spacing: .04em; }
.status-completed { color: #166534; background: var(--green-pale); border: 1px solid var(--green-border); }
.status-running { color: #075985; background: var(--blue-pale); border: 1px solid var(--blue-border); }
.status-failed { color: #991b1b; background: var(--red-pale); border: 1px solid var(--red-border); }
.status-interrupted, .status-stopped, .status-unknown { color: #92400e; background: var(--amber-pale); border: 1px solid var(--amber-border); }
.status-detail { color: var(--ink-secondary); font-size: 13px; }
.metrics { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); grid-auto-rows: 1fr; align-items: stretch; gap: 10px; margin: 16px 0 24px; }
.metric { min-width: 0; padding: 10px 12px; background: var(--surface); border: 1px solid var(--line); border-radius: 6px; box-shadow: 0 1px 3px rgba(15,23,42,0.04); border-top: 3px solid var(--teal); }
.metric-link { color: inherit; text-decoration: none; }
.metric-link:hover { color: inherit; border-color: var(--teal-dark); }
.metric-link:focus-visible { outline: 2px solid var(--teal); outline-offset: 2px; }
.metric-value { display: block; font-size: 20px; font-weight: 700; color: var(--ink); overflow-wrap: anywhere; line-height: 1.1; }
.metric-label { display: block; margin-top: 4px; color: var(--muted); font-size: 10px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.04em; }
.env-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(210px, 1fr)); gap: 10px; margin: 12px 0; }
.env-card { background: var(--surface); border: 1px solid var(--line); border-radius: 6px; padding: 10px 14px; box-shadow: 0 1px 2px rgba(15,23,42,0.03); }
.env-card strong { display: block; font-size: 11px; text-transform: uppercase; letter-spacing: 0.05em; color: var(--muted); margin-bottom: 4px; }
.env-card span { font-size: 13px; color: var(--ink); font-weight: 600; word-break: break-word; }
.notice { margin: 16px 0; border: 1px solid var(--amber-border); border-left: 4px solid var(--amber); background: var(--amber-pale); border-radius: 6px; padding: 12px 16px; color: #78350f; }
.notice.error { border-color: var(--red-border); border-left-color: var(--red); background: var(--red-pale); color: #7f1d1d; }
.notice.info { border-color: var(--blue-border); border-left-color: var(--blue); background: var(--blue-pale); color: #0c4a6e; }
.notice.success { border-color: var(--green-border); border-left-color: var(--green); background: var(--green-pale); color: #14532d; }
.notice p { margin: 4px 0 0; }
.data-table { width: 100%; border-collapse: separate; border-spacing: 0; background: var(--surface); border: 1px solid var(--line); border-radius: 6px; overflow: hidden; }
.data-table.progression-table { table-layout: fixed; }
.data-table.progression-table thead th:nth-child(1) { width: 9%; }
.data-table.progression-table thead th:nth-child(2) { width: 15%; }
.data-table.progression-table thead th:nth-child(3) { width: 10%; white-space: nowrap; }
.data-table.progression-table thead th:nth-child(4) { width: 8%; }
.data-table.progression-table thead th:nth-child(5) { width: 10%; }
.data-table.progression-table thead th:nth-child(6) { width: 12%; }
.data-table.progression-table thead th:nth-child(7) { width: 9%; }
.data-table.progression-table thead th:nth-child(8) { width: 12%; white-space: nowrap; }
.data-table.progression-table thead th:nth-child(9) { width: 15%; }
.data-table.progression-table th, .data-table.progression-table td { overflow-wrap: anywhere; }
.data-table.progression-table thead th { overflow-wrap: normal; }
.data-table.progression-table thead th:nth-child(3),
.data-table.progression-table thead th:nth-child(4),
.data-table.progression-table thead th:nth-child(5),
.data-table.progression-table thead th:nth-child(7),
.data-table.progression-table thead th:nth-child(8) { text-align: right; }
.data-table.progression-table td.numeric { white-space: normal; }
.data-table.progression-table tbody:nth-of-type(even) tr.progression-summary-row th,
.data-table.progression-table tbody:nth-of-type(even) tr.progression-summary-row td { background: #fafcff; }
.data-table.progression-table tr.progression-summary-row > th[scope="row"] { border-left: 3px solid var(--teal); padding-left: 9px; }
.data-table.progression-table tbody:not(:last-of-type) tr:last-child th,
.data-table.progression-table tbody:not(:last-of-type) tr:last-child td { border-bottom: 1px solid var(--line); }
.data-table.progression-table tr.progression-workflow-row td { padding: 6px 12px 10px; background: var(--surface); }
.data-table.progression-table .workflow-panel { padding: 8px 10px; border: 1px solid var(--line-light); border-left: 3px solid var(--teal); border-radius: 4px; background: var(--surface-alt); }
.data-table.progression-table .workflow-panel-heading { margin-bottom: 6px; color: var(--muted); font-size: 10px; font-weight: 700; letter-spacing: .04em; text-transform: uppercase; }
.data-table.progression-table .workflow-shared-label { flex-basis: 100px; }
.table-scroll { max-width: 100%; overflow-x: auto; margin-bottom: 14px; }
.data-table th, .data-table td { padding: 9px 12px; border-bottom: 1px solid var(--line); text-align: left; vertical-align: top; }
.data-table thead th { background: var(--surface-alt); color: var(--ink-secondary); font-size: 12px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.03em; }
.data-table tbody th { width: 220px; color: var(--muted); font-size: 12px; font-weight: 600; }
.data-table tbody th code { overflow-wrap: anywhere; }
.data-table td { overflow-wrap: break-word; word-break: normal; }
.data-table td code { overflow-wrap: anywhere; }
.data-table td.numeric { white-space: nowrap; text-align: right; font-variant-numeric: tabular-nums; }
.data-table td.nowrap { white-space: nowrap; }
.image-metrics-table { table-layout: fixed; }
.image-metrics-table tr > :nth-child(1) { width: 13%; }
.image-metrics-table tr > :nth-child(2) { width: 24%; }
.image-metrics-table tr > :nth-child(3) { width: 14%; }
.image-metrics-table tr > :nth-child(n+4) { width: 12.25%; }
.image-metrics-table th, .image-metrics-table td { padding: 6px 4px; }
.image-metrics-table tbody th { font-size: 12px; overflow-wrap: anywhere; }
.image-metrics-table td.numeric { white-space: normal; overflow-wrap: anywhere; font-size: 12px; }
.image-metrics-table thead th { font-size: 11px; letter-spacing: 0.02em; }
.image-metrics-table thead th:nth-child(n+4) { text-align: right; }
.image-restoring-beam { display: block; font-size: 12px; font-variant-numeric: tabular-nums; }
.image-restoring-beam .image-beam-angle { display: block; }
.cycle-label { color: var(--teal-deep); font-size: 14px; font-weight: 700; font-style: normal; }
.cycle-duration { font-size: 14px; font-weight: 400; }
.data-table tr:last-child th, .data-table tr:last-child td { border-bottom: none; }
.data-table tbody tr:nth-child(even) td, .data-table tbody tr:nth-child(even) th { background: #fafcff; }
code, pre, .mono { font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", monospace; }
code { font-size: .9em; background: var(--code); padding: 2px 5px; border-radius: 3px; }
pre { max-width: 100%; margin: 0; padding: 12px 14px; overflow: auto; background: var(--code); border: 1px solid var(--line); border-radius: 6px; white-space: pre-wrap; overflow-wrap: anywhere; font-size: 12px; }
.file-list { margin: 0; padding: 0; list-style: none; border: 1px solid var(--line); border-radius: 6px; overflow: hidden; }
.file-list li { display: flex; justify-content: space-between; align-items: baseline; gap: 14px; padding: 8px 12px; border-bottom: 1px solid var(--line); background: var(--surface); }
.file-list li:last-child { border-bottom: none; }
.file-list li:nth-child(even) { background: #fafcff; }
.file-size { flex: 0 0 auto; color: var(--muted); font-size: 12px; }
.search-row { display: flex; align-items: center; gap: 12px; margin: 12px 0; }
.search-row label { color: var(--muted); font-size: 13px; font-weight: 500; }
.search-row input { width: min(440px, 100%); min-height: 36px; border: 1px solid #cbd5e1; border-radius: 4px; padding: 7px 11px; color: var(--ink); background: white; font: inherit; }
.search-row input:focus { outline: 2px solid var(--teal); outline-offset: 1px; }
.pill-group { display: flex; flex-wrap: wrap; gap: 6px; margin: 10px 0 14px; align-items: center; }
.pill-label { font-size: 12px; font-weight: 600; color: var(--muted); margin-right: 4px; }
.pill-btn { border: 1px solid #cbd5e1; background: var(--surface); color: var(--ink-secondary); padding: 4px 12px; border-radius: 16px; font-size: 12px; font-weight: 600; cursor: pointer; transition: all 0.15s ease; }
.pill-btn:hover { background: var(--surface-alt); border-color: var(--muted); }
.pill-btn.active { background: var(--teal-dark); color: white; border-color: var(--teal-dark); }
.gallery { display: grid; grid-template-columns: repeat(auto-fill, minmax(240px, 1fr)); gap: 14px; }
figure { min-width: 0; margin: 0; border: 1px solid var(--line); border-radius: 6px; background: var(--surface); overflow: hidden; box-shadow: 0 1px 2px rgba(15,23,42,0.03); }
figure a { display: block; background: #e2e8f0; }
figure img { display: block; width: 100%; height: 180px; object-fit: contain; background: #ffffff; }
figcaption { padding: 8px 10px; font-size: 12px; overflow-wrap: anywhere; border-top: 1px solid var(--line); }
figcaption .caption-detail { display: block; color: var(--muted); margin-top: 3px; font-size: 11px; }
.blink-tool { border: 1px solid var(--line); border-radius: 6px; background: var(--surface); padding: 12px; margin: 14px 0; }
.blink-controls { display: flex; flex-wrap: wrap; align-items: end; gap: 12px; padding-bottom: 12px; }
.blink-controls label { display: grid; gap: 4px; min-width: 0; color: var(--muted); font-size: 12px; font-weight: 500; }
.blink-rate { min-width: 190px; }
.blink-rate input { width: 150px; vertical-align: middle; }
.blink-rate output { margin-left: 5px; color: var(--ink); font-weight: 600; }
.blink-controls button { min-height: 36px; border: 1px solid var(--teal-dark); border-radius: 4px; padding: 6px 14px; color: white; background: var(--teal-dark); font: inherit; font-size: 13px; font-weight: 600; cursor: pointer; }
.blink-controls button:hover { background: var(--teal-deep); }
.overlay-toggle { align-self: center; color: var(--ink-secondary); font-size: 13px; font-weight: 600; cursor: pointer; }
.overlay-toggle input { vertical-align: middle; margin: 0 4px 0 0; }
.blink-preview { margin: 0; }
.blink-preview img { height: min(65vh, 640px); object-fit: contain; }
.compare-tool { border: 1px solid var(--line); border-radius: 6px; background: var(--surface); padding: 14px; margin: 14px 0; }
.compare-controls { display: flex; flex-wrap: wrap; gap: 16px; align-items: center; margin-bottom: 12px; }
.compare-controls label { display: grid; gap: 4px; font-size: 12px; font-weight: 600; color: var(--ink-secondary); }
.compare-controls select { padding: 6px 10px; border: 1px solid #cbd5e1; border-radius: 4px; background: white; font: inherit; font-size: 13px; min-width: 220px; }
.compare-views { display: grid; grid-template-columns: 1fr 1fr; gap: 14px; }
.compare-views figure img { height: min(55vh, 520px); object-fit: contain; }
@media (max-width: 768px) { .compare-views { grid-template-columns: 1fr; } }
.dataset-card { background: var(--surface); border: 1px solid var(--line); border-radius: 6px; margin: 16px 0; padding: 16px; box-shadow: 0 1px 3px rgba(15,23,42,0.03); }
.dataset-header { display: flex; justify-content: space-between; align-items: baseline; flex-wrap: wrap; gap: 8px; border-bottom: 1px solid var(--line); padding-bottom: 8px; margin-bottom: 12px; }
.dataset-role { display: inline-block; padding: 2px 8px; border: 1px solid var(--line); border-radius: 999px; color: var(--muted); font-size: 11px; font-weight: 650; white-space: nowrap; }
.dataset-title { font-size: 15px; font-weight: 700; color: var(--teal-deep); margin: 0; font-family: ui-monospace, monospace; overflow-wrap: anywhere; }
.dataset-meta-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); gap: 10px; margin-bottom: 14px; }
.dataset-meta-item { font-size: 12px; }
.dataset-meta-item strong { display: block; color: var(--muted); text-transform: uppercase; font-size: 10px; letter-spacing: 0.04em; }
.dataset-meta-item span { color: var(--ink); font-weight: 600; overflow-wrap: anywhere; }
.dataset-plots-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 14px; margin-top: 12px; }
.ms-quality-placeholder { box-sizing: border-box; min-height: 235px; display: flex; flex-direction: column; align-items: center; justify-content: center; gap: 6px; padding: 16px; border: 1px dashed var(--line); border-radius: 6px; background: var(--surface); color: var(--muted); text-align: center; font-size: 13px; }
.ms-quality-placeholder strong { color: var(--ink-secondary); }
.image-metrics-chart { max-width: 100%; margin: 12px 0; overflow-x: auto; border: 1px solid var(--line); border-radius: 6px; background: var(--surface); }
.image-metrics-chart svg { display: block; font-family: inherit; }
.metric-chart-grid { stroke: var(--line); stroke-width: 1; }
.metric-chart-axis { fill: var(--muted); font-size: 11px; }
.metric-chart-title { fill: var(--ink); font-size: 13px; font-weight: 700; }
.metric-chart-line { fill: none; stroke-width: 2; stroke-linecap: round; stroke-linejoin: round; }
.metric-chart-line-max-image { stroke: var(--teal-dark); }
.metric-chart-line-min-image { stroke: #2563eb; stroke-dasharray: 6 3; }
.metric-chart-line-rms-noise { stroke: #b45309; stroke-dasharray: 2 3; }
.metric-chart-line-dynamic-range { stroke: #be123c; stroke-dasharray: 8 3 2 3; }
.metric-chart-point { stroke: var(--surface); stroke-width: 1; }
.metric-chart-point-max-image { fill: var(--teal); }
.metric-chart-point-min-image { fill: #2563eb; }
.metric-chart-point-rms-noise { fill: #b45309; }
.metric-chart-point-dynamic-range { fill: #be123c; }
.image-metrics-note { margin: 8px 0 12px; color: var(--muted); font-size: 12px; }
.bandpass-controls { display: flex; flex-wrap: wrap; align-items: flex-start; gap: 12px; margin: 14px 0; padding: 12px; border: 1px solid var(--line); border-radius: 6px; background: var(--surface); }
.bandpass-controls > label { display: grid; gap: 4px; min-width: 180px; color: var(--muted); font-size: 12px; font-weight: 600; }
.bandpass-controls > label select { min-height: 36px; padding: 6px 10px; border: 1px solid #cbd5e1; border-radius: 4px; background: white; color: var(--ink); font: inherit; }
.bandpass-range-group { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 10px; flex: 1 1 640px; max-width: 760px; min-width: 0; }
.bandpass-range-controls { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); align-content: start; gap: 5px 8px; min-width: 220px; margin: 0; padding: 6px 10px 8px; border: 1px solid var(--line-light); border-radius: 4px; }
.bandpass-range-controls legend { padding: 0 4px; color: var(--ink-secondary); font-size: 11px; font-weight: 700; }
.bandpass-range-controls label { display: grid; gap: 3px; min-width: 0; color: var(--muted); font-size: 10px; font-weight: 600; }
.bandpass-range-controls input { box-sizing: border-box; width: 100%; min-height: 34px; padding: 5px 7px; border: 1px solid #cbd5e1; border-radius: 4px; background: white; color: var(--ink); font: inherit; font-variant-numeric: tabular-nums; }
.bandpass-range-controls input:focus-visible { outline: 2px solid var(--blue); outline-offset: 1px; }
.bandpass-range-hint, .bandpass-range-error { grid-column: 1 / -1; margin: 0; font-size: 10px; }
.bandpass-range-hint { color: var(--muted); }
.bandpass-range-error { color: #7f1d1d; font-size: 11px; font-weight: 600; }
.bandpass-summary { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 10px; margin: 14px 0 20px; }
.bandpass-stat { min-width: 0; padding: 10px 12px; border: 1px solid var(--line); border-radius: 6px; background: var(--surface); }
.bandpass-stat strong { display: block; color: var(--muted); font-size: 10px; text-transform: uppercase; letter-spacing: .04em; }
.bandpass-stat span { display: block; margin-top: 3px; color: var(--ink); font-size: 15px; font-weight: 650; overflow-wrap: anywhere; }
.bandpass-chart-card { margin: 14px 0; padding: 12px; border: 1px solid var(--line); border-radius: 6px; background: var(--surface); }
.bandpass-chart-card h2 { margin: 0 0 2px; font-size: 16px; }
.bandpass-chart-card p { margin: 0 0 8px; color: var(--muted); font-size: 12px; }
.bandpass-chart { display: block; width: 100%; height: auto; overflow: visible; font-family: inherit; }
.bandpass-grid { stroke: var(--line); stroke-width: 1; }
.bandpass-axis { fill: var(--muted); font-size: 11px; }
.bandpass-axis-title { fill: var(--ink-secondary); font-size: 12px; font-weight: 600; }
.bandpass-line { fill: none; stroke-width: 1.7; stroke-linecap: round; stroke-linejoin: round; }
.bandpass-empty { fill: var(--muted); font-size: 14px; }
.bandpass-note { margin: 8px 0 14px; color: var(--muted); font-size: 12px; }
.bandpass-error { color: #7f1d1d; }
.step-badge { display: inline-block; padding: 2px 7px; border-radius: 4px; font-size: 11px; font-weight: 600; background: var(--teal-light); color: var(--teal-deep); border: 1px solid var(--teal-border); white-space: nowrap; }
.workflow-step-grid { display: grid; gap: 5px; min-width: 0; }
.workflow-step-aggregates { display: flex; flex-wrap: wrap; align-items: center; gap: 5px; min-width: 0; }
.workflow-step-aggregate { cursor: default; }
.workflow-step-aggregate-count { font-size: 10px; font-variant-numeric: tabular-nums; }
.workflow-detail-disclosure { min-width: 0; margin-top: 5px; border-top: 1px solid var(--line-light); }
.workflow-detail-summary { display: flex; flex-wrap: wrap; align-items: center; gap: 4px 8px; min-width: 0; padding: 5px 0 0; list-style: none; color: var(--ink-secondary); font-size: 11px; cursor: pointer; }
.workflow-detail-summary::-webkit-details-marker { display: none; }
.workflow-detail-summary::before { content: "\25B8"; flex: 0 0 10px; color: var(--muted); font-size: 12px; }
.workflow-detail-disclosure[open] > .workflow-detail-summary::before { content: "\25BE"; }
.workflow-detail-summary:focus-visible { outline: 2px solid var(--blue); outline-offset: 2px; border-radius: 2px; }
.workflow-detail-summary-label { font-weight: 700; }
.workflow-detail-summary-count { color: var(--muted); font-size: 10px; }
.workflow-detail-summary-hint { margin-left: auto; color: var(--muted); font-size: 10px; }
.workflow-detail-disclosure[open] .workflow-detail-summary-hint { display: none; }
.workflow-detail-content { display: grid; gap: 5px; padding-top: 6px; }
.image-command-trigger { appearance: none; display: inline; max-width: 100%; padding: 0; border: 0; background: transparent; color: var(--ink); font: inherit; font-size: 12px; text-align: left; text-decoration: underline dotted; text-underline-offset: 2px; overflow-wrap: anywhere; cursor: pointer; }
.image-command-trigger:focus-visible { outline: 2px solid var(--blue); outline-offset: 2px; border-radius: 2px; }
.image-command-name { font-size: 12px; }
.image-command-unavailable { display: block; margin-top: 4px; color: var(--muted); font-size: 11px; }
.workflow-ms-list-heading { display: flex; justify-content: space-between; gap: 8px; margin-top: 5px; color: var(--muted); font-size: 10px; font-weight: 700; letter-spacing: .04em; text-transform: uppercase; }
.workflow-ms-list { display: grid; grid-template-columns: minmax(0, 1fr); gap: 4px; min-width: 0; }
.workflow-ms-row { display: grid; grid-template-columns: minmax(180px, 32%) minmax(0, 1fr); align-items: start; gap: 8px; min-width: 0; padding: 5px 6px; border-top: 1px solid var(--line-light); }
.workflow-ms-name-container { min-width: 0; }
.workflow-ms-name { display: block; width: 100%; min-width: 0; padding: 0; overflow: hidden; border: 0; background: transparent; color: var(--ink-secondary); font: 11px/1.5 ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace; text-align: left; text-overflow: ellipsis; white-space: nowrap; cursor: pointer; }
.workflow-ms-name:hover { color: var(--teal-dark); text-decoration: underline dotted; text-underline-offset: 2px; }
.workflow-ms-name:focus-visible { outline: 2px solid var(--blue); outline-offset: 2px; border-radius: 2px; }
.workflow-ms-name-popover { position: fixed; top: 50%; left: 50%; transform: translate(-50%, -50%); box-sizing: border-box; width: min(640px, calc(100vw - 32px)); max-width: calc(100vw - 32px); max-height: min(70vh, 520px); margin: 0; padding: 12px 14px; overflow: auto; border: 1px solid var(--line); border-radius: 6px; background: var(--surface); color: var(--ink); box-shadow: 0 6px 18px rgba(15,23,42,.16); }
.workflow-ms-name-popover dl { margin: 0; }
.workflow-ms-name-popover dt { margin-top: 9px; color: var(--muted); font-size: 10px; font-weight: 700; letter-spacing: .04em; text-transform: uppercase; }
.workflow-ms-name-popover dt:first-child { margin-top: 0; }
.workflow-ms-name-popover dd { margin: 2px 0 0; color: var(--ink); font: 12px/1.5 ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace; overflow-wrap: anywhere; }
.workflow-ms-name-popover code { font: inherit; white-space: pre-wrap; overflow-wrap: anywhere; }
.workflow-ms-badges, .workflow-step-chain { display: flex; flex-wrap: wrap; align-items: center; gap: 4px 5px; min-width: 0; }
.workflow-step-item { display: inline-flex; align-items: center; gap: 5px; min-width: 0; }
.workflow-step-badge { display: inline-flex; align-items: baseline; gap: 4px; max-width: 100%; padding: 2px 6px; border: 1px solid var(--line); border-radius: 4px; background: var(--surface-alt); color: var(--ink-secondary); font-size: 11px; font-weight: 600; line-height: 1.35; white-space: nowrap; }
.workflow-step-badge-imaging { border-color: var(--workflow-imaging); background: var(--workflow-imaging-pale); color: var(--workflow-imaging-ink); }
.workflow-step-badge-predict { border-color: var(--workflow-predict); background: var(--workflow-predict-pale); color: var(--workflow-predict-ink); }
.workflow-step-badge-solve { border-color: var(--workflow-solve); background: var(--workflow-solve-pale); color: var(--workflow-solve-ink); }
.workflow-step-badge-apply { border-color: var(--workflow-applycal); background: var(--workflow-applycal-pale); color: var(--workflow-applycal-ink); }
.workflow-step-badge-average { border-color: var(--workflow-average); background: var(--workflow-average-pale); color: var(--workflow-average-ink); }
.workflow-step-badge-phaseup { border-color: var(--workflow-phaseup); background: var(--workflow-phaseup-pale); color: var(--workflow-phaseup-ink); }
.workflow-step-badge-phaseshift { border-color: var(--workflow-phaseshift); background: var(--workflow-phaseshift-pale); color: var(--workflow-phaseshift-ink); }
.workflow-step-badge-filter { border-color: var(--workflow-filter); background: var(--workflow-filter-pale); color: var(--workflow-filter-ink); }
.workflow-step-badge-aoflagger { border-color: var(--workflow-aoflagger); background: var(--workflow-aoflagger-pale); color: var(--workflow-aoflagger-ink); }
.workflow-command-trigger { appearance: none; font-family: inherit; text-align: left; cursor: pointer; }
.workflow-command-trigger:focus-visible { outline: 2px solid var(--blue); outline-offset: 2px; }
.workflow-command-text { position: fixed; top: 50%; left: 50%; transform: translate(-50%, -50%); box-sizing: border-box; width: min(760px, calc(100vw - 32px)); max-width: calc(100vw - 32px); max-height: min(70vh, 640px); margin: 0; padding: 12px 14px; overflow: auto; border: 1px solid var(--line); border-radius: 6px; background: var(--surface); color: var(--ink); box-shadow: 0 6px 18px rgba(15,23,42,.16); font: 12px/1.5 ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", monospace; white-space: pre-wrap; overflow-wrap: anywhere; }
.workflow-step-duration { color: var(--ink-secondary); font-size: 10px; font-weight: 500; }
.workflow-step-arrow { color: var(--muted); font-size: 12px; }
.workflow-shared-steps { display: flex; align-items: start; gap: 8px; padding: 0 0 4px; }
.workflow-shared-label { flex: 0 0 100px; color: var(--muted); font-size: 10px; font-weight: 700; text-transform: uppercase; }
@media (max-width: 600px) { .workflow-detail-summary-hint { margin-left: 18px; } .workflow-ms-list-heading { flex-wrap: wrap; } .workflow-ms-row { grid-template-columns: minmax(0, 1fr); gap: 3px; } .workflow-shared-steps { flex-wrap: wrap; gap: 5px; } .workflow-shared-label, .data-table.progression-table .workflow-shared-label { flex-basis: 85px; } }
.resource-summary-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); gap: 12px; margin-bottom: 16px; }
.resource-phase-note { margin: 8px 0 0; color: var(--muted); font-size: 12px; }
.resource-card { background: var(--surface); border: 1px solid var(--line); border-radius: 6px; padding: 12px 14px; box-shadow: 0 1px 3px rgba(15,23,42,0.03); }
.resource-card strong { display: block; font-size: 11px; text-transform: uppercase; color: var(--muted); letter-spacing: .05em; margin-bottom: 4px; }
.resource-card span { font-size: 18px; font-weight: 700; color: var(--ink); }
.resource-live-frame { display: block; width: 100%; height: 142px; margin: 0 0 14px; border: 1px solid var(--line); border-radius: 6px; background: var(--paper); }
.resource-chart-live-frame { display: block; width: 100%; height: 360px; margin: 0 0 14px; border: 1px solid var(--line); border-radius: 6px; background: var(--paper); }
.log-tail-live-frame { display: block; width: 100%; height: 480px; border: 1px solid var(--line); border-radius: 6px; background: var(--paper); }
.resource-chart-box { background: var(--surface); border: 1px solid var(--line); border-radius: 6px; padding: 16px; margin-bottom: 16px; box-shadow: 0 1px 3px rgba(15,23,42,0.03); }
.resource-chart-legend { display: flex; flex-wrap: wrap; gap: 16px; margin-bottom: 12px; font-size: 12px; font-weight: 600; }
.resource-range-switch { display: flex; flex-wrap: wrap; align-items: center; gap: 7px; margin: 0 0 10px; }
.resource-range-label { margin-right: 2px; color: var(--muted); font-size: 12px; font-weight: 600; }
.resource-range-button { min-height: 30px; padding: 4px 12px; border: 1px solid #cbd5e1; border-radius: 16px; color: var(--ink-secondary); background: var(--surface); font: inherit; font-size: 12px; font-weight: 600; cursor: pointer; }
.resource-range-button:hover { border-color: var(--teal); background: var(--surface-alt); }
.resource-range-button[aria-pressed="true"] { border-color: var(--teal-dark); color: white; background: var(--teal-dark); }
.resource-range-button:focus-visible { outline: 2px solid var(--teal); outline-offset: 2px; }
.resource-range-count { display: inline-flex; align-items: center; gap: 5px; color: var(--ink-secondary); font-size: 12px; font-weight: 600; }
.resource-range-count input { width: 82px; min-height: 30px; padding: 4px 7px; border: 1px solid #cbd5e1; border-radius: 5px; color: var(--ink); background: var(--surface); font: inherit; font-weight: 400; }
.resource-range-count input[aria-invalid="true"] { border-color: var(--red); }
.resource-range-count input:focus-visible { outline: 2px solid var(--teal); outline-offset: 2px; }
.resource-range-count-status { flex-basis: 100%; margin: 0 0 2px; color: var(--muted); font-size: 11px; font-weight: 400; }
.resource-range-count-status:empty { display: none; }
.resource-chart-range-view[hidden] { display: none; }
.legend-item { display: inline-flex; align-items: center; gap: 6px; }
.legend-swatch { width: 14px; height: 4px; border-radius: 2px; display: inline-block; }
.chart-container { width: 100%; overflow-x: auto; }
details { margin: 11px 0; border: 1px solid var(--line); border-radius: 6px; background: var(--surface); overflow: hidden; }
details > summary { cursor: pointer; padding: 10px 14px; color: var(--teal-deep); font-weight: 600; font-size: 14px; background: #fafcff; }
details[open] > summary { border-bottom: 1px solid var(--line); }
details > :not(summary) { margin: 12px 14px; }
details > .data-table, details > .file-list { margin: 0 0 12px; }
.stack-metric-details { margin: 0; border: 0; border-radius: 0; background: transparent; overflow: visible; }
.stack-metric-details > summary { padding: 0; border-bottom: 0; color: var(--muted); background: transparent; font-size: 11px; }
.stack-metric-details[open] > summary { border-bottom: 0; }
.stack-metric-details > div { margin: 6px 0 0; text-align: left; }
.stack-metric-details span { display: block; }
.stack-metric-chart { margin: 16px 0; }
.stack-metric-chart h4 { margin: 0 0 8px; color: var(--teal-deep); font-size: 14px; }
.log-events { display: grid; gap: 8px; }
.log-event { border-left: 3px solid var(--amber); border: 1px solid var(--amber-border); border-left-width: 4px; border-radius: 4px; padding: 8px 12px; background: var(--amber-pale); font-size: 13px; overflow-wrap: anywhere; }
.log-event.error { border-color: var(--red-border); border-left-color: var(--red); background: var(--red-pale); }
.log-event small { display: block; color: var(--muted); margin-bottom: 3px; font-size: 11px; }
.page-links { display: flex; flex-wrap: wrap; gap: 9px; margin: 16px 0; }
.page-links a { padding: 8px 12px; border: 1px solid var(--line); border-radius: 4px; background: white; text-decoration: none; font-size: 13px; font-weight: 500; color: var(--teal-dark); box-shadow: 0 1px 2px rgba(15,23,42,0.03); }
.page-links a:hover { background: var(--surface-alt); color: var(--teal-deep); }
.empty { padding: 16px; color: var(--muted); background: var(--surface); border: 1px dashed #cbd5e1; border-radius: 6px; font-size: 13px; }
[hidden] { display: none !important; }
footer { border-top: 1px solid var(--line); padding: 18px 0 26px; color: var(--muted); font-size: 12px; }
@media (max-width: 850px) { .metrics { grid-template-columns: repeat(3, minmax(0, 1fr)); } }
@media (max-width: 640px) { .bandpass-range-group { grid-template-columns: minmax(0, 1fr); flex-basis: 100%; max-width: none; } }
@media (max-width: 600px) {
  .masthead, nav, main, footer { width: min(100% - 24px, 1320px); }
  .masthead { padding-top: 18px; }
  h1 { font-size: 24px; }
  main { padding-top: 20px; }
  .metrics { grid-template-columns: repeat(2, minmax(0, 1fr)); }
    .resource-live-frame { height: 196px; }
    .resource-chart-live-frame { height: 250px; }
  .metric { padding: 12px; }
  .metric-value { font-size: 20px; }
  .data-table th, .data-table td { padding: 7px; }
  .data-table th { width: 125px; }
  .section-heading { display: block; }
  .file-list li { display: block; }
  .file-size { display: block; margin-top: 2px; }
}
"""


def _report_css():
    marker = "  /* RESOURCE_PHASE_COLORS */"
    if marker not in _CSS:
        raise RuntimeError("Report stylesheet is missing its resource phase color marker")
    phase_colors = "\n".join(
        "  --workflow-{}: {};".format(phase, color)
        for phase, (_, color) in RESOURCE_PHASE_STYLES.items()
    )
    return _CSS.replace(marker, phase_colors, 1) + """
.workflow-step-outcome { display: block; font-size: 10px; font-weight: 500; }
.direction-table .direction-varies { background: var(--blue-pale); }
.dirmap { margin: 6px 0 0; }
.dirmap-controls { display: flex; flex-wrap: wrap; gap: 12px; align-items: end; margin-bottom: 6px; }
.dirmap-cycle-select { display: grid; gap: 4px; font-size: 12px; font-weight: 600; color: var(--ink-secondary); }
.dirmap-cycle-select select { padding: 6px 10px; border: 1px solid #cbd5e1; border-radius: 4px; background: white; font: inherit; font-size: 13px; min-width: 220px; }
.dirmap-legend { display: flex; flex-wrap: wrap; align-items: center; gap: 6px 14px; margin: 4px 0 8px; color: var(--muted); font-size: 12px; }
.dirmap-key { display: inline-block; width: 12px; height: 12px; margin-right: -8px; border-radius: 50%; vertical-align: middle; }
.dirmap-key-old { background: var(--teal-dark); }
.dirmap-key-new { background: var(--amber); }
.dirmap-key-facet { border-radius: 0; height: 0; border-top: 2px solid #94a3b8; }
.dirmap-figure { position: relative; max-width: 720px; }
.dirmap-svg { display: block; width: 100%; height: auto; font-family: inherit; }
.dirmap-frame { fill: var(--surface); stroke: var(--line); }
.dirmap-grid { fill: none; stroke: #cbd5e1; stroke-width: 0.8; }
.dirmap-axis { fill: var(--muted); font-size: 11px; }
.dirmap-axis-title { fill: var(--ink-secondary); font-size: 12px; font-weight: 600; }
.dirmap-cycle[hidden] { display: none; }
.dirmap-facet { fill: none; stroke: #94a3b8; stroke-width: 1.2; }
.dirmap-marker { cursor: pointer; outline: none; }
.dirmap-marker circle { fill: var(--teal-dark); fill-opacity: 0.75; stroke: white; stroke-width: 1.5; }
.dirmap-marker.dirmap-new circle { fill: var(--amber); }
.dirmap-marker text { fill: var(--ink); font-size: 12px; font-weight: 700; paint-order: stroke; stroke: white; stroke-width: 3px; }
.dirmap-marker:hover circle, .dirmap-marker:focus circle, .dirmap-marker.dirmap-selected circle { stroke: var(--ink); stroke-width: 2.5; fill-opacity: 1; }
.dirmap-popup { position: absolute; z-index: 5; width: min(300px, 90%); padding: 10px 12px; border: 1px solid var(--line); border-radius: 6px; background: var(--surface); box-shadow: 0 6px 18px rgba(15,23,42,0.18); font-size: 12px; }
.dirmap-popup h3 { margin: 0 24px 6px 0; font-size: 15px; }
.dirmap-popup table { width: 100%; border-collapse: collapse; }
.dirmap-popup th { padding: 2px 8px 2px 0; color: var(--muted); font-weight: 600; text-align: left; vertical-align: top; white-space: nowrap; }
.dirmap-popup td { padding: 2px 0; overflow-wrap: anywhere; }
.dirmap-popup ul { margin: 6px 0 0; padding-left: 16px; }
.dirmap-popup-close { position: absolute; top: 6px; right: 8px; border: 0; background: none; color: var(--muted); font-size: 18px; line-height: 1; cursor: pointer; }
.gallery-group-heading { grid-column: 1 / -1; margin: 12px 0 0; padding-bottom: 4px; border-bottom: 1px solid var(--line); font-size: 15px; }
.direction-layout { grid-template-columns: minmax(0, 720px); }
.workflow-step-failed { color: var(--red); border: 1px solid var(--red-border); background: var(--red-pale); }
"""


_JS = r"""
window.addEventListener("message", function (event) {
    var frame = Array.prototype.find.call(
        document.querySelectorAll(".resource-chart-live-frame"),
        function (candidate) { return event.source === candidate.contentWindow; }
    );
    if (!frame) return;
    var data = event.data;
    if (!data) return;
    if (data.type === "facetselfcal-resource-chart-window") {
        // the full-range frame does not describe the recent-sample window
        if (data.fullRange) return;
        var container = frame.closest("[data-resource-range-container]");
        if (!container) return;
        var countInput = container.querySelector("[data-resource-sample-count]");
        var recentButton = container.querySelector(
            '[data-resource-range-button="recent"]'
        );
        var requestedCount = Number(data.requestedCount);
        var displayedCount = Number(data.displayedCount);
        var availableCount = Number(data.availableCount);
        if (
            !countInput ||
            requestedCount !== Number(countInput.value) ||
            !Number.isSafeInteger(displayedCount) ||
            displayedCount < 1 ||
            !Number.isSafeInteger(availableCount) ||
            availableCount < 1
        ) {
            return;
        }
        if (recentButton) {
            recentButton.textContent = "Latest " + displayedCount + " samples";
        }
        var countStatus = container.querySelector(
            "[data-resource-sample-count-status]"
        );
        if (countStatus) {
            countStatus.textContent =
                requestedCount > availableCount
                    ? "The live chart currently has only " +
                      availableCount +
                      " samples; showing all " +
                      availableCount +
                      "."
                    : "";
        }
        return;
    }
    if (data.type !== "facetselfcal-resource-chart-size") return;
    if (typeof data.height !== "number" || !Number.isFinite(data.height)) return;
    frame.style.height = Math.ceil(data.height) + "px";
});

document.addEventListener("DOMContentLoaded", function () {
    document.querySelectorAll(".resource-chart-live-frame").forEach(function (liveChartFrame) {
        var requestLiveChartSize = function () {
            liveChartFrame.contentWindow.postMessage(
                { type: "facetselfcal-resource-chart-size-request" },
                "*"
            );
        };
        liveChartFrame.addEventListener("load", requestLiveChartSize);
        requestLiveChartSize();
    });

    document.querySelectorAll("[data-resource-range-container]").forEach(function (container) {
        var buttons = container.querySelectorAll("[data-resource-range-button]");
        var views = container.querySelectorAll("[data-resource-range-view]");
        var countInput = container.querySelector("[data-resource-sample-count]");
        var countStatus = container.querySelector("[data-resource-sample-count-status]");
        var recentButton = container.querySelector(
            '[data-resource-range-button="recent"]'
        );
        var dataElement = container.querySelector("[data-resource-chart-data]");
        var chartData = null;
        if (dataElement) {
            try {
                chartData = window.facetselfcalDecodeResourceData(
                    JSON.parse(dataElement.textContent || "{}")
                );
            } catch (error) {
                if (countStatus) {
                    countStatus.textContent =
                        "Resource chart data could not be read.";
                }
                console.error("Could not parse report resource chart data.", error);
            }
        }
        var applySampleCount = function (updateLiveFrame) {
            if (!countInput || !recentButton) return;
            var requestedCount = Number(countInput.value);
            if (!Number.isSafeInteger(requestedCount) || requestedCount < 1) {
                countInput.setAttribute("aria-invalid", "true");
                countInput.setCustomValidity(
                    "Enter a positive whole number of samples."
                );
                if (countStatus) {
                    countStatus.textContent =
                        "Enter a positive whole number of samples.";
                }
                return;
            }
            if (!chartData || !Array.isArray(chartData.samples)) {
                countInput.setAttribute("aria-invalid", "true");
                if (countStatus) {
                    countStatus.textContent =
                        "Resource chart data is unavailable.";
                }
                return;
            }
            var availableCount = chartData.samples.length;
            if (!availableCount) {
                countInput.setAttribute("aria-invalid", "true");
                if (countStatus) {
                    countStatus.textContent =
                        "No resource samples are available.";
                }
                return;
            }
            countInput.removeAttribute("aria-invalid");
            countInput.setCustomValidity("");
            var effectiveCount = Math.min(requestedCount, availableCount);
            recentButton.textContent =
                "Latest " + effectiveCount + " samples";
            if (countStatus) {
                countStatus.textContent =
                    requestedCount > availableCount
                        ? "Only " +
                          availableCount +
                          " samples are available; showing all " +
                          availableCount +
                          "."
                        : "";
            }
            var recentTarget = container.querySelector(
                '[data-resource-range-view="recent"] [data-resource-chart-target]'
            );
            if (recentTarget) {
                if (typeof window.facetselfcalRenderResourceWindow !== "function") {
                    countInput.setAttribute("aria-invalid", "true");
                    if (countStatus) {
                        countStatus.textContent =
                            "The resource chart renderer is unavailable.";
                    }
                    return;
                }
                var renderedCount = window.facetselfcalRenderResourceWindow(
                    recentTarget,
                    chartData,
                    requestedCount
                );
                if (!renderedCount) {
                    countInput.setAttribute("aria-invalid", "true");
                    if (countStatus) {
                        countStatus.textContent =
                            "The recent resource chart could not be rendered.";
                    }
                    return;
                }
            }
            if (updateLiveFrame) {
                var liveFrame = container.querySelector(
                    '[data-resource-range-view="recent"] .resource-chart-live-frame'
                );
                if (liveFrame) {
                    var frameUrl = new URL(
                        liveFrame.getAttribute("src"),
                        document.baseURI
                    );
                    frameUrl.searchParams.set(
                        "sample_limit",
                        String(requestedCount)
                    );
                    if (liveFrame.src !== frameUrl.href) {
                        liveFrame.src = frameUrl.href;
                    }
                }
            }
        };
        if (countInput) {
            countInput.addEventListener("change", function () {
                applySampleCount(true);
            });
            applySampleCount(false);
        }
        buttons.forEach(function (button) {
            button.addEventListener("click", function () {
                var selectedRange = button.getAttribute("data-resource-range-button");
                var selectedView = null;
                buttons.forEach(function (candidate) {
                    var isSelected = candidate === button;
                    candidate.setAttribute("aria-pressed", isSelected ? "true" : "false");
                });
                views.forEach(function (view) {
                    var isSelected =
                        view.getAttribute("data-resource-range-view") === selectedRange;
                    view.hidden = !isSelected;
                    if (isSelected) selectedView = view;
                });
                // the status describes the recent-sample window only
                if (countStatus) countStatus.hidden = selectedRange !== "recent";
                var liveFrame = selectedView
                    ? selectedView.querySelector(".resource-chart-live-frame")
                    : null;
                if (liveFrame && liveFrame.contentWindow) {
                    liveFrame.contentWindow.postMessage(
                        { type: "facetselfcal-resource-chart-size-request" },
                        "*"
                    );
                }
            });
        });
    });

    document.querySelectorAll("[data-blink-tool]").forEach(function (tool) {
        var frames = JSON.parse(tool.getAttribute("data-blink-images") || "[]");
        var interval = tool.querySelector("[data-blink-interval]");
        var intervalOutput = tool.querySelector("[data-blink-interval-output]");
        var toggle = tool.querySelector("[data-blink-toggle]");
        var image = tool.querySelector("[data-blink-image]");
        var caption = tool.querySelector("[data-blink-caption]");
        var overlayToggle = tool.querySelector("[data-overlay-toggle]");
        var timer = null;
        var frameIndex = 0;
        var render = function () {
            var frame = frames[frameIndex];
            if (!frame) return;
            var showOverlays = !overlayToggle || overlayToggle.checked;
            image.src = !showOverlays && frame.plain ? frame.plain : frame.src;
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
        if (overlayToggle) overlayToggle.addEventListener("change", render);
        render();
    });

    document.querySelectorAll("[data-compare-tool]").forEach(function (tool) {
        var selectA = tool.querySelector("[data-compare-select-a]");
        var selectB = tool.querySelector("[data-compare-select-b]");
        var imgA = tool.querySelector("[data-compare-img-a]");
        var imgB = tool.querySelector("[data-compare-img-b]");
        var linkA = tool.querySelector("[data-compare-link-a]");
        var linkB = tool.querySelector("[data-compare-link-b]");
        var capA = tool.querySelector("[data-compare-caption-a]");
        var capB = tool.querySelector("[data-compare-caption-b]");
        var overlayToggle = tool.querySelector("[data-overlay-toggle]");
        var source = function (option) {
            var showOverlays = !overlayToggle || overlayToggle.checked;
            var plain = option.getAttribute("data-plain");
            return !showOverlays && plain ? plain : option.value;
        };
        var update = function () {
            if (selectA && imgA) {
                var optA = selectA.options[selectA.selectedIndex];
                if (optA) {
                    imgA.src = source(optA);
                    imgA.alt = optA.text;
                    if (linkA) linkA.href = source(optA);
                    if (capA) capA.textContent = optA.text;
                }
            }
            if (selectB && imgB) {
                var optB = selectB.options[selectB.selectedIndex];
                if (optB) {
                    imgB.src = source(optB);
                    imgB.alt = optB.text;
                    if (linkB) linkB.href = source(optB);
                    if (capB) capB.textContent = optB.text;
                }
            }
        };
        if (selectA) selectA.addEventListener("change", update);
        if (selectB) selectB.addEventListener("change", update);
        if (overlayToggle) overlayToggle.addEventListener("change", update);
        update();
    });

    document.querySelectorAll("[data-filter-target]").forEach(function (input) {
        var selector = input.getAttribute("data-filter-target");
        var container = input.closest("section") || input.closest("main");
        if (!container) return;
        var items = Array.from(container.querySelectorAll(selector));
        var groups = Array.from(container.querySelectorAll("details[data-filter-group]"));
        var buttons = Array.from(container.querySelectorAll(
            '[data-pill-target="' + selector + '"] .pill-btn'
        ));
        var selectedFilter = "";
        var update = function () {
            var query = input.value.trim().toLowerCase();
            items.forEach(function (item) {
                var text = item.getAttribute("data-search") || item.textContent || "";
                var tags = (item.getAttribute("data-filter-values") || "").split(/\s+/);
                item.hidden = (query.length > 0 && !text.toLowerCase().includes(query)) ||
                    (selectedFilter !== "" && !tags.includes(selectedFilter));
            });
            groups.forEach(function (group) {
                var members = Array.from(group.querySelectorAll(selector));
                if (!members.length) return;
                var visible = members.some(function (item) {
                    return !item.hidden;
                });
                group.hidden = !visible;
                if ((query || selectedFilter) && visible) group.open = true;
                if ((query || selectedFilter) && visible) {
                    members.forEach(function (item) {
                        if (item.hidden) return;
                        var parent = item.closest("details");
                        while (parent && container.contains(parent)) {
                            parent.open = true;
                            parent = parent.parentElement.closest("details");
                        }
                    });
                }
            });
        };
        buttons.forEach(function (button) {
            button.setAttribute("aria-pressed", button.classList.contains("active") ? "true" : "false");
            button.addEventListener("click", function () {
                var value = button.getAttribute("data-filter-value") || "";
                selectedFilter = value === "all" ? "" : value;
                buttons.forEach(function (candidate) {
                    candidate.classList.toggle("active", candidate === button);
                    candidate.setAttribute("aria-pressed", candidate === button ? "true" : "false");
                });
                update();
            });
        });
        input.addEventListener("input", update);
    });

    var bandpassDataNode = document.getElementById("bandpass-data");
    if (!bandpassDataNode) return;

    var bandpassData = JSON.parse(bandpassDataNode.textContent || "{}");
    var antennaSelect = document.getElementById("bandpass-antenna");
    var referenceAntennaSelect = document.getElementById("bandpass-reference-antenna");
    var polarizationSelect = document.getElementById("bandpass-polarization");
    var directionSelect = document.getElementById("bandpass-direction");
    var directionLabel = document.getElementById("bandpass-direction-label");
    var amplitudeChart = document.getElementById("bandpass-amplitude");
    var phaseChart = document.getElementById("bandpass-phase");
    var amplitudeRangeControls = {
        minimum: document.getElementById("bandpass-amplitude-y-min"),
        maximum: document.getElementById("bandpass-amplitude-y-max"),
        error: document.getElementById("bandpass-amplitude-range-error")
    };
    var phaseRangeControls = {
        minimum: document.getElementById("bandpass-phase-y-min"),
        maximum: document.getElementById("bandpass-phase-y-max"),
        error: document.getElementById("bandpass-phase-range-error")
    };
    var palette = ["#0f766e", "#2563eb", "#c2410c", "#7c3aed", "#be123c", "#4d7c0f"];
    var svgNamespace = "http://www.w3.org/2000/svg";

    var union = function (items) {
        var seen = new Set();
        return items.reduce(function (result, values) {
            values.forEach(function (value) {
                if (!seen.has(value)) {
                    seen.add(value);
                    result.push(value);
                }
            });
            return result;
        }, []);
    };
    var addOption = function (select, value, label) {
        var option = document.createElement("option");
        option.value = value;
        option.textContent = label;
        select.appendChild(option);
    };
    var appendSvg = function (svg, tag, attributes, text) {
        var element = document.createElementNS(svgNamespace, tag);
        Object.keys(attributes || {}).forEach(function (name) {
            element.setAttribute(name, attributes[name]);
        });
        if (text !== undefined) element.textContent = text;
        svg.appendChild(element);
        return element;
    };
    var formatTick = function (value) {
        return Number(value.toPrecision(4)).toString();
    };
    var setRangeError = function (controls, message) {
        controls.error.textContent = message;
        controls.error.hidden = !message;
        controls.minimum.setAttribute("aria-invalid", message ? "true" : "false");
        controls.maximum.setAttribute("aria-invalid", message ? "true" : "false");
    };
    var drawBandpassChart = function (
        svg, solution, metric, antenna, direction, polarizations, referenceAntenna,
        rangeControls
    ) {
        while (svg.firstChild) svg.removeChild(svg.firstChild);
        setRangeError(rangeControls, "");
        var width = 940;
        var height = 330;
        var margin = { left: 76, right: 22, top: 30, bottom: 58 };
        var plotWidth = width - margin.left - margin.right;
        var plotHeight = height - margin.top - margin.bottom;
        svg.setAttribute("viewBox", "0 0 " + width + " " + height);
        svg.setAttribute("role", "img");
        svg.setAttribute(
            "aria-label",
            (metric === "amplitude" ? "Amplitude" : "Phase") +
                " bandpass for antenna " + antenna + " in direction " + direction +
                (metric === "phase"
                    ? " relative to reference antenna " + referenceAntenna
                    : "")
        );

        if (!solution || !solution.series[direction]) {
            appendSvg(svg, "text", {
                x: width / 2,
                y: height / 2,
                "text-anchor": "middle",
                class: "bandpass-empty"
            }, "No " + metric + " solutions are available.");
            return;
        }

        var antennaData = solution.series[direction][antenna];
        var referenceData = referenceAntenna
            ? solution.series[direction][referenceAntenna]
            : null;
        var seriesList = polarizations.map(function (polarization) {
            var source = antennaData && antennaData[polarization];
            var values = source || [];
            var referenceValues = metric === "phase" && referenceData
                ? (referenceData[polarization] || [])
                : [];
            return {
                polarization: polarization,
                color: palette[Math.max(0, polarizationNames.indexOf(polarization)) % palette.length],
                points: solution.frequencies_mhz.map(function (frequency, channel) {
                    var value = values[channel];
                    if (value === null || value === undefined || !Number.isFinite(value)) {
                        return null;
                    }
                    if (metric === "phase") {
                        var referenceValue = referenceValues[channel];
                        if (
                            referenceValue === null ||
                            referenceValue === undefined ||
                            !Number.isFinite(referenceValue)
                        ) {
                            return null;
                        }
                        var phaseDifference = value - referenceValue;
                        value = Math.atan2(
                            Math.sin(phaseDifference),
                            Math.cos(phaseDifference)
                        );
                    }
                    return {
                        frequency: frequency,
                        value: metric === "phase" ? value * 180 / Math.PI : value
                    };
                })
            };
        }).filter(function (series) {
            return series.points.some(function (point) { return point !== null; });
        });

        if (!seriesList.length || !solution.frequencies_mhz.length) {
            appendSvg(svg, "text", {
                x: width / 2,
                y: height / 2,
                "text-anchor": "middle",
                class: "bandpass-empty"
            }, "No valid " + metric + " samples for this selection.");
            return;
        }

        var frequencies = solution.frequencies_mhz.filter(Number.isFinite);
        var xMin = frequencies.reduce(function (minimum, value) {
            return Math.min(minimum, value);
        }, Infinity);
        var xMax = frequencies.reduce(function (maximum, value) {
            return Math.max(maximum, value);
        }, -Infinity);
        if (xMin === xMax) {
            xMin -= 0.5;
            xMax += 0.5;
        }
        var finiteValues = [];
        seriesList.forEach(function (series) {
            series.points.forEach(function (point) {
                if (point) finiteValues.push(point.value);
            });
        });
        var yMin = metric === "phase" ? -180 : finiteValues.reduce(function (minimum, value) {
            return Math.min(minimum, value);
        }, Infinity);
        var yMax = metric === "phase" ? 180 : finiteValues.reduce(function (maximum, value) {
            return Math.max(maximum, value);
        }, -Infinity);
        if (metric === "amplitude") {
            var padding = (yMax - yMin) * 0.08 || Math.max(Math.abs(yMax) * 0.05, 0.05);
            yMin -= padding;
            yMax += padding;
            if (yMin === yMax) yMax = yMin + 1;
        }
        var minimumInput = rangeControls.minimum;
        var maximumInput = rangeControls.maximum;
        var minimumText = minimumInput.value.trim();
        var maximumText = maximumInput.value.trim();
        var requestedMinimum = minimumText === "" ? null : Number(minimumText);
        var requestedMaximum = maximumText === "" ? null : Number(maximumText);
        var rangeError = "";
        if (
            minimumInput.validity.badInput ||
            maximumInput.validity.badInput ||
            (requestedMinimum !== null && !Number.isFinite(requestedMinimum)) ||
            (requestedMaximum !== null && !Number.isFinite(requestedMaximum))
        ) {
            rangeError = "Enter valid numeric y-axis limits.";
        } else {
            var selectedMinimum = requestedMinimum === null ? yMin : requestedMinimum;
            var selectedMaximum = requestedMaximum === null ? yMax : requestedMaximum;
            if (
                !Number.isFinite(selectedMinimum) ||
                !Number.isFinite(selectedMaximum) ||
                selectedMinimum >= selectedMaximum
            ) {
                rangeError = "Minimum must be less than maximum for the current selection.";
            } else {
                yMin = selectedMinimum;
                yMax = selectedMaximum;
            }
        }
        if (rangeError) {
            setRangeError(rangeControls, rangeError);
            appendSvg(svg, "text", {
                x: width / 2,
                y: height / 2,
                "text-anchor": "middle",
                class: "bandpass-empty"
            }, "Invalid y-axis range. Check the minimum and maximum values.");
            return;
        }
        var x = function (frequency) {
            return margin.left + (frequency - xMin) / (xMax - xMin) * plotWidth;
        };
        var y = function (value) {
            return margin.top + (yMax - value) / (yMax - yMin) * plotHeight;
        };
        var plotClipId = svg.id + "-plot-clip";
        var clipPath = appendSvg(
            appendSvg(svg, "defs"),
            "clipPath",
            { id: plotClipId, clipPathUnits: "userSpaceOnUse" }
        );
        appendSvg(clipPath, "rect", {
            x: margin.left,
            y: margin.top,
            width: plotWidth,
            height: plotHeight
        });
        var plotClip = "url(#" + plotClipId + ")";

        for (var tickIndex = 0; tickIndex <= 4; tickIndex += 1) {
            var fraction = tickIndex / 4;
            var frequencyTick = xMin + fraction * (xMax - xMin);
            appendSvg(svg, "line", {
                x1: x(frequencyTick), y1: margin.top,
                x2: x(frequencyTick), y2: margin.top + plotHeight,
                class: "bandpass-grid"
            });
            appendSvg(svg, "text", {
                x: x(frequencyTick), y: margin.top + plotHeight + 22,
                "text-anchor": "middle", class: "bandpass-axis"
            }, formatTick(frequencyTick));

            var valueTick = yMax - fraction * (yMax - yMin);
            appendSvg(svg, "line", {
                x1: margin.left, y1: y(valueTick),
                x2: margin.left + plotWidth, y2: y(valueTick),
                class: "bandpass-grid"
            });
            appendSvg(svg, "text", {
                x: margin.left - 10, y: y(valueTick),
                "text-anchor": "end", "dominant-baseline": "middle",
                class: "bandpass-axis"
            }, formatTick(valueTick));
        }

        seriesList.forEach(function (series, seriesIndex) {
            var segment = [];
            var previousValue = null;
            var flushSegment = function () {
                if (segment.length === 1) {
                    appendSvg(svg, "circle", {
                        cx: segment[0][0], cy: segment[0][1], r: 2.5,
                        fill: series.color, "clip-path": plotClip
                    });
                } else if (segment.length > 1) {
                    var path = segment.map(function (point, pointIndex) {
                        return (pointIndex ? "L" : "M") + point[0].toFixed(2) + " " +
                            point[1].toFixed(2);
                    }).join(" ");
                    appendSvg(svg, "path", {
                        d: path,
                        stroke: series.color,
                        class: "bandpass-line",
                        "clip-path": plotClip
                    });
                }
                segment = [];
            };
            series.points.forEach(function (point) {
                if (!point) {
                    flushSegment();
                    previousValue = null;
                    return;
                }
                if (
                    metric === "phase" &&
                    previousValue !== null &&
                    Math.abs(point.value - previousValue) > 180
                ) {
                    flushSegment();
                }
                segment.push([x(point.frequency), y(point.value)]);
                previousValue = point.value;
            });
            flushSegment();

            var legendX = margin.left + seriesIndex * 112;
            appendSvg(svg, "line", {
                x1: legendX, y1: 14, x2: legendX + 20, y2: 14,
                stroke: series.color, class: "bandpass-line"
            });
            appendSvg(svg, "text", {
                x: legendX + 26, y: 18, class: "bandpass-axis"
            }, series.polarization);
        });

        appendSvg(svg, "text", {
            x: margin.left + plotWidth / 2,
            y: height - 8,
            "text-anchor": "middle",
            class: "bandpass-axis-title"
        }, "Frequency (MHz)");
        appendSvg(svg, "text", {
            x: 18,
            y: margin.top + plotHeight / 2,
            transform: "rotate(-90 18 " + (margin.top + plotHeight / 2) + ")",
            "text-anchor": "middle",
            class: "bandpass-axis-title"
        }, metric === "amplitude" ? "Gain amplitude" : "Relative phase (degrees)");
    };

    var antennaNames = union([
        bandpassData.amplitude ? bandpassData.amplitude.antennas : [],
        bandpassData.phase ? bandpassData.phase.antennas : []
    ]);
    var polarizationNames = union([
        bandpassData.amplitude ? bandpassData.amplitude.polarizations : [],
        bandpassData.phase ? bandpassData.phase.polarizations : []
    ]);
    var referenceAntennaNames = bandpassData.phase
        ? bandpassData.phase.antennas
        : [];
    var directionNames = union([
        bandpassData.amplitude ? bandpassData.amplitude.directions : [],
        bandpassData.phase ? bandpassData.phase.directions : []
    ]);
    antennaNames.forEach(function (name) { addOption(antennaSelect, name, name); });
    referenceAntennaNames.forEach(function (name) {
        addOption(referenceAntennaSelect, name, name);
    });
    if (!referenceAntennaNames.length) {
        referenceAntennaSelect.parentElement.hidden = true;
    }
    if (polarizationNames.length > 1) {
        addOption(polarizationSelect, "__all__", "All polarizations");
    }
    polarizationNames.forEach(function (name) { addOption(polarizationSelect, name, name); });
    directionNames.forEach(function (name) { addOption(directionSelect, name, name); });
    if (directionNames.length < 2) directionLabel.hidden = true;
    if (polarizationNames.length < 2) polarizationSelect.parentElement.hidden = true;

    var renderBandpass = function () {
        if (!antennaNames.length || !directionNames.length) return;
        var selectedPol = polarizationSelect.value;
        var selectedPolarizations = selectedPol === "__all__"
            ? polarizationNames
            : [selectedPol];
        drawBandpassChart(
            amplitudeChart, bandpassData.amplitude, "amplitude",
            antennaSelect.value, directionSelect.value, selectedPolarizations,
            null, amplitudeRangeControls
        );
        drawBandpassChart(
            phaseChart, bandpassData.phase, "phase",
            antennaSelect.value, directionSelect.value, selectedPolarizations,
            referenceAntennaSelect.value, phaseRangeControls
        );
    };
    antennaSelect.addEventListener("change", renderBandpass);
    referenceAntennaSelect.addEventListener("change", renderBandpass);
    polarizationSelect.addEventListener("change", renderBandpass);
    directionSelect.addEventListener("change", renderBandpass);
    amplitudeRangeControls.minimum.addEventListener("input", renderBandpass);
    amplitudeRangeControls.maximum.addEventListener("input", renderBandpass);
    phaseRangeControls.minimum.addEventListener("input", renderBandpass);
    phaseRangeControls.maximum.addEventListener("input", renderBandpass);
    renderBandpass();
});
document.addEventListener("DOMContentLoaded", function () {
    // Reorder solution plots by cycle, direction or solution type, with a
    // heading per group. "#sort=cycle&filter=dir-03" preselects both.
    var orders = {
        cycle: ["sortCycle", "sortDirection", "sortSoltype", "sortName"],
        direction: ["sortDirection", "sortCycle", "sortSoltype", "sortName"],
        soltype: ["sortSoltype", "sortCycle", "sortDirection", "sortName"]
    };
    var labels = {
        cycle: function (item) {
            var cycle = item.dataset.sortCycle;
            return cycle === "other" ? "Other plots" : "Cycle " + cycle;
        },
        direction: function (item) {
            return item.dataset.sortDirection ? "Dir" + item.dataset.sortDirection : "No direction";
        },
        soltype: function (item) { return item.dataset.sortSoltype; }
    };
    document.querySelectorAll("[data-sort-controls]").forEach(function (controls) {
        var gallery = document.querySelector(controls.getAttribute("data-sort-controls"));
        if (!gallery) return;
        var items = Array.from(gallery.querySelectorAll("[data-sort-name]"));
        var headings = [];
        var updateHeadings = function () {
            headings.forEach(function (heading) {
                var hidden = !heading.groupItems.some(function (item) { return !item.hidden; });
                if (heading.hidden !== hidden) heading.hidden = hidden;
            });
        };
        var apply = function (key) {
            if (!orders[key]) key = controls.getAttribute("data-sort-default");
            var fields = orders[key];
            headings.forEach(function (heading) { heading.remove(); });
            headings = [];
            items.sort(function (a, b) {
                for (var i = 0; i < fields.length; i++) {
                    var x = a.dataset[fields[i]] || "\uffff";
                    var y = b.dataset[fields[i]] || "\uffff";
                    if (x < y) return -1;
                    if (x > y) return 1;
                }
                return 0;
            });
            var currentLabel = null;
            var heading = null;
            items.forEach(function (item) {
                var label = labels[key](item);
                if (label !== currentLabel) {
                    heading = document.createElement("h3");
                    heading.className = "gallery-group-heading";
                    heading.textContent = label;
                    heading.groupItems = [];
                    gallery.appendChild(heading);
                    headings.push(heading);
                    currentLabel = label;
                }
                heading.groupItems.push(item);
                gallery.appendChild(item);
            });
            controls.querySelectorAll("[data-sort-key]").forEach(function (button) {
                var active = button.getAttribute("data-sort-key") === key;
                button.classList.toggle("active", active);
                button.setAttribute("aria-pressed", active ? "true" : "false");
            });
            updateHeadings();
        };
        new MutationObserver(updateHeadings).observe(gallery, {
            subtree: true, attributes: true, attributeFilter: ["hidden"]
        });
        controls.querySelectorAll("[data-sort-key]").forEach(function (button) {
            button.addEventListener("click", function () {
                apply(button.getAttribute("data-sort-key"));
            });
        });
        var params = new URLSearchParams(window.location.hash.slice(1));
        apply(params.get("sort"));
        var filter = params.get("filter");
        if (filter) {
            var section = gallery.closest("main") || document;
            var filterButton = Array.from(section.querySelectorAll(".pill-btn[data-filter-value]"))
                .find(function (button) { return button.getAttribute("data-filter-value") === filter; });
            if (filterButton) filterButton.click();
        }
    });
});
document.addEventListener("DOMContentLoaded", function () {
    // Direction map: cycle selector and a pop-up with the details of the
    // clicked direction.
    document.querySelectorAll("[data-dirmap]").forEach(function (map) {
        var data = JSON.parse(map.getAttribute("data-dirmap-directions") || "{}");
        var select = map.querySelector("[data-dirmap-cycle]");
        var popup = map.querySelector("[data-dirmap-popup]");
        var figure = map.querySelector(".dirmap-figure");
        var selected = null;
        var currentCycle = function () {
            var visible = map.querySelector(".dirmap-cycle:not([hidden])");
            return visible ? visible.getAttribute("data-cycle") : null;
        };
        var close = function () {
            popup.hidden = true;
            if (selected) selected.classList.remove("dirmap-selected");
            selected = null;
        };
        var row = function (label, value) {
            var tr = document.createElement("tr");
            var th = document.createElement("th");
            var td = document.createElement("td");
            th.textContent = label;
            td.textContent = value;
            tr.appendChild(th);
            tr.appendChild(td);
            return tr;
        };
        var open = function (marker) {
            var records = data[currentCycle()] || [];
            var record = records[Number(marker.getAttribute("data-index"))];
            if (!record) return;
            if (selected) selected.classList.remove("dirmap-selected");
            selected = marker;
            marker.classList.add("dirmap-selected");
            popup.textContent = "";
            var closeButton = document.createElement("button");
            closeButton.type = "button";
            closeButton.className = "dirmap-popup-close";
            closeButton.setAttribute("aria-label", "Close");
            closeButton.textContent = "\u00d7";
            closeButton.addEventListener("click", close);
            popup.appendChild(closeButton);
            var heading = document.createElement("h3");
            heading.textContent = record.name;
            popup.appendChild(heading);
            var table = document.createElement("table");
            table.appendChild(row("RA", record.ra));
            table.appendChild(row("Dec", record.dec));
            table.appendChild(row("From phase centre", record.distance));
            if (record.added) table.appendChild(row("Added", record.added));
            if (record.aflux) table.appendChild(row("Apparent flux", record.aflux));
            if (record.tflux) table.appendChild(row("Total flux", record.tflux));
            record.settings.forEach(function (setting) {
                table.appendChild(row(setting[0], setting[1]));
            });
            popup.appendChild(table);
            if (record.links.length) {
                var list = document.createElement("ul");
                record.links.forEach(function (link) {
                    var item = document.createElement("li");
                    var anchor = document.createElement("a");
                    anchor.href = link[1];
                    anchor.textContent = record.links.length > 1
                        ? "Solution plots: " + link[0]
                        : "Solution plots of " + record.name;
                    item.appendChild(anchor);
                    list.appendChild(item);
                });
                popup.appendChild(list);
            }
            popup.hidden = false;
            var box = figure.getBoundingClientRect();
            var mark = marker.querySelector("circle").getBoundingClientRect();
            var left = mark.right - box.left + 8;
            if (left + popup.offsetWidth > box.width) {
                left = Math.max(0, mark.left - box.left - popup.offsetWidth - 8);
            }
            var top = Math.min(
                Math.max(0, mark.top - box.top - 10),
                Math.max(0, box.height - popup.offsetHeight)
            );
            popup.style.left = left + "px";
            popup.style.top = top + "px";
        };
        map.querySelectorAll(".dirmap-marker").forEach(function (marker) {
            marker.addEventListener("click", function (event) {
                event.stopPropagation();
                open(marker);
            });
            marker.addEventListener("keydown", function (event) {
                if (event.key === "Enter" || event.key === " ") {
                    event.preventDefault();
                    open(marker);
                }
            });
        });
        popup.addEventListener("click", function (event) { event.stopPropagation(); });
        document.addEventListener("click", close);
        document.addEventListener("keydown", function (event) {
            if (event.key === "Escape") close();
        });
        if (select) {
            select.addEventListener("change", function () {
                close();
                // SVG elements ignore the hidden property; toggle the attribute.
                map.querySelectorAll(".dirmap-cycle").forEach(function (group) {
                    group.toggleAttribute("hidden", group.getAttribute("data-cycle") !== select.value);
                });
            });
        }
    });
});
"""


def _escape(value):
    return html.escape(str(value), quote=True)


def _bandpass_enabled(config):
    value = config.get("bandpass")
    if isinstance(value, str):
        return value.strip().casefold() == "true"
    return value is True


def _extraction_enabled(config):
    value = config.get("remove_outside_center")
    if isinstance(value, str):
        return value.strip().casefold() == "true"
    return value is True


def _dde_enabled(config):
    return config.get("DDE") is True


def _optional_pages(config):
    """Return the optional report pages that apply to this run."""
    pages = set()
    if _extraction_enabled(config):
        pages.add("extraction.html")
    if _dde_enabled(config):
        pages.add("directions.html")
    if _bandpass_enabled(config):
        pages.add("bandpass.html")
    return frozenset(pages)


def _stack_enabled(config):
    value = config.get("stack")
    if isinstance(value, str):
        return value.strip().casefold() == "true"
    return value is True


class _BandpassDataError(ValueError):
    pass


def _parse_config(path):
    config = {}
    if not path.is_file():
        return config
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError as exc:
        report_warning("Could not read configuration {}: {}".format(path, exc))
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


def _parse_first_list_literal(value):
    start = value.find("[")
    if start < 0:
        return None

    depth = 0
    quote = None
    escaped = False
    for end in range(start, len(value)):
        character = value[end]
        if quote is not None:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == quote:
                quote = None
            continue
        if character in {"'", '"'}:
            quote = character
        elif character == "[":
            depth += 1
        elif character == "]":
            depth -= 1
            if depth == 0:
                try:
                    result = ast.literal_eval(value[start : end + 1])
                except (SyntaxError, ValueError):
                    return None
                return result if isinstance(result, (list, tuple)) else None
    return None


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
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024 or unit == "TiB":
            return "{:.1f} {}".format(value, unit)
        value /= 1024
    return "{:.1f} TiB".format(value)


def _artifact_size(path):
    try:
        return _format_size(path.stat().st_size)
    except OSError as exc:
        report_warning("File size unavailable for {}: {}".format(path, exc))
        return "Unavailable"


def _format_duration(seconds):
    if seconds is None or seconds < 0:
        return "-"
    sec = int(round(seconds))
    mins, s = divmod(sec, 60)
    hours, m = divmod(mins, 60)
    if hours > 0:
        return "{:d}h {:d}m {:d}s".format(hours, m, s)
    elif m > 0:
        return "{:d}m {:d}s".format(m, s)
    return "{:d}s".format(s)


def _relative_url(path, base_dir):
    relative = os.path.relpath(str(path), str(base_dir))
    return quote(Path(relative).as_posix(), safe="/._-()")


def _page_shell(
    title,
    active_page,
    body,
    nested=False,
    subtitle="Offline processing report",
    optional_pages=(),
):
    prefix = "../" if nested else ""
    nav = []
    page_names = [
        (page, label) for page, label in _PAGE_NAMES.items()
        if page not in _OPTIONAL_PAGES or page in optional_pages
    ]
    for page, label in page_names:
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


def _section(title, content, note=None, section_id=None):
    note_html = "<p>{}</p>".format(_escape(note)) if note else ""
    id_attr = ' id="{}"'.format(_escape(section_id)) if section_id else ""
    return (
        '<section{}><div class="section-heading"><h2>{}</h2>{}</div>{}</section>'.format(
            id_attr, _escape(title), note_html, content
        )
    )


def _filter_input(target, label="Filter"):
    control_id = "report-filter-" + re.sub(r"[^a-zA-Z0-9_-]", "-", target).strip("-")
    return (
        '<div class="search-row"><label for="{}">{}</label>'
        '<input id="{}" type="search" data-filter-target="{}" '
        'placeholder="Type to filter" autocomplete="off"></div>'
    ).format(_escape(control_id), _escape(label), _escape(control_id), _escape(target))


def _config_value_html(value, key=None):
    if str(key).casefold() == "ms":
        paths = _as_list(value)
        items = [
            '<code title="{}">{}</code>'.format(
                _escape(path), _escape(Path(path).name or path)
            )
            for path in paths
        ]
        if len(paths) > 8:
            list_items = "".join("<li>{}</li>".format(item) for item in items)
            return "{} entries <details><summary>Show basenames</summary><ol>{}</ol></details>".format(
                len(paths), list_items
            )
        if items:
            return ", ".join(items)
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
                row_class, search_attr, _escape(key), _config_value_html(value, key)
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
    except OSError as exc:
        report_warning("Could not inspect artifact {}: {}".format(path, exc))
        return False


_STACK_IMAGE_IDENTITY_RE = re.compile(
    r"(?P<cycle>\d{3,})_stack(?P<stack>\d+)(?=[-_.]|$)",
    re.IGNORECASE,
)


def _stack_image_identity(value):
    text = str(value)
    name_match = re.search(
        r"(?:^|\s)-name(?:=|\s+)(?:\"([^\"]+)\"|'([^']+)'|([^\s]+))",
        text,
    )
    if name_match:
        text = next(group for group in name_match.groups() if group is not None)
    else:
        text = Path(text).name

    matches = list(_STACK_IMAGE_IDENTITY_RE.finditer(text))
    if not matches:
        return None
    match = matches[-1]
    return (
        str(int(match.group("cycle"))).zfill(3),
        str(int(match.group("stack"))).zfill(2),
    )


def _load_stack_imaging_groups(run_root):
    manifest_path = run_root / "misc" / "stack_imaging_groups.json"
    if not manifest_path.exists():
        return {}
    if not manifest_path.is_file():
        raise ValueError(
            "Stack imaging group manifest is not a file: {}".format(manifest_path)
        )
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(
            "Could not read stack imaging group manifest {}: {}".format(
                manifest_path, exc
            )
        ) from exc

    if (
        not isinstance(manifest, dict)
        or manifest.get("schema_version") != 1
        or not isinstance(manifest.get("groups"), list)
    ):
        raise ValueError(
            "Invalid stack imaging group manifest structure: {}".format(
                manifest_path
            )
        )

    groups = {}
    for group in manifest["groups"]:
        if not isinstance(group, dict):
            raise ValueError(
                "Invalid group entry in stack imaging group manifest: {}".format(
                    manifest_path
                )
            )
        stack_id = group.get("stack_id")
        members = group.get("measurement_sets")
        if (
            not isinstance(stack_id, str)
            or not stack_id.isdigit()
            or not isinstance(members, list)
            or not all(isinstance(member, str) for member in members)
        ):
            raise ValueError(
                "Invalid group entry in stack imaging group manifest: {}".format(
                    manifest_path
                )
            )
        normalized_stack_id = str(int(stack_id)).zfill(2)
        if normalized_stack_id in groups:
            raise ValueError(
                "Duplicate stack ID {} in {}".format(
                    normalized_stack_id, manifest_path
                )
            )
        groups[normalized_stack_id] = members
    return groups


def _stack_group_label(stack_id, groups):
    label = "Stack {}".format(stack_id)
    members = groups.get(str(stack_id).zfill(2), [])
    if not members:
        return label
    names = [Path(member).name for member in members]
    if len(names) > 2:
        member_text = "{}, {} +{} MS".format(names[0], names[1], len(names) - 2)
    else:
        member_text = ", ".join(names)
    return "{} · {}".format(label, member_text)


def _include_image_artifact(path, run_started_at, start_cycle, stack_mode=False):
    if stack_mode:
        stack_identity = _stack_image_identity(path.name)
        if stack_identity is not None:
            cycle = int(stack_identity[0])
            if cycle < start_cycle:
                return True
            return _artifact_is_current(path, run_started_at)
    # Use the last cycle-like number, as _image_cycle_identity does, so digits
    # in MS or target names earlier in the file name are not taken as the cycle.
    cycle_matches = re.findall(r"_(\d+)(?=-|\.|$)", path.name)
    if not cycle_matches or int(cycle_matches[-1]) < start_cycle:
        return True
    return _artifact_is_current(path, run_started_at)


def _filter_restart_cycle_timeline(records, run_started_at, start_cycle):
    if run_started_at is None:
        return records
    filtered = []
    for record in records:
        cycle = int(record["cycle"])
        if cycle < start_cycle:
            filtered.append(record)
            continue
        cycle_started_at = _log_timestamp_epoch(record.get("start_str", ""))
        if cycle_started_at is not None and cycle_started_at >= run_started_at:
            filtered.append(record)
    return filtered


def _filter_restart_image_metrics(
    records, run_started_at, start_cycle, current_image_names, stack_mode=False
):
    if run_started_at is None:
        return records
    filtered = []
    for record in records:
        cycle = record.get("cycle")
        try:
            cycle = int(cycle)
        except (TypeError, ValueError):
            cycle_identity = _image_cycle_identity(
                record.get("image", ""), stack_mode
            )
            cycle = int(cycle_identity) if cycle_identity is not None else None
        if cycle is None or cycle < start_cycle:
            filtered.append(record)
            continue
        metric_time = record.get("_timestamp")
        image_name = Path(record.get("image", "")).name.casefold()
        if (metric_time is not None and metric_time >= run_started_at) or image_name in current_image_names:
            filtered.append(record)
    return filtered


_DDE_DIAGNOSTIC_PATTERN = re.compile(
    r"(?:-errormap\d*(?:\.|-|$)|_artifact_errormap|-compactsource-flux)",
    re.IGNORECASE,
)
_ERRORMAP_INDEX_PATTERN = re.compile(r"-errormap(\d+)(?:\.|-|$)", re.IGNORECASE)


def _is_dde_diagnostic(value):
    """Return True for --DDE direction-finding products such as error maps.

    ``auto_direction`` in main.py writes calibration error maps
    (``<name>NNN-errormapK.fits``), their source catalogues and compact-source
    flux tables to ``errormaps_dd/``, with PNG previews in ``plots/``. They are
    used to find directions, not science images.
    """
    return _DDE_DIAGNOSTIC_PATTERN.search(Path(str(value)).name) is not None


_PLAIN_PLOT_SUFFIX = "-plain.png"


def _is_plain_plot(path):
    return path.name.lower().endswith(_PLAIN_PLOT_SUFFIX)


def _is_ateam_plot(path):
    return path.name.lower().startswith("ateam_")


def _is_ms_plot(path):
    name = path.name.lower()
    return (
        name.endswith(".time_coverage.png")
        or _is_ateam_plot(path)
        or (name.startswith("uvcoverage_") and name.endswith(".png"))
    )


def _scan_artifacts(
    run_root,
    run_started_at=None,
    current_run_cycles=None,
    start_cycle=0,
    stack_mode=False,
):
    overview_dir = run_root / "plots"
    all_raw_plots = sorted(
        (
            path for path in overview_dir.glob("*.png")
            if path.is_file()
        ),
        key=lambda path: path.name.lower(),
    ) if overview_dir.is_dir() else []

    raw_plots = [path for path in all_raw_plots if _artifact_is_current(path, run_started_at)]
    dde_plots = [
        p for p in all_raw_plots
        if _is_dde_diagnostic(p)
        and _include_image_artifact(
            p, run_started_at, start_cycle, stack_mode=stack_mode
        )
    ]
    # "<plot>-plain.png" is the same image as "<plot>.png" without overlays
    # (see plotimage_plain); it is offered as an alternative, not a plot of its own.
    plain_plots = {
        p.name[: -len(_PLAIN_PLOT_SUFFIX)].casefold(): p
        for p in all_raw_plots
        if _is_plain_plot(p)
        and _include_image_artifact(
            p, run_started_at, start_cycle, stack_mode=stack_mode
        )
    }
    all_overview_plots = [
        p for p in all_raw_plots
        if not _is_ms_plot(p)
        and not _is_dde_diagnostic(p)
        and not _is_plain_plot(p)
        and _include_image_artifact(
            p, run_started_at, start_cycle, stack_mode=stack_mode
        )
    ]
    overview_plots = [
        p for p in raw_plots
        if not _is_ms_plot(p) and not _is_dde_diagnostic(p) and not _is_plain_plot(p)
    ]
    all_ms_plot_files = [p for p in all_raw_plots if _is_ms_plot(p)]
    ms_plot_files = [
        p for p in all_raw_plots
        if _is_ms_plot(p)
        and (_is_ateam_plot(p) or _artifact_is_current(p, run_started_at))
    ]
    ms_json_files = sorted(
        (
            path for path in overview_dir.glob("*.json")
            if path.is_file()
            and not path.name.lower().endswith("-minmax-cache.json")
            and (_is_ateam_plot(path) or _artifact_is_current(path, run_started_at))
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
            if not path.is_file() or path.suffix.lower() != ".png":
                continue
            match = re.search(r"selfcalcycle(\d+)", path.name, re.IGNORECASE)
            cycle = str(int(match.group(1))).zfill(3) if match else "other"
            if cycle == "other":
                if not _artifact_is_current(path, run_started_at):
                    continue
            elif int(cycle) >= start_cycle:
                if not _artifact_is_current(path, run_started_at):
                    continue
                if current_run_cycles is not None and cycle not in current_run_cycles:
                    continue
            cycles[cycle].append(path)
            if cycle != "other":
                cycle_names.add(cycle)
        for paths in cycles.values():
            paths.sort(key=lambda path: path.name.lower())
        if cycles:
            calibration_sets.append((directory, dict(cycles)))

    fits_dir = run_root / "fits_images"
    all_fits_files = sorted(
        (
            path for path in fits_dir.rglob("*")
            if path.is_file()
            and (path.name.lower().endswith(".fits") or path.name.lower().endswith(".fits.gz"))
            and _include_image_artifact(
                path, run_started_at, start_cycle, stack_mode=stack_mode
            )
        ),
        key=lambda path: path.name.lower(),
    ) if fits_dir.is_dir() else []
    fits_files = [path for path in all_fits_files if _artifact_is_current(path, run_started_at)]

    solutions_dir = run_root / "h5_solutions"
    solution_files = []
    bandpass_files = []
    if solutions_dir.is_dir():
        for path in solutions_dir.rglob("*.h5"):
            if not path.is_file() or not _artifact_is_current(path, run_started_at):
                continue
            if any(
                part.casefold().startswith("bandpass_")
                for part in path.relative_to(solutions_dir).parts
            ):
                bandpass_files.append(path)
            match = re.search(r"selfcalcycle(\d+)", path.name, re.IGNORECASE)
            if match:
                cycle = str(int(match.group(1))).zfill(3)
                if current_run_cycles is not None and cycle not in current_run_cycles:
                    continue
                cycle_names.add(cycle)
            solution_files.append(path)
    solution_files.sort(key=lambda path: path.name.lower())
    bandpass_files.sort(key=lambda path: path.as_posix().casefold())

    def current_files(directory, pattern):
        directory = run_root / directory
        if not directory.is_dir():
            return []
        return sorted(
            (
                path for path in directory.glob(pattern)
                if path.is_file()
                and _include_image_artifact(
                    path, run_started_at, start_cycle, stack_mode=stack_mode
                )
            ),
            key=lambda path: path.name.lower(),
        )

    # Recognise MeasurementSets by their table.dat, since working copies have
    # names such as "<obs>.ms.copy.avg.extracted.copy" without a .ms suffix.
    ms_directories = sorted(
        (
            path for path in run_root.iterdir()
            if path.is_dir()
            and not path.name.lower().startswith("solution_plots_")
            and (path / "table.dat").is_file()
        ),
        key=lambda path: path.name.lower(),
    )

    return {
        "overview_plots": overview_plots,
        "all_overview_plots": all_overview_plots,
        "all_ms_plot_files": all_ms_plot_files,
        "ms_plot_files": ms_plot_files,
        "ms_json_files": ms_json_files,
        "calibration_sets": calibration_sets,
        "fits_files": fits_files,
        "all_fits_files": all_fits_files,
        "solution_files": solution_files,
        "bandpass_files": bandpass_files,
        "ms_directories": ms_directories,
        "dde_plots": dde_plots,
        "plain_plots": plain_plots,
        "dde_errormap_files": current_files("errormaps_dd", "*.fits"),
        "dde_direction_files": current_files("directions", "directions_*.txt"),
        "dde_region_files": current_files("facet_regions", "*.reg"),
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
    except OSError as exc:
        if path.exists():
            report_warning("Could not read run context {}: {}".format(path, exc))
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
        if not message.startswith(COMMAND_MARKER) and invocation_pattern.search(message):
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


_SAFE_SHELL_WORD = re.compile(r"[\w@%+=:,./-]+")


def _display_command(argv):
    """Join argv into a readable, copy-pasteable shell command.

    Arguments are quoted as a user would type them: values with brackets or
    quotes, such as ``--solint-list=['2min','10min']``, are wrapped in double
    quotes (``--solint-list="['2min','10min']"``) rather than shlex.join's
    nested single quotes.
    """
    def quote(value):
        if value and _SAFE_SHELL_WORD.fullmatch(value):
            return value
        if not re.search(r'["$`\\!]', value):
            return '"{}"'.format(value)
        return shlex.quote(value)

    words = []
    for arg in argv:
        option, separator, value = arg.partition("=")
        if separator and option.startswith("-") and _SAFE_SHELL_WORD.fullmatch(option):
            words.append(option + "=" + quote(value))
        else:
            words.append(quote(arg))
    return " ".join(words)


def _command_marker_record(message, timestamp):
    """Decode a startup command marker logged by facetselfcal."""
    try:
        payload = json.loads(message[len(COMMAND_MARKER):])
        argv = [str(item) for item in payload["argv"]]
    except (ValueError, KeyError, TypeError):
        return None
    return {
        "command": _display_command(argv),
        "cwd": payload.get("cwd"),
        "timestamp": timestamp,
        "source": "marker",
    }


def _current_command(logs, run_root):
    """Return the command line of the current run and where it was found.

    The startup marker in ``logs/selfcal.log`` is preferred. For logs from
    older versions the command line printed by ``logbasicinfo`` is used, and
    as a last resort a user-written ``facetselfcal.txt``.
    """
    records = logs.get("command_records") or []
    for source in ("marker", "log"):
        matching = [record for record in records if record["source"] == source]
        if matching:
            return matching[-1]
    command_path = run_root / "facetselfcal.txt"
    if command_path.is_file():
        try:
            command = command_path.read_text(encoding="utf-8", errors="replace").strip()
        except OSError as exc:
            report_warning("Could not read command record {}: {}".format(command_path, exc))
        else:
            if command:
                return {"command": command, "timestamp": "", "source": "file", "path": command_path}
    return None


def _ms_path_from_command(message):
    match = re.search(
        r'''(?:^|\s)msin=(?:"([^"]+)"|'([^']+)'|([^\s]+))''',
        message,
        re.IGNORECASE,
    )
    if match is None:
        return None
    return next((value.rstrip(",") for value in match.groups() if value), None)


def _ms_group_key(ms_path, run_root):
    if not ms_path:
        return None
    return os.path.normcase(_ms_identity_path(ms_path, run_root))


def _dp3_execution_marker_payload(message, marker):
    if not message.startswith(marker):
        return None
    try:
        payload = json.loads(message[len(marker):])
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def _normalized_dp3_command(command):
    return re.sub(
        r"^DP3\s+(?:solve|applycal):\s*",
        "",
        str(command).strip(),
        flags=re.IGNORECASE,
    )


def _dp3_workflow_step_record(kind, command, timestamp, run_root):
    ms_path = _ms_path_from_command(command)
    ms_key = _ms_group_key(ms_path, run_root)
    if kind in RESOURCE_PHASE_STYLES:
        name = RESOURCE_PHASE_STYLES[kind][0]
    elif kind == "solve":
        name = "Calibration solve (DP3)"
    else:
        name = "Apply solutions (DP3)"
    return {
        "kind": kind,
        "name": name,
        "timestamp": timestamp,
        "ms_path": ms_path,
        "ms_key": ms_key,
        "ms_name": Path(ms_key).name if ms_key else None,
        "command": command,
    }


def _workflow_steps_summary(step_details):
    if not step_details:
        return "-"

    shared_count = sum(
        step["kind"] in ("imaging", "predict") for step in step_details
    )
    ms_count = len(
        {
            step.get("ms_key")
            for step in step_details
            if step["kind"] not in ("imaging", "predict")
        }
    )
    step_count = len(step_details)
    summary = "{} step{}".format(
        step_count, "" if step_count == 1 else "s"
    )
    details = []
    if ms_count:
        details.append("{} MS".format(ms_count))
    if shared_count:
        details.append("{} shared".format(shared_count))
    if details:
        summary += " ({})".format(", ".join(details))
    return summary


def _workflow_steps_html(step_details, id_prefix="segment", heading="Workflow steps"):
    if not step_details:
        return "-"

    shared_steps = []
    ms_groups = {}
    command_index = 0
    for step in step_details:
        if step["kind"] in ("imaging", "predict"):
            shared_steps.append(step)
            continue
        group = ms_groups.setdefault(
            step.get("ms_key"),
            {"name": step.get("ms_name") or "MS not identified", "path": step.get("ms_path"), "steps": []},
        )
        group["steps"].append(step)

    step_labels = {
        "imaging": "WSClean imaging",
        "predict": "WSClean predict",
        "solve": "Solve",
        "apply": "Apply",
    }
    badge_classes = {
        "imaging": "workflow-step-badge-imaging",
        "predict": "workflow-step-badge-predict",
        "solve": "workflow-step-badge-solve",
        "apply": "workflow-step-badge-apply",
        "average": "workflow-step-badge-average",
        "phaseup": "workflow-step-badge-phaseup",
        "phaseshift": "workflow-step-badge-phaseshift",
        "filter": "workflow-step-badge-filter",
        "aoflagger": "workflow-step-badge-aoflagger",
    }

    def step_label(step):
        label = step_labels.get(step["kind"], step["name"])
        stack_label = step.get("stack_label")
        if stack_label:
            return "{} · {}".format(label, stack_label)
        return label

    def step_badge_class(step):
        return badge_classes.get(step["kind"], "workflow-step-badge-other")

    aggregates = {}
    for step in step_details:
        label = step_label(step)
        key = (step["kind"], label)
        aggregate = aggregates.setdefault(
            key, {"step": step, "label": label, "count": 0}
        )
        aggregate["count"] += 1

    aggregate_badges = []
    for aggregate in aggregates.values():
        aggregate_badges.append(
            '<span role="listitem" class="workflow-step-badge workflow-step-aggregate {}">'
            '{} <span class="workflow-step-aggregate-count">({}x)</span></span>'.format(
                step_badge_class(aggregate["step"]),
                _escape(aggregate["label"]),
                aggregate["count"],
            )
        )

    def render_badges(steps):
        nonlocal command_index
        badges = []
        for index, step in enumerate(steps):
            label = step_label(step)
            badge_class = step_badge_class(step)
            arrow = '<span class="workflow-step-arrow" aria-hidden="true">&rarr;</span>' if index + 1 < len(steps) else ""
            command_id = "workflow-command-{}-{}".format(
                id_prefix, command_index
            )
            command_index += 1
            title = "Click to view full command"
            if step.get("shared_command_duration"):
                title += "; duration is shared by all steps in this DP3 command"
            outcome = step.get("outcome", "unknown")
            outcome_text = ""
            if outcome == "failed":
                badge_class += " workflow-step-failed"
                outcome_text = "Failed (exit {})".format(step["returncode"])
            elif outcome == "in-progress":
                outcome_text = "In progress"
            elif outcome == "completed":
                outcome_text = "Completed"
            else:
                outcome_text = "Outcome not recorded"
            title += "; " + outcome_text
            badges.append(
                '<div class="workflow-step-item">'
                '<button type="button" class="workflow-step-badge workflow-command-trigger {}" '
                'popovertarget="{}" title="{}">{} <span class="workflow-step-duration">{}</span>'
                '<span class="workflow-step-outcome">{}</span></button>'
                '<pre class="workflow-command-text" id="{}" popover="auto">{}</pre>'
                '{}'
                '</div>'.format(
                    badge_class,
                    command_id,
                    _escape(title),
                    _escape(label),
                    _escape(step.get("duration") or "-"),
                    _escape(outcome_text),
                    command_id,
                    _escape(step.get("command") or step["name"]),
                    arrow,
                )
            )
        return "".join(badges)

    rows = [
        '<div class="workflow-panel" role="group" aria-label="{}">'
        '<div class="workflow-panel-heading">{}</div>'
        '<div class="workflow-step-grid">'
        '<div class="workflow-step-aggregates" role="list" aria-label="Aggregate step counts">{}</div>'.format(
            _escape(heading),
            _escape(heading),
            "".join(aggregate_badges),
        )
    ]
    if shared_steps or ms_groups:
        detail_counts = []
        if shared_steps:
            detail_counts.append(
                "{} shared step{}".format(
                    len(shared_steps), "" if len(shared_steps) == 1 else "s"
                )
            )
        if ms_groups:
            detail_counts.append(
                "{} measurement set{}".format(
                    len(ms_groups), "" if len(ms_groups) == 1 else "s"
                )
            )
        rows.append(
            '<details class="workflow-detail-disclosure">'
            '<summary class="workflow-detail-summary">'
            '<span class="workflow-detail-summary-label">Detailed steps</span>'
            '<span class="workflow-detail-summary-count">{}</span>'
            '<span class="workflow-detail-summary-hint">Click to expand</span>'
            '</summary><div class="workflow-detail-content">'.format(
                _escape(" · ".join(detail_counts))
            )
        )
    if shared_steps:
        rows.append(
            '<div class="workflow-shared-steps"><span class="workflow-shared-label">Shared</span>'
            '<div class="workflow-step-chain">{}</div></div>'.format(render_badges(shared_steps))
        )
    if ms_groups:
        rows.append(
            '<div class="workflow-ms-list-heading">Per measurement set</div>'
            '<div class="workflow-ms-list">'
        )
    for group_index, group in enumerate(ms_groups.values()):
        name_popover_id = "workflow-ms-name-{}-{}".format(
            id_prefix, group_index
        )
        group_name = group["name"]
        rows.append(
            '<div class="workflow-ms-row">'
            '<div class="workflow-ms-name-container">'
            '<button type="button" class="workflow-ms-name" popovertarget="{}" '
            'aria-controls="{}" aria-label="{}" title="{}">{}</button>'
            '<div class="workflow-ms-name-popover" id="{}" popover="auto">'
            '<dl><dt>Measurement set</dt><dd><code>{}</code></dd></dl>'
            '</div></div>'
            '<div class="workflow-ms-badges">{}</div></div>'.format(
                _escape(name_popover_id),
                _escape(name_popover_id),
                _escape(
                    "Show full measurement-set name for {}".format(group_name)
                ),
                "Click or tap to view the full measurement-set name",
                _escape(group_name),
                _escape(name_popover_id),
                _escape(group_name),
                render_badges(group["steps"]),
            )
        )
    if ms_groups:
        rows.append("</div>")
    if shared_steps or ms_groups:
        rows.append("</div></details>")
    rows.append("</div></div>")
    return "".join(rows)


def _scan_logs(run_root, stack_mode=False, stack_groups=None):
    candidates = [run_root / "logs" / "selfcal.log"]
    logs = []
    warnings = deque(maxlen=20)
    errors = deque(maxlen=20)
    warning_count = 0
    error_count = 0
    invocations = 0
    command_records = []
    timestamps = []

    host_info = {}
    cycles = {}
    current_cycle = None
    first_ts = None
    last_ts = None
    cycle_start_pattern = re.compile(
        r"Starting self-calibration cycle\s+(\d+)", re.IGNORECASE
    )
    setup_steps = []
    dp3_execution_steps = {}
    image_metric_pattern = re.compile(
        r"^(?P<image>.+?)\s+(?P<metric>Max image|Min image|RMS noise):\s*(?P<value>.+)$",
        re.IGNORECASE,
    )
    image_metric_fields = {
        label.casefold(): field for field, label in _IMAGE_METRIC_FIELDS
    }
    image_metric_records = {}
    image_commands = []
    flagging_stats = {}
    flagging_percentage_pattern = re.compile(
        r"^Flagging statistics for MS (?P<ms>.+): "
        r"(?P<percentage>\d+(?:\.\d+)?)% flagged "
        r"\((?P<flagged>\d+)/(?P<total>\d+) samples\)\."
    )
    fully_flagged_antennas_pattern = re.compile(
        r"^Fully flagged antennas for MS (?P<ms>.+): (?P<antennas>.*)$"
    )
    effective_config_names = {
        "soltype": "soltype_list",
        "soltypecycles": "soltypecycles_list",
        "solint": "solint_list",
        "smoothnessconstraint": "smoothnessconstraint_list",
        "nchan": "nchan_list",
    }
    effective_cycle_config = {}
    effective_cycle_config_errors = set()
    # Settings printed by logbasicinfo after facetselfcal resolved its
    # automatic choices (pixel scale, niter, ...); full_config.txt from older
    # versions holds the values before that step.
    effective_setting_names = {
        "Imsize": "imsize",
        "Pixelscale [arcsec]": "pixelscale",
        "Niter": "niter",
        "Widefield": "forwidefield",
    }
    effective_settings = {}
    stack_groups = stack_groups or {}

    def record_dp3_execution_start(payload, timestamp):
        execution_id = payload.get("id")
        command = payload.get("command")
        if (
            not isinstance(execution_id, str)
            or not execution_id
            or not isinstance(command, str)
        ):
            return

        step_kinds = dp3_workflow_step_kinds(command)
        destination = (
            cycles[current_cycle]["steps"]
            if current_cycle is not None
            else setup_steps
        )
        normalized_command = _normalized_dp3_command(command)
        shared_command = len(step_kinds) > 1 or any(
            kind in dp3_command_phases(command) for kind in step_kinds
        )
        execution_steps = []

        for kind in step_kinds:
            step = next(
                (
                    candidate
                    for candidate in reversed(destination)
                    if candidate["kind"] == kind
                    and candidate.get("execution_id") is None
                    and _normalized_dp3_command(candidate.get("command", ""))
                    == normalized_command
                    and candidate.get("timestamp") is not None
                    and 0
                    <= (timestamp - candidate["timestamp"]).total_seconds()
                    <= 5
                ),
                None,
            )
            if step is None:
                step = _dp3_workflow_step_record(
                    kind, command, timestamp, run_root
                )
                destination.append(step)
            else:
                step["timestamp"] = timestamp
                step["command"] = command

            step["execution_id"] = execution_id
            step["outcome"] = "in-progress"
            if shared_command:
                step["shared_command_duration"] = True
            execution_steps.append(step)

        dp3_execution_steps[execution_id] = execution_steps

    def record_dp3_execution_end(payload, timestamp):
        execution_id = payload.get("id")
        if not isinstance(execution_id, str) or timestamp is None:
            return
        for step in dp3_execution_steps.get(execution_id, ()):
            step["end_time"] = timestamp
            returncode = payload.get("returncode")
            if isinstance(returncode, int) and not isinstance(returncode, bool):
                step["returncode"] = returncode
                step["outcome"] = "completed" if returncode == 0 else "failed"
            else:
                step["outcome"] = "unknown"

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
                    ts_obj = None
                    if timestamp:
                        timestamps.append(timestamp)
                        try:
                            ts_obj = datetime.strptime(timestamp, "%m/%d/%Y %H:%M:%S")
                        except ValueError:
                            pass
                        if ts_obj and path.name == "selfcal.log":
                            if first_ts is None:
                                first_ts = ts_obj
                            last_ts = ts_obj

                    if level == "WARNING":
                        warning_count += 1
                        warnings.append((path, line_count, timestamp, message))
                    elif level in ("ERROR", "CRITICAL"):
                        error_count += 1
                        errors.append((path, line_count, timestamp, message))
                    if message.startswith(COMMAND_MARKER):
                        if path.name == "selfcal.log":
                            command_record = _command_marker_record(message, timestamp)
                            if command_record is not None:
                                command_records.append(command_record)
                    elif re.search(r"facetselfcal(?:\.py)?\b.*(?:\s-i(?:\s|=)|--config(?:\s|=))", line, re.IGNORECASE):
                        invocations += 1
                        if path.name == "selfcal.log":
                            command_records.append(
                                {"command": message.strip(), "timestamp": timestamp, "source": "log"}
                            )

                    if path.name == "selfcal.log":
                        if message.startswith(DP3_EXECUTION_START_MARKER):
                            payload = _dp3_execution_marker_payload(
                                message, DP3_EXECUTION_START_MARKER
                            )
                            if payload is not None and ts_obj is not None:
                                record_dp3_execution_start(payload, ts_obj)
                            continue

                        if message.startswith(DP3_EXECUTION_END_MARKER):
                            payload = _dp3_execution_marker_payload(
                                message, DP3_EXECUTION_END_MARKER
                            )
                            if payload is not None:
                                record_dp3_execution_end(payload, ts_obj)
                            continue

                        setting_name, separator, setting_value = message.partition(":")
                        if separator and setting_name.strip() in effective_setting_names:
                            try:
                                resolved_value = ast.literal_eval(setting_value.strip())
                            except (ValueError, SyntaxError):
                                resolved_value = setting_value.strip()
                            effective_settings[effective_setting_names[setting_name.strip()]] = resolved_value
                        config_name = effective_config_names.get(setting_name.strip())
                        if separator and config_name:
                            parsed_value = _parse_first_list_literal(setting_value)
                            if parsed_value is None:
                                effective_cycle_config.pop(config_name, None)
                                effective_cycle_config_errors.add(config_name)
                            else:
                                effective_cycle_config[config_name] = parsed_value
                                effective_cycle_config_errors.discard(config_name)

                        if message.startswith("Run host:"):
                            host_info["host"] = message.split(":", 1)[1].strip()
                        elif message.startswith("Operating system:"):
                            host_info["os"] = message.split(":", 1)[1].strip()
                        elif message.startswith("CPU count:"):
                            host_info["cpu"] = message.split(":", 1)[1].strip()
                        elif message.startswith("RAM:"):
                            host_info["ram"] = message.split(":", 1)[1].strip()
                        elif message.startswith("Disk at run directory"):
                            host_info["disk"] = message.split(":", 1)[1].strip()
                        elif message.startswith("VERSION:"):
                            host_info["version"] = message.split(":", 1)[1].strip()

                        flagging_match = flagging_percentage_pattern.match(message)
                        if flagging_match:
                            ms_path = flagging_match.group("ms").strip()
                            flagging_record = flagging_stats.setdefault(
                                _resolved_ms_path(ms_path, run_root),
                                {"ms_path": ms_path},
                            )
                            flagging_record.update(
                                {
                                    "flagged_percentage": flagging_match.group(
                                        "percentage"
                                    ),
                                    "flagged_samples": flagging_match.group("flagged"),
                                    "total_samples": flagging_match.group("total"),
                                }
                            )

                        antennas_match = fully_flagged_antennas_pattern.match(message)
                        if antennas_match:
                            ms_path = antennas_match.group("ms").strip()
                            flagging_record = flagging_stats.setdefault(
                                _resolved_ms_path(ms_path, run_root),
                                {"ms_path": ms_path},
                            )
                            flagging_record["fully_flagged_antennas"] = (
                                antennas_match.group("antennas").strip() or "None"
                            )

                        cm = cycle_start_pattern.search(message)
                        if cm:
                            current_cycle = str(int(cm.group(1))).zfill(3)
                            cycles[current_cycle] = {
                                "cycle": current_cycle,
                                "start_time": ts_obj,
                                "start_str": timestamp or "",
                                "end_time": None,
                                "steps": [],
                            }
                            continue

                        if message.casefold().startswith("wsclean imaging:"):
                            command = message.partition(":")[2].strip()
                            output_name = _wsclean_output_name(command)
                            if output_name:
                                stack_identity = (
                                    _stack_image_identity(output_name)
                                    if stack_mode
                                    else None
                                )
                                command_cycle = _image_cycle_identity(
                                    output_name, stack_mode
                                ) or current_cycle
                                command_record = {
                                    "cycle": command_cycle,
                                    "output_name": output_name,
                                    "command": command,
                                    "_timestamp": _log_timestamp_epoch(timestamp),
                                    "_order": line_count,
                                }
                                if stack_identity is not None:
                                    command_record["stack_id"] = stack_identity[1]
                                image_commands.append(command_record)

                        if ts_obj is not None:
                            dp3_phases = dp3_command_phases(message)
                            if dp3_phases:
                                ms_path = _ms_path_from_command(message)
                                ms_key = _ms_group_key(ms_path, run_root)
                                destination = (
                                    cycles[current_cycle]["steps"]
                                    if current_cycle is not None
                                    else setup_steps
                                )
                                for phase in dp3_phases:
                                    destination.append(
                                        {
                                            "kind": phase,
                                            "name": RESOURCE_PHASE_STYLES[phase][0],
                                            "timestamp": ts_obj,
                                            "ms_path": ms_path,
                                            "ms_key": ms_key,
                                            "ms_name": Path(ms_key).name if ms_key else None,
                                            "command": message,
                                            "shared_command_duration": True,
                                        }
                                    )

                        metric_match = image_metric_pattern.match(message)
                        if metric_match:
                            image_name = metric_match.group("image").strip()
                            stack_identity = (
                                _stack_image_identity(image_name)
                                if stack_mode
                                else None
                            )
                            metric_value = metric_match.group("value").strip()
                            try:
                                float(metric_value)
                            except ValueError:
                                pass
                            else:
                                metric_cycle = current_cycle
                                if metric_cycle is None:
                                    metric_cycle = _image_cycle_identity(
                                        image_name, stack_mode
                                    )
                                field = image_metric_fields[
                                    metric_match.group("metric").casefold()
                                ]
                                record_key = (metric_cycle or "", image_name)
                                # Error maps from --DDE direction finding are
                                # logged the same way but are not images to report.
                                if not _is_dde_diagnostic(image_name):
                                    record = image_metric_records.setdefault(
                                        record_key,
                                        {
                                            "cycle": metric_cycle,
                                            "image": image_name,
                                        },
                                    )
                                    if stack_identity is not None:
                                        record["stack_id"] = stack_identity[1]
                                        record["stack_label"] = _stack_group_label(
                                            stack_identity[1], stack_groups
                                        )
                                    record[field] = metric_value
                                    record["_timestamp"] = _log_timestamp_epoch(timestamp)
                                    record["_order"] = line_count

                        if current_cycle and ts_obj:
                            cdata = cycles[current_cycle]
                            step = None
                            folded_message = message.casefold()
                            if folded_message.startswith("wsclean imaging:"):
                                kind = "imaging"
                                command = message.partition(":")[2].strip()
                            elif folded_message.startswith(("predict step:", "dde predict step:")):
                                kind = "predict"
                                command = message.partition(":")[2].strip()
                            elif folded_message.startswith("wsclean "):
                                command = message
                                kind = (
                                    "predict"
                                    if re.search(r"(?:^|\s)-predict(?:\s|$)", message, re.IGNORECASE)
                                    else "imaging"
                                )
                            else:
                                kind = None
                                command = message

                            stack_identity = (
                                _stack_image_identity(command)
                                if stack_mode and kind is not None
                                else None
                            )
                            if kind is not None and (
                                stack_mode
                                or not any(
                                    existing["kind"] == kind
                                    for existing in cdata["steps"]
                                )
                            ):
                                step = {
                                    "kind": kind,
                                    "name": "WSClean {}".format(kind),
                                    "timestamp": ts_obj,
                                    "ms_path": None,
                                    "ms_key": None,
                                    "ms_name": None,
                                    "command": command,
                                }
                                if stack_identity is not None:
                                    step["stack_id"] = stack_identity[1]
                                    step["stack_label"] = _stack_group_label(
                                        stack_identity[1], stack_groups
                                    )
                            elif kind is None and "DP3 solve:" in message:
                                ms_path = _ms_path_from_command(message)
                                ms_key = _ms_group_key(ms_path, run_root)
                                step = {
                                    "kind": "solve",
                                    "name": "Calibration solve (DP3)",
                                    "timestamp": ts_obj,
                                    "ms_path": ms_path,
                                    "ms_key": ms_key,
                                    "ms_name": Path(ms_key).name if ms_key else None,
                                    "command": message,
                                }
                            elif kind is None and (
                                "DP3 applycal:" in message
                                or re.search(r"(?:^|\.)type=applycal\b", message, re.IGNORECASE)
                                or ("steps=[ac0]" in message and "applycal" in message)
                            ):
                                ms_path = _ms_path_from_command(message)
                                ms_key = _ms_group_key(ms_path, run_root)
                                step = {
                                    "kind": "apply",
                                    "name": "Apply solutions (DP3)",
                                    "timestamp": ts_obj,
                                    "ms_path": ms_path,
                                    "ms_key": ms_key,
                                    "ms_name": Path(ms_key).name if ms_key else None,
                                    "command": message,
                                }
                            if step is not None:
                                cdata["steps"].append(step)

        except OSError as exc:
            report_warning("Could not read run log {}: {}".format(path, exc))
            continue
        logs.append({"path": path, "lines": line_count, "tail": list(tail)})

    def build_step_details(steps, end_time):
        step_details = []
        for s_idx, step in enumerate(steps):
            step_ts = step["timestamp"]
            next_ts = step.get("end_time")
            if next_ts is None:
                next_ts = next(
                    (
                        candidate["timestamp"]
                        for candidate in steps[s_idx + 1 :]
                        if candidate["timestamp"] is not None
                        and candidate["timestamp"] > step_ts
                    ),
                    None,
                )
            if next_ts is None and end_time is not None and end_time > step_ts:
                next_ts = end_time
            s_dur = (
                (next_ts - step_ts).total_seconds()
                if next_ts is not None and next_ts >= step_ts
                else None
            )
            step_details.append({**step, "duration": _format_duration(s_dur)})
        return step_details

    cycle_keys = sorted(cycles.keys())
    for i, ck in enumerate(cycle_keys):
        cd = cycles[ck]
        if i + 1 < len(cycle_keys):
            cd["end_time"] = cycles[cycle_keys[i + 1]]["start_time"]
        else:
            cd["end_time"] = last_ts

        if cd["start_time"] and cd["end_time"] and cd["end_time"] >= cd["start_time"]:
            cd["duration_str"] = _format_duration((cd["end_time"] - cd["start_time"]).total_seconds())
        else:
            cd["duration_str"] = "-"

        cd["step_details"] = build_step_details(cd["steps"], cd["end_time"])

    setup_end_time = (
        cycles[cycle_keys[0]]["start_time"] if cycle_keys else last_ts
    )
    setup_start_time = min(
        (
            step["timestamp"]
            for step in setup_steps
            if step.get("timestamp") is not None
        ),
        default=None,
    )
    setup_start_str = (
        setup_start_time.strftime("%m/%d/%Y %H:%M:%S")
        if setup_start_time is not None
        else ""
    )
    setup_duration_str = (
        _format_duration((setup_end_time - setup_start_time).total_seconds())
        if setup_start_time is not None
        and setup_end_time is not None
        and setup_end_time >= setup_start_time
        else "-"
    )
    setup_step_details = build_step_details(setup_steps, setup_end_time)

    total_elapsed = None
    if first_ts and last_ts and last_ts >= first_ts:
        total_elapsed = _format_duration((last_ts - first_ts).total_seconds())

    return {
        "files": logs,
        "warnings": list(warnings),
        "errors": list(errors),
        "warning_count": warning_count,
        "error_count": error_count,
        "invocations": invocations,
        "command_records": command_records,
        "timestamps": timestamps,
        "host_info": host_info,
        "effective_cycle_config": effective_cycle_config,
        "effective_settings": effective_settings,
        "effective_cycle_config_errors": sorted(effective_cycle_config_errors),
        "flagging_stats": flagging_stats,
        "cycle_timeline": [cycles[k] for k in cycle_keys],
        "stack_groups": stack_groups if stack_mode else {},
        "setup_step_details": setup_step_details,
        "setup_start_str": setup_start_str,
        "setup_duration_str": setup_duration_str,
        "image_metrics": sorted(
            image_metric_records.values(), key=lambda record: record["_order"]
        ),
        "image_commands": image_commands,
        "total_elapsed": total_elapsed,
    }


def _clipped_image_rms(values, np):
    pixels = values[np.isfinite(values) & (np.abs(values) > 1e-7)]
    if not pixels.size:
        return float("nan")

    rms_old = np.std(pixels)
    median = np.median(pixels)
    rms = rms_old
    for _ in range(10):
        if rms_old == 0:
            return 0.0
        selected = pixels[np.abs(pixels - median) < rms_old * 3.0]
        if not selected.size:
            return float("nan")
        rms = np.std(selected)
        if rms_old != 0 and np.abs((rms - rms_old) / rms_old) < 1e-1:
            break
        rms_old = rms
    return rms


def _finite_metric(value):
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError, OverflowError):
        return False


def _valid_image_metric(field, value):
    return _finite_metric(value) and (field != "rms_noise" or float(value) >= 0)


def _restoring_beam_from_header(header):
    beam = {}
    for keyword, field, positive_only in (
        ("BMAJ", "bmaj", True),
        ("BMIN", "bmin", True),
        ("BPA", "bpa", False),
    ):
        if keyword not in header:
            beam[field + "_deg"] = None
            beam[field + "_status"] = "missing"
            continue
        try:
            value = float(header[keyword])
        except (TypeError, ValueError, OverflowError):
            value = float("nan")
        if not math.isfinite(value) or (positive_only and value <= 0):
            beam[field + "_deg"] = None
            beam[field + "_status"] = "invalid"
        else:
            beam[field + "_deg"] = value
            beam[field + "_status"] = "valid"
    return beam


def _unavailable_restoring_beam(reason="FITS header unavailable"):
    return {
        "bmaj_deg": None,
        "bmaj_status": "unavailable",
        "bmin_deg": None,
        "bmin_status": "unavailable",
        "bpa_deg": None,
        "bpa_status": "unavailable",
        "unavailable_reason": reason,
    }


def _restoring_beam_components(beam):
    major = beam.get("bmaj_deg") if beam.get("bmaj_status") == "valid" else None
    minor = beam.get("bmin_deg") if beam.get("bmin_status") == "valid" else None
    unit_reference = major if major is not None else minor
    if unit_reference is None:
        multiplier, unit = 1, "°"
    elif unit_reference < 1 / 60:
        multiplier, unit = 3600, "″"
    elif unit_reference < 1:
        multiplier, unit = 60, "′"
    else:
        multiplier, unit = 1, "°"

    def unavailable_component(field, keyword):
        status = beam.get(field + "_status")
        if status == "missing":
            reason = "missing keyword"
        elif status == "invalid":
            reason = "invalid value"
        else:
            reason = beam.get("unavailable_reason", "FITS header unavailable")
        return "{} unavailable ({})".format(keyword, reason)

    def format_axis(field, keyword):
        if beam.get(field + "_status") != "valid":
            return unavailable_component(field, keyword)
        value = beam[field + "_deg"]
        decimals = 2 if value * 3600 < 1 else 1
        return "{:.{}f}{}".format(value * multiplier, decimals, unit)

    major_text = format_axis("bmaj", "BMAJ")
    minor_text = format_axis("bmin", "BMIN")
    if beam.get("bpa_status") == "valid":
        angle_text = "PA {:.0f}°".format(beam["bpa_deg"])
    else:
        angle_text = unavailable_component("bpa", "BPA")
    return major_text, minor_text, angle_text


def _format_restoring_beam(beam):
    major_text, minor_text, angle_text = _restoring_beam_components(beam)
    return "{} × {}, {}".format(major_text, minor_text, angle_text)


def _restoring_beam_html(beam):
    major_text, minor_text, angle_text = _restoring_beam_components(beam)
    return (
        '<span class="image-restoring-beam">'
        '<span>{}</span> × <span>{}</span>'
        '<span class="image-beam-angle">{}</span>'
        '</span>'
    ).format(
        _escape(major_text),
        _escape(minor_text),
        _escape(angle_text),
    )


def _image_metrics_from_fits(
    fits_files, records, stack_mode=False, stack_groups=None
):
    primary_images = []
    stack_groups = stack_groups or {}
    for path in fits_files:
        name = path.name.casefold()
        if re.search(r"-\d{4}-image\.fits(?:\.gz)?$", name):
            continue
        if not name.endswith((
            "-mfs-image.fits",
            "-mfs-image.fits.gz",
            "-image.fits",
            "-image.fits.gz",
            ".app.restored.fits",
            ".app.restored.fits.gz",
        )):
            continue
        stack_identity = (
            _stack_image_identity(path.name) if stack_mode else None
        )
        if stack_identity is not None:
            primary_images.append((path, stack_identity[0], stack_identity[1]))
        else:
            cycle = _image_cycle_identity(path.name)
            if cycle is not None:
                primary_images.append((path, cycle, None))

    if not primary_images:
        return records

    try:
        import numpy as np
        from astropy.io import fits
    except ImportError as exc:
        report_warning("FITS image statistics are unavailable: {}.".format(exc))
        return records

    records_by_image = {
        Path(record.get("image", "")).name.casefold(): record
        for record in records
    }
    for path, cycle, stack_id in primary_images:
        record = records_by_image.get(path.name.casefold())
        fields = ("max_image", "min_image", "rms_noise")
        metrics_complete = record is not None and all(
            _valid_image_metric(field, record.get(field)) for field in fields
        )
        computed = None
        cache = None
        if not metrics_complete:
            cache = DerivedDataCache(path, "finite-fits-metrics-v1")
            computed = cache.load()
            if computed is not None and not all(
                _valid_image_metric(field, computed.get(field)) for field in fields
            ):
                report_warning("Ignoring invalid cached image statistics for {}.".format(path))
                computed = None

        needs_pixels = not metrics_complete and computed is None
        computed_from_fits = False
        beam = None
        fits_error = None
        try:
            with fits.open(path, memmap=False) as hdulist:
                beam = _restoring_beam_from_header(hdulist[0].header)
                if needs_pixels:
                    if hdulist[0].data is None:
                        raise ValueError("no primary image data")
                    values = np.asarray(hdulist[0].data).reshape(-1)
                    values = values[np.isfinite(values)]
                    if not values.size:
                        raise ValueError("no finite image pixels")
                    rms = _clipped_image_rms(values, np)
                    computed = {
                        "max_image": str(np.max(values)),
                        "min_image": str(np.min(values)),
                        "rms_noise": str(rms) if math.isfinite(rms) else None,
                    }
                    computed_from_fits = True
        except (OSError, ValueError, TypeError, IndexError) as exc:
            fits_error = exc
            if beam is None:
                beam = _unavailable_restoring_beam()
            report_warning("Restoring beam unavailable for {}: {}".format(path, exc))

        if record is not None:
            record["restoring_beam"] = beam
        if metrics_complete:
            if stack_id is not None:
                record["stack_id"] = stack_id
                record["stack_label"] = _stack_group_label(stack_id, stack_groups)
            continue

        if computed is None:
            if fits_error is not None:
                report_warning(
                    "Image statistics unavailable for {}: {}".format(path, fits_error)
                )
                continue
            report_warning(
                "Image statistics unavailable for {}: FITS statistics were not computed.".format(
                    path
                )
            )
            continue
        if computed["rms_noise"] is None:
            report_warning(
                "RMS noise unavailable for {}: no usable pixels after clipping.".format(
                    path
                )
            )
        elif computed_from_fits and cache is not None:
            cache.save(computed)

        if record is None:
            record = {
                "cycle": cycle,
                "image": str(path),
            }
            records.append(record)
            records_by_image[path.name.casefold()] = record
        record["restoring_beam"] = beam
        if stack_id is not None:
            record["stack_id"] = stack_id
            record["stack_label"] = _stack_group_label(stack_id, stack_groups)
        for field, value in computed.items():
            if not _valid_image_metric(field, record.get(field)):
                record[field] = value

    return records


def _wsclean_output_name(command):
    match = re.search(
        r"(?:^|\s)-name(?:=|\s+)(?:\"([^\"]+)\"|'([^']+)'|([^\s]+))",
        str(command),
    )
    if match is None:
        return None
    return next(group for group in match.groups() if group is not None)


def _normalized_image_output_name(value):
    name = Path(str(value).strip()).name.casefold()
    for suffix in (
        "-mfs-image.fits.gz",
        "-mfs-image.fits",
        "-image.fits.gz",
        "-image.fits",
        ".app.restored.fits.gz",
        ".app.restored.fits",
    ):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name


def _image_cycle_identity(value, stack_mode=False):
    if stack_mode:
        stack_identity = _stack_image_identity(value)
        if stack_identity is not None:
            return stack_identity[0]
    cycle_tokens = re.findall(
        r"(?<!\d)(\d{3,})(?=-|\.|$)", Path(str(value)).name
    )
    if cycle_tokens:
        return str(int(cycle_tokens[-1])).zfill(3)
    return None


def _normalized_cycle(value):
    try:
        return str(int(value)).zfill(3)
    except (TypeError, ValueError):
        return None


def _attach_wsclean_commands(records, commands, stack_mode=False):
    commands_by_image = defaultdict(list)
    for command in commands:
        output_name = command.get("output_name")
        image_key = _normalized_image_output_name(output_name) if output_name else ""
        if image_key:
            commands_by_image[image_key].append(command)

    for record in records:
        record.pop("wsclean_command", None)
        image_key = _normalized_image_output_name(record.get("image", ""))
        candidates = commands_by_image.get(image_key, [])
        if not candidates:
            continue

        record_cycle = _normalized_cycle(record.get("cycle"))
        if record_cycle is None:
            record_cycle = _image_cycle_identity(record.get("image", ""), stack_mode)
        record_stack = (
            str(record["stack_id"]).zfill(2)
            if record.get("stack_id") is not None
            else None
        )
        candidates = [
            command
            for command in candidates
            if not (
                record_cycle is not None
                and command.get("cycle") is not None
                and _normalized_cycle(command.get("cycle")) != record_cycle
            )
            and not (
                (record_stack is not None or command.get("stack_id") is not None)
                and record_stack != command.get("stack_id")
            )
        ]
        if not candidates:
            continue

        if record_cycle is None and len(
            {command.get("cycle") for command in candidates}
        ) > 1:
            continue
        if record_stack is None and len(
            {command.get("stack_id") for command in candidates}
        ) > 1:
            continue

        metric_timestamp = record.get("_timestamp")
        if metric_timestamp is not None:
            preceding = [
                command
                for command in candidates
                if command.get("_timestamp") is None
                or command["_timestamp"] <= metric_timestamp
            ]
            if preceding:
                candidates = preceding
            else:
                continue
        matched = max(
            candidates,
            key=lambda command: (
                command.get("_timestamp") is not None,
                command.get("_timestamp") or 0,
                command.get("_order", 0),
            ),
        )
        record["wsclean_command"] = matched["command"]
    return records


def _add_image_dynamic_range(records):
    records_with_dynamic_range = []
    for record in records:
        updated_record = dict(record)
        updated_record.pop("dynamic_range", None)
        try:
            maximum = float(updated_record["max_image"])
            minimum = float(updated_record["min_image"])
        except (KeyError, TypeError, ValueError):
            pass
        else:
            if math.isfinite(maximum) and math.isfinite(minimum) and minimum != 0:
                dynamic_range = maximum / abs(minimum)
                if math.isfinite(dynamic_range):
                    updated_record["dynamic_range"] = "{:.4g}".format(dynamic_range)
        records_with_dynamic_range.append(updated_record)
    return records_with_dynamic_range


def _scan_resource_phase_log(run_root, run_started_at=None, start_cycle=0):
    """Read explicit resource phase transitions from their CSV log."""
    csv_path = run_root / "logs" / "resource_phases.csv"
    if not csv_path.is_file():
        return None

    return read_phase_events(csv_path, run_started_at, start_cycle)


def _scan_resource_log(run_root, run_started_at=None, start_cycle=0):
    """Scan and aggregate process-tree resource records from logs/resource_usage.csv."""
    csv_path = run_root / "logs" / "resource_usage.csv"
    if not csv_path.is_file():
        return None

    valid_samples = read_resource_samples(csv_path, run_started_at, start_cycle)

    if not valid_samples:
        return None

    phase_events = _scan_resource_phase_log(run_root, run_started_at, start_cycle)
    phase_intervals = None
    if phase_events is not None:
        # Keep phase bars inside the sampled time range so they line up with
        # the CPU and RAM curves instead of stretching the time axis.
        phase_intervals = clip_phase_intervals_to_window(
            phase_intervals_from_events(
                phase_events, end_epoch=valid_samples[-1]["epoch"]
            ),
            valid_samples[0]["epoch"],
            valid_samples[-1]["epoch"],
        )

    peak_tree_rss_gib = max(s["tree_rss_gib"] for s in valid_samples)
    peak_tree_cpu_pct = max(s["tree_cpu_pct"] for s in valid_samples)
    avg_tree_cpu_pct = sum(s["tree_cpu_pct"] for s in valid_samples) / len(valid_samples)
    def system_peak(key):
        return max(
            (s[key] for s in valid_samples if s[key] is not None), default=None
        )
    peak_sys_ram_used_gib = system_peak("sys_ram_used_gib")
    sys_ram_total_gib = system_peak("sys_ram_total_gib")
    peak_sys_cpu_pct = system_peak("sys_cpu_pct")

    cycle_stats = {}
    for s in valid_samples:
        c = s["cycle"]
        keys = [c]
        if isinstance(c, int):
            keys.extend([str(c), str(c).zfill(3)])
        for k in dict.fromkeys(keys):
            if k not in cycle_stats:
                cycle_stats[k] = {
                    "cycle": c,
                    "peak_tree_rss_gib": s["tree_rss_gib"],
                    "peak_tree_cpu_pct": s["tree_cpu_pct"],
                    "cpu_sum": s["tree_cpu_pct"],
                    "count": 1,
                }
            else:
                cs = cycle_stats[k]
                cs["peak_tree_rss_gib"] = max(cs["peak_tree_rss_gib"], s["tree_rss_gib"])
                cs["peak_tree_cpu_pct"] = max(cs["peak_tree_cpu_pct"], s["tree_cpu_pct"])
                cs["cpu_sum"] += s["tree_cpu_pct"]
                cs["count"] += 1

    for cs in cycle_stats.values():
        cs["avg_tree_cpu_pct"] = cs["cpu_sum"] / cs["count"]

    chart_samples = downsample_resource_samples(
        valid_samples, _FULL_RESOURCE_CHART_POINT_LIMIT
    )

    return {
        "present": True,
        "samples": chart_samples,
        "all_samples": valid_samples,
        "recent_samples": valid_samples[-DEFAULT_RESOURCE_SAMPLE_COUNT:],
        "phase_intervals": phase_intervals,
        "sample_count": len(valid_samples),
        "cycle_stats": cycle_stats,
        "peak_tree_rss_gib": peak_tree_rss_gib,
        "peak_tree_cpu_pct": peak_tree_cpu_pct,
        "avg_tree_cpu_pct": avg_tree_cpu_pct,
        "peak_sys_ram_used_gib": peak_sys_ram_used_gib,
        "sys_ram_total_gib": sys_ram_total_gib,
        "peak_sys_cpu_pct": peak_sys_cpu_pct,
    }


def _generate_resource_svg(samples, phase_intervals=None):
    return generate_resource_svg(samples, phase_intervals)


def _resource_chart_legend_html():
    return (
        '<div class="resource-chart-legend">'
        '<span class="legend-item"><span class="legend-swatch" style="background:#0d9488;"></span> Process Tree CPU (% of one core)</span>'
        f'<span class="legend-item"><span class="legend-swatch" style="background:{RESOURCE_RAM_COLOR};"></span> Process Tree RAM (GiB)</span>'
        '<span class="legend-item" style="color:var(--muted);"><span class="legend-swatch" style="border-top:2px dashed #94a3b8; background:transparent;"></span> Cycle transition</span>'
        '</div>'
    )


def _render_resource_section(resources, include_chart=True, live_chart_frame=None):
    """Render summary cards, interactive chart, and cycle table for resources."""
    if not resources or not resources.get("samples"):
        return '<p class="empty">No resource usage samples recorded.</p>'

    def system_value(value, suffix):
        return "Unavailable" if value is None else format(value, suffix)
    cards = [
        f'<div class="resource-card"><strong>Peak Process RAM</strong><span>{resources["peak_tree_rss_gib"]:.2f} GiB</span></div>',
        f'<div class="resource-card"><strong>Peak Process CPU</strong><span>{resources["peak_tree_cpu_pct"]:.0f}%</span></div>',
        f'<div class="resource-card"><strong>Average Process CPU</strong><span>{resources["avg_tree_cpu_pct"]:.0f}%</span></div>',
        '<div class="resource-card"><strong>Peak System RAM</strong><span>{} / {} GiB</span></div>'.format(
            system_value(resources["peak_sys_ram_used_gib"], ".1f"),
            system_value(resources["sys_ram_total_gib"], ".1f"),
        ),
        '<div class="resource-card"><strong>Peak System CPU</strong><span>{}</span></div>'.format(
            "Unavailable" if resources["peak_sys_cpu_pct"] is None
            else "{:.0f}%".format(resources["peak_sys_cpu_pct"])
        ),
    ]

    chart_box = ""
    if include_chart:
        phase_intervals = resources.get("phase_intervals")
        full_samples = resources["samples"]
        all_samples = resources.get("all_samples") or full_samples
        recent_samples = resources.get("recent_samples") or all_samples[
            -DEFAULT_RESOURCE_SAMPLE_COUNT:
        ]
        # While monitoring is active the live chart page also draws the full range, so it is
        # not limited to the samples of the last report update
        full_view = (
            _resource_chart_live_iframe(
                "resource-chart-live.html?sample_limit=all",
                "Live process tree CPU and RAM chart, full time range",
            )
            if live_chart_frame
            else _resource_chart_legend_html()
            + '<div class="chart-container">{}</div>'.format(
                _generate_resource_svg(full_samples, phase_intervals)
            )
        )

        if len(all_samples) <= 1:
            chart_box = (
                live_chart_frame
                if live_chart_frame
                else '<div class="resource-chart-box">{}</div>'.format(full_view)
            )
        else:
            recent_intervals = clip_phase_intervals_to_window(
                phase_intervals,
                recent_samples[0]["epoch"],
                recent_samples[-1]["epoch"],
            )
            recent_view = (
                live_chart_frame
                if live_chart_frame
                else _resource_chart_legend_html()
                + '<div class="chart-container" data-resource-chart-target>{}</div>'.format(
                    _generate_resource_svg(recent_samples, recent_intervals)
                )
            )
            chart_data = resource_chart_data_json(all_samples, phase_intervals)
            chart_box = (
                '<div class="resource-chart-box">'
                '<div class="resource-chart-range" data-resource-range-container>'
                '<div class="resource-range-switch" role="group" '
                'aria-label="Resource chart time range">'
                '<span class="resource-range-label">Time range</span>'
                '<button type="button" class="resource-range-button" '
                'data-resource-range-button="full" aria-pressed="true">'
                'Full time range</button>'
                '<label class="resource-range-count">Recent window '
                '<input type="number" min="1" step="1" value="{}" '
                'data-resource-sample-count '
                'aria-label="Number of latest resource samples" '
                'aria-describedby="resource-sample-count-status">'
                ' samples</label>'
                '<button type="button" class="resource-range-button" '
                'data-resource-range-button="recent" aria-pressed="false">'
                'Latest {} samples</button>'
                '<span id="resource-sample-count-status" '
                'class="resource-range-count-status" '
                'data-resource-sample-count-status role="status" '
                'aria-live="polite" hidden></span></div>'
                '<script type="application/json" '
                'data-resource-chart-data>{}</script>'
                '<div class="resource-chart-range-view" '
                'data-resource-range-view="full">{}</div>'
                '<div class="resource-chart-range-view" '
                'data-resource-range-view="recent" hidden>{}</div>'
                '</div></div>'.format(
                    DEFAULT_RESOURCE_SAMPLE_COUNT,
                    len(recent_samples),
                    chart_data,
                    full_view,
                    recent_view,
                )
            )

    cycle_stats = resources.get("cycle_stats", {})
    unique_cycles = []
    seen = set()
    for s in resources.get("all_samples", resources["samples"]):
        c = s["cycle"]
        if c not in seen:
            seen.add(c)
            unique_cycles.append(c)

    cycle_table = ""
    if len(unique_cycles) > 1 or (unique_cycles and unique_cycles[0] not in ("", "init")):
        rows = []
        for c in unique_cycles:
            cs = cycle_stats.get(c)
            if not cs:
                continue
            rows.append(
                f'<tr><th scope="row">Cycle {_escape(str(c))}</th>'
                f'<td class="numeric">{cs["peak_tree_rss_gib"]:.2f} GiB</td>'
                f'<td class="numeric">{cs["peak_tree_cpu_pct"]:.0f}%</td>'
                f'<td class="numeric">{cs["avg_tree_cpu_pct"]:.0f}%</td>'
                f'<td class="numeric">{cs["count"]}</td></tr>'
            )
        if rows:
            cycle_table = (
                '<div class="table-scroll"><table class="data-table"><thead><tr>'
                '<th scope="col">Cycle</th>'
                '<th scope="col">Peak RAM</th>'
                '<th scope="col">Peak CPU</th>'
                '<th scope="col">Avg CPU</th>'
                '<th scope="col">Samples</th>'
                f'</tr></thead><tbody>{"".join(rows)}</tbody></table></div>'
            )

    dp3_phases = {"phaseup", "average", "phaseshift", "filter", "aoflagger"}
    phase_note = ""
    if any(
        interval.get("phase") in dp3_phases
        for interval in resources.get("phase_intervals") or []
    ):
        phase_note = (
            '<p class="resource-phase-note">DP3 step bars show the full parent-command '
            'interval; individual DP3 step timings are not available.</p>'
        )

    return (
        f'<div class="resource-summary-grid">{"".join(cards)}</div>\n'
        f'{chart_box}\n'
        f'{phase_note}\n'
        f'{cycle_table}'
    )


def _resource_live_frame(site_dir):
    if not (site_dir / "resource-live.html").is_file():
        return ""
    return (
        '<iframe class="resource-live-frame" src="resource-live.html" '
        'title="Live process and system CPU and RAM usage"></iframe>'
    )


def _resource_chart_live_frame(site_dir):
    chart_page = site_dir / "resource-chart-live.html"
    if not chart_page.is_file():
        return ""
    try:
        with chart_page.open(encoding="utf-8", errors="replace") as stream:
            page_head = stream.read(2048)
    except OSError as exc:
        report_warning("Live chart availability could not be read from {}: {}".format(chart_page, exc))
        return ""
    if '<meta http-equiv="refresh"' not in page_head:
        return ""
    return _resource_chart_live_iframe(
        "resource-chart-live.html", "Live process tree CPU and RAM chart"
    )


def _log_tail_live_frame(site_dir):
    """The resource monitor's live log excerpt, while it is refreshing (empty otherwise)."""
    page = site_dir / "log-tail-live.html"
    if not page.is_file():
        return ""
    try:
        with page.open(encoding="utf-8", errors="replace") as stream:
            page_head = stream.read(1024)
    except OSError as exc:
        report_warning("Live log availability could not be read from {}: {}".format(page, exc))
        return ""
    if '<meta http-equiv="refresh"' not in page_head:
        return ""
    return (
        '<iframe class="log-tail-live-frame" src="log-tail-live.html" '
        'title="Live excerpt of selfcal.log"></iframe>'
    )


def _resource_chart_live_iframe(src, title):
    return '<iframe class="resource-chart-live-frame" src="{}" title="{}"></iframe>'.format(
        _escape(src), _escape(title)
    )


def _get_cycle_config(config, cycle_idx):
    soltypes = _as_list(config.get("soltype_list"))
    soltypecycles = config.get("soltypecycles_list")
    solints = _as_list(config.get("solint_list"))
    smoothness = _as_list(config.get("smoothnessconstraint_list"))

    param_indices = []
    if isinstance(soltypecycles, (list, tuple)) and soltypecycles:
        for i, bound in enumerate(soltypecycles):
            try:
                if int(bound) <= cycle_idx:
                    param_indices.append(i)
            except (ValueError, TypeError):
                pass

    if not param_indices:
        param_indices = [0]

    def values_for_active_types(values):
        if not values:
            return ["-"]
        return [values[i] if i < len(values) else values[0] for i in param_indices]

    return {
        "soltype": values_for_active_types(soltypes),
        "solint": values_for_active_types(solints),
        "smoothness": values_for_active_types(smoothness),
    }


def _cycle_config_values_html(values, formatter=None):
    return "<br>".join(
        '<code>{}</code>'.format(_escape(formatter(value) if formatter else value))
        for value in values
    )


def _format_overview_interval(value):
    if value is None:
        return "-"
    return re.sub(r"(?<=\d)(?=[A-Za-z])", " ", str(value).strip())


def _format_overview_smoothness(value):
    if value is None:
        return "-"
    text = str(value).strip()
    if not text or text == "-":
        return "-"
    if re.search(r"\sMHz$", text, re.IGNORECASE):
        return re.sub(r"\s*MHz$", " MHz", text, flags=re.IGNORECASE)
    return "{} MHz".format(text)


def _measurement_set_metadata(run_root):
    path = run_root / "logs" / "selfcal.log"
    if not path.is_file():
        return {}

    metadata = {}
    current_metadata = None
    try:
        with path.open("r", encoding="utf-8", errors="replace") as stream:
            for line_index, raw_line in enumerate(stream):
                _, _, message = _parse_log_line(raw_line.rstrip("\r\n"))
                normalized_message = message.strip()
                if normalized_message == "================":
                    current_metadata = None
                    continue
                if normalized_message.startswith("===") and normalized_message.endswith("==="):
                    ms_path = normalized_message[3:-3].strip()
                    if ms_path:
                        # A later invocation (restart) logs the block again;
                        # keep the latest values and remember their position.
                        current_metadata = {"_source_path": ms_path, "_order": line_index}
                        metadata[_resolved_ms_path(ms_path, run_root)] = current_metadata
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
                        value = value.strip()
                        if field == "Number of channels":
                            try:
                                channel_count = float(value)
                            except ValueError:
                                report_warning(
                                    "Could not parse number of channels from "
                                    "{}: {}".format(path, value)
                                )
                            else:
                                if (
                                    math.isfinite(channel_count)
                                    and channel_count.is_integer()
                                ):
                                    value = str(int(channel_count))
                                else:
                                    report_warning(
                                        "Number of channels in {} is not an "
                                        "integer: {}".format(path, value)
                                    )
                        current_metadata[field] = value
                        break
    except OSError as exc:
        report_warning("Measurement-set metadata could not be read from {}: {}".format(path, exc))
        return metadata
    return metadata


def _measurement_set_metadata_for_path(path, metadata_by_ms, run_root):
    resolved_path = _resolved_ms_path(path, run_root)
    exact_match = metadata_by_ms.get(resolved_path)
    if exact_match is not None:
        return exact_match

    canonical_path = _canonical_ms_path(path, run_root)
    canonical_matches = [
        metadata
        for metadata_path, metadata in metadata_by_ms.items()
        if _canonical_ms_path(
            metadata.get("_source_path", metadata_path), run_root
        )
        == canonical_path
    ]
    if canonical_matches:
        return _latest_ms_metadata(canonical_matches)

    basename = _ms_identity_name(path)
    if not basename:
        return {}
    matches = [
        metadata
        for metadata_path, metadata in metadata_by_ms.items()
        if _ms_identity_name(
            Path(metadata.get("_source_path", metadata_path)).name
        )
        == basename
    ]
    return _latest_ms_metadata(matches) if matches else {}


def _latest_ms_metadata(matches):
    """Pick the most recently logged block among working copies of one MS.

    Invocations with different averaging log the same observation under
    different working-copy names (``.copy``, ``.copy.avg``); they all match the
    configured input, so the latest block describes the current run.
    """
    return max(matches, key=lambda metadata: metadata.get("_order", -1))


def _resolved_ms_path(path, run_root):
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = run_root / candidate
    return os.path.normpath(str(candidate))


def _canonical_ms_path(path, run_root):
    return _strip_ms_copy_avg_suffix(_resolved_ms_path(path, run_root))


def _strip_ms_copy_avg_suffix(path):
    return re.sub(r"(?:\.(?:copy|avg))+$", "", str(path), flags=re.IGNORECASE)


def _strip_ms_processing_suffixes(path):
    return re.sub(
        r"(?:\.(?:copy|avg|cut|phaseup))+$", "", str(path), flags=re.IGNORECASE
    )


def _ms_identity_path(path, run_root):
    return _strip_ms_processing_suffixes(_resolved_ms_path(path, run_root))


def _ms_identity_name(path):
    return Path(_strip_ms_processing_suffixes(path)).name.casefold()


def _ms_plot_name_candidates(ms_paths):
    names = []
    for ms_path in ms_paths:
        name = Path(ms_path).name
        while name:
            if name not in names:
                names.append(name)
            stripped_name = re.sub(
                r"\.(?:copy|avg|cut|phaseup)$", "", name, flags=re.IGNORECASE
            )
            if stripped_name == name:
                break
            name = stripped_name
    return tuple(names)


def _split_ms_parent_name(ms_path):
    name = _strip_ms_processing_suffixes(Path(ms_path).name)
    match = re.fullmatch(
        r"(?P<parent>.+)_chunk_\d+(?:\.ms)?", name, flags=re.IGNORECASE
    )
    return _strip_ms_processing_suffixes(match.group("parent")) if match else None


def _split_ms_parent_path(ms_path, run_root):
    resolved_path = _resolved_ms_path(ms_path, run_root)
    name = _strip_ms_processing_suffixes(Path(resolved_path).name)
    match = re.fullmatch(
        r"(?P<parent>.+)_chunk_\d+(?:\.ms)?", name, flags=re.IGNORECASE
    )
    if match is None:
        return None
    parent_path = Path(resolved_path).with_name(match.group("parent"))
    return _ms_identity_path(parent_path, run_root)


def _related_ms_plot_paths(path, ms_records, input_name_counts, run_root):
    identity_path = _ms_identity_path(path, run_root)
    identity_name = Path(identity_path).name.casefold()
    matching_records = []
    exact_matches = []
    for _, display_path in ms_records:
        family_path = _ms_identity_path(display_path, run_root)
        if Path(family_path).name.casefold() != identity_name:
            continue
        matching_records.append((family_path, display_path))
        if family_path == identity_path:
            exact_matches.append(display_path)

    if exact_matches:
        candidates = exact_matches
    elif input_name_counts.get(identity_name, 0) == 1:
        family_paths = {family_path for family_path, _ in matching_records}
        if len(family_paths) != 1:
            return ()
        candidates = [
            display_path
            for family_path, display_path in matching_records
            if family_path in family_paths
        ]
    else:
        return ()

    return tuple(dict.fromkeys(candidates))


def _match_ms_plots(
    ms_path, ms_plot_files, ms_json_files, related_ms_paths=()
):
    names = _ms_plot_name_candidates(
        (ms_path, *(related_ms_paths or ()))
    )
    plot_files_by_name = {path.name.casefold(): path for path in ms_plot_files}
    json_files_by_name = {path.name.casefold(): path for path in ms_json_files}

    def find_exact_match(files_by_name, prefix, suffix):
        for candidate in names:
            match = files_by_name.get(
                "{}{}{}".format(prefix, candidate, suffix).casefold()
            )
            if match is not None:
                return match
        return None

    return {
        "time_coverage": find_exact_match(
            plot_files_by_name, "", ".time_coverage.png"
        ),
        "uv_coverage": find_exact_match(
            plot_files_by_name, "UVcoverage_", ".png"
        ),
        "ateam_png": find_exact_match(plot_files_by_name, "Ateam_", ".png"),
        "ateam_json": find_exact_match(json_files_by_name, "Ateam_", ".json"),
    }


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
        '<figure class="{}" data-search="{}" data-filter-values="{}">'
        '<a href="{}" title="Open full-size image">'
        '<img src="{}" alt="{}" loading="lazy" decoding="async"></a>'
        '<figcaption><code>{}</code>{}</figcaption></figure>'
    ).format(
        _escape(item_class),
        _escape(search_text),
        _escape(extra_search),
        _escape(href),
        _escape(href),
        _escape(caption),
        _escape(caption),
        detail_html,
    )


def _quality_plot_placeholder(
    title, message="Not generated for this measurement set."
):
    return (
        '<div class="ms-quality-placeholder" role="status">'
        '<strong>{}</strong><span>{}</span></div>'
    ).format(_escape(title), _escape(message))


def _plain_plot(path, plain_plots):
    return (plain_plots or {}).get(path.stem.casefold())


def _overlay_toggle_html(paths, plain_plots):
    """Checkbox to switch between plain images and images with overlays."""
    if not any(_plain_plot(path, plain_plots) for path in paths):
        return ""
    return (
        '<label class="overlay-toggle"><span><input type="checkbox" data-overlay-toggle> '
        'Show axes, grid, mask and facets</span></label>'
    )


def _blink_controls(paths, base_dir, plain_plots=None):
    if len(paths) < 2:
        return ""

    frames = []
    for path in paths:
        frame = {"src": _relative_url(path, base_dir), "label": path.stem}
        plain = _plain_plot(path, plain_plots)
        if plain is not None:
            frame["plain"] = _relative_url(plain, base_dir)
        frames.append(frame)
    first_href = frames[0].get("plain", frames[0]["src"])
    return (
        '<div class="blink-tool" data-blink-tool data-blink-images="{}">'
        '<div class="blink-controls">'
        '<label class="blink-rate">Blink interval'
        '<span><input type="range" min="100" max="2000" step="100" value="500" '
        'data-blink-interval aria-label="Blink interval in milliseconds">'
        '<output data-blink-interval-output>500 ms</output></span></label>'
        '<button type="button" data-blink-toggle aria-pressed="false">Start blinking</button>'
        '{}'
        '</div>'
        '<figure class="blink-preview"><img data-blink-image src="{}" alt="{}">'
        '<figcaption><code data-blink-caption>{}</code></figcaption></figure>'
        '</div>'
    ).format(
        _escape(json.dumps(frames, separators=(",", ":"))),
        _overlay_toggle_html(paths, plain_plots),
        _escape(first_href),
        _escape(paths[0].stem),
        _escape(paths[0].stem),
    )


def _compare_controls(paths, base_dir, plain_plots=None):
    if len(paths) < 2:
        return ""

    def option(path, selected=False):
        plain = _plain_plot(path, plain_plots)
        return '<option value="{}"{}{}>{}</option>'.format(
            _escape(_relative_url(path, base_dir)),
            ' data-plain="{}"'.format(_escape(_relative_url(plain, base_dir))) if plain else "",
            ' selected="selected"' if selected else "",
            _escape(path.stem),
        )

    options_a = "".join(option(p) for p in paths)
    options_b = "".join(
        option(p, selected=idx == len(paths) - 1) for idx, p in enumerate(paths)
    )
    first_plain = _plain_plot(paths[0], plain_plots)
    last_plain = _plain_plot(paths[-1], plain_plots)
    first_href = _relative_url(first_plain or paths[0], base_dir)
    last_href = _relative_url(last_plain or paths[-1], base_dir)
    return (
        '<div class="compare-tool" data-compare-tool>'
        '<div class="compare-controls">'
        '<label><span>Reference image</span><select data-compare-select-a>{}</select></label>'
        '<label><span>Comparison image</span><select data-compare-select-b>{}</select></label>'
        '{}'
        '</div>'
        '<div class="compare-views">'
        '<figure><a href="{}" title="Open full size" data-compare-link-a><img data-compare-img-a src="{}" alt="{}"></a>'
        '<figcaption><code data-compare-caption-a>{}</code></figcaption></figure>'
        '<figure><a href="{}" title="Open full size" data-compare-link-b><img data-compare-img-b src="{}" alt="{}"></a>'
        '<figcaption><code data-compare-caption-b>{}</code></figcaption></figure>'
        '</div>'
        '</div>'
    ).format(
        options_a,
        options_b,
        _overlay_toggle_html(paths, plain_plots),
        _escape(first_href),
        _escape(first_href),
        _escape(paths[0].stem),
        _escape(paths[0].stem),
        _escape(last_href),
        _escape(last_href),
        _escape(paths[-1].stem),
        _escape(paths[-1].stem),
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


def _write_page(
    path,
    title,
    active,
    body,
    nested=False,
    subtitle="Offline processing report",
    optional_pages=(),
):
    path.parent.mkdir(parents=True, exist_ok=True)
    stage_page(
        path,
        _page_shell(
            title,
            active,
            body,
            nested=nested,
            subtitle=subtitle,
            optional_pages=optional_pages,
        ),
    )


def _jy_display_scale(values):
    magnitude = 0.0
    for value in values:
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(numeric):
            magnitude = max(magnitude, abs(numeric))

    if magnitude >= 1.0:
        return 1.0, "Jy"
    if magnitude >= 1e-3:
        return 1e3, "mJy"
    return 1e6, "\u03bcJy"


def _format_jy_value(value, scale=None, unit=None):
    if value is None:
        return "-"
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(numeric):
        return "{:.5g} Jy".format(numeric)
    if scale is None or unit is None:
        scale, unit = _jy_display_scale((numeric,))
    return "{:.5g} {}".format(numeric * scale, unit)


def _format_image_metric_value(field, value):
    if field in _JY_IMAGE_METRICS:
        return _format_jy_value(value)
    return str(value)


def _image_metric_cycle_cell(records, field, stack_mode=False):
    if stack_mode:
        latest_by_stack = {}
        for record in records:
            stack_id = record.get("stack_id")
            key = (
                ("stack", str(stack_id))
                if stack_id is not None
                else ("image", record.get("image", ""))
            )
            latest_by_stack[key] = record
        records = list(latest_by_stack.values())

    values = [
        '<span title="{}">{}</span>'.format(
            _escape(record["image"]),
            _escape(
                "{}: {}".format(
                    record.get("stack_label")
                    or "Stack {}".format(record.get("stack_id")),
                    _format_image_metric_value(field, record[field]),
                )
                if stack_mode and record.get("stack_id") is not None
                else _format_image_metric_value(field, record[field])
            ),
        )
        for record in records
        if record.get(field) is not None
    ]
    if stack_mode and len(values) > 1:
        return (
            '<td class="numeric"><details class="stack-metric-details">'
            '<summary>{} groups</summary><div>{}</div></details></td>'
        ).format(len(values), "<br>".join(values))
    return '<td class="numeric">{}</td>'.format(
        "<br>".join(values) if values else "&mdash;"
    )


def _primary_image_rank(record):
    """Rank image records so the flat-noise MFS image represents its cycle."""
    name = Path(str(record.get("image", ""))).name.casefold()
    for suffix in (".gz",):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
    if name.endswith("-mfs-image.fits"):
        return 2
    if name.endswith("-image.fits") and not re.search(r"-\d{4}-image\.fits$", name):
        return 1
    return 0


def _image_metrics_chart_by_cycle(records):
    # Prefer -MFS-image.fits, then -image.fits (one output channel), so extra
    # statistics lines logged later in a cycle do not replace the science image.
    latest_by_cycle = {}
    for record in records:
        cycle = record.get("cycle")
        if cycle is None:
            continue
        current = latest_by_cycle.get(cycle)
        if current is None or _primary_image_rank(record) >= _primary_image_rank(current):
            latest_by_cycle[cycle] = record

    cycles = sorted(
        latest_by_cycle,
        key=lambda value: (0, int(value)) if value.isdigit() else (1, value.casefold()),
    )
    if not cycles:
        return '<p class="empty">No cycle numbers were found for the logged image statistics.</p>'

    width = max(640, 125 + 82 * len(cycles))
    row_height = 112
    height = row_height * len(_IMAGE_METRIC_FIELDS) + 14
    plot_left = 108
    plot_right = width - 24
    if len(cycles) == 1:
        x_positions = {cycles[0]: (plot_left + plot_right) / 2}
    else:
        x_positions = {
            cycle: plot_left + index * (plot_right - plot_left) / (len(cycles) - 1)
            for index, cycle in enumerate(cycles)
        }

    parts = [
        '<div class="image-metrics-chart"><svg xmlns="http://www.w3.org/2000/svg" '
        'width="{}" height="{}" viewBox="0 0 {} {}" role="img" '
        'aria-label="Image statistics by self-calibration cycle">'.format(
            width, height, width, height
        )
    ]
    for row_index, (field, label) in enumerate(_IMAGE_METRIC_FIELDS):
        row_top = 8 + row_index * row_height
        plot_top = row_top + 36
        plot_bottom = row_top + 82
        metric_class = field.replace("_", "-")
        values = {}
        for cycle in cycles:
            try:
                value = float(latest_by_cycle[cycle].get(field))
            except (TypeError, ValueError):
                continue
            if math.isfinite(value):
                values[cycle] = value

        display_scale = 1.0
        display_unit = "Jy"
        display_label = label
        if field in _JY_IMAGE_METRICS and values:
            display_scale, display_unit = _jy_display_scale(values.values())
            display_label = "{} ({})".format(label, display_unit)

        parts.append(
            '<text x="8" y="{}" class="metric-chart-title">{}</text>'.format(
                row_top + 16, _escape(display_label)
            )
        )
        if not values:
            parts.append(
                '<text x="{}" y="{}" class="metric-chart-axis">No finite values</text>'.format(
                    plot_left, plot_top + 30
                )
            )
            continue

        min_value = min(values.values())
        max_value = max(values.values())
        value_range = max_value - min_value
        padding = value_range * 0.08 if value_range else max(abs(min_value) * 0.05, 1e-12)
        scale_min = min_value - padding
        scale_max = max_value + padding
        for grid_index in range(3):
            fraction = grid_index / 2
            y = plot_top + fraction * (plot_bottom - plot_top)
            grid_value = scale_max - fraction * (scale_max - scale_min)
            axis_value = (
                grid_value * display_scale
                if field in _JY_IMAGE_METRICS
                else grid_value
            )
            parts.append(
                '<line x1="{:.1f}" y1="{:.1f}" x2="{:.1f}" y2="{:.1f}" class="metric-chart-grid"/>'.format(
                    plot_left, y, plot_right, y
                )
            )
            parts.append(
                '<text x="{}" y="{:.1f}" text-anchor="end" dominant-baseline="middle" '
                'class="metric-chart-axis">{}</text>'.format(
                    plot_left - 8, y, _escape("{:.3g}".format(axis_value))
                )
            )

        points = []
        for cycle in cycles:
            if cycle not in values:
                continue
            value = values[cycle]
            x = x_positions[cycle]
            y = plot_bottom - (value - scale_min) / (scale_max - scale_min) * (
                plot_bottom - plot_top
            )
            points.append((x, y, cycle))

        if len(points) > 1:
            point_values = " ".join(
                "{:.1f},{:.1f}".format(x, y) for x, y, _ in points
            )
            parts.append(
                '<polyline points="{}" class="metric-chart-line metric-chart-line-{}"/>'.format(
                    point_values, metric_class
                )
            )
        for x, y, cycle in points:
            raw_value = latest_by_cycle[cycle].get(field, "")
            if field in _JY_IMAGE_METRICS:
                raw_value = _format_jy_value(
                    raw_value, display_scale, display_unit
                )
            parts.append(
                '<circle cx="{:.1f}" cy="{:.1f}" r="3.5" '
                'class="metric-chart-point metric-chart-point-{}">'
                '<title>Cycle {} - {}: {}</title></circle>'.format(
                    x,
                    y,
                    metric_class,
                    _escape(cycle),
                    _escape(label),
                    _escape(raw_value),
                )
            )
        for cycle in cycles:
            parts.append(
                '<text x="{:.1f}" y="{:.1f}" text-anchor="middle" '
                'class="metric-chart-axis">{}</text>'.format(
                    x_positions[cycle], plot_bottom + 18, _escape(cycle)
                )
            )

    parts.append("</svg></div>")
    return "".join(parts)


def _image_metrics_chart(records):
    stack_ids = {
        str(record["stack_id"])
        for record in records
        if record.get("stack_id") is not None
    }
    if not stack_ids:
        return _image_metrics_chart_by_cycle(records)

    records_by_stack = defaultdict(list)
    ungrouped_records = []
    for record in records:
        stack_id = record.get("stack_id")
        if stack_id is None:
            ungrouped_records.append(record)
        else:
            records_by_stack[str(stack_id)].append(record)

    charts = []
    for stack_id in sorted(
        stack_ids,
        key=lambda value: (0, int(value)) if value.isdigit() else (1, value),
    ):
        stack_records = records_by_stack[stack_id]
        label = next(
            (
                record["stack_label"]
                for record in stack_records
                if record.get("stack_label")
            ),
            "Stack {}".format(stack_id),
        )
        charts.append(
            '<div class="stack-metric-chart"><h4>{}</h4>{}</div>'.format(
                _escape(label),
                _image_metrics_chart_by_cycle(stack_records),
            )
        )
    if ungrouped_records:
        charts.append(
            '<div class="stack-metric-chart"><h4>Other images</h4>{}</div>'.format(
                _image_metrics_chart_by_cycle(ungrouped_records)
            )
        )
    return '<div class="stack-metric-charts">{}</div>'.format("".join(charts))


def _image_metrics_content(records):
    if not records:
        return '<p class="empty">No image extrema or RMS statistics were found in selfcal.log or the primary cycle FITS images.</p>'

    rows = []
    for row_index, record in enumerate(records):
        cycle = record.get("cycle")
        cycle_text = "Cycle {}".format(cycle) if cycle is not None else "Unknown"
        if record.get("stack_id") is not None:
            cycle_text = "{} · {}".format(
                cycle_text,
                record.get("stack_label")
                or "Stack {}".format(record["stack_id"]),
            )
        image_name = record.get("image", "")
        image_label = Path(image_name).name or image_name
        command = record.get("wsclean_command")
        if command:
            command_id = "image-wsclean-command-{}".format(row_index)
            image_cell = (
                '<button type="button" class="image-command-trigger" '
                'popovertarget="{}" aria-label="Show WSClean command for {}" '
                'title="Show WSClean command">{}</button>'
                '<pre class="workflow-command-text" id="{}" popover="auto">{}</pre>'
            ).format(
                command_id,
                _escape(image_label),
                _escape(image_label),
                command_id,
                _escape(command),
            )
        else:
            image_cell = (
                '<span class="image-command-name">{}</span>'
                '<span class="image-command-unavailable">'
                'WSClean command unavailable</span>'
            ).format(_escape(image_label))
        restoring_beam = record.get("restoring_beam")
        if not isinstance(restoring_beam, dict):
            restoring_beam = _unavailable_restoring_beam("FITS image unavailable")
        metric_cells = "".join(
            '<td class="numeric">{}</td>'.format(
                _escape(_format_image_metric_value(field, record.get(field, "-")))
            )
            for field, _ in _IMAGE_METRIC_FIELDS
        )
        rows.append(
            '<tr><th scope="row">{}</th><td>{}</td><td>{}</td>{}</tr>'.format(
                _escape(cycle_text),
                image_cell,
                _restoring_beam_html(restoring_beam),
                metric_cells,
            )
        )

    headers = "".join(
        '<th scope="col">{}</th>'.format(_escape(label))
        for _, label in _IMAGE_METRIC_FIELDS
    )
    metrics_note = (
        "Each chart uses the MFS image of each cycle within that imaging group; "
        "the table lists every image with metrics. Values come from selfcal.log "
        "when present, otherwise from the primary cycle FITS image."
        if any(record.get("stack_id") is not None for record in records)
        else "The chart uses the MFS image of each cycle; the table lists every "
        "image with metrics. Values come from selfcal.log when present, otherwise "
        "from the primary cycle FITS image."
    )
    return (
        _image_metrics_chart(records)
        + '<p class="image-metrics-note">{} Restoring-beam details are read from '
        "each image FITS header; unavailable values are identified explicitly.</p>".format(
            _escape(metrics_note)
        )
        + '<div class="table-scroll"><table class="data-table image-metrics-table"><thead><tr>'
        '<th scope="col">Cycle</th><th scope="col">Image</th>'
        '<th scope="col">Restoring beam</th>{}</tr></thead>'
        '<tbody>{}</tbody></table></div>'.format(headers, "".join(rows))
    )


def _overview_page(site_dir, run_root, config, artifacts, logs, status, error):
    title = str(config.get("imagename") or run_root.name)
    stack_mode = _stack_enabled(config)
    progression_config = dict(config)
    progression_config.update(logs.get("effective_cycle_config", {}))
    status_text, status_class = _status(status)
    ms_inputs = _as_list(config.get("ms"))
    telescope_val = config.get("telescope")
    if not telescope_val:
        for meta in _measurement_set_metadata(run_root).values():
            t = meta.get("Telescope")
            if t:
                telescope_val = t
                break
    if not telescope_val:
        telescope_val = "Unknown"

    try:
        start_cycle = max(0, int(config.get("start", 0)))
    except (TypeError, ValueError):
        start_cycle = 0
    observed_cycles = []
    for path in artifacts.get("solution_files", ()):
        if not path.name.lower().startswith("merged_"):
            continue
        match = re.search(r"selfcalcycle(\d+)", path.name, re.IGNORECASE)
        if match:
            observed_cycles.append(int(match.group(1)))
    cycle_count = max(
        [start_cycle] + [cycle + 1 for cycle in observed_cycles]
    )
    try:
        expected_cycles = int(config.get("stop"))
    except (TypeError, ValueError):
        expected_cycles = 0
    if expected_cycles > 0:
        cycle_count = min(cycle_count, expected_cycles)
        cycle_value = "{}/{}".format(cycle_count, expected_cycles)
    else:
        cycle_value = str(cycle_count)

    solve_type = (
        "Direction-dependent (DD)"
        if config.get("DDE") is True
        else "Direction independent (DI)"
    )
    facetselfcal_version = logs.get("host_info", {}).get("version") or "Not recorded"

    metrics = [
        (facetselfcal_version, "facetselfcal Version"),
        (len(ms_inputs), "Configured MS Datasets"),
        (cycle_value, "Solution Cycles"),
        (solve_type, "Solve Type"),
        (logs["warning_count"] + logs["error_count"], "Warnings and Errors"),
    ]
    if logs.get("total_elapsed"):
        metrics.append((logs["total_elapsed"], "Total Elapsed Time"))
    res = logs.get("resources")

    metric_html_parts = []
    for value, label in metrics:
        metric_content = (
            '<span class="metric-value">{}</span><span class="metric-label">{}</span>'
        ).format(_escape(value), _escape(label))
        if label == "Warnings and Errors":
            metric_html_parts.append(
                '<a class="metric metric-link" href="#warnings">{}</a>'.format(
                    metric_content
                )
            )
        else:
            metric_html_parts.append(
                '<div class="metric">{}</div>'.format(metric_content)
            )
    metric_html = "".join(metric_html_parts)
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
    live_resource_frame = _resource_live_frame(site_dir)
    if live_resource_frame:
        body_parts.append(
            _section("Live Resource Usage", live_resource_frame)
        )

    if error:
        body_parts.append(
            '<div class="notice error"><strong>Run error</strong><p>{}</p></div>'.format(
                _escape(error)
            )
        )
    if logs["invocations"] > 1:
        body_parts.append(
            '<div class="notice info"><strong>Combined run history</strong><p>The saved self-calibration log contains {} command records. This report summarizes the current output directory; some products may come from different run segments.</p></div>'.format(
                logs["invocations"]
            )
        )

    # Host & Execution Environment card
    host_info = logs.get("host_info", {})
    if host_info:
        env_cards = []
        if "host" in host_info:
            env_cards.append('<div class="env-card"><strong>Run Host</strong><span>{}</span></div>'.format(_escape(host_info["host"])))
        if "cpu" in host_info:
            env_cards.append('<div class="env-card"><strong>Processors</strong><span>{}</span></div>'.format(_escape(host_info["cpu"])))
        if "ram" in host_info:
            env_cards.append('<div class="env-card"><strong>RAM (Total / Avail)</strong><span>{}</span></div>'.format(_escape(host_info["ram"])))
        if "disk" in host_info:
            env_cards.append('<div class="env-card"><strong>Disk at Start</strong><span>{}</span></div>'.format(_escape(host_info["disk"])))
        if "os" in host_info:
            env_cards.append('<div class="env-card"><strong>Operating System</strong><span>{}</span></div>'.format(_escape(host_info["os"])))
        if res and res.get("peak_tree_rss_gib") is not None:
            env_cards.append('<div class="env-card"><strong>Peak Process RAM</strong><span>{:.1f} GiB</span></div>'.format(res["peak_tree_rss_gib"]))
        if res and res.get("peak_tree_cpu_pct") is not None:
            env_cards.append('<div class="env-card"><strong>Peak Process CPU</strong><span>{:.0f}%</span></div>'.format(res["peak_tree_cpu_pct"]))
        if env_cards:
            body_parts.append(_section("Execution Environment", '<div class="env-grid">{}</div>'.format("".join(env_cards)), "Startup system snapshot from logs/selfcal.log."))

    setup_step_details = logs.get("setup_step_details", [])
    cycle_timeline = logs.get("cycle_timeline", [])
    image_metrics_by_cycle = defaultdict(list)
    for image_metric in logs.get("image_metrics", []):
        cycle = image_metric.get("cycle")
        if cycle is not None:
            image_metrics_by_cycle[cycle].append(image_metric)
    if cycle_timeline or setup_step_details:
        row_groups = []
        if setup_step_details:
            preparation_steps_html = _workflow_steps_html(
                setup_step_details,
                "setup",
                "Workflow steps for Preparation",
            )
            preparation_start = logs.get("setup_start_str")
            preparation_duration = logs.get("setup_duration_str")
            preparation_metric_cells = "".join(
                _image_metric_cycle_cell([], field)
                for field, _ in _CYCLE_IMAGE_METRIC_FIELDS
            )
            preparation_rows = [
                '<tr class="progression-summary-row">'
                '<th scope="row"><span class="cycle-label">Preparation</span></th>'
                '<td>{}</td>'
                '<td class="numeric"><span class="cycle-duration">{}</span></td>'
                '{}'
                '<td class="nowrap">&mdash;</td>'
                '<td class="numeric">&mdash;</td>'
                '<td class="numeric">&mdash;</td>'
                '<td>{}</td>'
                '</tr>'.format(
                    _escape(preparation_start) if preparation_start else "&mdash;",
                    _escape(preparation_duration)
                    if preparation_duration and preparation_duration != "-"
                    else "&mdash;",
                    preparation_metric_cells,
                    _escape(_workflow_steps_summary(setup_step_details)),
                )
            ]
            preparation_rows.append(
                '<tr class="progression-workflow-row"><td colspan="9">{}</td></tr>'.format(
                    preparation_steps_html
                )
            )
            row_groups.append(
                '<tbody class="progression-cycle-group">{}</tbody>'.format(
                    "".join(preparation_rows)
                )
            )
        for cdata in cycle_timeline:
            c_int = int(cdata["cycle"])
            c_cfg = _get_cycle_config(progression_config, c_int)
            cycle_step_details = cdata.get("step_details", [])
            steps_html = _workflow_steps_html(
                cycle_step_details,
                "cycle-{}".format(cdata["cycle"]),
                "Workflow steps for Cycle {}".format(cdata["cycle"]),
            )
            cycle_metric_records = image_metrics_by_cycle.get(cdata["cycle"], [])
            metric_cells = "".join(
                _image_metric_cycle_cell(
                    cycle_metric_records, field, stack_mode=stack_mode
                )
                for field, _ in _CYCLE_IMAGE_METRIC_FIELDS
            )
            cycle_rows = [
                '<tr class="progression-summary-row">'
                '<th scope="row"><span class="cycle-label">Cycle {}</span></th>'
                '<td>{}</td>'
                '<td class="numeric"><span class="cycle-duration">{}</span></td>'
                '{}'
                '<td>{}</td>'
                '<td class="numeric">{}</td>'
                '<td class="numeric">{}</td>'
                '<td>{}</td>'
                '</tr>'.format(
                    _escape(cdata["cycle"]),
                    _escape(cdata["start_str"]),
                    _escape(cdata["duration_str"]),
                    metric_cells,
                    _cycle_config_values_html(c_cfg["soltype"]),
                    _cycle_config_values_html(c_cfg["solint"], _format_overview_interval),
                    _cycle_config_values_html(c_cfg["smoothness"], _format_overview_smoothness),
                    _escape(_workflow_steps_summary(cycle_step_details)),
                )
            ]
            if cycle_step_details:
                cycle_rows.append(
                    '<tr class="progression-workflow-row"><td colspan="9">{}</td></tr>'.format(
                        steps_html
                    )
                )
            row_groups.append(
                '<tbody class="progression-cycle-group">{}</tbody>'.format(
                    "".join(cycle_rows)
                )
            )
        timeline_html = (
            '<div class="table-scroll"><table class="data-table progression-table"><thead><tr>'
            '<th scope="col">Cycle</th>'
            '<th scope="col">Start Time</th>'
            '<th scope="col">Duration</th>'
            '<th scope="col">RMS noise</th>'
            '<th scope="col">Dynamic range</th>'
            '<th scope="col">Solution Type</th>'
            '<th scope="col">Interval</th>'
            '<th scope="col">Smoothness</th>'
            '<th scope="col">Workflow Steps</th>'
            '</tr></thead>{}</table></div>'.format("".join(row_groups))
        )
        timeline_note = (
            "Timing, parameters, and logged image statistics per calibration cycle. "
            "Preparation lists DP3 work before the first self-calibration cycle when recorded."
        )
        if stack_mode:
            timeline_note += (
                " In stack mode, WSClean steps and image statistics are identified "
                "by their imaging group."
            )
        effective_settings = logs.get("effective_cycle_config", {})
        if effective_settings:
            timeline_note += (
                " Solver settings use effective values recorded in logs/selfcal.log "
                "when available; missing settings fall back to full_config.txt."
            )
        else:
            timeline_note += (
                " No effective solver settings were found in logs/selfcal.log; "
                "the table uses full_config.txt."
            )
        setting_errors = logs.get("effective_cycle_config_errors", [])
        if setting_errors:
            timeline_note += (
                " Could not parse runtime values for {}; those values fall back to "
                "full_config.txt.".format(", ".join(setting_errors))
            )
        all_step_details = setup_step_details + [
            step
            for cdata in cycle_timeline
            for step in cdata.get("step_details", [])
        ]
        if any(
            step.get("shared_command_duration")
            for step in all_step_details
        ):
            timeline_note += (
                " DP3 steps in a combined command share its full duration; "
                "individual DP3 step timings are not available."
            )
        body_parts.append(
            _section("Self-Calibration Progression", timeline_html, timeline_note)
        )

    summary_config = dict(config)
    logged_settings = dict(logs.get("effective_settings", {}))
    logged_settings.update(logs.get("effective_cycle_config", {}))
    summary_config.update(logged_settings)
    if (
        _dde_enabled(config)
        and summary_config.get("facetdirections") in (None, "None")
        and artifacts.get("dde_direction_files")
    ):
        summary_config["facetdirections"] = "Automatic (directions/directions_NNN.txt)"
    summary_html = _config_table(summary_config, _SUMMARY_KEYS)
    summary_note = "Selected values from full_config.txt"
    logged_keys = [key for key in _SUMMARY_KEYS if key in logged_settings]
    if logged_keys:
        summary_note += (
            "; {} are the values logged in selfcal.log after facetselfcal set its "
            "automatic choices".format(", ".join(logged_keys))
        )
    summary_note += "."
    if _dde_enabled(config):
        summary_note += (
            " With --DDE, solution intervals and smoothness can differ per direction; "
            "see the Directions page."
        )
    command_record = logs.get("command_record")
    if command_record:
        summary_html = (
            '<details><summary>Command line</summary><pre>{}</pre></details>'.format(
                _escape(command_record["command"])
            )
            + summary_html
        )
    body_parts.append(_section("Run configuration", summary_html, summary_note))

    overview_plots = artifacts.get("all_overview_plots", artifacts["overview_plots"])
    if overview_plots:
        selected = [p for p in overview_plots if p.name.lower().startswith("im_")]
        if not selected:
            selected = overview_plots
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
                    len(selected)
                ),
            )
        )

    issues_content = _render_log_events(
        logs["warnings"], logs["warning_count"], is_error=False
    )
    body_parts.append(_section("Warnings", issues_content, section_id="warnings"))
    errors_content = _render_log_events(logs["errors"], logs["error_count"], is_error=True)
    body_parts.append(_section("Recent errors", errors_content))
    body_parts.append(
        '<div class="page-links"><a href="calibration.html">Calibration plots: {} files</a><a href="datasets.html">Measurement-set details</a><a href="run-details.html">Full configuration and logs</a></div>'.format(
            sum(len(paths) for _, cycles in artifacts["calibration_sets"] for paths in cycles.values())
        )
    )

    subtitle_parts = ["Run overview", run_root.name]
    if telescope_val and telescope_val != "Unknown":
        subtitle_parts.append("Telescope: {}".format(telescope_val))
    if logs.get("total_elapsed"):
        subtitle_parts.append("Elapsed: {}".format(logs["total_elapsed"]))

    _write_page(
        site_dir / "index.html",
        title,
        "index.html",
        "\n".join(body_parts),
        subtitle=" / ".join(subtitle_parts),
        optional_pages=_optional_pages(config),
    )


def _imaging_page(site_dir, run_root, config, artifacts, logs):
    title = str(config.get("imagename") or run_root.name)
    stack_mode = _stack_enabled(config)
    stack_groups = logs.get("stack_groups", {})
    plots = artifacts.get("all_overview_plots", artifacts["overview_plots"])
    plain_plots = artifacts.get("plain_plots", {})
    fits_files = artifacts.get("all_fits_files", artifacts["fits_files"])
    stacked_progression = defaultdict(list)
    if stack_mode:
        for path in plots:
            stack_identity = _stack_image_identity(path.name)
            if stack_identity is not None:
                cycle, stack_id = stack_identity
                stacked_progression[stack_id].append((cycle, path))

    stacked_paths = {
        path
        for frames in stacked_progression.values()
        for _, path in frames
    }
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
        if (
            path.name.lower().startswith("im_") or path in sequence_paths
        )
        and path not in stacked_paths
    ]
    progression.sort(key=lambda path: sequence_order.get(path, (path.stem.casefold(), -1)))
    diagnostics = [
        path for path in plots
        if path not in progression and path not in stacked_paths
    ]
    sections = []
    if plots:
        sections.append(_filter_input(".image-entry", "Filter plot names"))
        for stack_id in sorted(
            stacked_progression,
            key=lambda value: (0, int(value)) if value.isdigit() else (1, value),
        ):
            frames = sorted(
                stacked_progression[stack_id],
                key=lambda frame: (int(frame[0]), frame[1].name.casefold()),
            )
            group_paths = [path for _, path in frames]
            stack_label = _stack_group_label(stack_id, stack_groups)
            figures = "".join(
                _figure(
                    path,
                    site_dir,
                    path.stem,
                    detail="{} · Cycle {}".format(stack_label, cycle),
                )
                for cycle, path in frames
            )
            stack_sections = []
            if len(group_paths) > 1:
                stack_sections.append(
                    '<details data-filter-group open><summary>Side-by-side comparison</summary>{}</details>'.format(
                        _compare_controls(group_paths, site_dir, plain_plots)
                    )
                )
                stack_sections.append(
                    '<details data-filter-group open><summary>Multi-frame image blinking</summary>{}</details>'.format(
                        _blink_controls(group_paths, site_dir, plain_plots)
                    )
                )
            stack_sections.append(
                '<div class="gallery">{}</div>'.format(figures)
            )
            sections.append(
                '<details class="stack-image-group" data-filter-group open>'
                '<summary>{} · Image progression ({})</summary>{}</details>'.format(
                    _escape(stack_label),
                    len(group_paths),
                    "".join(stack_sections),
                )
            )
        if progression:
            figures = "".join(
                _figure(path, site_dir, path.stem, detail="Imaging overview")
                for path in progression
            )
            blink_controls = _blink_controls(progression, site_dir, plain_plots)
            compare_controls = _compare_controls(progression, site_dir, plain_plots)
            sections.append(
                '<details data-filter-group open><summary>Side-by-side cycle comparison</summary>{}</details>'.format(
                    compare_controls
                )
            )
            sections.append(
                '<details data-filter-group open><summary>Multi-frame image blinking</summary>{}</details>'.format(
                    blink_controls
                )
            )
            sections.append(
                '<details data-filter-group open><summary>Image progression gallery ({})</summary><div class="gallery">{}</div></details>'.format(
                    len(progression), figures
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
    fits_cycles = set()

    for path in fits_files:
        stack_identity = (
            _stack_image_identity(path.name) if stack_mode else None
        )
        if stack_identity is not None:
            fits_cycles.add(stack_identity[0])
        else:
            cm = re.search(r"_(\d{3,})(?:-|\.|$)", path.name)
            if cm:
                fits_cycles.add(cm.group(1))
        for category, pattern in product_categories:
            if pattern.search(path.name):
                fits_groups[category].append(path)
                break
        else:
            fits_groups["Other FITS products"].append(path)

    fit_sections = []
    if fits_files:
        fit_sections.append(_filter_input(".fits-entry", "Filter FITS filenames"))

        # Cycle filter pills for FITS products
        if fits_cycles:
            pills = ['<div class="pill-group" data-pill-group data-pill-target=".fits-entry">']
            pills.append('<span class="pill-label">Filter by Cycle:</span>')
            pills.append('<button type="button" class="pill-btn active" data-filter-value="all">All cycles</button>')
            for cy in sorted(fits_cycles):
                pills.append('<button type="button" class="pill-btn" data-filter-value="cycle-{}">Cycle {}</button>'.format(cy, cy))
            pills.append('</div>')
            fit_sections.append("".join(pills))

        category_order = [category for category, _ in product_categories]
        category_order.append("Other FITS products")
        for category in category_order:
            paths = fits_groups.get(category, [])
            if not paths:
                continue
            if stack_mode:
                paths_by_stack = defaultdict(list)
                ungrouped_paths = []
                for path in paths:
                    stack_identity = _stack_image_identity(path.name)
                    if stack_identity is None:
                        ungrouped_paths.append(path)
                    else:
                        paths_by_stack[stack_identity[1]].append(path)

                stack_sections = []
                for stack_id in sorted(
                    paths_by_stack,
                    key=lambda value: (
                        (0, int(value)) if value.isdigit() else (1, value)
                    ),
                ):
                    stack_label = _stack_group_label(stack_id, stack_groups)
                    entries = []
                    for path in paths_by_stack[stack_id]:
                        cycle = _stack_image_identity(path.name)[0]
                        cy_tag = "cycle-{}".format(cycle)
                        search_val = "{} {} {}".format(
                            path.name, cy_tag, stack_label
                        )
                        entries.append(
                            '<li class="fits-entry" data-search="{}" data-filter-values="{}">{}'
                            '<span class="file-size">{}</span></li>'.format(
                                _escape(search_val),
                                _escape(cy_tag),
                                _link(path, site_dir),
                                _escape(_artifact_size(path)),
                            )
                        )
                    stack_sections.append(
                        '<details class="fits-stack-group">'
                        '<summary>{} ({})</summary>'
                        '<ul class="file-list">{}</ul></details>'.format(
                            _escape(stack_label),
                            len(entries),
                            "".join(entries),
                        )
                    )

                if ungrouped_paths:
                    entries = []
                    for path in ungrouped_paths:
                        cycle_match = re.search(
                            r"_(\d{3,})(?:-|\.|$)", path.name
                        )
                        cycle = cycle_match.group(1) if cycle_match else None
                        cy_tag = "cycle-{}".format(cycle) if cycle else ""
                        search_val = "{} {}".format(path.name, cy_tag).strip()
                        entries.append(
                            '<li class="fits-entry" data-search="{}" data-filter-values="{}">{}'
                            '<span class="file-size">{}</span></li>'.format(
                                _escape(search_val),
                                _escape(cy_tag),
                                _link(path, site_dir),
                                _escape(_artifact_size(path)),
                            )
                        )
                    stack_sections.append(
                        '<ul class="file-list">{}</ul>'.format("".join(entries))
                    )
                fit_sections.append(
                    '<details data-filter-group{}><summary>{} ({})</summary>{}</details>'.format(
                        " open" if not fit_sections else "",
                        _escape(category),
                        len(paths),
                        "".join(stack_sections),
                    )
                )
            else:
                entries = []
                for path in paths:
                    cm = re.search(r"_(\d{3,})(?:-|\.|$)", path.name)
                    cy_tag = "cycle-{}".format(cm.group(1)) if cm else ""
                    search_val = "{} {}".format(path.name, cy_tag).strip()
                    entries.append(
                        '<li class="fits-entry" data-search="{}" data-filter-values="{}">{}<span class="file-size">{}</span></li>'.format(
                            _escape(search_val),
                            _escape(cy_tag),
                            _link(path, site_dir),
                            _escape(_artifact_size(path)),
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

    page_intro = (
        "PNG previews open locally at full size. FITS files are linked as products; "
        "use an astronomy FITS viewer to inspect their pixel data."
    )
    if stack_mode:
        page_intro += (
            " Stacked-run previews and image-statistics charts are grouped by "
            "imaging group; comparisons stay within each group."
        )
    body = (
        '<p class="page-intro">{}</p>\n'.format(_escape(page_intro))
        + _section("Image Progression & Comparisons", "".join(sections)) + "\n"
        + _section(
            "Image Statistics & Trends",
            _image_metrics_content(logs.get("image_metrics", [])),
            "Maximum/minimum image values and RMS noise parsed from selfcal.log.",
        ) + "\n"
        + _section("FITS products", "".join(fit_sections), "{} files found.".format(len(fits_files)))
    )
    _write_page(
        site_dir / "imaging.html",
        title,
        "imaging.html",
        body,
        subtitle="Imaging & FITS products / {}".format(run_root.name),
        optional_pages=_optional_pages(config),
    )


def _cycle_sort_key(value):
    if value.isdigit():
        return (0, int(value))
    return (1, value)


_POLARIZATION_PRODUCTS = {"XX", "YY", "XY", "YX", "RR", "LL", "RL", "LR", "Q", "U", "V"}


def _plot_direction_index(name):
    """Return the direction index encoded in a solution plot file name.

    Merged DDE plots use ``dirDirNN`` and per-solve plots use
    ``dir[MODEL_DATA_DDN]``; both refer to the same direction index.
    """
    match = re.search(r"MODEL_DATA_DD(\d+)", name, re.IGNORECASE)
    if match is None:
        match = re.search(r"(?:dir|dil)(\d+)", name, re.IGNORECASE)
    return int(match.group(1)) if match else None


def _plot_page(
    site_dir,
    run_root,
    title,
    dataset_name,
    dataset_slug,
    heading,
    entries,
    optional_pages=(),
    default_sort="soltype",
):
    """Write a page with solution plots that can be filtered and reordered.

    ``entries`` is a list of ``(path, cycle)`` pairs. The plots can be ordered
    by self-calibration cycle, by direction (for --DDE solves) or by solution
    type; the order and a filter can be preselected with a URL fragment such as
    ``#sort=cycle&filter=dir-03``.
    """
    calibration_dir = site_dir / "calibration"
    page_path = calibration_dir / "{}.html".format(dataset_slug)
    figures = []
    has_phase = False
    has_amp = False
    has_pol = False
    directions = set()
    cycles = set()
    soltypes = set()

    for path, cycle in entries:
        name = path.name
        tags = []
        lower = name.lower()
        filter_values = []
        cycle_title = "Cycle {}".format(cycle) if cycle != "other" else "Other plots"
        if "amp" in lower:
            tags.append("amplitude")
            filter_values.append("amplitude")
            has_amp = True
        if "phase" in lower:
            tags.append("phase")
            filter_values.append("phase")
            has_phase = True
        polarization = re.search(r"pol([a-z0-9]+)", name, re.IGNORECASE)
        # "_polI" marks Stokes I solutions (e.g. merged DDE plots); only
        # correlation-specific or polarization solution plots count here.
        if (
            "poldiff" in lower
            or re.search(r"fulljones|rotation|leakage", lower)
            or (
                polarization
                and polarization.group(1).upper() in _POLARIZATION_PRODUCTS
            )
        ):
            tags.append("polarization")
            filter_values.append("polarization")
            has_pol = True
        direction = _plot_direction_index(name)
        if direction is not None:
            tags.append("direction {:02d}".format(direction))
            filter_values.append("dir-{:02d}".format(direction))
            directions.add(direction)
        if polarization:
            label = polarization.group(1)
            tags.append("Stokes I" if label.upper() == "I" else "pol-{}".format(label))
        soltype_match = re.match(r"(.+?)_(?:sky)?selfcalcycle\d+", name, re.IGNORECASE)
        soltype = soltype_match.group(1) if soltype_match else path.stem
        cycles.add(cycle)
        soltypes.add(soltype)
        detail = " / ".join([cycle_title] + tags)
        figure = _figure(
            path,
            calibration_dir,
            name,
            detail=detail,
            extra_search="{} {} {}".format(dataset_name, cycle_title, " ".join(tags)),
            item_class="plot-item",
        )
        figure = figure.replace(
            'data-filter-values="{}"'.format(
                _escape("{} {} {}".format(dataset_name, cycle_title, " ".join(tags)))
            ),
            'data-filter-values="{}" data-sort-cycle="{}" data-sort-direction="{}" '
            'data-sort-soltype="{}" data-sort-name="{}"'.format(
                _escape(" ".join(filter_values)),
                _escape(cycle),
                "" if direction is None else "{:02d}".format(direction),
                _escape(soltype),
                _escape(lower),
            ),
            1,
        )
        figures.append(figure)

    # Filter pills for plot types and directions
    pills = ['<div class="pill-group" data-pill-group data-pill-target=".plot-item">']
    pills.append('<span class="pill-label">Show:</span>')
    pills.append('<button type="button" class="pill-btn active" data-filter-value="all">All plots ({})</button>'.format(len(entries)))
    if has_phase:
        pills.append('<button type="button" class="pill-btn" data-filter-value="phase">Phase</button>')
    if has_amp:
        pills.append('<button type="button" class="pill-btn" data-filter-value="amplitude">Amplitude</button>')
    if has_pol:
        pills.append('<button type="button" class="pill-btn" data-filter-value="polarization">Polarization</button>')
    for direction in sorted(directions):
        pills.append(
            '<button type="button" class="pill-btn" data-filter-value="dir-{0:02d}">Dir{0:02d}</button>'.format(direction)
        )
    pills.append('</div>')

    sort_options = []
    if len(cycles) > 1:
        sort_options.append(("cycle", "Cycle"))
    if directions:
        sort_options.append(("direction", "Direction"))
    if len(soltypes) > 1 or not sort_options:
        sort_options.append(("soltype", "Solution type"))
    if default_sort not in {key for key, _ in sort_options}:
        default_sort = sort_options[0][0]
    sort_controls = (
        '<div class="pill-group" data-sort-controls="#solution-plot-gallery" data-sort-default="{}">'
        '<span class="pill-label">Order by:</span>{}</div>'
    ).format(
        default_sort,
        "".join(
            '<button type="button" class="pill-btn{}" data-sort-key="{}">{}</button>'.format(
                " active" if key == default_sort else "", key, label
            )
            for key, label in sort_options
        ),
    )

    body = (
        '<p class="page-intro"><a href="../calibration.html">&larr; Back to Calibration index</a> / {} / {}. '
        'This page contains {} plots and loads thumbnails lazily.</p>'.format(
            _escape(dataset_name), _escape(heading), len(entries)
        )
        + sort_controls
        + "".join(pills)
        + _filter_input(".plot-item", "Search plot filenames")
        + '<div class="gallery" id="solution-plot-gallery">{}</div>'.format("".join(figures))
    )
    _write_page(
        page_path,
        title,
        "calibration.html",
        body,
        nested=True,
        subtitle="Calibration plots / {} / {}".format(dataset_name, heading),
        optional_pages=optional_pages,
    )
    return page_path


def _calibration_page(site_dir, run_root, config, artifacts):
    title = str(config.get("imagename") or run_root.name)
    optional_pages = _optional_pages(config)
    ms_inputs = _as_list(config.get("ms"))
    calibration_sets = artifacts["calibration_sets"]
    all_cycles = sorted(
        {cycle for _, cycles in calibration_sets for cycle in cycles},
        key=_cycle_sort_key,
    )
    if not calibration_sets:
        body = (
            '<p class="empty">No calibration plots are available.</p>'
            '<p class="page-intro">For restart runs using <code>--start=N</code>, '
            'existing plots from cycles before N are included when present; '
            'plots from cycle N onward are shown only if produced during the current run.</p>'
        )
        _write_page(
            site_dir / "calibration.html",
            title,
            "calibration.html",
            body,
            optional_pages=optional_pages,
        )
        return

    rows = []
    direction_counts = []
    cycle_headers = "".join("<th scope=\"col\">Cycle {}</th>".format(_escape(cycle)) for cycle in all_cycles)
    for index, (directory, cycles) in enumerate(calibration_sets, start=1):
        dataset_slug = "ms-{:02d}".format(index)
        dataset_name = _configured_ms_label(
            directory.name.removeprefix("solution_plots_"), ms_inputs
        )
        cells = []
        all_entries = []
        for cycle in all_cycles:
            paths = cycles.get(cycle, [])
            if not paths:
                cells.append('<td class="muted">-</td>')
                continue
            all_entries.extend((path, cycle) for path in paths)
            cycle_slug = "cycle-{}".format(cycle)
            page_slug = "{}-{}.html".format(dataset_slug, cycle_slug)
            _plot_page(
                site_dir,
                run_root,
                title,
                dataset_name,
                page_slug[:-5],
                "Cycle {}".format(cycle) if cycle != "other" else "Other plots",
                [(path, cycle) for path in paths],
                optional_pages=optional_pages,
            )
            cells.append(
                '<td><a href="calibration/{}">{} plots</a></td>'.format(
                    _escape(page_slug), len(paths)
                )
            )
        all_slug = "{}-all".format(dataset_slug)
        _plot_page(
            site_dir,
            run_root,
            title,
            dataset_name,
            all_slug,
            "All cycles",
            all_entries,
            optional_pages=optional_pages,
            default_sort="cycle",
        )
        rows.append(
            '<tr class="dataset-row" data-search="{}"><th scope="row"><code>{}</code></th>{}'
            '<td><a href="calibration/{}.html">{} plots</a></td></tr>'.format(
                _escape(dataset_name), _escape(dataset_name), "".join(cells),
                _escape(all_slug), len(all_entries),
            )
        )
        counts = defaultdict(lambda: [0, set()])
        for path, cycle in all_entries:
            direction = _plot_direction_index(path.name)
            if direction is not None:
                counts[direction][0] += 1
                counts[direction][1].add(cycle)
        if counts:
            direction_counts.append((dataset_name, all_slug, counts))

    sections = [
        _section(
            "Solutions by cycle",
            _filter_input(".dataset-row", "Filter measurement sets")
            + '<div class="table-scroll"><table class="data-table"><thead><tr><th scope="col">Measurement set</th>{}<th scope="col">All cycles</th></tr></thead><tbody>{}</tbody></table></div>'.format(
                cycle_headers, "".join(rows)
            ),
            "Open a cycle page, or all cycles of a measurement set on one page, to view, filter and reorder the solutions.",
        )
    ]
    if direction_counts:
        directions = sorted({d for _, _, counts in direction_counts for d in counts})
        header = "".join(
            '<th scope="col"><code>{}</code></th>'.format(_escape(name))
            for name, _, _ in direction_counts
        )
        direction_rows = []
        for direction in directions:
            cells = []
            for _, all_slug, counts in direction_counts:
                if direction not in counts:
                    cells.append('<td class="muted">-</td>')
                    continue
                count, direction_cycles = counts[direction]
                cells.append(
                    '<td><a href="calibration/{}.html#sort=cycle&amp;filter=dir-{:02d}">{} plots in {} cycles</a></td>'.format(
                        _escape(all_slug), direction, count, len(direction_cycles)
                    )
                )
            direction_rows.append(
                '<tr><th scope="row">Dir{:02d}</th>{}</tr>'.format(direction, "".join(cells))
            )
        sections.append(_section(
            "Solutions by direction",
            '<div class="table-scroll"><table class="data-table"><thead><tr><th scope="col">Direction</th>{}</tr></thead><tbody>{}</tbody></table></div>'.format(
                header, "".join(direction_rows)
            ),
            "Each link opens all solutions of one direction, ordered by cycle. "
            "The direction positions are listed on the Directions page.",
        ))

    body = (
        '<p class="page-intro">Calibration plots are split into local pages per measurement set. Plots can be filtered by type or direction and ordered by cycle, direction or solution type.</p>\n'
        + "\n".join(sections)
    )
    _write_page(
        site_dir / "calibration.html",
        title,
        "calibration.html",
        body,
        subtitle="Calibration solutions / {}".format(run_root.name),
        optional_pages=optional_pages,
    )


def _read_bandpass_soltab(solset, solution_name, np):
    if solution_name not in solset._v_children:
        return None
    group = solset._v_children[solution_name]
    children = group._v_children
    required_axes = ("time", "freq", "ant", "pol")
    missing_axes = [axis for axis in required_axes if axis not in children]
    if "val" not in children:
        raise _BandpassDataError("{} has no value array.".format(solution_name))
    if missing_axes:
        raise _BandpassDataError(
            "{} is missing axis arrays: {}.".format(
                solution_name, ", ".join(missing_axes)
            )
        )

    frequencies = np.asarray(children["freq"][:], dtype=float)
    antennas = [_decode_bandpass_label(value) for value in children["ant"][:]]
    polarizations = [_decode_bandpass_label(value) for value in children["pol"][:]]
    times = children["time"][:]
    has_direction_axis = "dir" in children
    directions = (
        [_decode_bandpass_label(value) for value in children["dir"][:]]
        if has_direction_axis
        else ["Direction"]
    )
    if not frequencies.size or not antennas or not polarizations or not directions:
        raise _BandpassDataError(
            "{} has an empty frequency, antenna, direction, or polarization axis.".format(
                solution_name
            )
        )
    if len(set(antennas)) != len(antennas):
        raise _BandpassDataError("{} has duplicate antenna labels.".format(solution_name))
    if len(set(polarizations)) != len(polarizations):
        raise _BandpassDataError(
            "{} has duplicate polarization labels.".format(solution_name)
        )
    if len(set(directions)) != len(directions):
        raise _BandpassDataError(
            "{} has duplicate direction labels.".format(solution_name)
        )

    values = np.asarray(children["val"][:], dtype=float)
    expected_shape = (
        len(times),
        len(frequencies),
        len(antennas),
        len(directions),
        len(polarizations),
    )
    if values.ndim == 4 and not has_direction_axis:
        values = np.expand_dims(values, axis=3)
    if values.shape != expected_shape:
        raise _BandpassDataError(
            "{} has value shape {}, expected {}.".format(
                solution_name, values.shape, expected_shape
            )
        )
    if not len(times):
        raise _BandpassDataError("{} has no time samples.".format(solution_name))

    weights = None
    if "weight" in children:
        weights = np.asarray(children["weight"][:], dtype=float)
        if weights.ndim == 4 and not has_direction_axis:
            weights = np.expand_dims(weights, axis=3)
        if weights.shape != expected_shape:
            raise _BandpassDataError(
                "{} has weight shape {}, expected {}.".format(
                    solution_name, weights.shape, expected_shape
                )
            )

    finite_frequency_indices = np.flatnonzero(np.isfinite(frequencies))
    if not finite_frequency_indices.size:
        raise _BandpassDataError(
            "{} has no finite frequency values.".format(solution_name)
        )
    original_valid = np.isfinite(values) & np.isfinite(frequencies)[None, :, None, None, None]
    if weights is not None:
        original_valid &= np.isfinite(weights) & (weights > 0)
    valid_count = int(np.count_nonzero(original_valid))
    total_count = int(values.size)
    frequency_order = finite_frequency_indices[
        np.argsort(frequencies[finite_frequency_indices])
    ]
    frequencies_mhz = (frequencies[frequency_order] / 1e6).tolist()
    values = values[:, frequency_order, :, :, :]
    if weights is not None:
        weights = weights[:, frequency_order, :, :, :]

    series = {}
    for direction_index, direction in enumerate(directions):
        direction_values = values[:, :, :, direction_index, :]
        valid = np.isfinite(direction_values)
        if weights is not None:
            direction_weights = weights[:, :, :, direction_index, :]
            valid &= np.isfinite(direction_weights) & (direction_weights > 0)

        if solution_name == "amplitude000":
            masked_values = np.ma.array(direction_values, mask=~valid)
            reduced = np.ma.median(masked_values, axis=0).filled(np.nan)
        else:
            cosine_sum = np.sum(
                np.cos(np.where(valid, direction_values, 0.0)) * valid, axis=0
            )
            sine_sum = np.sum(
                np.sin(np.where(valid, direction_values, 0.0)) * valid, axis=0
            )
            valid_times = np.sum(valid, axis=0)
            coherence = np.hypot(cosine_sum, sine_sum) / np.maximum(valid_times, 1)
            reduced = np.arctan2(sine_sum, cosine_sum)
            reduced[(valid_times == 0) | (coherence < 1e-8)] = np.nan

        reduced = np.asarray(reduced, dtype=float)
        antenna_series = {}
        for antenna_index, antenna in enumerate(antennas):
            antenna_series[antenna] = {
                polarization: [
                    float(value) if math.isfinite(float(value)) else None
                    for value in reduced[:, antenna_index, polarization_index]
                ]
                for polarization_index, polarization in enumerate(polarizations)
            }
        series[direction] = antenna_series

    return {
        "frequencies_mhz": [float(value) for value in frequencies_mhz],
        "directions": directions,
        "antennas": antennas,
        "polarizations": polarizations,
        "series": series,
        "weights_available": weights is not None,
        "valid_count": valid_count,
        "total_count": total_count,
    }


def _decode_bandpass_label(value):
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _read_bandpass_h5(path):
    cache = DerivedDataCache(path, "original-sample-bandpass-v1")
    saved = cache.load()
    if saved is not None:
        if _valid_bandpass_cache(saved):
            return saved
        report_warning("Ignoring invalid cached bandpass data for {}.".format(path))
    try:
        import numpy as np
        import tables
    except ImportError as exc:
        raise _BandpassDataError(
            "Reading bandpass H5 files requires NumPy and PyTables."
        ) from exc

    try:
        with tables.open_file(path, mode="r") as handle:
            if "sol000" not in handle.root._v_children:
                raise _BandpassDataError("H5Parm has no sol000 solution set.")
            solset = handle.root._v_children["sol000"]
            solutions = {}
            for solution_name, key in (
                ("amplitude000", "amplitude"),
                ("phase000", "phase"),
            ):
                solution = _read_bandpass_soltab(solset, solution_name, np)
                if solution is not None:
                    solutions[key] = solution
            if not solutions:
                raise _BandpassDataError(
                    "H5Parm contains neither amplitude000 nor phase000 solutions."
                )
            cache.save(solutions)
            return solutions
    except _BandpassDataError:
        raise
    except (
        tables.HDF5ExtError,
        tables.NoSuchNodeError,
        OSError,
        ValueError,
        TypeError,
        AttributeError,
        IndexError,
    ) as exc:
        raise _BandpassDataError(
            "Could not read {}: {}".format(path.name, exc)
        ) from exc


def _valid_bandpass_cache(solutions):
    if not solutions or not set(solutions).issubset({"amplitude", "phase"}):
        return False
    try:
        for solution in solutions.values():
            if not isinstance(solution, dict):
                return False
            for field in ("valid_count", "total_count"):
                if not isinstance(solution[field], int) or solution[field] < 0:
                    return False
            if solution["valid_count"] > solution["total_count"]:
                return False
            if not isinstance(solution["weights_available"], bool):
                return False
            frequencies = solution["frequencies_mhz"]
            if not isinstance(frequencies, list) or not frequencies or not all(
                isinstance(value, (int, float)) and not isinstance(value, bool)
                and math.isfinite(value) for value in frequencies
            ):
                return False
            for key in ("directions", "antennas", "polarizations"):
                labels = solution[key]
                if not isinstance(labels, list) or not labels or not all(
                    isinstance(value, str) for value in labels
                ) or len(set(labels)) != len(labels):
                    return False
            for direction in solution["directions"]:
                for antenna in solution["antennas"]:
                    for polarization in solution["polarizations"]:
                        values = solution["series"][direction][antenna][polarization]
                        if not isinstance(values, list) or len(values) != len(frequencies) or not all(
                            value is None or (
                                isinstance(value, (int, float)) and not isinstance(value, bool)
                                and math.isfinite(value)
                            ) for value in values
                        ):
                            return False
    except (KeyError, TypeError):
        return False
    return True


def _configured_ms_label(path, ms_inputs):
    name = Path(path).name
    identity_name = _ms_identity_name(name)
    matches = [
        Path(measurement_set).name
        for measurement_set in ms_inputs
        if _ms_identity_name(Path(measurement_set).name) == identity_name
    ]
    return matches[0] if len(matches) == 1 else name


def _bandpass_label(path, ms_inputs):
    filename = path.name.casefold()
    direct_matches = [
        Path(measurement_set).name
        for measurement_set in ms_inputs
        if Path(measurement_set).name
        and filename.endswith(
            (Path(measurement_set).name + ".h5").casefold()
        )
    ]
    if len(direct_matches) == 1:
        return direct_matches[0]
    label = path.stem
    if label.casefold().startswith("bandpass_"):
        label = label[len("bandpass_"):]
    label = re.sub(r"^(?:sky)?selfcalcycle\d+_", "", label, flags=re.IGNORECASE)
    return _configured_ms_label(label or path.name, ms_inputs)


def _bandpass_summary(solutions):
    antennas = []
    polarizations = []
    directions = []
    frequencies = []
    total_count = 0
    valid_count = 0
    weights_available = True
    channels = []
    for name in ("amplitude", "phase"):
        solution = solutions.get(name)
        if solution is None:
            continue
        for key, target in (
            ("antennas", antennas),
            ("polarizations", polarizations),
            ("directions", directions),
        ):
            for value in solution[key]:
                if value not in target:
                    target.append(value)
        frequencies.extend(solution["frequencies_mhz"])
        channels.append((name, len(solution["frequencies_mhz"])))
        total_count += solution["total_count"]
        valid_count += solution["valid_count"]
        weights_available &= solution["weights_available"]

    flagged_percent = (
        100.0 * (total_count - valid_count) / total_count
        if weights_available and total_count
        else None
    )
    return {
        "antennas": antennas,
        "polarizations": polarizations,
        "directions": directions,
        "channels": channels,
        "frequency_min_mhz": min(frequencies) if frequencies else None,
        "frequency_max_mhz": max(frequencies) if frequencies else None,
        "flagged_percent": flagged_percent,
        "weights_available": weights_available,
    }


def _bandpass_frequency_range(summary):
    frequency_min = summary["frequency_min_mhz"]
    frequency_max = summary["frequency_max_mhz"]
    if frequency_min is None or frequency_max is None:
        return "Unknown"
    if frequency_min == frequency_max:
        return "{:.5f} MHz".format(frequency_min)
    return "{:.5f} to {:.5f} MHz".format(frequency_min, frequency_max)


def _bandpass_detail_page(
    site_dir,
    title,
    dataset_name,
    page_name,
    solutions,
    summary,
    optional_pages=(),
):
    bandpass_dir = site_dir / "bandpass"
    page_path = bandpass_dir / page_name
    channels_text = " / ".join(
        "{} {}".format(name, count) for name, count in summary["channels"]
    )
    flagged_text = (
        "{:.1f}%".format(summary["flagged_percent"])
        if summary["flagged_percent"] is not None
        else "Unavailable"
    )
    stats = (
        ("Frequency range", _bandpass_frequency_range(summary)),
        ("Channels", channels_text or "Unavailable"),
        ("Antennas", str(len(summary["antennas"]))),
        ("Polarizations", ", ".join(summary["polarizations"]) or "Unavailable"),
        ("Directions", str(len(summary["directions"]))),
        ("Flagged / invalid original samples", flagged_text),
    )
    summary_html = '<div class="bandpass-summary">{}</div>'.format(
        "".join(
            '<div class="bandpass-stat"><strong>{}</strong><span>{}</span></div>'.format(
                _escape(label), _escape(value)
            )
            for label, value in stats
        )
    )
    controls_html = (
        '<div class="bandpass-controls">'
        '<label for="bandpass-antenna">Antenna<select id="bandpass-antenna"></select></label>'
        '<label for="bandpass-reference-antenna">Phase reference antenna'
        '<select id="bandpass-reference-antenna"></select></label>'
        '<label for="bandpass-polarization">Polarization<select id="bandpass-polarization"></select></label>'
        '<label id="bandpass-direction-label" for="bandpass-direction">'
        'Direction<select id="bandpass-direction"></select></label>'
        '<div class="bandpass-range-group" role="group" aria-label="Plot y-axis ranges">'
        '<fieldset class="bandpass-range-controls">'
        '<legend>Amplitude Y range</legend>'
        '<label for="bandpass-amplitude-y-min">Min'
        '<input id="bandpass-amplitude-y-min" type="number" step="any" inputmode="decimal" '
        'placeholder="Auto" aria-describedby="bandpass-amplitude-range-hint '
        'bandpass-amplitude-range-error"></label>'
        '<label for="bandpass-amplitude-y-max">Max'
        '<input id="bandpass-amplitude-y-max" type="number" step="any" inputmode="decimal" '
        'placeholder="Auto" aria-describedby="bandpass-amplitude-range-hint '
        'bandpass-amplitude-range-error"></label>'
        '<p id="bandpass-amplitude-range-hint" class="bandpass-range-hint">'
        'Blank endpoints use automatic scaling.</p>'
        '<p id="bandpass-amplitude-range-error" class="bandpass-range-error" '
        'role="status" aria-live="polite" hidden></p>'
        '</fieldset>'
        '<fieldset class="bandpass-range-controls">'
        '<legend>Phase Y range (degrees)</legend>'
        '<label for="bandpass-phase-y-min">Min'
        '<input id="bandpass-phase-y-min" type="number" step="any" inputmode="decimal" '
        'placeholder="-180" aria-describedby="bandpass-phase-range-hint '
        'bandpass-phase-range-error"></label>'
        '<label for="bandpass-phase-y-max">Max'
        '<input id="bandpass-phase-y-max" type="number" step="any" inputmode="decimal" '
        'placeholder="180" aria-describedby="bandpass-phase-range-hint '
        'bandpass-phase-range-error"></label>'
        '<p id="bandpass-phase-range-hint" class="bandpass-range-hint">'
        'Blank endpoints keep the default -180 to 180 degree range.</p>'
        '<p id="bandpass-phase-range-error" class="bandpass-range-error" '
        'role="status" aria-live="polite" hidden></p>'
        '</fieldset>'
        '</div>'
        '</div>'
    )
    charts_html = (
        '<div class="bandpass-chart-card"><h2>Amplitude</h2>'
        '<p>Dimensionless gain amplitude versus frequency.</p>'
        '<svg id="bandpass-amplitude" class="bandpass-chart" viewBox="0 0 940 330"></svg></div>'
        '<div class="bandpass-chart-card"><h2>Phase</h2>'
        '<p>Phase relative to the selected reference antenna, wrapped to [-180, 180] degrees. The reference itself is zero.</p>'
        '<svg id="bandpass-phase" class="bandpass-chart" viewBox="0 0 940 330"></svg></div>'
    )
    note = (
        "Each curve is one antenna and polarization. Time samples are collapsed "
        "to a median amplitude and circular-mean phase. The phase plot subtracts "
        "the selected reference antenna at each frequency for the same direction "
        "and polarization; flagged or invalid target/reference samples are left as gaps. "
        "The flagged/invalid percentage counts original time-frequency-antenna-direction-polarization "
        "samples before collapse, not missing plot points or incoherent reduced phases."
    )
    if not summary["weights_available"]:
        note += " At least one solution table has no weight/flag axis, so only non-finite samples can be omitted."
    data_json = json.dumps(solutions, separators=(",", ":"), ensure_ascii=True, allow_nan=False)
    data_json = (
        data_json.replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("&", "\\u0026")
    )
    body = (
        '<p class="page-intro"><a href="../bandpass.html">&larr; Back to Bandpass index</a> / {}.</p>'.format(
            _escape(dataset_name)
        )
        + summary_html
        + controls_html
        + charts_html
        + '<p class="bandpass-note">{}</p>'.format(_escape(note))
        + '<script type="application/json" id="bandpass-data">{}</script>'.format(
            data_json
        )
    )
    _write_page(
        page_path,
        title,
        "bandpass.html",
        body,
        nested=True,
        subtitle="Bandpass solutions / {}".format(dataset_name),
        optional_pages=optional_pages,
    )
    return page_path


def _bandpass_page(site_dir, run_root, config, artifacts):
    title = str(config.get("imagename") or run_root.name)
    optional_pages = _optional_pages(config)
    rows = []
    ms_inputs = _as_list(config.get("ms"))
    for index, path in enumerate(artifacts.get("bandpass_files", []), start=1):
        dataset_name = _bandpass_label(path, ms_inputs)
        try:
            solutions = _read_bandpass_h5(path)
        except _BandpassDataError as exc:
            report_warning("Bandpass data unavailable for {}: {}".format(path, exc))
            rows.append(
                '<tr><th scope="row"><code>{}</code></th>'
                '<td colspan="6" class="bandpass-error">Bandpass data unavailable: {}</td></tr>'.format(
                    _escape(dataset_name), _escape(exc)
                )
            )
            continue

        summary = _bandpass_summary(solutions)
        detail_page = _bandpass_detail_page(
            site_dir,
            title,
            dataset_name,
            "ms-{:02d}.html".format(index),
            solutions,
            summary,
            optional_pages=optional_pages,
        )
        channel_text = " / ".join(
            "{} {}".format(name, count) for name, count in summary["channels"]
        )
        flagged_text = (
            "{:.1f}%".format(summary["flagged_percent"])
            if summary["flagged_percent"] is not None
            else "Unavailable"
        )
        rows.append(
            '<tr><th scope="row"><code>{}</code></th>'
            '<td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td>'
            '<td><a href="{}">View 1D plots</a></td></tr>'.format(
                _escape(dataset_name),
                len(summary["antennas"]),
                _escape(", ".join(summary["polarizations"]) or "Unavailable"),
                _escape(channel_text or "Unavailable"),
                _escape(_bandpass_frequency_range(summary)),
                _escape(flagged_text),
                _escape(_relative_url(detail_page, site_dir)),
            )
        )

    if rows:
        solutions_html = (
            '<div class="table-scroll"><table class="data-table"><thead><tr>'
            '<th scope="col">Measurement set</th><th scope="col">Antennas</th>'
            '<th scope="col">Polarizations</th><th scope="col">Channels</th>'
            '<th scope="col">Frequency coverage</th><th scope="col">Flagged / invalid original samples</th>'
            '<th scope="col">Plots</th></tr></thead><tbody>{}</tbody></table></div>'.format(
                "".join(rows)
            )
        )
    else:
        solutions_html = (
            '<p class="empty">Bandpass was enabled, but no per-measurement-set '
            'bandpass H5 files were found under h5_solutions/.</p>'
        )
    body = (
        '<p class="page-intro">Select a measurement set to inspect its bandpass. '
        'Each detail page shows one-dimensional amplitude and phase curves versus '
        'frequency for a selected antenna and polarization.</p>\n'
        + _section(
            "Bandpass solutions",
            solutions_html,
            "{} H5 product{} found.".format(
                len(artifacts.get("bandpass_files", [])),
                "" if len(artifacts.get("bandpass_files", [])) == 1 else "s",
            ),
        )
    )
    _write_page(
        site_dir / "bandpass.html",
        title,
        "bandpass.html",
        body,
        subtitle="Bandpass solutions / {}".format(run_root.name),
        optional_pages=optional_pages,
    )


def _datasets_page(site_dir, run_root, config, artifacts, logs=None):
    title = str(config.get("imagename") or run_root.name)
    inputs = _as_list(config.get("ms"))
    report_logs = logs or {}
    flagging_stats = report_logs.get("flagging_stats", {})
    flagging_stats_by_canonical = defaultdict(list)
    flagging_stats_by_basename = defaultdict(list)
    for logged_path, record in flagging_stats.items():
        source_path = record.get("ms_path", logged_path)
        resolved_path = _resolved_ms_path(source_path, run_root)
        logged_canonical_path = _canonical_ms_path(source_path, run_root)
        flagging_stats_by_canonical[logged_canonical_path].append(
            (resolved_path, record)
        )
        flagging_stats_by_basename[
            _ms_identity_name(source_path)
        ].append((resolved_path, record))

    metadata_by_ms = _measurement_set_metadata(run_root)
    cycle_steps = [
        step
        for cycle in report_logs.get("cycle_timeline") or ()
        for step in cycle.get("step_details") or ()
    ]
    setup_steps = list(report_logs.get("setup_step_details") or ())
    candidate_ms_records = [
        (
            _canonical_ms_path(
                metadata.get("_source_path", ms_path), run_root
            ),
            _resolved_ms_path(
                metadata.get("_source_path", ms_path), run_root
            ),
        )
        for ms_path, metadata in metadata_by_ms.items()
    ]
    candidate_ms_records.extend(
        (
            _canonical_ms_path(step["ms_path"], run_root),
            _resolved_ms_path(step["ms_path"], run_root),
        )
        for step in cycle_steps
        if step.get("ms_path")
    )
    candidate_ms_records.extend(
        (
            _canonical_ms_path(path, run_root),
            _resolved_ms_path(record.get("ms_path", path), run_root),
        )
        for path, record in flagging_stats.items()
    )
    candidate_ms_records.extend(
        (
            _canonical_ms_path(step["ms_path"], run_root),
            _resolved_ms_path(step["ms_path"], run_root),
        )
        for step in setup_steps
        if step.get("ms_path")
    )
    seen_ms_paths = set()
    selfcal_ms_records = []
    for canonical_path, display_path in candidate_ms_records:
        if canonical_path in seen_ms_paths:
            continue
        seen_ms_paths.add(canonical_path)
        selfcal_ms_records.append((canonical_path, display_path))

    input_name_counts = defaultdict(int)
    for input_path in inputs:
        input_name_counts[
            _ms_identity_name(_canonical_ms_path(input_path, run_root))
        ] += 1

    split_ms_by_input_name = defaultdict(dict)
    split_ms_by_input_path = defaultdict(dict)
    split_records = {}
    for canonical_path, display_path in selfcal_ms_records:
        parent_name = _split_ms_parent_name(canonical_path)
        if parent_name:
            split_record = (canonical_path, display_path)
            split_ms_by_input_name[parent_name.casefold()].setdefault(
                Path(canonical_path).name.casefold(),
                split_record,
            )
            parent_path = _split_ms_parent_path(display_path, run_root)
            if parent_path:
                split_ms_by_input_path[parent_path].setdefault(
                    canonical_path, split_record
                )
            split_record_name = Path(canonical_path).name.casefold()
            split_record_key = (
                (parent_name.casefold(), split_record_name)
                if input_name_counts[parent_name.casefold()] == 1
                else canonical_path
            )
            split_records.setdefault(
                split_record_key,
                (canonical_path, parent_name, display_path),
            )

    candidate_family_paths_by_name = defaultdict(set)
    for _, display_path in candidate_ms_records:
        family_path = _ms_identity_path(display_path, run_root)
        identity_name = Path(family_path).name.casefold()
        if identity_name:
            candidate_family_paths_by_name[identity_name].add(family_path)

    dataset_entries = []
    assigned_split_paths = set()
    for input_path in inputs:
        input_path = str(input_path)
        canonical_input_path = _canonical_ms_path(input_path, run_root)
        input_name = _ms_identity_name(canonical_input_path)
        exact_split_paths = split_ms_by_input_path.get(
            _ms_identity_path(input_path, run_root), {}
        )
        if exact_split_paths:
            split_paths = list(exact_split_paths.values())
        elif input_name_counts[input_name] == 1:
            split_paths = list(
                split_ms_by_input_name.get(input_name, {}).values()
            )
        else:
            split_paths = []
        split_paths = [
            paths for paths in split_paths if paths[0] not in assigned_split_paths
        ]
        split_paths.sort(key=lambda paths: Path(paths[1]).name.casefold())
        assigned_split_paths.update(paths[0] for paths in split_paths)
        dataset_entries.append(
            {
                "path": input_path,
                "canonical_path": canonical_input_path,
                "role": "Input MS" if split_paths else "Input / self-calibration MS",
                "is_input": True,
                "parent_name": None,
                "split_paths": split_paths,
            }
        )
        dataset_entries.extend(
            {
                "path": display_path,
                "canonical_path": canonical_path,
                "role": "Self-calibration MS",
                "is_input": False,
                "parent_name": Path(input_path).name,
                "split_paths": [],
            }
            for canonical_path, display_path in split_paths
        )

    dataset_entries.extend(
        {
            "path": display_path,
            "canonical_path": canonical_path,
            "role": "Unmatched self-calibration MS",
            "is_input": False,
            "parent_name": "Ambiguous input name: {}".format(parent_name),
            "split_paths": [],
        }
        for canonical_path, parent_name, display_path in split_records.values()
        if canonical_path not in assigned_split_paths
    )

    has_vla_dataset = any(
        _measurement_set_metadata_for_path(
            entry["path"], metadata_by_ms, run_root
        )
        .get("Telescope", "").strip().upper() in {"VLA", "EVLA"}
        for entry in dataset_entries
    )
    metadata_fields = tuple(
        (field, labels)
        for field, labels in _MS_METADATA_FIELDS
        if field != "VLA configuration" or has_vla_dataset
    )
    column_count = len(metadata_fields) + 2
    dataset_rows = []
    ms_cards = []

    ms_plot_files = artifacts.get("ms_plot_files", [])
    all_ms_plot_files = artifacts.get("all_ms_plot_files") or ms_plot_files
    ms_json_files = artifacts.get("ms_json_files", [])

    for entry in dataset_entries:
        path = entry["path"]
        canonical = entry["canonical_path"]
        ms_display_name = Path(path).name or path
        metadata = dict(
            _measurement_set_metadata_for_path(path, metadata_by_ms, run_root)
        )
        metadata.pop("_order", None)
        resolved_path = _resolved_ms_path(path, run_root)
        flagging_record = flagging_stats.get(resolved_path)
        if flagging_record is None:
            canonical_matches = flagging_stats_by_canonical.get(
                canonical, []
            )
            if len(canonical_matches) == 1:
                flagging_record = canonical_matches[0][1]
        if flagging_record is None:
            basename_matches = flagging_stats_by_basename.get(
                _ms_identity_name(path), []
            )
            if len(basename_matches) == 1:
                flagging_record = basename_matches[0][1]
        telescope = (
            metadata.get("Telescope") or config.get("telescope") or ""
        ).strip().upper()
        search_text = " ".join([path, entry["role"]] + list(metadata.values()))
        metadata_cells = "".join(
            "<td>{}</td>".format(
                _escape(
                    metadata.get(
                        field,
                        "Not applicable"
                        if field == "VLA configuration"
                        and has_vla_dataset
                        and telescope
                        and telescope not in {"VLA", "EVLA"}
                        else "Not recorded",
                    )
                )
            )
            for field, _ in metadata_fields
        )
        dataset_rows.append(
            '<tr class="dataset-row" data-search="{}"><th scope="row"><code title="{}">{}</code></th>'
            '<td><span class="dataset-role">{}</span></td>{}</tr>'.format(
                _escape(search_text),
                _escape(path),
                _escape(ms_display_name),
                _escape(entry["role"]),
                metadata_cells,
            )
        )

        related_ms_paths = _related_ms_plot_paths(
            path, candidate_ms_records, input_name_counts, run_root
        )
        if metadata.get("_source_path"):
            related_ms_paths = (
                metadata["_source_path"],
                *related_ms_paths,
            )
        identity_name = _ms_identity_name(path)
        candidate_family_count = len(
            candidate_family_paths_by_name.get(identity_name, ())
        )
        split_parent_name = (
            _split_ms_parent_name(canonical)
            if not entry["is_input"]
            else None
        )
        split_parent_name = (
            _ms_identity_name(split_parent_name)
            if split_parent_name
            else ""
        )
        ambiguous_plot_identity = (
            (entry["is_input"] and input_name_counts[identity_name] > 1)
            or (
                candidate_family_count > 1
                and (
                    (entry["is_input"] and not related_ms_paths)
                    or (
                        not entry["is_input"]
                        and input_name_counts[split_parent_name] > 1
                    )
                )
            )
        )
        if ambiguous_plot_identity:
            matched_plots = {
                "time_coverage": None,
                "uv_coverage": None,
                "ateam_png": None,
                "ateam_json": None,
            }
        else:
            matched_plots = _match_ms_plots(
                path,
                all_ms_plot_files if entry["is_input"] else ms_plot_files,
                ms_json_files,
                related_ms_paths=related_ms_paths,
            )
        plot_placeholder_message = (
            "Multiple measurement sets share this basename; matching plots "
            "cannot be attributed safely."
            if ambiguous_plot_identity
            else "Not generated for this measurement set."
        )
        meta_items = []
        for field, _ in metadata_fields:
            if field in metadata:
                meta_items.append(
                    '<div class="dataset-meta-item"><strong>{}</strong><span>{}</span></div>'.format(
                        _escape(field), _escape(metadata[field])
                    )
                )
        if entry["split_paths"]:
            split_count = len(entry["split_paths"])
            split_label = "self-calibration measurement set"
            if split_count != 1:
                split_label += "s"
            meta_items.append(
                '<div class="dataset-meta-item"><strong>Self-calibration splits</strong><span>{} {}</span></div>'.format(
                    split_count, split_label
                )
            )
        elif not entry["is_input"]:
            meta_items.append(
                '<div class="dataset-meta-item"><strong>Input MS</strong><span>{}</span></div>'.format(
                    _escape(entry["parent_name"])
                )
            )
        if flagging_record is not None:
            percentage = flagging_record.get("flagged_percentage")
            if percentage is not None:
                percentage_value = "{}% ({} of {} samples)".format(
                    percentage,
                    flagging_record.get("flagged_samples", "?"),
                    flagging_record.get("total_samples", "?"),
                )
            else:
                percentage_value = "Not recorded"
            meta_items.extend(
                (
                    '<div class="dataset-meta-item"><strong>Flagged visibility</strong><span>{}</span></div>'.format(
                        _escape(percentage_value)
                    ),
                    '<div class="dataset-meta-item"><strong>Fully flagged antennas</strong><span>{}</span></div>'.format(
                        _escape(
                            flagging_record.get(
                                "fully_flagged_antennas", "Not recorded"
                            )
                        )
                    ),
                )
            )

        plot_figures = []
        if matched_plots["time_coverage"]:
            plot_figures.append(
                _figure(
                    matched_plots["time_coverage"],
                    site_dir,
                    matched_plots["time_coverage"].stem,
                    detail="Time coverage & gaps",
                    item_class="ms-quality-plot",
                )
            )
        else:
            plot_figures.append(
                _quality_plot_placeholder(
                    "Time-coverage plot", plot_placeholder_message
                )
            )
        if matched_plots["uv_coverage"]:
            plot_figures.append(
                _figure(
                    matched_plots["uv_coverage"],
                    site_dir,
                    matched_plots["uv_coverage"].stem,
                    detail="UV coverage (u-v)",
                    item_class="ms-quality-plot",
                )
            )
        else:
            plot_figures.append(
                _quality_plot_placeholder(
                    "UV-coverage plot", plot_placeholder_message
                )
            )
        if matched_plots["ateam_png"]:
            plot_figures.append(
                _figure(
                    matched_plots["ateam_png"],
                    site_dir,
                    matched_plots["ateam_png"].stem,
                    detail="A-team source separation & elevation",
                    item_class="ms-quality-plot",
                )
            )
        else:
            plot_figures.append(
                _quality_plot_placeholder("A-team plot", plot_placeholder_message)
            )

        ateam_note = ""
        if matched_plots["ateam_json"]:
            try:
                ateam_data = json.loads(matched_plots["ateam_json"].read_text(encoding="utf-8", errors="replace"))
                if not isinstance(ateam_data, (dict, list)):
                    raise ValueError("expected an A-team source collection")
                if not ateam_data:
                    ateam_note = '<p class="caption-detail" style="margin-top:6px;">A-team check: No interfering A-team sources within separation threshold.</p>'
                else:
                    sources_str = ", ".join(str(s) for s in ateam_data)
                    ateam_note = '<p class="caption-detail" style="margin-top:6px; color:var(--amber);">A-team check: Potentially interfering sources: {}</p>'.format(_escape(sources_str))
            except (OSError, ValueError, TypeError) as exc:
                report_warning("Could not read A-team data {}: {}".format(matched_plots["ateam_json"], exc))
                ateam_note = '<p class="notice">A-team check unavailable: saved data could not be read.</p>'

        plots_content = '<div class="dataset-plots-grid">{}</div>{}'.format(
            "".join(plot_figures), ateam_note
        )
        metadata_content = (
            '<div class="dataset-meta-grid">{}</div>'.format(
                "".join(meta_items)
            )
            if meta_items
            else ""
        )
        if not metadata and flagging_record is None:
            metadata_content += (
                '<p class="empty">No observational metadata parsed from log.</p>'
            )

        ms_cards.append(
            '<div class="dataset-card">'
            '<div class="dataset-header"><h3 class="dataset-title" title="{}">{}</h3>'
            '<span class="dataset-role">{}</span></div>'
            '{}'
            '<h4>Data Quality & Observation Coverage Plots</h4>'
            '{}'
            '</div>'.format(
                _escape(path),
                _escape(ms_display_name),
                _escape(entry["role"]),
                metadata_content,
                plots_content,
            )
        )

    if not dataset_rows:
        dataset_rows.append(
            '<tr><td colspan="{}">No input MS list was found in full_config.txt.</td></tr>'.format(
                column_count
            )
        )

    body = (
        '<p class="page-intro">Configured input paths come from full_config.txt. When an input is split for self-calibration, both the input and its self-calibration measurement sets are listed. Observation metadata is extracted from logs/selfcal.log; available input time-coverage and UV-coverage plots are shown even if generated before the current run.</p>\n'
        + _section(
            "Input and Self-Calibration Measurement Sets",
            _filter_input(".dataset-row", "Filter measurement sets")
            + '<div class="table-scroll"><table class="data-table"><thead><tr><th scope="col">Measurement Set</th><th scope="col">Role</th>{}</tr></thead><tbody>{}</tbody></table></div>'.format(
                "".join(
                    '<th scope="col">{}</th>'.format(_escape(field))
                    for field, _ in metadata_fields
                ),
                "".join(dataset_rows),
            ),
        ) + "\n"
        + _section("Measurement Set Data Quality & Coverage", "".join(ms_cards) if ms_cards else '<p class="empty">No measurement sets configured.</p>')
    )
    _write_page(
        site_dir / "datasets.html",
        title,
        "datasets.html",
        body,
        subtitle="Measurement sets & coverage / {}".format(run_root.name),
        optional_pages=_optional_pages(config),
    )


def _current_extraction_text(path, run_started_at):
    if not path.is_file() or not _artifact_is_current(path, run_started_at):
        return None
    try:
        return path.read_text(encoding="utf-8", errors="replace").strip()
    except OSError as exc:
        report_warning("Could not read extraction metadata {}: {}".format(path, exc))
        return None


def _extraction_image_size(run_root, run_started_at):
    text = _current_extraction_text(
        run_root / "misc" / "imsize_after_extract.txt", run_started_at
    )
    if text is None:
        return None
    try:
        image_size = int(text)
    except ValueError:
        report_warning(
            "Invalid post-extraction image size in misc/imsize_after_extract.txt."
        )
        return None
    if image_size <= 0:
        report_warning(
            "Post-extraction image size in misc/imsize_after_extract.txt must be positive."
        )
        return None
    return image_size


def _extracted_measurement_set_names(run_root, run_started_at):
    text = _current_extraction_text(
        run_root / "misc" / "extracted_ms.txt", run_started_at
    )
    if not text:
        return []
    return [line.strip() for line in text.splitlines() if line.strip()]


def _latest_pre_extraction_image(config, artifacts):
    image_prefix = Path(str(config.get("imagename") or "")).name.casefold()
    candidates = []
    for path in artifacts.get("fits_files", []):
        name = path.name.casefold()
        if image_prefix and not name.startswith(image_prefix):
            continue
        if re.search(r"-\d{4}-image\.fits(?:\.gz)?$", name):
            continue
        if name.endswith(
            (
                "-mfs-image.fits",
                "-mfs-image.fits.gz",
                "-image.fits",
                "-image.fits.gz",
            )
        ):
            candidates.append(path)
    if not candidates:
        return None

    def image_order(path):
        match = re.search(r"(\d{3,})(?=-mfs-image|-image)", path.name, re.IGNORECASE)
        cycle = int(match.group(1)) if match else -1
        return cycle, path.name.casefold()

    return max(candidates, key=image_order)


def _plotimage_brightness_limits(run_root, image_path, image_metrics, np, fits):
    reference_name = image_path.name
    cycle_match = re.search(
        r"(\d{3,})(?=-mfs-image|-image)", image_path.name, re.IGNORECASE
    )
    reference_path = image_path
    if cycle_match is not None:
        reference_name = (
            image_path.name[: cycle_match.start(1)]
            + "000"
            + image_path.name[cycle_match.end(1) :]
        )
        candidate_names = (reference_name, reference_name + ".gz")
        reference_candidates = [
            image_path.parent / name for name in candidate_names
        ]
        if not any(path.is_file() for path in reference_candidates):
            fits_directory = run_root / "fits_images"
            reference_candidates.extend(
                path
                for name in candidate_names
                for path in fits_directory.rglob(name)
            )
        reference_path = next(
            (path for path in reference_candidates if path.is_file()),
            image_path,
        )

    noise = next(
        (
            float(record["rms_noise"])
            for record in reversed(image_metrics)
            if Path(str(record.get("image", ""))).name.casefold()
            == reference_name.casefold()
            and _valid_image_metric("rms_noise", record.get("rms_noise"))
            and float(record["rms_noise"]) > 0
        ),
        None,
    )
    if noise is not None:
        noise_source = "selfcal.log"
    else:
        try:
            with fits.open(reference_path, memmap=False) as hdulist:
                if hdulist[0].data is None:
                    raise ValueError("the primary FITS HDU has no image data")
                values = np.asarray(hdulist[0].data).reshape(-1)
            noise = _clipped_image_rms(values, np)
            noise_source = "FITS image"
        except (OSError, ValueError, TypeError, IndexError) as exc:
            report_warning(
                "Could not determine plotimage brightness scaling from {}: {}".format(
                    reference_path, exc
                )
            )
            return None, "Brightness scaling unavailable."

    if not math.isfinite(noise) or noise <= 0:
        report_warning(
            "Could not determine plotimage brightness scaling from {}: invalid RMS noise.".format(
                reference_path
            )
        )
        return None, "Brightness scaling unavailable."
    return (-3.0 * noise, 10.0 * noise), (
        "Brightness limits match plotimage (-3 to +10 × RMS from {} in {}).".format(
            reference_name if noise_source == "selfcal.log" else reference_path.name,
            noise_source,
        )
    )


def _measurement_set_extraction_reference(run_root, config):
    measurement_sets = _as_list(config.get("ms"))
    if not measurement_sets:
        return None
    try:
        import numpy as np
        from casacore.tables import table  # type: ignore
    except ImportError as exc:
        report_warning(
            "Measurement Set metadata is unavailable for the extraction field outline: {}".format(
                exc
            )
        )
        return None

    for measurement_set in measurement_sets:
        path = _resolved_ms_path(measurement_set, run_root)
        try:
            with table(str(path) + "::FIELD", readonly=True, ack=False) as field_table:
                phase_directions = np.asarray(field_table.getcol("PHASE_DIR"))
            phase_direction = phase_directions.reshape(-1, 2)[0]
            if not np.all(np.isfinite(phase_direction)):
                raise ValueError("non-finite phase center")
        except (OSError, RuntimeError, ValueError, TypeError, IndexError) as exc:
            report_warning(
                "Could not read the phase center from {}: {}".format(path, exc)
            )
            continue

        frequency = None
        try:
            with table(
                str(path) + "::SPECTRAL_WINDOW", readonly=True, ack=False
            ) as spectral_window:
                frequencies = np.asarray(
                    spectral_window.getcol("CHAN_FREQ"), dtype=float
                ).reshape(-1)
            finite_frequencies = frequencies[np.isfinite(frequencies)]
            if finite_frequencies.size:
                frequency = float(np.median(finite_frequencies))
        except (OSError, RuntimeError, ValueError, TypeError, IndexError) as exc:
            report_warning(
                "Could not read the observing frequency from {}: {}".format(
                    path, exc
                )
            )
        return (
            float(math.degrees(phase_direction[0]) % 360.0),
            float(math.degrees(phase_direction[1])),
            frequency,
        )
    return None


def _default_extraction_box_degrees(frequency):
    if frequency is None or not math.isfinite(frequency):
        return None
    if frequency < 500e6:
        return 1.5
    if frequency < 1.0e9:
        return 3.0
    if frequency < 1.7e9:
        return 2.0
    if frequency < 4.0e9:
        return 1.5
    return None


def _extraction_region_file(region_setting, run_root):
    if region_setting is None:
        return None, None
    value = str(region_setting).strip()
    if value.casefold() in ("auto", "keepall"):
        return None, None
    try:
        float(value)
    except ValueError:
        path = Path(value).expanduser()
        if not path.is_absolute():
            path = run_root / path
        if path.is_file():
            return path, None
        return None, "Configured extraction region file was not found: {}".format(path)
    return None, None


def _render_extraction_visualization(
    site_dir,
    run_root,
    config,
    artifacts,
    run_started_at,
    extracted_image_size,
    image_metrics,
):
    image_path = _latest_pre_extraction_image(config, artifacts)
    reference = None
    try:
        import numpy as np
        from astropy.io import fits
        from astropy.wcs import WCS
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure
        from matplotlib.patches import Rectangle
    except ImportError as exc:
        report_warning("Extraction visualization is unavailable: {}".format(exc))
        return None, "Visualization unavailable because a plotting dependency is missing."

    image = None
    image_header = None
    if image_path is not None:
        try:
            with fits.open(image_path, memmap=False) as hdulist:
                if hdulist[0].data is None:
                    raise ValueError("the primary FITS HDU has no image data")
                image = np.asarray(hdulist[0].data)
                image_header = hdulist[0].header.copy()
            while image.ndim > 2:
                image = image[0]
            if image.ndim != 2:
                raise ValueError("the FITS image does not contain a 2D celestial plane")
            wcs = WCS(image_header).celestial
            if not wcs.has_celestial:
                raise ValueError("the FITS image has no celestial WCS")
            image_height, image_width = image.shape
            try:
                pixel_scale = float(config["pixelscale"])
            except (KeyError, TypeError, ValueError, OverflowError):
                pixel_scale = None
            image_frequency = image_header.get("CRVAL3")
            background_note = "Background image: {}".format(image_path.name)
        except (OSError, ValueError, TypeError, IndexError) as exc:
            report_warning(
                "Could not use pre-extraction image {}: {}".format(image_path, exc)
            )
            image = None
            image_header = None
            image_path = None
            wcs = None

    if image_path is None:
        try:
            image_width = int(float(config["imsize"]))
            image_height = image_width
            pixel_scale = float(config["pixelscale"])
        except (KeyError, TypeError, ValueError, OverflowError):
            report_warning(
                "Cannot draw the extraction field outline without valid imsize and pixelscale settings."
            )
            return None, "Field-outline visualization unavailable: image size or pixel size is not configured."
        if image_width <= 0 or pixel_scale <= 0 or not math.isfinite(pixel_scale):
            report_warning(
                "Cannot draw the extraction field outline from non-positive imsize or pixelscale settings."
            )
            return None, "Field-outline visualization unavailable: invalid image geometry."

        reference = _measurement_set_extraction_reference(run_root, config)
        wcs = WCS(naxis=2)
        wcs.wcs.crpix = [(image_width + 1) / 2.0, (image_height + 1) / 2.0]
        if reference is None:
            wcs.wcs.crval = [0.0, 0.0]
            wcs.wcs.cdelt = [-pixel_scale, pixel_scale]
            wcs.wcs.ctype = ["LINEAR", "LINEAR"]
            wcs.wcs.cunit = ["arcsec", "arcsec"]
            background_note = (
                "No suitable pre-extraction image is available; showing the "
                "field outline in relative coordinates."
            )
        else:
            wcs.wcs.crval = [reference[0], reference[1]]
            wcs.wcs.cdelt = [-pixel_scale / 3600.0, pixel_scale / 3600.0]
            wcs.wcs.ctype = ["RA---SIN", "DEC--SIN"]
            background_note = (
                "No suitable pre-extraction image is available; showing the "
                "field outline at the Measurement Set phase center."
            )
        image_header = wcs.to_header()
        image_frequency = reference[2] if reference is not None else None

    if image is not None:
        try:
            from astropy.wcs.utils import proj_plane_pixel_scales

            scales = np.asarray(proj_plane_pixel_scales(wcs), dtype=float)
            if scales.size and np.all(np.isfinite(scales)):
                pixel_scale = float(np.mean(np.abs(scales)) * 3600.0)
        except (ImportError, ValueError, TypeError):
            pixel_scale = None

    region_setting = config.get("remove_outside_center_box")
    region_file, region_error = _extraction_region_file(region_setting, run_root)
    region_note = ""
    region_size_pixels = None
    region_file_overlay = False
    region_header = wcs.to_header(relax=True)
    region_header["NAXIS"] = 2
    region_header["NAXIS1"] = image_width
    region_header["NAXIS2"] = image_height
    if region_error:
        report_warning(region_error)
        region_note = region_error
        if extracted_image_size is not None:
            region_size_pixels = float(extracted_image_size)
            region_note += " Showing the post-extraction size footprint."
    elif region_file is not None:
        try:
            import pyregion

            shapes = pyregion.open(str(region_file)).as_imagecoord(region_header)
            patches, texts = shapes.get_mpl_patches_texts(origin=0)
            region_artists = (patches, texts)
            region_file_overlay = True
            region_note = "Extraction region: {}".format(region_file.name)
        except (ImportError, OSError, ValueError, TypeError, KeyError) as exc:
            report_warning(
                "Could not render extraction region {}: {}".format(region_file, exc)
            )
            region_artists = ([], [])
            region_note = "Extraction-region overlay unavailable."
            if extracted_image_size is not None:
                region_size_pixels = float(extracted_image_size)
                region_note += " Showing the post-extraction size footprint."
    else:
        region_artists = ([], [])
        setting_text = str(region_setting).strip().casefold() if region_setting is not None else ""
        if setting_text == "keepall":
            region_size_pixels = float(min(image_width, image_height))
            region_note = "Extraction region: entire field (keepall)."
        elif setting_text == "auto" and extracted_image_size is None:
            region_note = "Automatic extraction region is pending resolution by the workflow."
        elif extracted_image_size is not None:
            region_size_pixels = float(extracted_image_size)
            region_note = "Extraction region: {} × {} pixels.".format(
                extracted_image_size, extracted_image_size
            )
        else:
            try:
                box_degrees = (
                    float(region_setting)
                    if region_setting is not None
                    else _default_extraction_box_degrees(
                        float(image_frequency)
                        if image_frequency is not None
                        else None
                    )
                )
            except (TypeError, ValueError):
                box_degrees = None
            if box_degrees is not None and pixel_scale is not None and pixel_scale > 0:
                region_size_pixels = box_degrees * 3600.0 / pixel_scale
                region_note = "Extraction region: {:.4g}° square.".format(box_degrees)
            elif setting_text == "auto":
                region_note = "Automatic extraction region is pending resolution by the workflow."
            else:
                region_note = "Extraction region dimensions will be available after extraction."

    plot_path = site_dir / "assets" / "extraction-region.png"
    try:
        from matplotlib.patches import Rectangle

        figure = Figure(figsize=(7.5, 6.0), constrained_layout=True)
        FigureCanvasAgg(figure)
        axes = figure.add_subplot(111, projection=wcs)
        if image is not None:
            brightness_limits, brightness_note = _plotimage_brightness_limits(
                run_root, image_path, image_metrics, np, fits
            )
            if brightness_limits is None:
                axes.imshow(image, origin="lower", cmap="bone", interpolation="nearest")
            else:
                axes.imshow(
                    image,
                    origin="lower",
                    cmap="bone",
                    interpolation="nearest",
                    vmin=brightness_limits[0],
                    vmax=brightness_limits[1],
                )
            background_note += " " + brightness_note
        else:
            axes.set_facecolor("#f8fafc")
        axes.add_patch(
            Rectangle(
                (-0.5, -0.5),
                image_width,
                image_height,
                fill=False,
                edgecolor="#2563eb",
                linewidth=1.4,
                label="Field of view",
            )
        )
        if region_file_overlay:
            for artist in region_artists[0]:
                artist.set_edgecolor("#dc2626")
                artist.set_facecolor("none")
                artist.set_linewidth(1.6)
                axes.add_patch(artist)
            for artist in region_artists[1]:
                artist.set_color("#dc2626")
                axes.add_artist(artist)
        elif region_size_pixels is not None:
            center_x = float(wcs.wcs.crpix[0] - 1.0)
            center_y = float(wcs.wcs.crpix[1] - 1.0)
            axes.add_patch(
                Rectangle(
                    (
                        center_x - region_size_pixels / 2.0,
                        center_y - region_size_pixels / 2.0,
                    ),
                    region_size_pixels,
                    region_size_pixels,
                    fill=False,
                    edgecolor="#dc2626",
                    linewidth=1.6,
                    label="Extraction region",
                )
            )
        axes.set_xlim(-0.5, image_width - 0.5)
        axes.set_ylim(-0.5, image_height - 0.5)
        axes.set_aspect("equal", adjustable="box")
        axes.set_title("Field of view and extraction region")
        if reference is None and image_path is None:
            axes.set_xlabel("Offset from field center (arcsec)")
            axes.set_ylabel("Offset from field center (arcsec)")
        else:
            axes.set_xlabel("Right Ascension (J2000)")
            axes.set_ylabel("Declination (J2000)")
        axes.grid(color="#64748b", alpha=0.35, linestyle=":")
        if region_note:
            axes.text(
                0.02,
                0.02,
                region_note,
                transform=axes.transAxes,
                fontsize=8,
                color="#991b1b" if "pending" in region_note.casefold() else "#334155",
                verticalalignment="bottom",
                bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "none"},
            )
        plot_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(plot_path, dpi=140, format="png")
        figure.clear()
    except (OSError, ValueError, TypeError, RuntimeError) as exc:
        report_warning("Could not render extraction field visualization: {}".format(exc))
        return None, "Field-of-view visualization unavailable."

    detail = background_note
    if region_note:
        detail += " " + region_note
    return plot_path, detail


def _extraction_page(
    site_dir, run_root, config, artifacts, run_started_at, logs
):
    title = str(config.get("imagename") or run_root.name)
    option_keys = (
        ("--remove-outside-center", "remove_outside_center"),
        ("--remove-outside-center-region", "remove_outside_center_box"),
        ("--remove-outside-center-auto-minboxsize", "remove_outside_center_auto_minboxsize"),
        ("--remove-outside-center-avgtimestep", "remove_outside_center_avgtimestep"),
        ("--remove-outside-center-noddcor", "remove_outside_center_noddcor"),
        ("--remove-outside-center-avgfreqstep", "remove_outside_center_avgfreqstep"),
    )
    options = {
        label: config.get(key, "Unavailable") for label, key in option_keys
    }
    settings_html = _config_table(options, keys=[label for label, _ in option_keys])

    extracted_image_size = _extraction_image_size(run_root, run_started_at)
    pixel_scale = config.get("pixelscale", "Unavailable")
    if extracted_image_size is None:
        image_size_text = "Pending"
    else:
        image_size_text = "{} × {} pixels".format(
            extracted_image_size, extracted_image_size
        )
    if pixel_scale is None:
        pixel_scale_text = "Unavailable"
    else:
        pixel_scale_text = "{} arcsec/pixel".format(_display_value(pixel_scale))
    output_names = _extracted_measurement_set_names(run_root, run_started_at)
    if output_names:
        output_names_html = "<ul>{}</ul>".format(
            "".join("<li><code>{}</code></li>".format(_escape(name)) for name in output_names)
        )
    else:
        output_names_html = (
            '<p class="empty">Output Measurement Set names will be listed after extraction completes.</p>'
        )
    summary_rows = (
        ("Post-extraction image size to use", image_size_text),
        ("Self-calibration pixel size", pixel_scale_text),
        ("Output Measurement Sets", output_names_html),
    )
    summary_html = '<table class="data-table"><tbody>{}</tbody></table>'.format(
        "".join(
            '<tr><th scope="row">{}</th><td>{}</td></tr>'.format(
                _escape(label),
                value if label == "Output Measurement Sets" else _escape(value),
            )
            for label, value in summary_rows
        )
    )

    plot_path, plot_detail = _render_extraction_visualization(
        site_dir,
        run_root,
        config,
        artifacts,
        run_started_at,
        extracted_image_size,
        logs.get("image_metrics", []),
    )
    if plot_path is None:
        visualization_html = _quality_plot_placeholder(
            "Field-of-view visualization", plot_detail
        )
    else:
        visualization_html = _figure(
            plot_path,
            site_dir,
            "Extraction region within the field of view",
            detail=plot_detail,
        )
    body = (
        '<p class="page-intro">Extraction settings and products for this run. '
        'The visualization uses the latest available pre-extraction self-calibration image; '
        'when none is available, it shows the configured field outline.</p>\n'
        + _section("Extraction summary", summary_html)
        + _section("Extraction settings", settings_html)
        + _section("Field-of-view visualization", visualization_html)
    )
    _write_page(
        site_dir / "extraction.html",
        title,
        "extraction.html",
        body,
        subtitle="Extraction / {}".format(run_root.name),
        optional_pages=_optional_pages(config),
    )


_DIRECTION_FIELDS = ("start", "solints", "smoothness", "soltypes")
_DIRECTION_FIELD_LABELS = {
    "start": "Start cycle",
    "solints": "Solution intervals",
    "smoothness": "Smoothness constraints",
    "soltypes": "Solve per soltype",
}


def _parse_directions_file(path):
    """Parse a facet directions file.

    ``write_facet_directions`` in main.py writes one direction per line as
    ``RA DEC start solints smoothness soltypelist_includedir # direction DirNN``.
    Files supplied with ``--facetdirections`` may only contain RA and DEC.
    """
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError as exc:
        report_warning("Could not read directions file {}: {}".format(path, exc))
        return []
    directions = []
    for line in lines:
        text, _, comment = line.partition("#")
        tokens = re.findall(r"\[[^\]]*\]|\S+", text)
        if len(tokens) < 2:
            continue
        try:
            ra, dec = float(tokens[0]), float(tokens[1])
        except ValueError:
            continue
        name_match = re.search(r"direction\s+(\S+)", comment, re.IGNORECASE)
        record = {
            "name": name_match.group(1) if name_match else "Dir{:02d}".format(len(directions)),
            "ra": ra,
            "dec": dec,
        }
        for key, value in zip(_DIRECTION_FIELDS, tokens[2:]):
            record[key] = value
        directions.append(record)
    return directions


def _direction_value_text(value):
    if value is None:
        return "-"
    try:
        parsed = ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return str(value)
    if isinstance(parsed, (list, tuple)):
        return ", ".join(
            ("yes" if item else "no") if isinstance(item, bool) else str(item)
            for item in parsed
        )
    return str(parsed)


def _format_sexagesimal(value, hours=False):
    value = float(value)
    if hours:
        value = (value % 360.0) / 15.0
        sign = ""
    else:
        sign = "-" if value < 0 else "+"
        value = abs(value)
    whole = int(value)
    minutes = int((value - whole) * 60.0)
    seconds = ((value - whole) * 60.0 - minutes) * 60.0
    if round(seconds, 1) >= 60.0:
        seconds = 0.0
        minutes += 1
    if minutes >= 60:
        minutes -= 60
        whole += 1
    if hours:
        return "{:02d}h{:02d}m{:04.1f}s".format(whole % 24, minutes, seconds)
    return "{}{:02d}d{:02d}m{:04.1f}s".format(sign, whole, minutes, seconds)


def _same_position(a, b, tolerance_deg=1.0 / 3600.0):
    ra_difference = (a[0] - b[0] + 180.0) % 360.0 - 180.0
    return (
        abs(ra_difference * math.cos(math.radians(a[1]))) <= tolerance_deg
        and abs(a[1] - b[1]) <= tolerance_deg
    )


def _read_catalog_summary(path):
    """Return the row count and positions/fluxes of a PyBDSF-style catalogue."""
    cache = DerivedDataCache(path, "dde-catalog-summary-v1")
    cached = cache.load()
    if cached is not None:
        return cached
    try:
        from astropy.io import fits
    except ImportError as exc:
        report_warning("Catalogue summaries are unavailable: {}.".format(exc))
        return None
    try:
        with fits.open(path, memmap=False) as hdul:
            data = hdul[1].data
            columns = set(data.columns.names) if data is not None else set()
            rows = []
            if data is not None and {"RA", "DEC"} <= columns:
                for row in data[:50]:
                    rows.append({
                        "ra": float(row["RA"]),
                        "dec": float(row["DEC"]),
                        "aflux": float(row["AFLUX"]) if "AFLUX" in columns else None,
                        "tflux": float(row["TFLUX"]) if "TFLUX" in columns else None,
                    })
            summary = {"count": 0 if data is None else len(data), "rows": rows}
    except (OSError, IndexError, KeyError, TypeError, ValueError) as exc:
        report_warning("Could not read catalogue {}: {}".format(path, exc))
        return None
    cache.save(summary)
    return summary


def _directions_by_cycle(run_root, config, artifacts):
    """Return {cycle: [direction, ...]} and a description of the source."""
    by_cycle = {}
    for path in artifacts.get("dde_direction_files", []):
        cycle = _image_cycle_identity(path.name)
        if cycle is not None:
            directions = _parse_directions_file(path)
            if directions:
                by_cycle[str(int(cycle)).zfill(3)] = directions
    if by_cycle:
        return by_cycle, "directions/directions_NNN.txt"
    facetdirections = config.get("facetdirections")
    if isinstance(facetdirections, str) and facetdirections not in ("", "None"):
        path = Path(facetdirections).expanduser()
        if not path.is_absolute():
            path = run_root / path
        if path.is_file():
            directions = _parse_directions_file(path)
            if directions:
                return {"all": directions}, str(facetdirections)
    return {}, None


def _errormap_cycle_files(run_root, artifacts):
    """Group errormaps_dd/ products and error-map plots by cycle."""
    grouped = defaultdict(lambda: {"files": [], "plots": []})
    for path in artifacts.get("dde_errormap_files", []):
        cycle = _image_cycle_identity(path.name)
        if cycle is not None:
            grouped[str(int(cycle)).zfill(3)]["files"].append(path)
    for path in artifacts.get("dde_plots", []):
        cycle = _image_cycle_identity(path.name)
        if cycle is not None:
            grouped[str(int(cycle)).zfill(3)]["plots"].append(path)
    return dict(grouped)


def _directions_table_html(directions, previous_positions, catalog):
    varying = {
        key for key in _DIRECTION_FIELDS
        if len({direction.get(key) for direction in directions}) > 1
    }
    has_flux = bool(catalog and any(row.get("tflux") is not None for row in catalog["rows"]))
    headers = ["Direction", "RA", "Dec"]
    if previous_positions is not None:
        headers.append("Added")
    present_fields = [
        key for key in _DIRECTION_FIELDS
        if any(direction.get(key) is not None for direction in directions)
    ]
    headers.extend(_DIRECTION_FIELD_LABELS[key] for key in present_fields)
    if has_flux:
        headers.extend(["Apparent flux", "Total flux"])
    header_html = "".join(
        '<th scope="col"{}>{}</th>'.format(
            ' class="direction-varies" title="Differs between directions"'
            if label in {_DIRECTION_FIELD_LABELS[key] for key in varying} else "",
            _escape(label),
        )
        for label in headers
    )
    rows = []
    for direction in directions:
        position = (direction["ra"], direction["dec"])
        cells = [
            '<th scope="row">{}</th>'.format(_escape(direction["name"])),
            '<td title="{:.6f}">{}</td>'.format(direction["ra"], _escape(_format_sexagesimal(direction["ra"], hours=True))),
            '<td title="{:.6f}">{}</td>'.format(direction["dec"], _escape(_format_sexagesimal(direction["dec"]))),
        ]
        if previous_positions is not None:
            cells.append('<td>{}</td>'.format(_escape(previous_positions.get(direction["name"], "-"))))
        for key in present_fields:
            cells.append('<td{}>{}</td>'.format(
                ' class="direction-varies"' if key in varying else "",
                _escape(_direction_value_text(direction.get(key))),
            ))
        if has_flux:
            match = next(
                (row for row in catalog["rows"] if _same_position(position, (row["ra"], row["dec"]))),
                None,
            )
            for field in ("aflux", "tflux"):
                value = match.get(field) if match else None
                cells.append('<td class="numeric">{}</td>'.format(
                    _escape(_format_jy_value(value)) if value is not None else "-"
                ))
        rows.append("<tr>{}</tr>".format("".join(cells)))
    return (
        '<div class="table-scroll"><table class="data-table direction-table">'
        "<thead><tr>{}</tr></thead><tbody>{}</tbody></table></div>"
    ).format(header_html, "".join(rows))


_DIRECTION_MAP_SIZE = 560
_DIRECTION_MAP_MARGIN = (16, 16, 44, 78)  # top, right, bottom, left


def _direction_map_wcs(artifacts):
    """Return (celestial WCS, nx, ny) of the latest primary image, or None."""
    candidates = []
    for path in artifacts.get("all_fits_files", artifacts.get("fits_files", [])):
        rank = _primary_image_rank({"image": path.name})
        cycle = _image_cycle_identity(path.name)
        if rank and cycle is not None and "-pb" not in path.name.lower():
            candidates.append((rank, int(cycle), path))
    if not candidates:
        return None
    try:
        import warnings
        from astropy.io import fits
        from astropy.wcs import WCS, FITSFixedWarning
    except ImportError as exc:
        report_warning("Direction map is unavailable: {}.".format(exc))
        return None
    path = max(candidates, key=lambda item: (item[1], item[0]))[2]
    try:
        header = fits.getheader(path)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FITSFixedWarning)
            wcs = WCS(header).celestial
        return wcs, int(header["NAXIS1"]), int(header["NAXIS2"])
    except Exception as exc:  # astropy raises several exception types for bad headers
        report_warning("Could not read the WCS of {}: {}".format(path, exc))
        return None


def _read_facet_polygons(path):
    polygons = []
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        report_warning("Could not read facet regions {}: {}".format(path, exc))
        return polygons
    for match in re.finditer(r"polygon\(([^)]*)\)", text, re.IGNORECASE):
        try:
            values = [float(value) for value in match.group(1).split(",")]
        except ValueError:
            continue
        if len(values) >= 6 and len(values) % 2 == 0:
            polygons.append(list(zip(values[0::2], values[1::2])))
    return polygons


def _nice_grid_step(span, candidates):
    for step in candidates:
        if span / step <= 6:
            return step
    return candidates[-1]


def _format_ra_label(ra_deg, step_seconds):
    seconds = round(((ra_deg % 360.0) / 15.0) * 3600.0) % 86400
    hours, remainder = divmod(int(seconds), 3600)
    minutes, secs = divmod(remainder, 60)
    if step_seconds % 60 == 0:
        return "{:02d}h{:02d}m".format(hours, minutes)
    return "{:02d}h{:02d}m{:02d}s".format(hours, minutes, secs)


def _format_dec_label(dec_deg, step_arcsec):
    sign = "-" if dec_deg < 0 else "+"
    total = round(abs(dec_deg) * 3600.0)
    degrees, remainder = divmod(int(total), 3600)
    minutes, secs = divmod(remainder, 60)
    if step_arcsec % 60 == 0:
        return "{}{:02d}°{:02d}′".format(sign, degrees, minutes)
    return "{}{:02d}°{:02d}′{:02d}″".format(sign, degrees, minutes, secs)


def _angular_distance_deg(ra1, dec1, ra2, dec2):
    ra1, dec1, ra2, dec2 = (math.radians(value) for value in (ra1, dec1, ra2, dec2))
    value = (
        math.sin(dec1) * math.sin(dec2)
        + math.cos(dec1) * math.cos(dec2) * math.cos(ra1 - ra2)
    )
    return math.degrees(math.acos(max(-1.0, min(1.0, value))))


def _direction_map_html(site_dir, artifacts, directions_by_cycle, cycle_keys, first_seen, catalogs):
    """Interactive SVG map of the directions per cycle on an RA/Dec grid."""
    geometry = _direction_map_wcs(artifacts)
    if geometry is None or not directions_by_cycle:
        return ""
    wcs, nx, ny = geometry
    size = _DIRECTION_MAP_SIZE
    top, right, bottom, left = _DIRECTION_MAP_MARGIN
    scale = size / float(max(nx, ny))
    plot_w, plot_h = nx * scale, ny * scale

    def to_svg(ra_values, dec_values):
        pixels = wcs.all_world2pix(ra_values, dec_values, 0)
        return [
            (left + (x + 0.5) * scale, top + plot_h - (y + 0.5) * scale, x, y)
            for x, y in zip(pixels[0], pixels[1])
        ]

    def inside(x, y):
        return -0.5 <= x <= nx - 0.5 and -0.5 <= y <= ny - 0.5

    # World coordinates along the image edge give the grid ranges.
    edge = [(x, 0) for x in range(0, nx, max(1, nx // 40))] + [(x, ny - 1) for x in range(0, nx, max(1, nx // 40))]
    edge += [(0, y) for y in range(0, ny, max(1, ny // 40))] + [(nx - 1, y) for y in range(0, ny, max(1, ny // 40))]
    world = wcs.all_pix2world([point[0] for point in edge], [point[1] for point in edge], 0)
    centre = wcs.all_pix2world([(nx - 1) / 2.0], [(ny - 1) / 2.0], 0)
    centre_ra, centre_dec = float(centre[0][0]), float(centre[1][0])
    ra_offsets = [((float(ra) - centre_ra + 180.0) % 360.0) - 180.0 for ra in world[0]]
    dec_min, dec_max = float(min(world[1])), float(max(world[1]))
    ra_min, ra_max = centre_ra + min(ra_offsets), centre_ra + max(ra_offsets)

    ra_step_s = _nice_grid_step((ra_max - ra_min) * 240.0, [5, 10, 15, 20, 30, 60, 120, 300, 600, 900, 1200, 1800, 3600, 7200])
    dec_step_as = _nice_grid_step((dec_max - dec_min) * 3600.0, [10, 15, 30, 60, 120, 300, 600, 900, 1200, 1800, 3600, 7200, 18000, 36000])
    ra_step, dec_step = ra_step_s / 240.0, dec_step_as / 3600.0

    grid = []
    labels = []
    samples = 80

    def add_line(points, label, axis):
        segment = []
        segments = []
        for sx, sy, x, y in points:
            if inside(x, y):
                segment.append((sx, sy))
            elif segment:
                segments.append(segment)
                segment = []
        if segment:
            segments.append(segment)
        for segment in segments:
            if len(segment) > 1:
                grid.append('<polyline class="dirmap-grid" points="{}"/>'.format(
                    " ".join("{:.1f},{:.1f}".format(sx, sy) for sx, sy in segment)
                ))
        visible = [point for segment in segments for point in segment]
        if not visible:
            return
        if axis == "ra":
            sx, sy = max(visible, key=lambda point: point[1])
            if sy >= top + plot_h - 0.06 * plot_h:
                labels.append('<text class="dirmap-axis" x="{:.1f}" y="{:.1f}" text-anchor="middle">{}</text>'.format(
                    sx, top + plot_h + 16, _escape(label)))
        else:
            sx, sy = min(visible, key=lambda point: point[0])
            if sx <= left + 0.06 * plot_w:
                labels.append('<text class="dirmap-axis" x="{:.1f}" y="{:.1f}" text-anchor="end" dominant-baseline="middle">{}</text>'.format(
                    left - 6, sy, _escape(label)))

    ra_value = math.floor(ra_min / ra_step) * ra_step
    while ra_value <= ra_max + 1e-9:
        decs = [dec_min + (dec_max - dec_min) * index / samples for index in range(samples + 1)]
        add_line(to_svg([ra_value] * len(decs), decs), _format_ra_label(ra_value, ra_step_s), "ra")
        ra_value += ra_step
    dec_value = math.floor(dec_min / dec_step) * dec_step
    while dec_value <= dec_max + 1e-9:
        ras = [ra_min + (ra_max - ra_min) * index / samples for index in range(samples + 1)]
        add_line(to_svg(ras, [dec_value] * len(ras)), _format_dec_label(dec_value, dec_step_as), "dec")
        dec_value += dec_step

    # facet_regions/facets_NNN.reg holds the facet layout of cycle NNN;
    # older runs only have facets.reg, the layout of the latest cycle.
    region_files = {path.name.lower(): path for path in artifacts.get("dde_region_files", [])}

    def facet_svg_for(cycle, is_latest):
        path = region_files.get("facets_{}.reg".format(cycle))
        if path is None and is_latest:
            path = region_files.get("facets.reg")
        if path is None:
            return ""
        shapes = []
        for polygon in _read_facet_polygons(path):
            points = to_svg([point[0] for point in polygon], [point[1] for point in polygon])
            shapes.append('<polygon class="dirmap-facet" points="{}"/>'.format(
                " ".join("{:.1f},{:.1f}".format(sx, sy) for sx, sy, _, _ in points)
            ))
        return "".join(shapes)

    calibration_links = [
        (
            directory.name.removeprefix("solution_plots_"),
            "calibration/ms-{:02d}-all.html".format(index),
        )
        for index, (directory, _) in enumerate(artifacts.get("calibration_sets", []), start=1)
    ]

    keys = cycle_keys or ["all"]
    latest = keys[-1]
    facet_svgs = {cycle: facet_svg_for(cycle, cycle == latest) for cycle in keys}
    cycle_groups = []
    cycle_data = {}
    for cycle in keys:
        directions = directions_by_cycle[cycle]
        catalog = catalogs.get(cycle)
        fluxes = []
        records = []
        for direction in directions:
            match = None
            if catalog:
                match = next(
                    (row for row in catalog["rows"]
                     if _same_position((direction["ra"], direction["dec"]), (row["ra"], row["dec"]))),
                    None,
                )
            tflux = match.get("tflux") if match else None
            aflux = match.get("aflux") if match else None
            fluxes.append(tflux)
            index_match = re.search(r"(\d+)$", direction["name"])
            direction_index = int(index_match.group(1)) if index_match else None
            added = first_seen.get(direction["name"]) if cycle_keys else None
            record = {
                "name": direction["name"],
                "ra": _format_sexagesimal(direction["ra"], hours=True),
                "dec": _format_sexagesimal(direction["dec"]),
                "distance": "{:.1f} arcmin".format(
                    60.0 * _angular_distance_deg(direction["ra"], direction["dec"], centre_ra, centre_dec)
                ),
                "added": added,
                "settings": [
                    [_DIRECTION_FIELD_LABELS[key], _direction_value_text(direction.get(key))]
                    for key in _DIRECTION_FIELDS if direction.get(key) is not None
                ],
                "aflux": _format_jy_value(aflux) if aflux is not None else None,
                "tflux": _format_jy_value(tflux) if tflux is not None else None,
                "links": [
                    [name, "{}#sort=cycle&filter=dir-{:02d}".format(href, direction_index)]
                    for name, href in calibration_links
                ] if direction_index is not None else [],
            }
            records.append(record)
        cycle_data[cycle] = records

        known = [flux for flux in fluxes if flux is not None and flux > 0]
        max_flux = max(known) if known else None
        markers = []
        for index, (direction, flux) in enumerate(zip(directions, fluxes)):
            sx, sy, x, y = to_svg([direction["ra"]], [direction["dec"]])[0]
            radius = 7.0
            if max_flux and flux:
                radius = 5.0 + 11.0 * math.sqrt(max(flux, 0.0) / max_flux)
            new = cycle_keys and first_seen.get(direction["name"]) == "Cycle {}".format(cycle)
            # Put the label on the left of markers near the right edge, and
            # below markers near the top edge, so it stays inside the plot.
            label_right = sx + radius + 48 < left + plot_w
            label_x = sx + radius + 3 if label_right else sx - radius - 3
            label_y = sy - radius + 2 if sy - radius - 12 > top else sy + radius + 12
            markers.append(
                '<g class="dirmap-marker{}" data-index="{}" tabindex="0" role="button" '
                'aria-label="{}"><circle cx="{:.1f}" cy="{:.1f}" r="{:.1f}"/>'
                '<text x="{:.1f}" y="{:.1f}" text-anchor="{}">{}</text></g>'.format(
                    " dirmap-new" if new else "", index, _escape(direction["name"]),
                    sx, sy, radius, label_x, label_y,
                    "start" if label_right else "end", _escape(direction["name"]),
                )
            )
        cycle_groups.append(
            '<g class="dirmap-cycle" data-cycle="{}"{}>{}{}</g>'.format(
                _escape(cycle),
                "" if cycle == latest else " hidden",
                '<g class="dirmap-facets">{}</g>'.format(facet_svgs[cycle]) if facet_svgs[cycle] else "",
                "".join(markers),
            )
        )

    width = left + plot_w + right
    height = top + plot_h + bottom
    svg = (
        '<svg class="dirmap-svg" viewBox="0 0 {w:.0f} {h:.0f}" role="img" '
        'aria-label="Direction positions on an RA/Dec grid">'
        '<defs><clipPath id="dirmap-clip"><rect x="{l}" y="{t}" width="{pw:.1f}" height="{ph:.1f}"/></clipPath></defs>'
        '<rect class="dirmap-frame" x="{l}" y="{t}" width="{pw:.1f}" height="{ph:.1f}"/>'
        '<g clip-path="url(#dirmap-clip)">{grid}</g>{labels}'
        '<text class="dirmap-axis-title" x="{cx:.1f}" y="{by:.1f}" text-anchor="middle">Right Ascension (J2000)</text>'
        '<text class="dirmap-axis-title" transform="translate(14 {cy:.1f}) rotate(-90)" text-anchor="middle">Declination (J2000)</text>'
        '<g clip-path="url(#dirmap-clip)">{cycles}</g>'
        '</svg>'
    ).format(
        w=width, h=height, l=left, t=top, pw=plot_w, ph=plot_h,
        grid="".join(grid), labels="".join(labels), cycles="".join(cycle_groups),
        cx=left + plot_w / 2.0, by=height - 6, cy=top + plot_h / 2.0,
    )

    selector = ""
    if len(keys) > 1:
        selector = (
            '<label class="dirmap-cycle-select"><span>Self-calibration cycle</span>'
            '<select data-dirmap-cycle>{}</select></label>'
        ).format("".join(
            '<option value="{0}"{1}>Cycle {0} ({2} directions)</option>'.format(
                _escape(cycle), " selected" if cycle == latest else "", len(directions_by_cycle[cycle])
            )
            for cycle in reversed(keys)
        ))
    legend = (
        '<p class="dirmap-legend"><span class="dirmap-key dirmap-key-old"></span>Direction'
        '<span class="dirmap-key dirmap-key-new"></span>Added in the selected cycle'
        '{}<span>Marker size scales with total flux. Click a direction for details.</span></p>'
    ).format(
        ""
        if not any(facet_svgs.values())
        else '<span class="dirmap-key dirmap-key-facet"></span>Facet boundaries'
        if all(facet_svgs.values())
        else '<span class="dirmap-key dirmap-key-facet"></span>Facet boundaries '
        "(only for cycles with a saved facet layout)"
    )
    return (
        '<div class="dirmap" data-dirmap data-dirmap-directions="{}">'
        '<div class="dirmap-controls">{}</div>{}'
        '<div class="dirmap-figure">{}'
        '<div class="dirmap-popup" data-dirmap-popup hidden role="dialog" aria-live="polite"></div>'
        '</div></div>'
    ).format(
        _escape(json.dumps(cycle_data, separators=(",", ":"))),
        selector,
        legend,
        svg,
    )


def _directions_page(site_dir, run_root, config, artifacts):
    title = str(config.get("imagename") or run_root.name)
    optional_pages = _optional_pages(config)
    sections = []

    directions_by_cycle, directions_source = _directions_by_cycle(run_root, config, artifacts)
    errormap_cycles = _errormap_cycle_files(run_root, artifacts)
    cycle_keys = sorted(
        (key for key in directions_by_cycle if key != "all"), key=_cycle_sort_key
    )

    # Directions of each cycle, latest cycle first.
    if directions_by_cycle:
        first_seen = {}
        for cycle in cycle_keys:
            for direction in directions_by_cycle[cycle]:
                first_seen.setdefault(direction["name"], "Cycle {}".format(cycle))
        catalogs = {}
        for cycle in cycle_keys or ["all"]:
            catalog_paths = [
                path for path in errormap_cycles.get(cycle, {}).get("files", [])
                if path.name.lower().endswith("-errormap.srl.filtered.fits")
            ]
            if catalog_paths:
                catalogs[cycle] = _read_catalog_summary(catalog_paths[0])

        direction_map = _direction_map_html(
            site_dir, artifacts, directions_by_cycle, cycle_keys, first_seen, catalogs
        )
        if direction_map:
            sections.append(_section(
                "Direction overview",
                direction_map,
                "Direction positions within the image area for the selected cycle.",
            ))

        tables = []
        for index, cycle in enumerate(reversed(cycle_keys or ["all"])):
            directions = directions_by_cycle[cycle]
            catalog = catalogs.get(cycle)
            table = _directions_table_html(
                directions, first_seen if cycle_keys else None, catalog
            )
            label = (
                "Cycle {} / {} directions".format(cycle, len(directions))
                if cycle != "all" else "{} directions".format(len(directions))
            )
            tables.append(
                '<details{}><summary>{}</summary>{}</details>'.format(
                    " open" if index == 0 else "", _escape(label), table
                )
            )
        sections.append(_section(
            "Directions",
            "".join(tables),
            "Read from {}. Highlighted columns differ between directions. Fluxes "
            "are the apparent and primary-beam corrected flux of the source in each "
            "direction, from the filtered error-map catalogue.".format(directions_source),
        ))
    else:
        sections.append(_section(
            "Directions",
            '<p class="empty">No facet directions file was found for this run.</p>',
        ))

    # How directions were selected (--auto): counts per cycle and error maps.
    if errormap_cycles:
        rows = []
        previous_names = set()
        for cycle in sorted(set(errormap_cycles) | set(cycle_keys), key=_cycle_sort_key):
            names = [direction["name"] for direction in directions_by_cycle.get(cycle, [])]
            added = [name for name in names if name not in previous_names]
            previous_names.update(names)
            detected = "-"
            for path in errormap_cycles.get(cycle, {}).get("files", []):
                if path.name.lower().endswith("-errormap.merged.srl.fits"):
                    summary = _read_catalog_summary(path)
                    if summary is not None:
                        detected = str(summary["count"])
            rows.append(
                "<tr><th scope=\"row\">Cycle {}</th><td class=\"numeric\">{}</td>"
                "<td class=\"numeric\">{}</td><td>{}</td></tr>".format(
                    _escape(cycle),
                    _escape(detected),
                    len(names) if names else "-",
                    _escape(", ".join(added) if added else "-"),
                )
            )
        selection_table = (
            '<div class="table-scroll"><table class="data-table"><thead><tr>'
            '<th scope="col">Cycle</th><th scope="col">Artefact sources detected</th>'
            '<th scope="col">Directions</th><th scope="col">Added</th>'
            "</tr></thead><tbody>{}</tbody></table></div>"
        ).format("".join(rows))

        figures = []
        cycles_with_plots = []
        for cycle in sorted(errormap_cycles, key=_cycle_sort_key):
            plots = errormap_cycles[cycle]["plots"]
            if plots:
                cycles_with_plots.append(cycle)
            for path in plots:
                index = _ERRORMAP_INDEX_PATTERN.search(path.name)
                figures.append(_figure(
                    path,
                    site_dir,
                    path.name,
                    detail="Cycle {} / error map {}".format(
                        cycle, index.group(1) if index else "?"
                    ),
                    extra_search="cycle-{}".format(cycle),
                    item_class="errormap-entry",
                ))
        gallery = ""
        if figures:
            pills = ['<div class="pill-group" data-pill-group data-pill-target=".errormap-entry">']
            pills.append('<span class="pill-label">Cycle:</span>')
            pills.append('<button type="button" class="pill-btn active" data-filter-value="all">All cycles</button>')
            for cycle in cycles_with_plots:
                pills.append(
                    '<button type="button" class="pill-btn" data-filter-value="cycle-{0}">Cycle {0}</button>'.format(
                        _escape(cycle)
                    )
                )
            pills.append("</div>")
            gallery = (
                "".join(pills)
                + _filter_input(".errormap-entry", "Search error-map plots")
                + '<div class="gallery">{}</div>'.format("".join(figures))
            )

        file_groups = []
        for cycle in sorted(errormap_cycles, key=_cycle_sort_key):
            files = errormap_cycles[cycle]["files"]
            if files:
                file_groups.append(
                    '<details><summary>Cycle {} / {} files</summary><ul class="file-list">{}</ul></details>'.format(
                        _escape(cycle),
                        len(files),
                        "".join(
                            '<li>{}<span class="file-size">{}</span></li>'.format(
                                _link(path, site_dir), _escape(_artifact_size(path))
                            )
                            for path in files
                        ),
                    )
                )
        sections.append(_section(
            "Direction selection",
            selection_table + gallery + "".join(file_groups),
            "Calibration error maps highlight residual artefacts in the image at three "
            "spatial scales. A strong artefact pattern marks a bright source with "
            "calibration errors; these sources become the directions for the next "
            "cycle (red circles in the plots). The errormaps_dd/ files are the error "
            "maps, their source catalogues and the compact-source flux catalogue.",
        ))

    # Facet layout: the latest image plot has the facet regions overlaid.
    layout_parts = []
    numbered_plots = []
    for path in artifacts.get("all_overview_plots", artifacts["overview_plots"]):
        match = re.fullmatch(r"(.+)_([0-9]+)", path.stem)
        if match:
            numbered_plots.append((int(match.group(2)), path))
    if numbered_plots:
        cycle, path = max(numbered_plots, key=lambda item: item[0])
        layout_parts.append('<div class="gallery direction-layout">{}</div>'.format(_figure(
            path, site_dir, path.name, detail="Cycle {:03d} image with facet regions".format(cycle)
        )))
    layout_links = list(artifacts.get("dde_region_files", []))
    facets_fits = run_root / "fits_images" / "facets.fits"
    if facets_fits.is_file():
        layout_links.append(facets_fits)
    if layout_links:
        layout_parts.append('<ul class="file-list">{}</ul>'.format("".join(
            '<li>{}<span class="file-size">{}</span></li>'.format(
                _link(path, site_dir, path.relative_to(run_root).as_posix()),
                _escape(_artifact_size(path)),
            )
            for path in layout_links
        )))
    if layout_parts:
        sections.append(_section(
            "Facet layout",
            "".join(layout_parts),
            "facets.reg defines the facet assigned to each direction for imaging; "
            "directions_NNN.reg marks the direction centres per cycle.",
        ))

    body = (
        '<p class="page-intro">Direction-dependent calibration: the directions solved for, '
        "how they were selected and the facet layout. Solution plots per direction "
        'are on the <a href="calibration.html">Calibration</a> pages.</p>\n'
        + "\n".join(sections)
    )
    _write_page(
        site_dir / "directions.html",
        title,
        "directions.html",
        body,
        subtitle="Directions / {}".format(run_root.name),
        optional_pages=optional_pages,
    )


def _run_details_page(site_dir, run_root, config, artifacts, logs, command_record, status, error):
    title = str(config.get("imagename") or run_root.name)
    file_links = []
    relatives = ["full_config.txt", "logs/selfcal.log", "logs/resource_usage.csv", "logs/resource_phases.csv"]
    if command_record and command_record["source"] == "file":
        relatives.insert(1, "facetselfcal.txt")
    for relative in relatives:
        path = run_root / relative
        if path.is_file():
            file_links.append(
                '<li>{}<span class="file-size">{}</span></li>'.format(
                    _link(path, site_dir), _escape(_artifact_size(path))
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
    log_tails = _log_tail_live_frame(site_dir) or (
        "".join(tail_sections) if tail_sections else '<p class="empty">No logs were found.</p>'
    )

    config_filter = _filter_input(".config-row", "Filter configuration options")
    config_table = _config_table(config, filterable=True)
    command_section = '<p class="empty">No command line was recorded for this run.</p>'
    source_note = None
    if command_record:
        source_note = {
            "marker": "Logged by facetselfcal at startup{}.",
            "log": "Taken from logs/selfcal.log{}.",
            "file": "Taken from the user-supplied facetselfcal.txt; this run did not log its command line.",
        }[command_record["source"]].format(
            " ({})".format(command_record["timestamp"]) if command_record.get("timestamp") else ""
        )
        if command_record.get("cwd"):
            source_note += " Working directory: {}.".format(command_record["cwd"])
        command_section = "<pre>{}</pre>".format(_escape(command_record["command"]))

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
        status_block += '<div class="notice info"><strong>Combined log</strong><p>{} command-line invocation records were found. The log may span multiple processing segments.</p></div>'.format(logs["invocations"])

    # Execution Environment card
    env_section = ""
    host_info = logs.get("host_info", {})
    if host_info:
        env_cards = []
        if "host" in host_info:
            env_cards.append('<div class="env-card"><strong>Run Host</strong><span>{}</span></div>'.format(_escape(host_info["host"])))
        if "cpu" in host_info:
            env_cards.append('<div class="env-card"><strong>Processors</strong><span>{}</span></div>'.format(_escape(host_info["cpu"])))
        if "ram" in host_info:
            env_cards.append('<div class="env-card"><strong>RAM (Total / Avail)</strong><span>{}</span></div>'.format(_escape(host_info["ram"])))
        if "disk" in host_info:
            env_cards.append('<div class="env-card"><strong>Disk at Start</strong><span>{}</span></div>'.format(_escape(host_info["disk"])))
        if "os" in host_info:
            env_cards.append('<div class="env-card"><strong>Operating System</strong><span>{}</span></div>'.format(_escape(host_info["os"])))
        if "version" in host_info:
            env_cards.append('<div class="env-card"><strong>facetselfcal Version</strong><span>{}</span></div>'.format(_escape(host_info["version"])))
        res = logs.get("resources")
        if res and res.get("peak_tree_rss_gib") is not None:
            env_cards.append('<div class="env-card"><strong>Peak Process RAM</strong><span>{:.1f} GiB</span></div>'.format(res["peak_tree_rss_gib"]))
        if res and res.get("peak_tree_cpu_pct") is not None:
            env_cards.append('<div class="env-card"><strong>Peak Process CPU</strong><span>{:.0f}%</span></div>'.format(res["peak_tree_cpu_pct"]))
        if env_cards:
            env_section = _section("Execution Environment", '<div class="env-grid">{}</div>'.format("".join(env_cards)))

    resource_section = ""
    res = logs.get("resources")
    resource_content = _resource_live_frame(site_dir)
    live_chart_frame = _resource_chart_live_frame(site_dir)
    if res and res.get("samples"):
        resource_content += _render_resource_section(
            res,
            live_chart_frame=live_chart_frame or None,
        )
    else:
        resource_content += live_chart_frame
    if resource_content:
        resource_note = (
            "The latest-sample window defaults to 300 samples and can be adjusted in the chart controls. While monitoring is active both time ranges come from the live chart, which refreshes about every 15 seconds. Phase bars use explicit workflow boundaries; process-tree CPU is relative to one core and can exceed 100%."
            if live_chart_frame
            else "Full time range shows all recorded samples; the latest-sample window defaults to 300 samples and can be adjusted in the chart controls. Phase bars use explicit workflow boundaries; process-tree CPU is relative to one core and can exceed 100%."
        )
        resource_section = _section(
            "Resource Utilization (CPU & RAM)",
            resource_content,
            resource_note,
        )

    body = (
        '<p class="page-intro">The report references the run configuration and logs in place. Values are shown as recorded, without interpreting instrument-specific settings.</p>\n'
        + status_block + "\n"
        + env_section + "\n"
        + (resource_section + "\n" if resource_section else "")
        + _section("Configuration files and logs", links_html) + "\n"
        + _section("Recorded command", command_section, source_note) + "\n"
        + _section("Saved configuration", config_filter + config_table) + "\n"
        + _section("Artifact inventory", inventory) + "\n"
        + _section("Warning log records", _render_log_events(logs["warnings"], logs["warning_count"])) + "\n"
        + _section("Error log records", _render_log_events(logs["errors"], logs["error_count"], is_error=True)) + "\n"
        + _section("Recent log excerpts", log_tails, "At most the last 80 lines of each log are included here; while monitoring is active they refresh about every 15 seconds. The full logs are linked above.")
    )
    _write_page(
        site_dir / "run-details.html",
        title,
        "run-details.html",
        body,
        subtitle="Run details & logs / {}".format(run_root.name),
        optional_pages=_optional_pages(config),
    )


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
    with report_context(site_dir / ".cache") as (diagnostics, pages):
        _build_html_overview(run_root, site_dir, status, error)
        notice = ""
        if diagnostics:
            notice = (
                '<aside class="notice" role="status"><strong>Report input diagnostics</strong>'
                '<ul>{}</ul></aside>'
            ).format("".join("<li>{}</li>".format(_escape(item)) for item in diagnostics))
        atomic_write_text(asset_dir / "report.css", _report_css())
        atomic_write_text(asset_dir / "report.js", resource_chart_browser_js() + "\n" + _JS)
        generation = datetime.now().isoformat()
        for path, content in sorted(
            pages.items(), key=lambda item: (item[0] == site_dir / "index.html", str(item[0]))
        ):
            content = content.replace(
                "<main>", '<main data-report-generated="{}">{}'.format(_escape(generation), notice), 1
            )
            atomic_write_text(path, content)
    return site_dir / "index.html"


def _build_html_overview(run_root, site_dir, status, error):
    config = _parse_config(run_root / "full_config.txt")
    stack_mode = _stack_enabled(config)
    stack_groups = _load_stack_imaging_groups(run_root) if stack_mode else {}
    run_started_at, current_run_cycles = _current_run_context(run_root)
    try:
        start_cycle = int(config.get("start", 0))
    except (TypeError, ValueError):
        start_cycle = 0
    artifacts = _scan_artifacts(
        run_root,
        run_started_at,
        current_run_cycles,
        start_cycle=start_cycle,
        stack_mode=stack_mode,
    )
    logs = _scan_logs(
        run_root,
        stack_mode=stack_mode,
        stack_groups=stack_groups,
    )
    logs["resources"] = _scan_resource_log(run_root, run_started_at, start_cycle)
    command_record = _current_command(logs, run_root)
    logs["command_record"] = command_record
    current_image_names = {path.name.casefold() for path in artifacts["fits_files"]}
    logs["cycle_timeline"] = _filter_restart_cycle_timeline(
        logs["cycle_timeline"], run_started_at, start_cycle
    )
    logs["image_metrics"] = _filter_restart_image_metrics(
        logs["image_metrics"],
        run_started_at,
        start_cycle,
        current_image_names,
        stack_mode=stack_mode,
    )
    logs["image_metrics"] = _image_metrics_from_fits(
        artifacts.get("all_fits_files", artifacts["fits_files"]),
        logs.get("image_metrics", []),
        stack_mode=stack_mode,
        stack_groups=stack_groups,
    )
    logs["image_metrics"] = _attach_wsclean_commands(
        logs["image_metrics"],
        logs.get("image_commands", []),
        stack_mode=stack_mode,
    )
    logs["image_metrics"] = _add_image_dynamic_range(logs["image_metrics"])

    _overview_page(site_dir, run_root, config, artifacts, logs, status, error)
    _imaging_page(site_dir, run_root, config, artifacts, logs)
    if _extraction_enabled(config):
        _extraction_page(
            site_dir, run_root, config, artifacts, run_started_at, logs
        )
    if _dde_enabled(config):
        _directions_page(site_dir, run_root, config, artifacts)
    _calibration_page(site_dir, run_root, config, artifacts)
    if _bandpass_enabled(config):
        _bandpass_page(site_dir, run_root, config, artifacts)
    _datasets_page(site_dir, run_root, config, artifacts, logs)
    _run_details_page(site_dir, run_root, config, artifacts, logs, command_record, status, error)


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
