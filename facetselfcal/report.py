"""Generate a browsable, offline HTML report from a facetselfcal run directory."""

import argparse
import ast
import csv
import html
import json
import math
import os
import re
from collections import defaultdict, deque
from datetime import datetime
from pathlib import Path
from urllib.parse import quote

from .resource_chart import (
    RESOURCE_PHASE_STYLES,
    RESOURCE_RAM_COLOR,
    dp3_command_phases,
    generate_resource_svg,
    phase_intervals_from_events,
)


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
.resource-chart-box { background: var(--surface); border: 1px solid var(--line); border-radius: 6px; padding: 16px; margin-bottom: 16px; box-shadow: 0 1px 3px rgba(15,23,42,0.03); }
.resource-chart-legend { display: flex; flex-wrap: wrap; gap: 16px; margin-bottom: 12px; font-size: 12px; font-weight: 600; }
.legend-item { display: inline-flex; align-items: center; gap: 6px; }
.legend-swatch { width: 14px; height: 4px; border-radius: 2px; display: inline-block; }
.chart-container { width: 100%; overflow-x: auto; }
details { margin: 11px 0; border: 1px solid var(--line); border-radius: 6px; background: var(--surface); overflow: hidden; }
details > summary { cursor: pointer; padding: 10px 14px; color: var(--teal-deep); font-weight: 600; font-size: 14px; background: #fafcff; }
details[open] > summary { border-bottom: 1px solid var(--line); }
details > :not(summary) { margin: 12px 14px; }
details > .data-table, details > .file-list { margin: 0 0 12px; }
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
    return _CSS.replace(marker, phase_colors, 1)


_JS = r"""
window.addEventListener("message", function (event) {
    var frame = document.querySelector(".resource-chart-live-frame");
    if (!frame || event.source !== frame.contentWindow) return;
    var data = event.data;
    if (!data || data.type !== "facetselfcal-resource-chart-size") return;
    if (typeof data.height !== "number" || !Number.isFinite(data.height)) return;
    frame.style.height = Math.ceil(data.height) + "px";
});

document.addEventListener("DOMContentLoaded", function () {
    var liveChartFrame = document.querySelector(".resource-chart-live-frame");
    if (liveChartFrame) {
        var requestLiveChartSize = function () {
            liveChartFrame.contentWindow.postMessage(
                { type: "facetselfcal-resource-chart-size-request" },
                "*"
            );
        };
        liveChartFrame.addEventListener("load", requestLiveChartSize);
        requestLiveChartSize();
    }

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

    document.querySelectorAll("[data-compare-tool]").forEach(function (tool) {
        var selectA = tool.querySelector("[data-compare-select-a]");
        var selectB = tool.querySelector("[data-compare-select-b]");
        var imgA = tool.querySelector("[data-compare-img-a]");
        var imgB = tool.querySelector("[data-compare-img-b]");
        var linkA = tool.querySelector("[data-compare-link-a]");
        var linkB = tool.querySelector("[data-compare-link-b]");
        var capA = tool.querySelector("[data-compare-caption-a]");
        var capB = tool.querySelector("[data-compare-caption-b]");
        var update = function () {
            if (selectA && imgA) {
                var optA = selectA.options[selectA.selectedIndex];
                if (optA) {
                    imgA.src = optA.value;
                    imgA.alt = optA.text;
                    if (linkA) linkA.href = optA.value;
                    if (capA) capA.textContent = optA.text;
                }
            }
            if (selectB && imgB) {
                var optB = selectB.options[selectB.selectedIndex];
                if (optB) {
                    imgB.src = optB.value;
                    imgB.alt = optB.text;
                    if (linkB) linkB.href = optB.value;
                    if (capB) capB.textContent = optB.text;
                }
            }
        };
        if (selectA) selectA.addEventListener("change", update);
        if (selectB) selectB.addEventListener("change", update);
        update();
    });

    document.querySelectorAll("[data-pill-group]").forEach(function (group) {
        var targetSelector = group.getAttribute("data-pill-target");
        var items = Array.from(document.querySelectorAll(targetSelector));
        var buttons = Array.from(group.querySelectorAll(".pill-btn"));
        buttons.forEach(function (btn) {
            btn.addEventListener("click", function () {
                buttons.forEach(function (b) { b.classList.remove("active"); });
                btn.classList.add("active");
                var filterVal = (btn.getAttribute("data-filter-value") || "").trim().toLowerCase();
                items.forEach(function (item) {
                    var search = (item.getAttribute("data-search") || item.textContent || "").toLowerCase();
                    item.hidden = filterVal !== "" && filterVal !== "all" && !search.includes(filterVal);
                });
                document.querySelectorAll("details[data-filter-group]").forEach(function (g) {
                    var visible = Array.from(g.querySelectorAll(targetSelector)).some(function (i) { return !i.hidden; });
                    g.hidden = !visible;
                    if (filterVal && filterVal !== "all" && visible) g.open = true;
                });
            });
        });
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
"""


def _escape(value):
    return html.escape(str(value), quote=True)


def _bandpass_enabled(config):
    value = config.get("bandpass")
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
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024 or unit == "TB":
            return "{:.1f} {}".format(value, unit)
        value /= 1024
    return "{:.1f} TB".format(value)


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
    return quote(Path(relative).as_posix(), safe="/._-()[]")


def _page_shell(
    title,
    active_page,
    body,
    nested=False,
    subtitle="Offline processing report",
    bandpass_enabled=False,
):
    prefix = "../" if nested else ""
    nav = []
    page_names = list(_PAGE_NAMES.items())
    if bandpass_enabled:
        page_names.insert(3, ("bandpass.html", "Bandpass"))
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
    return (
        '<div class="search-row"><label for="report-filter">{}</label>'
        '<input id="report-filter" type="search" data-filter-target="{}" '
        'placeholder="Type to filter" autocomplete="off"></div>'
    ).format(_escape(label), _escape(target))


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
    except OSError:
        return True


def _include_image_artifact(path, run_started_at, start_cycle):
    cycle_match = re.search(r"_(\d+)(?:-|\.|$)", path.name)
    if cycle_match is None or int(cycle_match.group(1)) < start_cycle:
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


def _filter_restart_image_metrics(records, run_started_at, start_cycle, current_image_names):
    if run_started_at is None:
        return records
    filtered = []
    for record in records:
        cycle = record.get("cycle")
        try:
            cycle = int(cycle)
        except (TypeError, ValueError):
            cycle_match = re.search(r"_(\d+)(?:-|\.|$)", Path(record.get("image", "")).name)
            cycle = int(cycle_match.group(1)) if cycle_match else None
        if cycle is None or cycle < start_cycle:
            filtered.append(record)
            continue
        metric_time = record.get("_timestamp")
        image_name = Path(record.get("image", "")).name.casefold()
        if (metric_time is not None and metric_time >= run_started_at) or image_name in current_image_names:
            filtered.append(record)
    return filtered


def _is_ateam_plot(path):
    return path.name.lower().startswith("ateam_")


def _is_ms_plot(path):
    name = path.name.lower()
    return name.endswith(".time_coverage.png") or _is_ateam_plot(path)


def _scan_artifacts(run_root, run_started_at=None, current_run_cycles=None, start_cycle=0):
    overview_dir = run_root / "plots"
    all_raw_plots = sorted(
        (
            path for path in overview_dir.glob("*.png")
            if path.is_file()
        ),
        key=lambda path: path.name.lower(),
    ) if overview_dir.is_dir() else []

    raw_plots = [path for path in all_raw_plots if _artifact_is_current(path, run_started_at)]
    all_overview_plots = [
        p for p in all_raw_plots
        if not _is_ms_plot(p) and _include_image_artifact(p, run_started_at, start_cycle)
    ]
    overview_plots = [p for p in raw_plots if not _is_ms_plot(p)]
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
            and _include_image_artifact(path, run_started_at, start_cycle)
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
                cycle = match.group(1)
                if current_run_cycles is not None and cycle not in current_run_cycles:
                    continue
                cycle_names.add(cycle)
            solution_files.append(path)
    solution_files.sort(key=lambda path: path.name.lower())
    bandpass_files.sort(key=lambda path: path.as_posix().casefold())

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


def _ms_path_from_command(message):
    match = re.search(
        r'''(?:^|\s)msin=(?:"([^"]+)"|'([^']+)'|([^\s]+))''',
        message,
        re.IGNORECASE,
    )
    if match is None:
        return None
    return next((value.rstrip(",") for value in match.groups() if value), None)


def _ms_group_key(ms_path):
    if not ms_path:
        return None
    normalized = os.path.normcase(os.path.normpath(ms_path.rstrip("/")))
    return re.sub(r"(?:\.(?:copy|avg))+$", "", normalized, flags=re.IGNORECASE)


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
        return step_labels.get(step["kind"], step["name"])

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
            badges.append(
                '<div class="workflow-step-item">'
                '<button type="button" class="workflow-step-badge workflow-command-trigger {}" '
                'popovertarget="{}" title="{}">{} <span class="workflow-step-duration">{}</span></button>'
                '<pre class="workflow-command-text" id="{}" popover="auto">{}</pre>'
                '{}'
                '</div>'.format(
                    badge_class,
                    command_id,
                    _escape(title),
                    _escape(label),
                    _escape(step.get("duration") or "-"),
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


def _scan_logs(run_root):
    candidates = [run_root / "logs" / "selfcal.log", run_root / "h5plot.log"]
    logs = []
    warnings = deque(maxlen=20)
    errors = deque(maxlen=20)
    warning_count = 0
    error_count = 0
    invocations = 0
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
    image_metric_pattern = re.compile(
        r"^(?P<image>.+?)\s+(?P<metric>Max image|Min image|RMS noise):\s*(?P<value>.+)$",
        re.IGNORECASE,
    )
    image_metric_fields = {
        label.casefold(): field for field, label in _IMAGE_METRIC_FIELDS
    }
    image_metric_records = {}
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
    }
    effective_cycle_config = {}
    effective_cycle_config_errors = set()

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
                        if ts_obj:
                            if first_ts is None:
                                first_ts = ts_obj
                            last_ts = ts_obj

                    if level == "WARNING":
                        warning_count += 1
                        warnings.append((path, line_count, timestamp, message))
                    elif level in ("ERROR", "CRITICAL"):
                        error_count += 1
                        errors.append((path, line_count, timestamp, message))
                    if re.search(r"facetselfcal(?:\.py)?\b.*(?:\s-i(?:\s|=)|--config(?:\s|=))", line, re.IGNORECASE):
                        invocations += 1

                    if path.name == "selfcal.log":
                        setting_name, separator, setting_value = message.partition(":")
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
                                _canonical_ms_path(ms_path, run_root),
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
                                _canonical_ms_path(ms_path, run_root),
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

                        if ts_obj is not None:
                            dp3_phases = dp3_command_phases(message)
                            if dp3_phases:
                                ms_path = _ms_path_from_command(message)
                                ms_key = _ms_group_key(ms_path)
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
                            metric_value = metric_match.group("value").strip()
                            try:
                                float(metric_value)
                            except ValueError:
                                pass
                            else:
                                metric_cycle = current_cycle
                                if metric_cycle is None:
                                    cycle_tokens = re.findall(
                                        r"_(\d+)(?=-|\.|$)", Path(image_name).name
                                    )
                                    if cycle_tokens:
                                        metric_cycle = str(int(cycle_tokens[-1])).zfill(3)
                                field = image_metric_fields[
                                    metric_match.group("metric").casefold()
                                ]
                                record_key = (metric_cycle or "", image_name)
                                record = image_metric_records.setdefault(
                                    record_key,
                                    {
                                        "cycle": metric_cycle,
                                        "image": image_name,
                                    },
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

                            if kind is not None and not any(
                                existing["kind"] == kind for existing in cdata["steps"]
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
                            elif kind is None and "DP3 solve:" in message:
                                ms_path = _ms_path_from_command(message)
                                ms_key = _ms_group_key(ms_path)
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
                                ms_key = _ms_group_key(ms_path)
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

        except OSError:
            continue
        logs.append({"path": path, "lines": line_count, "tail": list(tail)})

    def build_step_details(steps, end_time):
        step_details = []
        for s_idx, step in enumerate(steps):
            step_ts = step["timestamp"]
            next_ts = next(
                (
                    candidate["timestamp"]
                    for candidate in steps[s_idx + 1 :]
                    if candidate["timestamp"] is not None
                    and candidate["timestamp"] > step_ts
                ),
                end_time,
            )
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
        "timestamps": timestamps,
        "host_info": host_info,
        "effective_cycle_config": effective_cycle_config,
        "effective_cycle_config_errors": sorted(effective_cycle_config_errors),
        "flagging_stats": flagging_stats,
        "cycle_timeline": [cycles[k] for k in cycle_keys],
        "setup_step_details": setup_step_details,
        "setup_start_str": setup_start_str,
        "setup_duration_str": setup_duration_str,
        "image_metrics": sorted(
            image_metric_records.values(), key=lambda record: record["_order"]
        ),
        "total_elapsed": total_elapsed,
    }


def _clipped_image_rms(values, np):
    pixels = values[np.abs(values) > 1e-7]
    if not pixels.size:
        return float("nan")

    rms_old = np.std(pixels)
    median = np.median(pixels)
    rms = rms_old
    for _ in range(10):
        selected = pixels[np.abs(pixels - median) < rms_old * 3.0]
        if not selected.size:
            return float("nan")
        rms = np.std(selected)
        if rms_old != 0 and np.abs((rms - rms_old) / rms_old) < 1e-1:
            break
        rms_old = rms
    return rms


def _image_metrics_from_fits(fits_files, records):
    primary_images = []
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
        cycle_match = re.search(r"_(\d{3})(?=-|\.|$)", path.name)
        if cycle_match:
            primary_images.append((path, str(int(cycle_match.group(1))).zfill(3)))

    if not primary_images:
        return records

    try:
        import numpy as np
        from astropy.io import fits
    except ImportError:
        return records

    records_by_image = {
        Path(record.get("image", "")).name.casefold(): record
        for record in records
    }
    for path, cycle in primary_images:
        record = records_by_image.get(path.name.casefold())
        try:
            with fits.open(path, memmap=False) as hdulist:
                if hdulist[0].data is None:
                    continue
                values = np.asarray(hdulist[0].data).reshape(-1)
                if not values.size:
                    continue
                computed = {
                    "max_image": str(np.max(values)),
                    "min_image": str(np.min(values)),
                    "rms_noise": str(_clipped_image_rms(values, np)),
                }
        except Exception:
            continue

        if record is None:
            record = {
                "cycle": cycle,
                "image": str(path),
            }
            records.append(record)
            records_by_image[path.name.casefold()] = record
        for field, value in computed.items():
            record.setdefault(field, value)

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

    events = []
    try:
        with csv_path.open("r", encoding="utf-8", errors="replace") as stream:
            for row in csv.DictReader(stream):
                try:
                    epoch = float(row["epoch"])
                    cycle_raw = row.get("cycle", "").strip()
                    cycle = (
                        None
                        if cycle_raw.casefold() in {"", "none"}
                        else int(cycle_raw)
                    )
                    phase = row.get("phase", "").strip().lower()
                    event = row.get("event", "").strip().lower()
                except (KeyError, ValueError, TypeError):
                    continue

                if phase not in RESOURCE_PHASE_STYLES or event not in {"start", "end"}:
                    continue
                if start_cycle > 0 and run_started_at is not None and cycle is not None:
                    if cycle >= start_cycle and epoch < run_started_at:
                        continue

                events.append({
                    "timestamp": row.get("timestamp", ""),
                    "epoch": epoch,
                    "cycle": cycle,
                    "phase": phase,
                    "event": event,
                })
    except OSError:
        return []

    return events


def _scan_resource_log(run_root, run_started_at=None, start_cycle=0):
    """Scan and aggregate process-tree resource records from logs/resource_usage.csv."""
    csv_path = run_root / "logs" / "resource_usage.csv"
    if not csv_path.is_file():
        return None

    valid_samples = []
    try:
        with csv_path.open("r", encoding="utf-8", errors="replace") as stream:
            reader = csv.DictReader(stream)
            for row in reader:
                try:
                    epoch = float(row["epoch"])
                    tree_cpu = float(row["tree_cpu_pct"])
                    tree_rss = float(row["tree_rss_gib"])
                    sys_cpu = float(row.get("sys_cpu_pct", 0.0))
                    sys_ram_used = float(row.get("sys_ram_used_gib", 0.0))
                    sys_ram_total = float(row.get("sys_ram_total_gib", 0.0))
                except (KeyError, ValueError, TypeError):
                    continue

                cycle_raw = row.get("cycle", "").strip()
                try:
                    cycle_val = int(cycle_raw)
                except ValueError:
                    cycle_val = cycle_raw

                if start_cycle > 0 and run_started_at is not None:
                    if isinstance(cycle_val, int) and cycle_val >= start_cycle:
                        if epoch < run_started_at:
                            continue

                valid_samples.append({
                    "timestamp": row.get("timestamp", ""),
                    "epoch": epoch,
                    "cycle": cycle_val,
                    "tree_cpu_pct": tree_cpu,
                    "tree_rss_gib": tree_rss,
                    "sys_cpu_pct": sys_cpu,
                    "sys_ram_used_gib": sys_ram_used,
                    "sys_ram_total_gib": sys_ram_total,
                })
    except OSError:
        return None

    if not valid_samples:
        return None

    phase_events = _scan_resource_phase_log(run_root, run_started_at, start_cycle)
    phase_intervals = None
    if phase_events is not None:
        phase_end_epoch = max(
            valid_samples[-1]["epoch"],
            phase_events[-1]["epoch"] if phase_events else valid_samples[-1]["epoch"],
        )
        phase_intervals = phase_intervals_from_events(
            phase_events, end_epoch=phase_end_epoch
        )

    peak_tree_rss_gib = max(s["tree_rss_gib"] for s in valid_samples)
    peak_tree_cpu_pct = max(s["tree_cpu_pct"] for s in valid_samples)
    avg_tree_cpu_pct = sum(s["tree_cpu_pct"] for s in valid_samples) / len(valid_samples)
    peak_sys_ram_used_gib = max(s["sys_ram_used_gib"] for s in valid_samples)
    sys_ram_total_gib = max(s["sys_ram_total_gib"] for s in valid_samples)
    peak_sys_cpu_pct = max(s["sys_cpu_pct"] for s in valid_samples)

    cycle_stats = {}
    for s in valid_samples:
        c = s["cycle"]
        keys = [c]
        if isinstance(c, int):
            keys.extend([str(c), str(c).zfill(3)])
        for k in keys:
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

    if len(valid_samples) > 300:
        step = math.ceil(len(valid_samples) / 300)
        chart_samples = valid_samples[::step]
        if chart_samples[-1] is not valid_samples[-1]:
            chart_samples.append(valid_samples[-1])
    else:
        chart_samples = valid_samples

    return {
        "present": True,
        "samples": chart_samples,
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


def _render_resource_section(resources, include_chart=True):
    """Render summary cards, interactive chart, and cycle table for resources."""
    if not resources or not resources.get("samples"):
        return '<p class="empty">No resource usage samples recorded.</p>'

    cards = [
        f'<div class="resource-card"><strong>Peak Process RAM</strong><span>{resources["peak_tree_rss_gib"]:.2f} GiB</span></div>',
        f'<div class="resource-card"><strong>Peak Process CPU</strong><span>{resources["peak_tree_cpu_pct"]:.0f}%</span></div>',
        f'<div class="resource-card"><strong>Average Process CPU</strong><span>{resources["avg_tree_cpu_pct"]:.0f}%</span></div>',
        f'<div class="resource-card"><strong>Peak System RAM</strong><span>{resources["peak_sys_ram_used_gib"]:.1f} / {resources["sys_ram_total_gib"]:.1f} GiB</span></div>',
        f'<div class="resource-card"><strong>Peak System CPU</strong><span>{resources["peak_sys_cpu_pct"]:.0f}%</span></div>',
    ]

    chart_box = ""
    if include_chart:
        phase_intervals = resources.get("phase_intervals")
        svg_chart = _generate_resource_svg(resources["samples"], phase_intervals)
        legend_html = (
            '<div class="resource-chart-legend">'
            '<span class="legend-item"><span class="legend-swatch" style="background:#0d9488;"></span> Process Tree CPU (% of one core)</span>'
            f'<span class="legend-item"><span class="legend-swatch" style="background:{RESOURCE_RAM_COLOR};"></span> Process Tree RAM (GiB)</span>'
            '<span class="legend-item" style="color:var(--muted);"><span class="legend-swatch" style="border-top:2px dashed #94a3b8; background:transparent;"></span> Cycle transition</span>'
            '</div>'
        )
        chart_box = (
            '<div class="resource-chart-box">'
            f'{legend_html}'
            f'<div class="chart-container">{svg_chart}</div>'
            '</div>'
        )

    cycle_stats = resources.get("cycle_stats", {})
    unique_cycles = []
    seen = set()
    for s in resources["samples"]:
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
    except OSError:
        return ""
    if '<meta http-equiv="refresh"' not in page_head:
        return ""
    return (
        '<iframe class="resource-chart-live-frame" src="resource-chart-live.html" '
        'title="Live process tree CPU and RAM chart"></iframe>'
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
            for raw_line in stream:
                _, _, message = _parse_log_line(raw_line.rstrip("\r\n"))
                normalized_message = message.strip()
                if normalized_message == "================":
                    current_metadata = None
                    continue
                if normalized_message.startswith("===") and normalized_message.endswith("==="):
                    ms_path = normalized_message[3:-3].strip()
                    if ms_path:
                        current_metadata = {"_source_path": ms_path}
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


def _resolved_ms_path(path, run_root):
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = run_root / candidate
    return os.path.normpath(str(candidate))


def _canonical_ms_path(path, run_root):
    return _strip_ms_copy_avg_suffix(_resolved_ms_path(path, run_root))


def _strip_ms_copy_avg_suffix(path):
    return re.sub(r"(?:\.(?:copy|avg))+$", "", str(path), flags=re.IGNORECASE)


def _split_ms_parent_name(ms_path):
    name = _strip_ms_copy_avg_suffix(Path(ms_path).name)
    match = re.fullmatch(
        r"(?P<parent>.+)_chunk_\d+(?:\.ms)?", name, flags=re.IGNORECASE
    )
    return _strip_ms_copy_avg_suffix(match.group("parent")) if match else None


def _match_ms_plots(ms_path, ms_plot_files, ms_json_files):
    name = Path(ms_path).name
    names = tuple(dict.fromkeys((name, _strip_ms_copy_avg_suffix(name))))
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


def _quality_plot_placeholder(title):
    return (
        '<div class="ms-quality-placeholder" role="status">'
        '<strong>{}</strong><span>Not generated for this measurement set.</span></div>'
    ).format(_escape(title))


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


def _compare_controls(paths, base_dir):
    if len(paths) < 2:
        return ""

    options_a = "".join(
        '<option value="{}">{}</option>'.format(
            _escape(_relative_url(p, base_dir)), _escape(p.stem)
        )
        for p in paths
    )
    options_b = "".join(
        '<option value="{}"{}>{}</option>'.format(
            _escape(_relative_url(p, base_dir)),
            ' selected="selected"' if idx == len(paths) - 1 else "",
            _escape(p.stem),
        )
        for idx, p in enumerate(paths)
    )
    first_href = _relative_url(paths[0], base_dir)
    last_href = _relative_url(paths[-1], base_dir)
    return (
        '<div class="compare-tool" data-compare-tool>'
        '<div class="compare-controls">'
        '<label><span>Reference image</span><select data-compare-select-a>{}</select></label>'
        '<label><span>Comparison image</span><select data-compare-select-b>{}</select></label>'
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
    bandpass_enabled=False,
):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        _page_shell(
            title,
            active,
            body,
            nested=nested,
            subtitle=subtitle,
            bandpass_enabled=bandpass_enabled,
        ),
        encoding="utf-8",
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


def _image_metric_cycle_cell(records, field):
    values = [
        '<span title="{}">{}</span>'.format(
            _escape(record["image"]),
            _escape(_format_image_metric_value(field, record[field])),
        )
        for record in records
        if record.get(field) is not None
    ]
    return '<td class="numeric">{}</td>'.format(
        "<br>".join(values) if values else "&mdash;"
    )


def _image_metrics_chart(records):
    latest_by_cycle = {}
    for record in records:
        cycle = record.get("cycle")
        if cycle is not None:
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


def _image_metrics_content(records):
    if not records:
        return '<p class="empty">No image extrema or RMS statistics were found in selfcal.log or the primary cycle FITS images.</p>'

    rows = []
    for record in records:
        cycle = record.get("cycle")
        cycle_text = "Cycle {}".format(cycle) if cycle is not None else "Unknown"
        image_name = record.get("image", "")
        image_label = Path(image_name).name or image_name
        metric_cells = "".join(
            '<td class="numeric">{}</td>'.format(
                _escape(_format_image_metric_value(field, record.get(field, "-")))
            )
            for field, _ in _IMAGE_METRIC_FIELDS
        )
        rows.append(
            '<tr><th scope="row">{}</th><td><code title="{}">{}</code></td>{}</tr>'.format(
                _escape(cycle_text),
                _escape(image_name),
                _escape(image_label),
                metric_cells,
            )
        )

    headers = "".join(
        '<th scope="col">{}</th>'.format(_escape(label))
        for _, label in _IMAGE_METRIC_FIELDS
    )
    return (
        _image_metrics_chart(records)
        + '<p class="image-metrics-note">The chart uses the last image for each cycle; the table lists every image with metrics. Values come from selfcal.log when present, otherwise from the primary cycle FITS image.</p>'
        + '<div class="table-scroll"><table class="data-table"><thead><tr>'
        '<th scope="col">Cycle</th><th scope="col">Image</th>{}</tr></thead>'
        '<tbody>{}</tbody></table></div>'.format(headers, "".join(rows))
    )


def _overview_page(site_dir, run_root, config, artifacts, logs, status, error):
    title = str(config.get("imagename") or run_root.name)
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
                _image_metric_cycle_cell(cycle_metric_records, field)
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

    summary_html = _config_table(config, _SUMMARY_KEYS)
    body_parts.append(_section("Run configuration", summary_html, "Selected values from full_config.txt."))

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
        bandpass_enabled=_bandpass_enabled(config),
    )


def _imaging_page(site_dir, run_root, config, artifacts, logs):
    title = str(config.get("imagename") or run_root.name)
    plots = artifacts.get("all_overview_plots", artifacts["overview_plots"])
    fits_files = artifacts.get("all_fits_files", artifacts["fits_files"])
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
            compare_controls = _compare_controls(progression, site_dir)
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
        cm = re.search(r"_(0\d{2})(?:-|\.|$)", path.name)
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
            entries = []
            for path in paths:
                cm = re.search(r"_(0\d{2})(?:-|\.|$)", path.name)
                cy_tag = "cycle-{}".format(cm.group(1)) if cm else ""
                search_val = "{} {}".format(path.name, cy_tag).strip()
                entries.append(
                    '<li class="fits-entry" data-search="{}">{}<span class="file-size">{}</span></li>'.format(
                        _escape(search_val),
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
        '<p class="page-intro">PNG previews open locally at full size. FITS files are linked as products; use an astronomy FITS viewer to inspect their pixel data.</p>\n'
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
        bandpass_enabled=_bandpass_enabled(config),
    )


def _cycle_sort_key(value):
    if value.isdigit():
        return (0, int(value))
    return (1, value)


def _plot_page(
    site_dir,
    run_root,
    title,
    dataset_name,
    dataset_slug,
    cycle,
    paths,
    bandpass_enabled=False,
):
    calibration_dir = site_dir / "calibration"
    page_path = calibration_dir / "{}.html".format(dataset_slug)
    cycle_title = "Cycle {}".format(cycle) if cycle != "other" else "Other plots"
    figures = []
    has_phase = False
    has_amp = False
    has_pol = False

    for path in paths:
        name = path.name
        tags = []
        lower = name.lower()
        if "amp" in lower:
            tags.append("amplitude")
            has_amp = True
        if "phase" in lower:
            tags.append("phase")
            has_phase = True
        if "poldiff" in lower or "pol" in lower:
            tags.append("polarization")
            has_pol = True
        direction = re.search(r"(?:dir|dil)(\d+)", name, re.IGNORECASE)
        if direction:
            tags.append("direction {}".format(direction.group(1)))
        polarization = re.search(r"pol([a-z0-9]+)", name, re.IGNORECASE)
        if polarization:
            tags.append("pol-{}".format(polarization.group(1)))
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

    # Filter pills for plot types
    pills = ['<div class="pill-group" data-pill-group data-pill-target=".plot-item">']
    pills.append('<span class="pill-label">Filter plot types:</span>')
    pills.append('<button type="button" class="pill-btn active" data-filter-value="all">All plots ({})</button>'.format(len(paths)))
    if has_phase:
        pills.append('<button type="button" class="pill-btn" data-filter-value="phase">Phase</button>')
    if has_amp:
        pills.append('<button type="button" class="pill-btn" data-filter-value="amplitude">Amplitude</button>')
    if has_pol:
        pills.append('<button type="button" class="pill-btn" data-filter-value="polarization">Polarization</button>')
    pills.append('</div>')

    body = (
        '<p class="page-intro"><a href="../calibration.html">&larr; Back to Calibration index</a> / {} / {}. '
        'This page contains {} plots and loads thumbnails lazily.</p>'.format(
            _escape(dataset_name), _escape(cycle_title), len(paths)
        )
        + "".join(pills)
        + _filter_input(".plot-item", "Search plot filenames")
        + '<div class="gallery">{}</div>'.format("".join(figures))
    )
    _write_page(
        page_path,
        title,
        "calibration.html",
        body,
        nested=True,
        subtitle="Calibration plots / {} / {}".format(dataset_name, cycle_title),
        bandpass_enabled=bandpass_enabled,
    )
    return page_path


def _calibration_page(site_dir, run_root, config, artifacts):
    title = str(config.get("imagename") or run_root.name)
    bandpass_enabled = _bandpass_enabled(config)
    calibration_sets = artifacts["calibration_sets"]
    all_cycles = sorted(
        {cycle for _, cycles in calibration_sets for cycle in cycles},
        key=_cycle_sort_key,
    )
    if not calibration_sets:
        body = (
            '<p class="empty">No calibration plots are (yet) available. </p>'
        )
        _write_page(
            site_dir / "calibration.html",
            title,
            "calibration.html",
            body,
            bandpass_enabled=bandpass_enabled,
        )
        return

    rows = []
    cycle_headers = "".join("<th scope=\"col\">Cycle {}</th>".format(_escape(cycle)) for cycle in all_cycles)
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
                bandpass_enabled=bandpass_enabled,
            )
            cells.append(
                '<td><a href="calibration/{}">{} plots</a></td>'.format(
                    _escape(page_slug), len(paths)
                )
            )
        rows.append(
            '<tr class="dataset-row" data-search="{}"><th scope="row"><code>{}</code></th>{}</tr>'.format(
                _escape(dataset_name), _escape(dataset_name), "".join(cells)
            )
        )

    body = (
        '<p class="page-intro">Calibration plots are split into one local page per measurement set and solution cycle. Open a cycle page to view and filter phase, amplitude, and polarization solutions.</p>\n'
        + _filter_input(".dataset-row", "Filter measurement sets") + "\n"
        + '<div class="table-scroll"><table class="data-table"><thead><tr><th scope="col">Measurement set</th>{}</tr></thead><tbody>{}</tbody></table></div>'.format(
            cycle_headers, "".join(rows)
        )
    )
    _write_page(
        site_dir / "calibration.html",
        title,
        "calibration.html",
        body,
        subtitle="Calibration solutions / {}".format(run_root.name),
        bandpass_enabled=bandpass_enabled,
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
    frequency_order = finite_frequency_indices[
        np.argsort(frequencies[finite_frequency_indices])
    ]
    frequencies_mhz = (frequencies[frequency_order] / 1e6).tolist()
    values = values[:, frequency_order, :, :, :]
    if weights is not None:
        weights = weights[:, frequency_order, :, :, :]

    series = {}
    valid_count = 0
    total_count = 0
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
                np.where(valid, np.cos(direction_values), 0.0), axis=0
            )
            sine_sum = np.sum(
                np.where(valid, np.sin(direction_values), 0.0), axis=0
            )
            valid_times = np.sum(valid, axis=0)
            coherence = np.hypot(cosine_sum, sine_sum) / np.maximum(valid_times, 1)
            reduced = np.arctan2(sine_sum, cosine_sum)
            reduced[(valid_times == 0) | (coherence < 1e-8)] = np.nan

        reduced = np.asarray(reduced, dtype=float)
        valid_reduced = np.isfinite(reduced)
        valid_count += int(np.count_nonzero(valid_reduced))
        total_count += int(reduced.size)
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
            return solutions
    except _BandpassDataError:
        raise
    except (
        tables.HDF5ExtError,
        tables.NoSuchNodeError,
        OSError,
        ValueError,
        TypeError,
    ) as exc:
        raise _BandpassDataError(
            "Could not read {}: {}".format(path.name, exc)
        ) from exc


def _bandpass_label(path, ms_inputs):
    filename = path.name.casefold()
    for measurement_set in ms_inputs:
        basename = Path(measurement_set).name
        if basename and filename.endswith((basename + ".h5").casefold()):
            return basename
    label = path.stem
    if label.casefold().startswith("bandpass_"):
        label = label[len("bandpass_"):]
    label = re.sub(r"^(?:sky)?selfcalcycle\d+_", "", label, flags=re.IGNORECASE)
    return label or path.name


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
    site_dir, title, dataset_name, page_name, solutions, summary
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
        ("Flagged samples", flagged_text),
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
        "and polarization; flagged or invalid target/reference samples are left as gaps."
    )
    if not summary["weights_available"]:
        note += " At least one solution table has no weight/flag axis, so only non-finite samples can be omitted."
    data_json = json.dumps(solutions, separators=(",", ":"), ensure_ascii=True)
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
        bandpass_enabled=True,
    )
    return page_path


def _bandpass_page(site_dir, run_root, config, artifacts):
    title = str(config.get("imagename") or run_root.name)
    rows = []
    ms_inputs = _as_list(config.get("ms"))
    for index, path in enumerate(artifacts.get("bandpass_files", []), start=1):
        dataset_name = _bandpass_label(path, ms_inputs)
        try:
            solutions = _read_bandpass_h5(path)
        except _BandpassDataError as exc:
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
            '<th scope="col">Frequency coverage</th><th scope="col">Flagged</th>'
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
        bandpass_enabled=True,
    )


def _datasets_page(site_dir, run_root, config, artifacts, logs=None):
    title = str(config.get("imagename") or run_root.name)
    inputs = _as_list(config.get("ms"))
    report_logs = logs or {}
    flagging_stats = report_logs.get("flagging_stats", {})
    flagging_stats_by_basename = defaultdict(list)
    for logged_path, record in flagging_stats.items():
        logged_canonical_path = _canonical_ms_path(logged_path, run_root)
        flagging_stats_by_basename[
            Path(logged_canonical_path).name.casefold()
        ].append((logged_canonical_path, record))

    metadata_by_ms = _measurement_set_metadata(run_root)
    cycle_steps = [
        step
        for cycle in report_logs.get("cycle_timeline") or ()
        for step in cycle.get("step_details") or ()
    ]
    setup_steps = list(report_logs.get("setup_step_details") or ())
    candidate_ms_records = [
        (
            ms_path,
            _resolved_ms_path(metadata.get("_source_path", ms_path), run_root),
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

    split_ms_by_input_name = defaultdict(dict)
    for canonical_path, display_path in selfcal_ms_records:
        parent_name = _split_ms_parent_name(canonical_path)
        if parent_name:
            split_ms_by_input_name[parent_name.casefold()].setdefault(
                Path(canonical_path).name.casefold(),
                (canonical_path, display_path),
            )

    dataset_entries = []
    for input_path in inputs:
        input_path = str(input_path)
        canonical_input_path = _canonical_ms_path(input_path, run_root)
        input_name = _strip_ms_copy_avg_suffix(
            Path(canonical_input_path).name
        )
        split_paths = sorted(
            split_ms_by_input_name.get(input_name.casefold(), {}).values(),
            key=lambda paths: Path(paths[1]).name.casefold(),
        )
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

    has_vla_dataset = any(
        metadata_by_ms.get(entry["canonical_path"], {})
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
        metadata = dict(metadata_by_ms.get(canonical, {}))
        flagging_record = flagging_stats.get(canonical)
        if flagging_record is None:
            basename_matches = flagging_stats_by_basename.get(
                Path(canonical).name.casefold(), []
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

        matched_plots = _match_ms_plots(
            path,
            all_ms_plot_files if entry["is_input"] else ms_plot_files,
            ms_json_files,
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
            plot_figures.append(_quality_plot_placeholder("Time-coverage plot"))
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
            plot_figures.append(_quality_plot_placeholder("A-team plot"))

        ateam_note = ""
        if matched_plots["ateam_json"]:
            try:
                ateam_data = json.loads(matched_plots["ateam_json"].read_text(encoding="utf-8", errors="replace"))
                if not ateam_data:
                    ateam_note = '<p class="caption-detail" style="margin-top:6px;">A-team check: No interfering A-team sources within separation threshold.</p>'
                else:
                    sources_str = ", ".join(str(s) for s in ateam_data)
                    ateam_note = '<p class="caption-detail" style="margin-top:6px; color:var(--amber);">A-team check: Potentially interfering sources: {}</p>'.format(_escape(sources_str))
            except Exception:
                pass

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
        '<p class="page-intro">Configured input paths come from full_config.txt. When an input is split for self-calibration, both the input and its self-calibration measurement sets are listed. Observation metadata is extracted from logs/selfcal.log; available input time-coverage plots are shown even if generated before the current run.</p>\n'
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
        bandpass_enabled=_bandpass_enabled(config),
    )


def _run_details_page(site_dir, run_root, config, artifacts, logs, command_text, status, error):
    title = str(config.get("imagename") or run_root.name)
    file_links = []
    for relative in ("full_config.txt", "facetselfcal.txt", "logs/selfcal.log", "logs/resource_usage.csv", "logs/resource_phases.csv", "h5plot.log"):
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
        command_section = '<details open><summary>Recorded command line</summary><pre>{}</pre></details>'.format(
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
    resource_content += live_chart_frame
    if res and res.get("samples"):
        resource_content += _render_resource_section(
            res, include_chart=not bool(live_chart_frame)
        )
    if resource_content:
        resource_note = (
            "Live usage and chart refresh about every 15 seconds while monitoring is active. Phase bars use explicit workflow boundaries; process-tree CPU is relative to one core and can exceed 100%."
            if live_chart_frame
            else "Current usage refreshes separately; phase bars use explicit workflow boundaries. Process-tree CPU is relative to one core and can exceed 100%."
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
        + _section("Recorded command", command_section or '<p class="empty">No facetselfcal.txt command record was found.</p>') + "\n"
        + _section("Saved configuration", config_filter + config_table) + "\n"
        + _section("Artifact inventory", inventory) + "\n"
        + _section("Warning log records", _render_log_events(logs["warnings"], logs["warning_count"])) + "\n"
        + _section("Error log records", _render_log_events(logs["errors"], logs["error_count"], is_error=True)) + "\n"
        + _section("Recent log excerpts", log_tails, "At most the last 80 lines of each log are included here; the full logs are linked above.")
    )
    _write_page(
        site_dir / "run-details.html",
        title,
        "run-details.html",
        body,
        subtitle="Run details & logs / {}".format(run_root.name),
        bandpass_enabled=_bandpass_enabled(config),
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
    asset_dir.mkdir(parents=True, exist_ok=True)
    (asset_dir / "report.css").write_text(_report_css(), encoding="utf-8")
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
    try:
        start_cycle = int(config.get("start", 0))
    except (TypeError, ValueError):
        start_cycle = 0
    artifacts = _scan_artifacts(
        run_root, run_started_at, current_run_cycles, start_cycle=start_cycle
    )
    logs = _scan_logs(run_root)
    logs["resources"] = _scan_resource_log(run_root, run_started_at, start_cycle)
    current_image_names = {path.name.casefold() for path in artifacts["fits_files"]}
    logs["cycle_timeline"] = _filter_restart_cycle_timeline(
        logs["cycle_timeline"], run_started_at, start_cycle
    )
    logs["image_metrics"] = _filter_restart_image_metrics(
        logs["image_metrics"], run_started_at, start_cycle, current_image_names
    )
    logs["image_metrics"] = _image_metrics_from_fits(
        artifacts.get("all_fits_files", artifacts["fits_files"]),
        logs.get("image_metrics", []),
    )
    logs["image_metrics"] = _add_image_dynamic_range(logs["image_metrics"])

    _overview_page(site_dir, run_root, config, artifacts, logs, status, error)
    _imaging_page(site_dir, run_root, config, artifacts, logs)
    _calibration_page(site_dir, run_root, config, artifacts)
    if _bandpass_enabled(config):
        _bandpass_page(site_dir, run_root, config, artifacts)
    _datasets_page(site_dir, run_root, config, artifacts, logs)
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
