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

from .resource_chart import generate_resource_svg


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
.table-scroll { max-width: 100%; overflow-x: auto; margin-bottom: 14px; }
.data-table th, .data-table td { padding: 9px 12px; border-bottom: 1px solid var(--line); text-align: left; vertical-align: top; }
.data-table thead th { background: var(--surface-alt); color: var(--ink-secondary); font-size: 12px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.03em; }
.data-table tbody th { width: 220px; color: var(--muted); font-size: 12px; font-weight: 600; }
.data-table tbody th code { overflow-wrap: anywhere; }
.cycle-label { font-size: 14px; font-weight: 700; font-style: italic; }
.cycle-duration { font-size: 14px; font-weight: 400; }
.data-table tr:last-child th, .data-table tr:last-child td { border-bottom: none; }
.data-table tbody tr:nth-child(even) td, .data-table tbody tr:nth-child(even) th { background: #fafcff; }
.data-table td { overflow-wrap: anywhere; }
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
.step-badge { display: inline-block; padding: 2px 7px; border-radius: 4px; font-size: 11px; font-weight: 600; background: var(--teal-light); color: var(--teal-deep); border: 1px solid var(--teal-border); white-space: nowrap; }
.workflow-step-grid { display: grid; gap: 5px; min-width: 400px; }
.workflow-ms-row { display: grid; grid-template-columns: minmax(100px, 220px) minmax(0, 1fr); align-items: start; gap: 8px; padding-top: 4px; border-top: 1px solid var(--line-light); }
.workflow-ms-row:first-child { border-top: 0; padding-top: 0; }
.workflow-ms-name { min-width: 0; overflow: hidden; color: var(--ink-secondary); font: 11px/1.5 ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace; text-overflow: ellipsis; white-space: nowrap; }
.workflow-ms-badges, .workflow-step-chain { display: flex; flex-wrap: wrap; align-items: center; gap: 4px 5px; min-width: 0; }
.workflow-step-item { display: inline-flex; align-items: center; gap: 5px; min-width: 0; }
.workflow-step-badge { display: inline-flex; align-items: baseline; gap: 4px; max-width: 100%; padding: 2px 6px; border: 1px solid var(--line); border-radius: 4px; background: var(--surface-alt); color: var(--ink-secondary); font-size: 11px; font-weight: 600; line-height: 1.35; white-space: nowrap; }
.workflow-step-badge-imaging { border-color: var(--blue-border); background: var(--blue-pale); color: #075985; }
.workflow-step-badge-solve { border-color: var(--amber-border); background: var(--amber-pale); color: #92400e; }
.workflow-step-badge-apply { border-color: var(--teal-border); background: var(--teal-light); color: var(--teal-deep); }
.workflow-command-trigger { appearance: none; font-family: inherit; text-align: left; cursor: pointer; }
.workflow-command-trigger:focus-visible { outline: 2px solid var(--blue); outline-offset: 2px; }
.workflow-command-text { position: fixed; top: 50%; left: 50%; transform: translate(-50%, -50%); box-sizing: border-box; width: min(760px, calc(100vw - 32px)); max-width: calc(100vw - 32px); max-height: min(70vh, 640px); margin: 0; padding: 12px 14px; overflow: auto; border: 1px solid var(--line); border-radius: 6px; background: var(--surface); color: var(--ink); box-shadow: 0 6px 18px rgba(15,23,42,.16); font: 12px/1.5 ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", monospace; white-space: pre-wrap; overflow-wrap: anywhere; }
.workflow-step-duration { color: var(--ink-secondary); font-size: 10px; font-weight: 500; }
.workflow-step-arrow { color: var(--muted); font-size: 12px; }
.workflow-shared-steps { display: flex; align-items: start; gap: 8px; padding-bottom: 4px; }
.workflow-shared-label { flex: 0 0 100px; color: var(--muted); font-size: 10px; font-weight: 700; text-transform: uppercase; }
@media (max-width: 600px) { .workflow-ms-row { grid-template-columns: minmax(85px, 140px) minmax(0, 1fr); gap: 5px; } .workflow-shared-label { flex-basis: 85px; } }
.resource-summary-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); gap: 12px; margin-bottom: 16px; }
.resource-card { background: var(--surface); border: 1px solid var(--line); border-radius: 6px; padding: 12px 14px; box-shadow: 0 1px 3px rgba(15,23,42,0.03); }
.resource-card strong { display: block; font-size: 11px; text-transform: uppercase; color: var(--muted); letter-spacing: .05em; margin-bottom: 4px; }
.resource-card span { font-size: 18px; font-weight: 700; color: var(--ink); }
.resource-live-frame { display: block; width: 100%; height: 142px; margin: 0 0 14px; border: 1px solid var(--line); border-radius: 6px; background: var(--paper); }
.resource-chart-live-frame { display: block; width: 100%; height: 560px; margin: 0 0 14px; border: 1px solid var(--line); border-radius: 6px; background: var(--paper); }
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
        "all_overview_plots": all_overview_plots,
        "ms_plot_files": ms_plot_files,
        "ms_json_files": ms_json_files,
        "calibration_sets": calibration_sets,
        "fits_files": fits_files,
        "all_fits_files": all_fits_files,
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


def _workflow_steps_html(step_details, id_prefix="segment"):
    if not step_details:
        return "-"

    shared_steps = []
    ms_groups = {}
    command_index = 0
    for step in step_details:
        if step["kind"] == "imaging":
            shared_steps.append(step)
            continue
        group = ms_groups.setdefault(
            step.get("ms_key"),
            {"name": step.get("ms_name") or "MS not identified", "path": step.get("ms_path"), "steps": []},
        )
        group["steps"].append(step)

    def render_badges(steps):
        nonlocal command_index
        badges = []
        for index, step in enumerate(steps):
            label = {"imaging": "Imaging", "solve": "Solve", "apply": "Apply"}.get(
                step["kind"], step["name"]
            )
            badge_class = {
                "imaging": "workflow-step-badge-imaging",
                "solve": "workflow-step-badge-solve",
                "apply": "workflow-step-badge-apply",
            }.get(step["kind"], "workflow-step-badge-other")
            arrow = '<span class="workflow-step-arrow" aria-hidden="true">&rarr;</span>' if index + 1 < len(steps) else ""
            command_id = "workflow-command-{}-{}".format(
                id_prefix, command_index
            )
            command_index += 1
            badges.append(
                '<div class="workflow-step-item">'
                '<button type="button" class="workflow-step-badge workflow-command-trigger {}" '
                'popovertarget="{}" title="Click to view full command">{} <span class="workflow-step-duration">{}</span></button>'
                '<pre class="workflow-command-text" id="{}" popover="auto">{}</pre>'
                '{}'
                '</div>'.format(
                    badge_class,
                    command_id,
                    _escape(label),
                    _escape(step.get("duration") or "-"),
                    command_id,
                    _escape(step.get("command") or step["name"]),
                    arrow,
                )
            )
        return "".join(badges)

    rows = ['<div class="workflow-step-grid">']
    if shared_steps:
        rows.append(
            '<div class="workflow-shared-steps"><span class="workflow-shared-label">Shared</span>'
            '<div class="workflow-step-chain">{}</div></div>'.format(render_badges(shared_steps))
        )
    for group in ms_groups.values():
        rows.append(
            '<div class="workflow-ms-row"><span class="workflow-ms-name" title="{}">{}</span>'
            '<div class="workflow-ms-badges">{}</div></div>'.format(
                _escape(group["path"] or group["name"]),
                _escape(group["name"]),
                render_badges(group["steps"]),
            )
        )
    rows.append("</div>")
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
    actionable_warnings = []
    cycles = {}
    current_cycle = None
    first_ts = None
    last_ts = None
    cycle_start_pattern = re.compile(
        r"Starting self-calibration cycle\s+(\d+)", re.IGNORECASE
    )
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

                        if "bandwidth smearing" in message.lower() or "try to increase your frequency resolution" in message.lower():
                            if message not in actionable_warnings:
                                actionable_warnings.append(message)

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
                            if message.startswith("wsclean ") and not any(
                                existing["kind"] == "imaging" for existing in cdata["steps"]
                            ):
                                step = {
                                    "kind": "imaging",
                                    "name": "Imaging (wsclean)",
                                    "timestamp": ts_obj,
                                    "ms_path": None,
                                    "ms_key": None,
                                    "ms_name": None,
                                    "command": message,
                                }
                            elif "DP3 solve:" in message:
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
                            elif (
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

        step_details = []
        for s_idx, step in enumerate(cd["steps"]):
            step_ts = step["timestamp"]
            next_ts = (
                cd["steps"][s_idx + 1]["timestamp"]
                if s_idx + 1 < len(cd["steps"])
                else cd["end_time"]
            )
            s_dur = (next_ts - step_ts).total_seconds() if next_ts and next_ts >= step_ts else None
            step_details.append({**step, "duration": _format_duration(s_dur)})
        cd["step_details"] = step_details

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
        "actionable_warnings": actionable_warnings,
        "flagging_stats": flagging_stats,
        "cycle_timeline": [cycles[k] for k in cycle_keys],
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
        "sample_count": len(valid_samples),
        "cycle_stats": cycle_stats,
        "peak_tree_rss_gib": peak_tree_rss_gib,
        "peak_tree_cpu_pct": peak_tree_cpu_pct,
        "avg_tree_cpu_pct": avg_tree_cpu_pct,
        "peak_sys_ram_used_gib": peak_sys_ram_used_gib,
        "sys_ram_total_gib": sys_ram_total_gib,
        "peak_sys_cpu_pct": peak_sys_cpu_pct,
    }


def _generate_resource_svg(samples):
    return generate_resource_svg(samples)


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
        svg_chart = _generate_resource_svg(resources["samples"])
        legend_html = (
            '<div class="resource-chart-legend">'
            '<span class="legend-item"><span class="legend-swatch" style="background:#0d9488;"></span> Process Tree CPU (%)</span>'
            '<span class="legend-item"><span class="legend-swatch" style="background:#d97706;"></span> Process Tree RAM (GiB)</span>'
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
                f'<td>{cs["peak_tree_rss_gib"]:.2f} GiB</td>'
                f'<td>{cs["peak_tree_cpu_pct"]:.0f}%</td>'
                f'<td>{cs["avg_tree_cpu_pct"]:.0f}%</td>'
                f'<td>{cs["count"]}</td></tr>'
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

    return (
        f'<div class="resource-summary-grid">{"".join(cards)}</div>\n'
        f'{chart_box}\n'
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


def _match_ms_plots(ms_path, ms_plot_files, ms_json_files):
    name = Path(ms_path).name
    clean_name = re.sub(r"(?:\.(?:copy|avg))+$", "", name, flags=re.IGNORECASE)
    matched = {"time_coverage": None, "ateam_png": None, "ateam_json": None}
    for p in ms_plot_files:
        p_name = p.name
        if f"{name}.time_coverage" in p_name or f"{clean_name}.time_coverage" in p_name:
            matched["time_coverage"] = p
        elif f"ateam_{name.lower()}" in p_name.lower() or f"ateam_{clean_name.lower()}" in p_name.lower():
            matched["ateam_png"] = p
    for j in ms_json_files:
        j_name = j.name
        if f"ateam_{name.lower()}" in j_name.lower() or f"ateam_{clean_name.lower()}" in j_name.lower():
            matched["ateam_json"] = j
    return matched


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


def _write_page(path, title, active, body, nested=False, subtitle="Offline processing report"):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        _page_shell(title, active, body, nested=nested, subtitle=subtitle),
        encoding="utf-8",
    )


def _image_metric_cycle_cell(records, field):
    values = [
        '<span title="{}">{}</span>'.format(
            _escape(record["image"]), _escape(record[field])
        )
        for record in records
        if record.get(field) is not None
    ]
    return "<td>{}</td>".format("<br>".join(values) if values else "&mdash;")


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

        parts.append(
            '<text x="8" y="{}" class="metric-chart-title">{}</text>'.format(
                row_top + 16, _escape(label)
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
            parts.append(
                '<line x1="{:.1f}" y1="{:.1f}" x2="{:.1f}" y2="{:.1f}" class="metric-chart-grid"/>'.format(
                    plot_left, y, plot_right, y
                )
            )
            parts.append(
                '<text x="{}" y="{:.1f}" text-anchor="end" dominant-baseline="middle" '
                'class="metric-chart-axis">{}</text>'.format(
                    plot_left - 8, y, _escape("{:.3g}".format(grid_value))
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
            "<td>{}</td>".format(_escape(record.get(field, "-")))
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
    live_resource_frame = _resource_live_frame(site_dir)
    if live_resource_frame:
        body_parts.append(
            _section("Live Resource Usage", live_resource_frame)
        )

    # Actionable alerts (e.g. bandwidth smearing)
    actionable = logs.get("actionable_warnings", [])
    if actionable:
        items = "".join("<p>{}</p>".format(_escape(w)) for w in actionable)
        body_parts.append(
            '<div class="notice"><strong>Observational / Data Quality Alert</strong>{}</div>'.format(items)
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

    # Cycle Timeline Table
    cycle_timeline = logs.get("cycle_timeline", [])
    image_metrics_by_cycle = defaultdict(list)
    for image_metric in logs.get("image_metrics", []):
        cycle = image_metric.get("cycle")
        if cycle is not None:
            image_metrics_by_cycle[cycle].append(image_metric)
    if cycle_timeline:
        rows = []
        for cdata in cycle_timeline:
            c_int = int(cdata["cycle"])
            c_cfg = _get_cycle_config(config, c_int)
            steps_html = _workflow_steps_html(
                cdata.get("step_details", []), "cycle-{}".format(cdata["cycle"])
            )
            cycle_metric_records = image_metrics_by_cycle.get(cdata["cycle"], [])
            metric_cells = "".join(
                _image_metric_cycle_cell(cycle_metric_records, field)
                for field, _ in _CYCLE_IMAGE_METRIC_FIELDS
            )
            rows.append(
                '<tr>'
                '<th scope="row"><span class="cycle-label">Cycle {}</span></th>'
                '<td>{}</td>'
                '<td><span class="cycle-duration">{}</span></td>'
                '{}'
                '<td>{}</td>'
                '<td>{}</td>'
                '<td>{}</td>'
                '<td>{}</td>'
                '</tr>'.format(
                    _escape(cdata["cycle"]),
                    _escape(cdata["start_str"]),
                    _escape(cdata["duration_str"]),
                    metric_cells,
                    _cycle_config_values_html(c_cfg["soltype"]),
                    _cycle_config_values_html(c_cfg["solint"], _format_overview_interval),
                    _cycle_config_values_html(c_cfg["smoothness"], _format_overview_smoothness),
                    steps_html,
                )
            )
        timeline_html = (
            '<div class="table-scroll"><table class="data-table"><thead><tr>'
            '<th scope="col">Cycle</th>'
            '<th scope="col">Start Time</th>'
            '<th scope="col">Duration</th>'
            '<th scope="col">RMS noise</th>'
            '<th scope="col">Dynamic range</th>'
            '<th scope="col">Solution Type</th>'
            '<th scope="col">Interval</th>'
            '<th scope="col">Smoothness</th>'
            '<th scope="col">Workflow Steps</th>'
            '</tr></thead><tbody>{}</tbody></table></div>'.format("".join(rows))
        )
        body_parts.append(_section("Self-Calibration Progression", timeline_html, "Timing, parameters, and logged image statistics per calibration cycle."))

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
    body_parts.append(_section("Recent warnings", issues_content))
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
    _write_page(site_dir / "imaging.html", title, "imaging.html", body, subtitle="Imaging & FITS products / {}".format(run_root.name))


def _cycle_sort_key(value):
    if value.isdigit():
        return (0, int(value))
    return (1, value)


def _plot_page(site_dir, run_root, title, dataset_name, dataset_slug, cycle, paths):
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
    _write_page(site_dir / "calibration.html", title, "calibration.html", body, subtitle="Calibration solutions / {}".format(run_root.name))


def _datasets_page(site_dir, run_root, config, artifacts, logs=None):
    title = str(config.get("imagename") or run_root.name)
    inputs = _as_list(config.get("ms"))
    flagging_stats = (logs or {}).get("flagging_stats", {})
    flagging_stats_by_basename = defaultdict(list)
    for logged_path, record in flagging_stats.items():
        logged_canonical_path = _canonical_ms_path(logged_path, run_root)
        flagging_stats_by_basename[
            Path(logged_canonical_path).name.casefold()
        ].append((logged_canonical_path, record))

    metadata_by_ms = _measurement_set_metadata(run_root)
    has_vla_dataset = any(
        metadata_by_ms.get(_canonical_ms_path(path, run_root), {})
        .get("Telescope", "").strip().upper() in {"VLA", "EVLA"}
        for path in inputs
    )
    metadata_fields = tuple(
        (field, labels)
        for field, labels in _MS_METADATA_FIELDS
        if field != "VLA configuration" or has_vla_dataset
    )
    column_count = len(metadata_fields) + 1
    input_rows = []
    ms_cards = []

    ms_plot_files = artifacts.get("ms_plot_files", [])
    ms_json_files = artifacts.get("ms_json_files", [])

    for path in inputs:
        canonical = _canonical_ms_path(path, run_root)
        ms_display_name = Path(path).name or path
        metadata = dict(metadata_by_ms.get(canonical, {}))
        flagging_record = flagging_stats.get(canonical)
        if flagging_record is None:
            basename_matches = flagging_stats_by_basename.get(
                Path(canonical).name.casefold(), []
            )
            if len(basename_matches) == 1:
                flagging_record = basename_matches[0][1]
        telescope = metadata.get("Telescope", "").strip().upper()
        search_text = " ".join([path] + list(metadata.values()))
        metadata_cells = "".join(
            "<td>{}</td>".format(
                _escape(
                    metadata.get(
                        field,
                        "Not applicable"
                        if field == "VLA configuration"
                        and has_vla_dataset
                        and telescope not in {"VLA", "EVLA"}
                        else "Not recorded",
                    )
                )
            )
            for field, _ in metadata_fields
        )
        input_rows.append(
            '<tr class="dataset-row" data-search="{}"><th scope="row"><code title="{}">{}</code></th>{}</tr>'.format(
                _escape(search_text),
                _escape(path),
                _escape(ms_display_name),
                metadata_cells,
            )
        )

        # Build quality card per measurement set
        matched_plots = _match_ms_plots(path, ms_plot_files, ms_json_files)
        meta_items = []
        for field, _ in metadata_fields:
            if field in metadata:
                meta_items.append(
                    '<div class="dataset-meta-item"><strong>{}</strong><span>{}</span></div>'.format(
                        _escape(field), _escape(metadata[field])
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

        ms_cards.append(
            '<div class="dataset-card">'
            '<div class="dataset-header"><h3 class="dataset-title" title="{}">{}</h3></div>'
            '<div class="dataset-meta-grid">{}</div>'
            '<h4>Data Quality & Observation Coverage Plots</h4>'
            '{}'
            '</div>'.format(
                _escape(path),
                _escape(ms_display_name),
                "".join(meta_items) if meta_items else '<p class="empty">No observational metadata parsed from log.</p>',
                plots_content,
            )
        )

    if not input_rows:
        input_rows.append(
            '<tr><td colspan="{}">No input MS list was found in full_config.txt.</td></tr>'.format(
                column_count
            )
        )

    body = (
        '<p class="page-intro">Input paths come from full_config.txt. Observation metadata is extracted from logs/selfcal.log. Time coverage and A-team elevation diagnostics are shown below each dataset.</p>\n'
        + _section(
            "Configured Measurement Sets Table",
            _filter_input(".dataset-row", "Filter configured inputs")
            + '<div class="table-scroll"><table class="data-table"><thead><tr><th scope="col">Measurement Set</th>{}</tr></thead><tbody>{}</tbody></table></div>'.format(
                "".join(
                    '<th scope="col">{}</th>'.format(_escape(field))
                    for field, _ in metadata_fields
                ),
                "".join(input_rows),
            ),
        ) + "\n"
        + _section("Measurement Set Data Quality & Coverage", "".join(ms_cards) if ms_cards else '<p class="empty">No measurement sets configured.</p>')
    )
    _write_page(site_dir / "datasets.html", title, "datasets.html", body, subtitle="Measurement sets & coverage / {}".format(run_root.name))


def _run_details_page(site_dir, run_root, config, artifacts, logs, command_text, status, error):
    title = str(config.get("imagename") or run_root.name)
    file_links = []
    for relative in ("full_config.txt", "facetselfcal.txt", "logs/selfcal.log", "logs/resource_usage.csv", "h5plot.log"):
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
            "Live usage and chart refresh about every 15 seconds while monitoring is active; summary statistics use logged samples."
            if live_chart_frame
            else "Current usage refreshes separately; the full chart summarizes logged samples."
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
    _write_page(site_dir / "run-details.html", title, "run-details.html", body, subtitle="Run details & logs / {}".format(run_root.name))


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
