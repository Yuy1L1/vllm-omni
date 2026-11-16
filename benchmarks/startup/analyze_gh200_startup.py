#!/usr/bin/env python3
"""Analyze Qwen3-TTS startup results produced by qwen3_tts_startup.py.

This script intentionally separates an EXCLUSIVE outer startup stack
(Python imports / Omni init / residual) from OVERLAPPING instrumented
inner-stage spans (timelines). Nested spans must not be summed.
Gantt-style timelines

Usage:
    python analyze_gh200_startup.py ~/vllm-omni-startup/results/gh200

Dependencies: matplotlib (python -m pip install matplotlib)
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

MODE_ORDER = {"default": 0, "no-cg": 1, "eager": 2}
CACHE_ORDER = {"cold": 0, "warm": 1}


def choose_summaries(root: Path):
    """Choose latest successfully completed r01-<mode> from all batches."""
    chosen = {}
    for path in sorted(root.glob("qwen3-tts-matrix-*/r01-*/summary.json")):
        try:
            payload = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        for run in payload.get("runs", []):
            mode = run.get("mode")
            state = run.get("cache_state")
            if mode not in MODE_ORDER or state not in CACHE_ORDER:
                continue
            if run.get("status") != "ok" or run.get("startup_s") is None:
                continue
            # lexicographic result batch directory is chronological by timestamp
            stamp = path.parent.parent.name
            key = (mode, state)
            if key not in chosen or stamp > chosen[key][0]:
                chosen[key] = (stamp, path, run, payload.get("spans", []))
    return chosen


def write_csv(path, rows, fields):
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def make_outer_rows(chosen):
    rows = []
    for (mode, state), (stamp, source, run, spans) in chosen.items():
        child = run.get("child_result") or {}
        total = float(run["startup_s"])
        imports = float(child.get("python_imports_s") or 0)
        omni = float(child.get("engine_init_s") or 0)
        residual = total - imports - omni
        if residual < -0.1:
            print(f"WARNING negative residual for {mode}/{state}: {residual:.3f}s; review raw data")
        # Zero only tiny floating point / independent-clock rounding differences.
        if -0.1 < residual < 0:
            residual = 0.0
        rt = child.get("request_times_s") or []
        cache = (run.get("repeat_indicators") or {}).get("inductor_fxgraph_cache") or {}
        rows.append({
            "mode": mode,
            "cache": state,
            "startup_total_s": total,
            "python_imports_s": imports,
            "omni_init_s": omni,
            "residual_s": residual,
            "first_request_s": rt[0] if rt else "",
            "steady_median_s": child.get("steady_median_s") or "",
            "fx_cache_hits_logged": cache.get("hits", 0),
            "fx_cache_misses_logged": cache.get("misses", 0),
            "fx_cache_bypasses_logged": cache.get("bypasses", 0),
            "batch": stamp,
            "source_json": str(source),
        })
    return sorted(rows, key=lambda r: (MODE_ORDER[r["mode"]], CACHE_ORDER[r["cache"]]))


def plot_outer_stack(rows, output):
    fig, ax = plt.subplots(figsize=(11, max(4, .74 * len(rows) + 1.5)))
    components = [
        ("python_imports_s", "Python imports", "#5b8db8"),
        ("omni_init_s", "Omni initialization (opaque)", "#e4a35c"),
        ("residual_s", "Residual / boundary overhead", "#a6adb5"),
    ]
    for idx, r in enumerate(rows):
        left = 0.0
        for key, label, color in components:
            duration = max(0, float(r[key]))
            ax.barh(idx, duration, left=left, height=.65, color=color, edgecolor="white", linewidth=.6)
            if duration >= 14:
                ax.text(left + duration / 2, idx, f"{duration:.1f}s", va="center", ha="center", fontsize=8)
            left += duration
        ax.text(float(r["startup_total_s"]) + 2, idx, f'{r["startup_total_s"]:.1f}s', va="center", fontsize=9)
    ax.set_yticks(range(len(rows)), [f'{r["mode"]} / {r["cache"]}' for r in rows])
    ax.invert_yaxis()
    ax.set_xlim(0, max(r["startup_total_s"] for r in rows) * 1.15)
    ax.set_xlabel("Startup wall time (s)")
    ax.set_title("GH200 startup: exclusive OUTER-level stack")
    ax.legend(handles=[Patch(facecolor=c, label=label) for _, label, c in components],
              loc="lower right", fontsize=8)
    ax.grid(axis="x", alpha=.18)
    ax.set_axisbelow(True)
    fig.text(.02, .015,
             "Omni initialization CONTAINS loading, compilation, graph capture, memory profiling, etc. "
             "It is not an attribution of these inner phases.", fontsize=8)
    fig.tight_layout(rect=(0, .045, 1, 1))
    fig.savefig(output, dpi=170)
    plt.close(fig)


def phase_category(phase):
    p = phase.lower()
    if "compile" in p: return "Compile"
    if "graph" in p: return "CUDA graph"
    if p == "load_to_engine_init": return "Stage coarse interval"
    if "load" in p or "weight" in p: return "Model load / weights"
    if "memory_profile" in p: return "Memory profile"
    if "worker_spawn" in p or "device_and" in p: return "Worker / distributed"
    if "engine_init" in p: return "Engine init (parent interval)"
    if "ready_handoff" in p: return "Ready handoff"
    if "python_import" in p: return "Python import"
    return "Other"


COLORS = {
    "Python import": "#4879b4",
    "Worker / distributed": "#9c83b8",
    "Model load / weights": "#4d9c7b",
    "Stage coarse interval": "#c7c7c7",
    "Compile": "#d68b43",
    "CUDA graph": "#cf6a75",
    "Memory profile": "#82b7ae",
    "Engine init (parent interval)": "#a0a0a0",
    "Ready handoff": "#9b88a2",
    "Other": "#b7b7b7",
}


def plot_timeline(mode, state, run, spans, output):
    total = float(run["startup_s"])  # inside-child measurement (not controller time)
    ready_stamps = [float(s.get("end_s", 0)) for s in spans
                    if s.get("mode") == mode
                    and s.get("cache_state") == state
                    and s.get("repetition") == run.get("repetition")
                    and s.get("phase") == "startup"]
    # Span positions use CONTROLLER timestamps from process launch.
    # The child-reported startup duration starts later, after interpreter boot.
    ready_wall = max(ready_stamps) if ready_stamps else total
    selected = []
    for s in spans:
        if s.get("mode") != mode or s.get("cache_state") != state:
            continue
        if s.get("repetition") != run.get("repetition"):
            continue
        phase = str(s.get("phase", "unknown"))
        if phase in {"startup", "request"}:
            continue
        lo = max(0, float(s.get("start_s", 0)))
        hi = min(ready_wall, float(s.get("end_s", 0)))
        if hi - lo <= 0.025:
            continue
        selected.append((s.get("stage"), s.get("replica"), phase, lo, hi, str(s.get("source", ""))))

    if not selected:
        print(f"NO spans in {mode}/{state}; skipping timeline")
        return
    selected.sort(key=lambda x: (x[0] is not None, x[0] if x[0] is not None else -1,
                                 x[1] if x[1] is not None else -1, x[3], x[2]))
    fig, ax = plt.subplots(figsize=(13, max(5, .43 * len(selected) + 2)))
    used = set()
    for idx, (stage, replica, phase, lo, hi, source) in enumerate(selected):
        cat = phase_category(phase)
        used.add(cat)
        ax.barh(idx, hi-lo, left=lo, height=.65, color=COLORS[cat],
                alpha=.55 if "parent" in cat.lower() or cat == "Stage coarse interval" else .92,
                edgecolor="white", linewidth=.5)
        if hi-lo >= 7:
            ax.text((lo+hi)/2, idx, f"{hi-lo:.1f}s", ha="center", va="center", fontsize=7)
    labels = [f'{"S"+str(stage) if stage is not None else "global"}'
              f'{"/r"+str(replica) if replica is not None else ""} · {phase}'
              for stage, replica, phase, lo, hi, src in selected]
    ax.set_yticks(range(len(labels)), labels, fontsize=8)
    ax.invert_yaxis()
    ax.axvline(ready_wall, color="black", ls="--", lw=1)
    ax.set_xlim(0, max(ready_wall * 1.03, 1))
    ax.set_xlabel("Seconds after child launch (controller receive timestamps)")
    ax.set_title(f"{mode} / {state} — overlapping spans, child startup={total:.2f}s, controller ready at {ready_wall:.2f}s")
    ax.grid(axis="x", alpha=.2)
    ax.set_axisbelow(True)
    ax.legend(handles=[Patch(facecolor=COLORS[k], label=k) for k in COLORS if k in used],
              fontsize=8, loc="lower right", ncol=2)
    fig.text(.025, .015,
             "Bars can overlap or nest; log-reported durations are anchored to controller receipt time. "
             "Do NOT sum bars. The controller clock also includes child-process boot time.", fontsize=8)
    fig.tight_layout(rect=(0, .04, 1, 1))
    fig.savefig(output, dpi=170)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results_root", type=Path, help="Directory containing qwen3-tts-matrix-* batches")
    parser.add_argument("--out", type=Path, default=None, help="Analysis output (default: <results_root>/analysis)")
    args = parser.parse_args()
    root = args.results_root.expanduser().resolve()
    out = (args.out or root / "analysis").expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    chosen = choose_summaries(root)
    if not chosen:
        raise SystemExit(f"No completed r01-*/summary.json found under {root}")
    rows = make_outer_rows(chosen)
    write_csv(out / "outer_startup_breakdown.csv", rows, list(rows[0]))
    plot_outer_stack(rows, out / "outer_startup_stacked.png")
    all_spans = []
    for (mode, state), (stamp, source, run, spans) in sorted(chosen.items()):
        for s in spans:
            if s.get("mode") == mode and s.get("cache_state") == state and s.get("repetition") == run.get("repetition"):
                all_spans.append({**s, "source_json": str(source)})
        plot_timeline(mode, state, run, spans, out / f"timeline_{mode}_{state}.png")
    if all_spans:
        fields = ["mode", "repetition", "cache_state", "stage", "replica", "phase", "start_s", "end_s", "duration_s", "source", "source_json"]
        write_csv(out / "spans_flat.csv", all_spans, fields)
    print(f"\nResults: {out}")
    print("Plots: outer_startup_stacked.png and timeline_<mode>_<cache>.png")
    print("CSVs:  outer_startup_breakdown.csv and spans_flat.csv")
    print("\nSelected runs:")
    for r in rows:
        print(f'  {r["mode"]:<7} {r["cache"]:<4} startup={r["startup_total_s"]:8.3f}s '
              f'FX-hit/miss={r["fx_cache_hits_logged"]}/{r["fx_cache_misses_logged"]}  {r["batch"]}')
    print("\nReminder: compile/model-load/CUDA-graph spans overlap, and reported durations are log-anchored. "
          "They must NOT be directly summed or interpreted as exclusive startup costs.")


if __name__ == "__main__":
    main()
