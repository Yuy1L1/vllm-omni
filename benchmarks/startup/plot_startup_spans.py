#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Plot process-local startup spans from one completed matrix batch."""

import argparse
import csv
import json
from collections import defaultdict
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("batch_dir", type=Path)
    parser.add_argument("--mode", choices=["default", "no-cg", "eager"])
    parser.add_argument("--cache", choices=["cold", "warm"])
    parser.add_argument("--start", type=float, default=0, help="Left edge in seconds from controller launch")
    parser.add_argument("--end", type=float, help="Right edge in seconds from controller launch")
    parser.add_argument("--out", type=Path, help="Default: <batch_dir>/analysis-fine-grained")
    args = parser.parse_args()
    if args.end is not None and args.end <= args.start:
        parser.error("--end must be larger than --start")
    output = args.out or args.batch_dir / "analysis-fine-grained"
    output.mkdir(parents=True, exist_ok=True)
    exporter = Path(__file__).with_name("export_startup_spans.py")
    colors = {"0": "#4477aa", "1": "#ee7733", "": "#888888"}
    count = 0
    for pair in sorted((args.batch_dir / "startup-spans").glob("r*-*")):
        repetition, mode = pair.name.split("-", 1)
        if args.mode and mode != args.mode:
            continue
        for cache in ("cold", "warm"):
            if args.cache and cache != args.cache:
                continue
            stem = f"{mode}-r01-{cache}"
            trace_dir = pair / stem
            events = args.batch_dir / pair.name / f"{stem}.events.jsonl"
            if not trace_dir.is_dir() or not events.is_file():
                continue
            subprocess.run([sys.executable, str(exporter), str(trace_dir), "--events", str(events)], check=True)
            with (trace_dir / "spans.csv").open() as stream:
                rows = list(csv.DictReader(stream))
            by_id = {row["span_id"]: row for row in rows}

            def stage(row):
                while not row.get("stage") and row.get("parent_span_id") in by_id:
                    row = by_id[row["parent_span_id"]]
                return row.get("stage", "")

            def depth(row):
                value = 0
                while row.get("parent_span_id") in by_id:
                    value += 1
                    row = by_id[row["parent_span_id"]]
                return value

            # Preserve all raw records; a separate residual table is useful for
            # finding the next instrumentation target within a parent context.
            with (output / f"residuals_{mode}_{repetition}_{cache}.csv").open("w") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]) if rows else ["span_id"])
                writer.writeheader()
                writer.writerows(sorted(rows, key=lambda row: float(row["exclusive_s"]), reverse=True))
            event_rows = [json.loads(line) for line in events.read_text().splitlines() if line.strip()]
            frontend_pid = str(next(event["pid"] for event in event_rows if event["event"] == "run_start"))
            ready = next((event for event in event_rows if event["event"] == "startup_ready"), None)
            by_pid = defaultdict(list)
            for row in rows:
                by_pid[row["pid"]].append(row)
            by_pid.setdefault(frontend_pid, [])
            children = defaultdict(list)
            for row in rows:
                parent = by_id.get(row.get("parent_span_id"))
                if parent and (parent["pid"], parent["tid"]) == (row["pid"], row["tid"]):
                    children[parent["span_id"]].append(row)
            visible = {row["span_id"] for row in rows if float(row["end_s"]) >= args.start
                       and (args.end is None or float(row["start_s"]) <= args.end)}
            # Preserve ancestors when zooming so every visible leaf keeps its context.
            for key in list(visible):
                parent = by_id[key].get("parent_span_id")
                while parent in by_id:
                    visible.add(parent)
                    parent = by_id[parent].get("parent_span_id")
            layout = []
            important = {"model_load", "profile_run", "flashinfer_sampling_ninja", "graph_capture", "compile_warmup"}

            def visit(row, prefix="", last=True):
                if row["span_id"] not in visible:
                    return
                index = len(layout)
                layout.append({"kind": "span", "row": row, "prefix": prefix + ("└─ " if last else "├─ ")})
                nested = sorted((child for child in children[row["span_id"]] if child["span_id"] in visible),
                                key=lambda child: (float(child["start_s"]), -float(child["end_s"])))
                for number, child in enumerate(nested):
                    visit(child, prefix + ("   " if last else "│  "), number == len(nested) - 1)
                layout[index]["last_descendant"] = len(layout) - 1

            def process_key(pid):
                values = by_pid[pid]
                return (0 if pid == frontend_pid else 1, min((stage(row) for row in values), default=""), int(pid))

            for pid in sorted(by_pid, key=process_key):
                process_rows = by_pid[pid]
                roots = [row for row in process_rows if not by_id.get(row.get("parent_span_id"))
                         or by_id[row["parent_span_id"]]["pid"] != pid]
                if pid != frontend_pid and not any(row["span_id"] in visible for row in process_rows):
                    continue
                ppid = process_rows[0].get("ppid", "?") if process_rows else "?"
                if pid == frontend_pid:
                    title = f"Frontend / PID {pid}  ← benchmark controller PID {ppid}"
                else:
                    identity = next((stage(row) for row in process_rows if stage(row)), "?")
                    parent_label = "Frontend" if ppid == frontend_pid else "process"
                    title = f"Stage {identity} / PID {pid}  ← {parent_label} PID {ppid}"
                layout.append({"kind": "header", "label": title})
                for number, row in enumerate(sorted(roots, key=lambda row: float(row["start_s"]))):
                    visit(row, "", number == len(roots) - 1)
                if pid == frontend_pid and ready:
                    layout.append({"kind": "ready", "time": float(ready["t_rel_s"])})
                layout.append({"kind": "gap"})
            if not any(item["kind"] == "span" for item in layout):
                continue
            fig, ax = plt.subplots(figsize=(20, max(5, 0.30 * len(layout) + 2)))
            end_limit = args.end if args.end is not None else max(
                max(float(row["end_s"]) for row in rows), float(ready["t_rel_s"]) if ready else 0
            ) * 1.03
            for index, item in enumerate(layout):
                if item["kind"] == "header":
                    ax.axhspan(index - .48, index + .48, color="#e8edf2", zorder=0)
                    ax.text(-.60, index, item["label"], transform=ax.get_yaxis_transform(),
                            fontsize=10, fontweight="bold", ha="left", va="center", clip_on=False)
                    continue
                if item["kind"] == "gap":
                    continue
                if item["kind"] == "ready":
                    ax.plot(item["time"], index, marker="D", color="#444444", markersize=5)
                    ax.text(-.60, index, "└─ startup_ready (frontend log received)",
                            transform=ax.get_yaxis_transform(), fontfamily="DejaVu Sans Mono",
                            fontsize=8, ha="left", va="center", clip_on=False)
                    continue
                row = item["row"]
                begin, end = float(row["start_s"]), float(row["end_s"])
                level = depth(row)
                is_parent = bool(children[row["span_id"]])
                if is_parent and row["name"] in {"executor_init", "kv_cache_init", "profile_run", "compile_warmup"}:
                    ax.add_patch(Rectangle((begin, index - .43), end - begin,
                                           item["last_descendant"] - index + .86,
                                           color="#dfe3e8", alpha=.20, linewidth=0, zorder=0))
                base = matplotlib.colors.to_rgb("#777777" if row["pid"] == frontend_pid else colors.get(stage(row), "#888888"))
                whiten = min(.62, .12 * level + (0 if is_parent else .22))
                color = tuple(channel * (1 - whiten) + whiten for channel in base)
                ax.barh(index, end - begin, left=begin, height=.72 if is_parent else .36,
                        color=color, edgecolor=base if is_parent else "none", linewidth=.6, zorder=3)
                if end - begin < (end_limit - args.start) * .001:
                    ax.plot(begin, index, "|", color=color, markersize=4, zorder=4)
                detail = row.get("scope") or row.get("sampling_pass") or row.get("lock_mode") or ""
                detail = {"core_config_profile_allocate_warmup": "core", "worker_allocation": "allocation"}.get(detail, detail)
                label = item["prefix"] + row["name"] + (f" [{detail}]" if detail else "")
                if row["name"] in important:
                    label += f"  I={float(row['inclusive_s']):.2f}s E={float(row['exclusive_s']):.2f}s"
                ax.text(-.60, index, label, transform=ax.get_yaxis_transform(), fontfamily="DejaVu Sans Mono",
                        fontsize=8, ha="left", va="center", clip_on=False,
                        fontweight="bold" if is_parent else "normal", color="#263238")
            ax.set_yticks([])
            ax.set_ylim(len(layout) - .5, -1)
            ax.set_xlim(args.start, end_limit)
            ax.set_xlabel("Seconds from controller launch · host monotonic clock")
            ax.set_title(f"{mode} / {cache} / {repetition} — startup span hierarchy\n"
                         "Process headers: OS PID/PPID · Tree: local parent_span_id · I: inclusive, E: exclusive (not additive)")
            ax.grid(axis="x", alpha=.2)
            ax.legend(handles=[Patch(color=colors["0"], label="Stage 0"),
                               Patch(color=colors["1"], label="Stage 1"),
                               Patch(color="#777777", label="Frontend")], loc="lower right", fontsize=8)
            fig.subplots_adjust(left=.39, right=.985, top=.94, bottom=.055)
            suffix = "" if args.start == 0 and args.end is None else f"_{args.start:g}-{args.end if args.end is not None else 'end'}s"
            destination = output / f"timeline_{mode}_{repetition}_{cache}{suffix}.png"
            fig.savefig(destination, dpi=180)
            plt.close(fig)
            print(destination)
            count += 1
    if not count:
        parser.error("No matching spans found; check the batch path and filters")


if __name__ == "__main__":
    main()
