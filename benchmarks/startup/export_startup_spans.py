#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Export one launch's process-local JSONL spans to CSV and Chrome trace."""

import argparse
import csv
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace_dir", type=Path, help="One mode/repetition/cache launch directory")
    parser.add_argument("--events", type=Path, help="Matching benchmark events JSONL (origin and ready cutoff)")
    args = parser.parse_args()
    spans = []
    for path in sorted(args.trace_dir.glob("*.jsonl")):
        spans.extend(json.loads(line) for line in path.read_text().splitlines() if line.strip())
    if not spans:
        parser.error("No span records found")
    if len({s["hostname"] for s in spans}) != 1:
        parser.error("Monotonic timestamps cannot align different hosts")
    origin = min(s["start_ns"] for s in spans)
    cutoff = None
    launch = {}
    if args.events:
        events = [json.loads(line) for line in args.events.read_text().splitlines() if line.strip()]
        run_start = next(e for e in events if e["event"] == "run_start")
        origin = run_start["start_ns"]
        launch = {key: run_start.get(key) for key in ("mode", "repetition", "cache_state")}
        ready = [e["t_rel_s"] for e in events if e["event"] == "startup_ready"]
        if ready:
            cutoff = origin + int(min(ready) * 1e9)
    if cutoff is not None:
        spans = [s for s in spans if s["end_ns"] <= cutoff]
    spans.sort(key=lambda s: (s["start_ns"], s["end_ns"]))
    rows, trace = [], []
    for span in spans:
        if span.get("name") == "phase_lock_wait" and span.get("mode") in {"SH", "EX"}:
            span = {**span, "lock_mode": span["mode"]}
        span = {**span, **launch}
        row = dict(span)
        row["start_s"] = (span["start_ns"] - origin) / 1e9
        row["end_s"] = (span["end_ns"] - origin) / 1e9
        rows.append(row)
        trace.append({
            "name": span["name"], "cat": ".".join(str(launch.get(key) or "unknown") for key in ("mode", "cache_state")), "ph": "X",
            "pid": span["pid"], "tid": span["tid"],
            "ts": (span["start_ns"] - origin) / 1e3,
            "dur": (span["end_ns"] - span["start_ns"]) / 1e3,
            "args": span,
        })
    bootstrap = []
    for path in sorted((args.trace_dir / "bootstrap").glob("*.jsonl")):
        bootstrap.extend(json.loads(line) for line in path.read_text().splitlines() if line.strip())
    launches = {}
    for mark in bootstrap:
        launches.setdefault(mark["launch_id"], {})[mark["event"]] = mark
        trace.append({"name": mark["event"], "cat": "bootstrap", "ph": "i", "s": "t",
                      "pid": mark["pid"], "tid": mark["tid"],
                      "ts": (mark["timestamp_ns"] - origin) / 1e3,
                      "args": {**mark, **launch}})
    for launch_id, marks in launches.items():
        begin, child = marks.get("parent_start_begin"), marks.get("child_target_entry")
        returned = marks.get("parent_start_return")
        if begin and returned:
            trace.append({"name": "process_start_call", "cat": "bootstrap", "ph": "X",
                          "pid": begin["pid"], "tid": begin["tid"],
                          "ts": (begin["timestamp_ns"] - origin) / 1e3,
                          "dur": (returned["timestamp_ns"] - begin["timestamp_ns"]) / 1e3,
                          "args": {**begin, **launch}})
        if begin and child:
            for mark, phase in ((begin, "s"), (child, "f")):
                flow = {"name": "process_launch", "cat": "bootstrap", "ph": phase,
                        "id": launch_id, "pid": mark["pid"], "tid": mark["tid"],
                        "ts": (mark["timestamp_ns"] - origin) / 1e3,
                        "args": {"launch_id": launch_id, "stage": child["stage"],
                                 "start_method": child["start_method"]}}
                if phase == "f":
                    flow["bp"] = "e"
                trace.append(flow)
    columns = sorted({key for row in rows for key in row})
    with (args.trace_dir / "spans.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    (args.trace_dir / "trace.json").write_text(json.dumps({"traceEvents": trace, "displayTimeUnit": "ms", "startup_launch": launch}))
    print(f"Exported {len(rows)} spans to {args.trace_dir}")


if __name__ == "__main__":
    main()
