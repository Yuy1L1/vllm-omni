#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Plot the retained single-run Qwen3-TTS startup matrix."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


MODES = ("default", "no-cg", "eager")
CACHE_STATES = ("cold", "warm")
PHASES = (
    "python_init",
    "worker_orchestration_other",
    "model_loading",
    "engine_init_residual",
    "torch_compile_observed",
    "cuda_graph_capture_observed",
)
COLORS = {
    "python_init": "#4C78A8",
    "worker_orchestration_other": "#BAB0AC",
    "model_loading": "#F58518",
    "engine_init_residual": "#54A24B",
    "torch_compile_observed": "#E45756",
    "cuda_graph_capture_observed": "#B279A2",
}
LABELS = {
    "python_init": "Python initialization/import",
    "worker_orchestration_other": "Worker/orchestration + uninstrumented",
    "model_loading": "Model loading",
    "engine_init_residual": "Engine-init residual",
    "torch_compile_observed": "torch.compile (observed)",
    "cuda_graph_capture_observed": "CUDA Graph capture (observed)",
}
INTERVAL_PRIORITY = (
    "python_init",
    "model_loading",
    "cuda_graph_capture_observed",
    "torch_compile_observed",
    "engine_init_residual",
)
EVENT_TO_PHASE = {
    "python_imports_done": "python_init",
    "model_load_done": "model_loading",
    "graph_capture_done": "cuda_graph_capture_observed",
    "compile_done": "torch_compile_observed",
    "model_compile_done": "torch_compile_observed",
    "engine_init_done": "engine_init_residual",
}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def partition_startup(events: list[dict[str, Any]]) -> tuple[float, dict[str, float]]:
    ready = next(event for event in events if event["event"] == "startup_ready")
    window_end = float(ready["t_rel_s"])
    startup_s = float(ready["duration_s"])
    window_start = window_end - startup_s
    intervals: dict[str, list[tuple[float, float]]] = {phase: [] for phase in INTERVAL_PRIORITY}

    for event in events:
        phase = EVENT_TO_PHASE.get(event["event"])
        duration = event.get("duration_s")
        if phase is None or duration is None:
            continue
        start = max(window_start, float(event["t_rel_s"]) - float(duration))
        end = min(window_end, float(event["t_rel_s"]))
        if end > start:
            intervals[phase].append((start, end))

    boundaries = {window_start, window_end}
    for phase_intervals in intervals.values():
        for start, end in phase_intervals:
            boundaries.add(start)
            boundaries.add(end)

    durations = {phase: 0.0 for phase in PHASES}
    ordered = sorted(boundaries)
    for start, end in zip(ordered, ordered[1:]):
        midpoint = (start + end) / 2
        owner = "worker_orchestration_other"
        for phase in INTERVAL_PRIORITY:
            if any(left <= midpoint < right for left, right in intervals[phase]):
                owner = phase
                break
        durations[owner] += end - start
    return startup_s, durations


def collect(source_dirs: dict[str, Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for mode in MODES:
        summary = json.loads((source_dirs[mode] / "summary.json").read_text())
        run_by_cache = {run["cache_state"]: run for run in summary["runs"]}
        for cache_state in CACHE_STATES:
            events_path = source_dirs[mode] / f"{mode}-{cache_state}.events.jsonl"
            startup_s, durations = partition_startup(load_jsonl(events_path))
            child = run_by_cache[cache_state]["child_result"]
            request_times = child.get("request_times_s") or []
            rows.append(
                {
                    "mode": mode,
                    "cache_state": cache_state,
                    "startup_total_s": startup_s,
                    "first_request_s": request_times[0] if request_times else None,
                    "steady_median_s": child.get("steady_median_s"),
                    **durations,
                }
            )
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def draw_png(path: Path, rows: list[dict[str, Any]]) -> None:
    from PIL import Image, ImageDraw, ImageFont

    width, height = 1350, 800
    left, top, bottom = 95, 110, 660
    plot_height = bottom - top
    y_max = 400
    bar_width = 105
    x_positions = (145, 270, 455, 580, 765, 890)

    def y(value: float) -> float:
        return bottom - value / y_max * plot_height

    def font(size: int, bold: bool = False) -> Any:
        name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
        candidate = Path("/usr/share/fonts/truetype/dejavu") / name
        try:
            return ImageFont.truetype(str(candidate), size)
        except OSError:
            return ImageFont.load_default()

    def centered_text(draw: Any, center_x: float, top_y: float, text: str, text_font: Any, fill: str) -> None:
        bounds = draw.textbbox((0, 0), text, font=text_font)
        draw.text((center_x - (bounds[2] - bounds[0]) / 2, top_y), text, font=text_font, fill=fill)

    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    draw.text((65, 28), "Qwen3-TTS startup wall-clock attribution", fill="#222222", font=font(26, True))
    draw.text(
        (65, 65),
        "Single retained launch per bar on one B300 (physical GPU 6)",
        fill="#555555",
        font=font(16),
    )

    for tick in range(0, y_max + 1, 50):
        tick_y = y(tick)
        draw.line((left, tick_y, 1025, tick_y), fill="#dddddd", width=1)
        draw.text((left - 45, tick_y - 8), str(tick), fill="#555555", font=font(13))

    for index, (x, row) in enumerate(zip(x_positions, rows)):
        cumulative = 0.0
        for phase in PHASES:
            value = float(row[phase])
            segment_top = y(cumulative + value)
            segment_bottom = y(cumulative)
            draw.rectangle(
                (x, segment_top, x + bar_width, segment_bottom),
                fill=COLORS[phase],
                outline="white",
            )
            if value >= 10.0:
                label = f"{value:.1f}"
                label_font = font(12, True)
                bounds = draw.textbbox((0, 0), label, font=label_font)
                text_y = segment_top + (segment_bottom - segment_top - (bounds[3] - bounds[1])) / 2
                text_color = "#333333" if phase == "worker_orchestration_other" else "white"
                centered_text(draw, x + bar_width / 2, text_y, label, label_font, text_color)
            cumulative += value

        total = float(row["startup_total_s"])
        centered_text(draw, x + bar_width / 2, y(total) - 25, f"{total:.1f}s", font(13, True), "#111111")
        centered_text(
            draw,
            x + bar_width / 2,
            bottom + 10,
            str(row["cache_state"]).capitalize(),
            font(14),
            "#333333",
        )
        if index % 2 == 0:
            group_center = (x_positions[index] + x_positions[index + 1] + bar_width) / 2
            mode_label = "no-CG" if row["mode"] == "no-cg" else str(row["mode"]).capitalize()
            centered_text(draw, group_center, bottom + 42, mode_label, font(18, True), "#222222")

    legend_x, legend_y = 1060, 125
    for index, phase in enumerate(PHASES):
        item_y = legend_y + index * 48
        draw.rectangle((legend_x, item_y, legend_x + 20, item_y + 20), fill=COLORS[phase])
        if phase == "worker_orchestration_other":
            draw.text((legend_x + 30, item_y), "Worker/orchestration +", fill="#333333", font=font(13))
            draw.text((legend_x + 30, item_y + 16), "uninstrumented", fill="#333333", font=font(13))
        elif phase == "engine_init_residual":
            draw.text((legend_x + 30, item_y), "Engine-init residual", fill="#333333", font=font(13))
            draw.text((legend_x + 30, item_y + 16), "(profile/KV/warmup/JIT)", fill="#333333", font=font(13))
        else:
            draw.text((legend_x + 30, item_y + 2), LABELS[phase], fill="#333333", font=font(13))

    draw.text(
        (65, 730),
        "Intervals are unioned across stages; observed compile/CG intervals are removed from engine-init residual.",
        fill="#555555",
        font=font(13),
    )
    draw.text(
        (65, 754),
        "Engine-init residual can include uninstrumented JIT/compilation. First-request latency is outside this chart.",
        fill="#555555",
        font=font(13),
    )
    image.save(path)


def write_readme(path: Path, rows: list[dict[str, Any]], sources: dict[str, Path]) -> None:
    lines = [
        "# Qwen3-TTS retained single-run breakdown",
        "",
        "| Mode | Cache | Startup | First request | Steady median |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['mode']} | {row['cache_state']} | {row['startup_total_s']:.3f}s | "
            f"{row['first_request_s']:.3f}s | {row['steady_median_s']:.3f}s |"
        )
    lines.extend(
        [
            "",
            "## Method",
            "",
            "- One retained launch per bar; these are observations, not medians.",
            "- The startup window is `startup_ready.duration_s`.",
            "- Reported intervals are projected onto the controller monotonic timeline and unioned across stages.",
            "- Observed `torch.compile` and CUDA Graph capture intervals take precedence "
            "over the enclosing engine-init interval.",
            "- Engine-init residual may contain memory profiling, KV-cache creation, warmup, "
            "and uninstrumented JIT/compilation.",
            "- Worker/orchestration + uninstrumented is the remaining startup wall-clock time.",
            "- First-request and steady-state latency are reported in the table but are not part of the startup stack.",
            "",
            "## Sources",
            "",
        ]
    )
    for mode in MODES:
        lines.append(f"- `{mode}`: `{sources[mode]}`")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--default-dir", type=Path, required=True)
    parser.add_argument("--no-cg-dir", type=Path, required=True)
    parser.add_argument("--eager-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    sources = {
        "default": args.default_dir.resolve(),
        "no-cg": args.no_cg_dir.resolve(),
        "eager": args.eager_dir.resolve(),
    }
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = collect(sources)
    write_csv(output_dir / "startup_breakdown_single_run.csv", rows)
    draw_png(output_dir / "startup_breakdown_single_run.png", rows)
    write_readme(output_dir / "README.md", rows, sources)
    print(f"Wrote single-run breakdown to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
