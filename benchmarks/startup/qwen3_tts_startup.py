#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Qwen3-TTS startup attribution benchmark for B300/GH200 and other GPUs.

Each selected mode runs a fresh child process for each cold and warm launch.
The controller creates an empty TorchInductor cache for each mode/repetition,
then reuses it for the immediately following warm launch. The accompanying
shell wrapper also isolates vLLM and FlashInfer JIT caches per cold/warm pair.
Running this Python file alone does NOT reset vLLM/FlashInfer caches.

HF model files, OS page cache, and CUDA driver cache are not cleared. Therefore
"cold" means empty benchmark-controlled compiler caches, not cold machine RAM.
Sampling backend overrides, when supplied via environment, affect timing.

The controller uses only the Python standard library and does not initialize
Torch/CUDA. It timestamps every log line on receipt and records stage-aware
intervals that can overlap; don't add overlapping stage intervals.
"""
from __future__ import annotations
import argparse
import datetime as dt
import json
import os
import platform
import queue
import re
import shutil
import signal
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

DEFAULT_MODEL = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"

DEFAULT_DEPLOY_CONFIG = "vllm_omni/deploy/qwen3_tts.yaml"
# Stage 1's inner Code2Wav graph is controlled by enforce_eager, independently
# of vLLM's CompilationConfig.cudagraph_mode. There is currently no narrower
# deploy knob, hence the Stage-1 eager override in the no-cg ablation.
MODES: dict[str, dict[str, Any]] = {
    "default": {
        "stage_overrides": {},
        "disable_torch_compile": False,
        "meaning": "repository defaults: torch.compile plus configured outer/inner CUDA Graphs",
    },
    "no-cg": {
        "stage_overrides": {
            "0": {"compilation_config": {"cudagraph_mode": "NONE"}},
            "1": {
                "enforce_eager": True,
                "compilation_config": {"cudagraph_mode": "NONE"},
            },
        },
        "disable_torch_compile": False,
        "meaning": (
            "no known CUDA Graph capture; Stage 0 keeps torch.compile, while Stage 1 must use "
            "enforce_eager to disable its model-internal Code2Wav graph"
        ),
    },
    "eager": {
        "stage_overrides": {
            "0": {
                "enforce_eager": True,
                "compilation_config": {"cudagraph_mode": "NONE"},
            },
            "1": {
                "enforce_eager": True,
                "compilation_config": {"cudagraph_mode": "NONE"},
            },
        },
        "disable_torch_compile": True,
        "meaning": "enforce_eager on both stages and TORCH_COMPILE_DISABLE=1 for model-local compile calls",
    },
}

STAGE_PREFIX = re.compile(r"StageEngineCoreProc_stage(?P<stage>\d+)_replica(?P<replica>\d+)")

STAGE_RUNTIME = re.compile(r"\[StageRuntime\] Stage (?P<stage>\d+)(?: replica (?P<replica>\d+))? initialized")

STAGE_LAUNCH = re.compile(r"Stage-(?P<stage>\d+) set runtime devices")

def _duration(pattern: str) -> re.Pattern[str]:
    return re.compile(pattern, re.IGNORECASE)
EVENT_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("worker_proc_up", re.compile(r"world_size=\d+ rank=\d+")),
    ("model_load_start", re.compile(r"Starting to load model")),
    ("weights_load_done", _duration(r"Loading weights took (?P<duration>[\d.]+) seconds")),
    (
        "model_load_done",
        _duration(r"Model loading took (?P<memory_gib>[\d.]+) GiB memory and (?P<duration>[\d.]+) seconds"),
    ),
    (
        "memory_profile_done",
        _duration(r"memory profil(?:e|ing).*?(?:took|takes) (?P<duration>[\d.]+) s(?:ec(?:ond)?s?)?"),
    ),
    ("compile_done", _duration(r"torch\.compile takes (?P<duration>[\d.]+) s in total")),
    (
        "model_compile_done",
        _duration(
            r"(?P<label>[\w\.-]+): compiled region took (?P<duration>[\d.]+) s; "
            r"Inductor cache .*?: (?P<hits>\d+) hits, (?P<misses>\d+) misses, (?P<bypasses>\d+) bypasses"
        ),
    ),
    (
        "graph_capture_done",
        _duration(r"Graph capturing finished in (?P<duration>[\d.]+) secs, took (?P<memory_gib>[\d.]+) GiB"),
    ),
    (
        "aux_graph_memory",
        _duration(r"Estimated auxiliary CUDA graph memory: (?P<memory_gib>[\d.]+) GiB"),
    ),
    (
        "mtp_graph_memory",
        _duration(r"Estimated talker MTP CUDA graph memory: (?P<memory_gib>[\d.]+) GiB"),
    ),
    (
        "graph_memory_reserved",
        _duration(r"Reserved (?P<memory_gib>[\d.]+) GiB for CUDA graphs"),
    ),
    (
        "engine_init_done",
        _duration(r"init engine \(profile, create kv cache, warmup model\) took (?P<duration>[\d.]+) s"),
    ),
    ("decoder_graph_enabled", re.compile(r"Code2Wav decoder CUDA Graph enabled")),
    ("decoder_graph_disabled", re.compile(r"Code2Wav CUDA Graph disabled")),
    (
        "hf_config_load",
        re.compile(r"loading configuration file (?P<config_path>.*?config\.json)(?:\s|$)", re.IGNORECASE),
    ),
]

def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="milliseconds")

def run_text(command: list[str], cwd: Path) -> str:
    try:
        result = subprocess.run(command, cwd=cwd, text=True, capture_output=True, timeout=15, check=False)
        return (result.stdout or result.stderr).strip()
    except (OSError, subprocess.TimeoutExpired):
        return "unavailable"

def hardware_snapshot(cwd: Path) -> dict[str, Any]:
    return {
        "captured_at_utc": utc_now(),
        "hostname": platform.node(),
        "python": platform.python_version(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "nvidia_smi_gpus": run_text(
            [
                "nvidia-smi",
                "--query-gpu=index,name,uuid,memory.total,driver_version",
                "--format=csv,noheader",
            ],
            cwd,
        ),
        "nvidia_smi_compute_apps": run_text(
            [
                "nvidia-smi",
                "--query-compute-apps=gpu_uuid,pid,used_memory,process_name",
                "--format=csv,noheader",
            ],
            cwd,
        ),
        "git_head": run_text(["git", "rev-parse", "HEAD"], cwd),
        "git_status_short": run_text(["git", "status", "--short"], cwd),
    }

def stage_identity(line: str) -> tuple[int | None, int | None]:
    match = STAGE_PREFIX.search(line)
    if match:
        return int(match.group("stage")), int(match.group("replica"))
    match = STAGE_RUNTIME.search(line) or STAGE_LAUNCH.search(line)
    if match:
        replica = match.groupdict().get("replica")
        return int(match.group("stage")), int(replica) if replica is not None else None
    return None, None

def classify_line(line: str) -> tuple[str | None, dict[str, Any]]:
    if line.startswith("STARTUP_BENCH_EVENT "):
        try:
            payload = json.loads(line.removeprefix("STARTUP_BENCH_EVENT "))
        except json.JSONDecodeError:
            return None, {}
        event = payload.pop("event", None)
        return event, payload
    if STAGE_LAUNCH.search(line):
        return "stage_launch", {}
    if STAGE_RUNTIME.search(line):
        return "stage_ready", {}
    if "[Omni] Initializing with model" in line or "[OmniEngine] Initializing with model" in line:
        return "omni_init_start", {}
    if "AsyncOmniEngine initialized in" in line:
        match = re.search(r"initialized in (?P<duration>[\d.]+) seconds", line)
        return "omni_ready", ({"duration_s": float(match.group("duration"))} if match else {})
    for name, pattern in EVENT_PATTERNS:
        match = pattern.search(line)
        if not match:
            continue
        fields: dict[str, Any] = {}
        for key, value in match.groupdict().items():
            if value is None:
                continue
            if key in {"duration", "memory_gib"}:
                fields[key] = float(value)
            elif value.isdigit():
                fields[key] = int(value)
            else:
                fields[key] = value
        if "duration" in fields:
            fields["duration_s"] = fields.pop("duration")
        return name, fields
    return None, {}

def reader_thread(stream: Any, output: queue.Queue[str | None]) -> None:
    try:
        for line in iter(stream.readline, ""):
            output.put(line)
    finally:
        output.put(None)

def terminate_process_group(proc: subprocess.Popen[str]) -> None:
    if proc.poll() is not None:
        return
    try:
        os.killpg(proc.pid, signal.SIGTERM)
        proc.wait(timeout=15)
    except (ProcessLookupError, subprocess.TimeoutExpired):
        if proc.poll() is None:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait(timeout=15)

def measured_launch(
    *,
    command: list[str],
    env: dict[str, str],
    cwd: Path,
    log_path: Path,
    events_path: Path,
    mode: str,
    repetition: int,
    cache_state: str,
    timeout_s: float,
) -> dict[str, Any]:
    started_ns = time.perf_counter_ns()
    proc = subprocess.Popen(
        command,
        cwd=cwd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        start_new_session=True,
    )
    assert proc.stdout is not None
    lines: queue.Queue[str | None] = queue.Queue()
    thread = threading.Thread(target=reader_thread, args=(proc.stdout, lines), daemon=True)
    thread.start()
    deadline = time.monotonic() + timeout_s
    timed_out = False
    child_result: dict[str, Any] | None = None
    event_count = 0
    with log_path.open("w", encoding="utf-8") as log_file, events_path.open("w", encoding="utf-8") as events_file:
        synthetic = {
            "event": "run_start",
            "mode": mode,
            "repetition": repetition,
            "cache_state": cache_state,
            "wall_time_utc": utc_now(),
            "t_rel_s": 0.0,
            "pid": proc.pid,
        }
        events_file.write(json.dumps(synthetic, sort_keys=True) + "\n")
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                timed_out = True
                terminate_process_group(proc)
            try:
                item = lines.get(timeout=max(0.05, min(0.25, remaining)) if not timed_out else 0.25)
            except queue.Empty:
                if timed_out or (proc.poll() is not None and not thread.is_alive()):
                    break
                continue
            if item is None:
                break
            received_ns = time.perf_counter_ns()
            rel_s = (received_ns - started_ns) / 1e9
            wall_time = utc_now()
            text_line = item.rstrip("\n")
            log_file.write(f"[{wall_time} +{rel_s:.6f}s] {text_line}\n")
            log_file.flush()
            print(f"[{mode}/r{repetition:02d}/{cache_state} +{rel_s:8.3f}s] {text_line}", flush=True)
            if text_line.startswith("STARTUP_BENCH_RESULT "):
                try:
                    child_result = json.loads(text_line.removeprefix("STARTUP_BENCH_RESULT "))
                except json.JSONDecodeError:
                    pass
            event, fields = classify_line(text_line)
            if event is None:
                continue
            stage, replica = stage_identity(text_line)
            record = {
                "event": event,
                "mode": mode,
                "repetition": repetition,
                "cache_state": cache_state,
                "wall_time_utc": wall_time,
                "t_rel_s": round(rel_s, 6),
                "stage": stage,
                "replica": replica,
                "timestamp_source": "controller_receive_monotonic",
                **fields,
            }
            events_file.write(json.dumps(record, sort_keys=True) + "\n")
            events_file.flush()
            event_count += 1
    if proc.poll() is None:
        terminate_process_group(proc)
    thread.join(timeout=2)
    elapsed_s = (time.perf_counter_ns() - started_ns) / 1e9
    return {
        "mode": mode,
        "repetition": repetition,
        "cache_state": cache_state,
        "returncode": proc.returncode,
        "timed_out": timed_out,
        "controller_process_s": round(elapsed_s, 6),
        "startup_s": child_result.get("startup_s") if child_result else None,
        "child_result": child_result,
        "event_count": event_count,
        "log": str(log_path),
        "events": str(events_path),
        "status": "ok" if proc.returncode == 0 and child_result is not None else "failed",
    }

def load_events(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]

def derive_spans(run: dict[str, Any], events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    spans: list[dict[str, Any]] = []
    marks: dict[tuple[int, int | None, str], dict[str, Any]] = {}
    for event in events:
        stage = event.get("stage")
        if stage is not None:
            marks.setdefault((stage, event.get("replica"), event["event"]), event)
        duration = event.get("duration_s")
        if duration is not None and event.get("event") in {
            "python_imports_done",
            "startup_ready",
            "weights_load_done",
            "model_load_done",
            "memory_profile_done",
            "compile_done",
            "model_compile_done",
            "graph_capture_done",
            "engine_init_done",
            "request_done",
        }:
            end = float(event["t_rel_s"])
            spans.append(
                {
                    "mode": run["mode"],
                    "repetition": run["repetition"],
                    "cache_state": run["cache_state"],
                    "stage": stage,
                    "replica": event.get("replica"),
                    "phase": event["event"].removesuffix("_done"),
                    "start_s": round(max(0.0, end - float(duration)), 6),
                    "end_s": round(end, 6),
                    "duration_s": float(duration),
                    "source": "reported_duration_anchored_to_controller_receive_time",
                }
            )
    # Coarse stage intervals use observed milestones. They are useful even
    # when a vLLM version does not print an exact duration for a sub-phase.
    phases = [
        ("worker_spawn_import", "stage_launch", "worker_proc_up"),
        ("device_and_distributed_init", "worker_proc_up", "model_load_start"),
        ("load_to_engine_init", "model_load_start", "engine_init_done"),
        ("ready_handoff", "engine_init_done", "stage_ready"),
    ]
    identities = {(stage, replica) for stage, replica, _ in marks}
    for stage, replica in sorted(identities, key=lambda value: (value[0], value[1] is None, value[1] or -1)):
        for phase, begin_name, end_name in phases:
            begin = marks.get((stage, replica, begin_name))
            end = marks.get((stage, replica, end_name))
            # Launch/ready messages may be emitted by a parent without replica id.
            begin = begin or marks.get((stage, None, begin_name))
            end = end or marks.get((stage, None, end_name))
            if not begin or not end:
                continue
            start_s, end_s = float(begin["t_rel_s"]), float(end["t_rel_s"])
            if end_s < start_s:
                continue
            spans.append(
                {
                    "mode": run["mode"],
                    "repetition": run["repetition"],
                    "cache_state": run["cache_state"],
                    "stage": stage,
                    "replica": replica,
                    "phase": phase,
                    "start_s": round(start_s, 6),
                    "end_s": round(end_s, 6),
                    "duration_s": round(end_s - start_s, 6),
                    "source": "controller_receive_milestones",
                }
            )
    return spans

def summarize_events(events: list[dict[str, Any]]) -> dict[str, Any]:
    compile_hits = compile_misses = compile_bypasses = 0
    hf_config_loads: dict[str, int] = {}
    graph_memory: list[dict[str, Any]] = []
    worker_processes: set[tuple[int | None, int | None]] = set()
    ready_times = [float(event["t_rel_s"]) for event in events if event["event"] == "startup_ready"]
    startup_end_s = min(ready_times) if ready_times else float("inf")
    for event in events:
        if float(event.get("t_rel_s", 0.0)) > startup_end_s:
            continue
        if event["event"] == "model_compile_done":
            compile_hits += int(event.get("hits", 0))
            compile_misses += int(event.get("misses", 0))
            compile_bypasses += int(event.get("bypasses", 0))
        if event["event"] == "hf_config_load":
            key = str(event.get("stage"))
            hf_config_loads[key] = hf_config_loads.get(key, 0) + 1
        if event["event"] == "worker_proc_up":
            worker_processes.add((event.get("stage"), event.get("replica")))
        if event["event"] in {"aux_graph_memory", "mtp_graph_memory", "graph_memory_reserved"}:
            graph_memory.append(
                {
                    "event": event["event"],
                    "stage": event.get("stage"),
                    "replica": event.get("replica"),
                    "memory_gib": event.get("memory_gib"),
                }
            )
    return {
        "inductor_fxgraph_cache": {
            "hits": compile_hits,
            "misses": compile_misses,
            "bypasses": compile_bypasses,
        },
        "hf_config_load_count_by_stage": hf_config_loads,
        "worker_process_count": len(worker_processes),
        "graph_memory_observations": graph_memory,
    }

def write_summary(
    output_dir: Path,
    metadata: dict[str, Any],
    runs: list[dict[str, Any]],
    spans: list[dict[str, Any]],
) -> None:
    payload = {"metadata": metadata, "runs": runs, "spans": spans}
    (output_dir / "summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with (output_dir / "spans.jsonl").open("w", encoding="utf-8") as handle:
        for span in spans:
            handle.write(json.dumps(span, sort_keys=True) + "\n")
    def metric(run: dict[str, Any], name: str) -> float | None:
        if run.get("status") != "ok":
            return None
        if name == "startup":
            value = run.get("startup_s")
        else:
            child = run.get("child_result") or {}
            if name == "first_request":
                request_times = child.get("request_times_s") or []
                value = request_times[0] if request_times else None
            elif name == "steady":
                value = child.get("steady_median_s")
            else:
                raise ValueError(name)
        return float(value) if value is not None else None
    def median_text(group: list[dict[str, Any]], name: str) -> str:
        values = [value for run in group if (value := metric(run, name)) is not None]
        return f"{statistics.median(values):.3f}" if values else ""
    lines = [
        "# Qwen3-TTS startup matrix",
        "",
        "Cold means an empty benchmark-controlled TorchInductor cache; "
        "warm reuses the immediately preceding cold launch's caches. "
        "vLLM and FlashInfer caches are inherited from the parent (the wrapper isolates them). "
        "HF model files, OS page cache, and GPU driver caches are inherited. "
        "Cross-stage spans may overlap and must not be added.",
        "",
        "## Aggregate across repetitions",
        "",
        "| mode | cache | ok/total | startup median (s) | first request median (s) | steady median (s) |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for mode in metadata["mode_order"]:
        for cache_state in ("cold", "warm"):
            group = [run for run in runs if run["mode"] == mode and run["cache_state"] == cache_state]
            if not group:
                continue
            ok = sum(run["status"] == "ok" for run in group)
            lines.append(
                f"| {mode} | {cache_state} | {ok}/{len(group)} | {median_text(group, 'startup')} | "
                f"{median_text(group, 'first_request')} | {median_text(group, 'steady')} |"
            )
    lines.extend(
        [
        "",
        "## Individual launches",
        "",
        "| mode | repetition | cache | status | startup (s) | first request (s) | "
        "steady median (s) | FX hits/misses | workers |",
        "| --- | ---: | --- | --- | ---: | ---: | ---: | --- | ---: |",
        ]
    )
    for run in runs:
        startup = "" if run.get("startup_s") is None else f"{run['startup_s']:.3f}"
        child = run.get("child_result") or {}
        request_times = child.get("request_times_s") or []
        first_request = f"{request_times[0]:.3f}" if request_times else ""
        steady = child.get("steady_median_s")
        steady_text = f"{steady:.3f}" if steady is not None else ""
        repeated = run.get("repeat_indicators") or {}
        cache = repeated.get("inductor_fxgraph_cache") or {}
        lines.append(
            f"| {run['mode']} | {run['repetition']} | {run['cache_state']} | {run['status']} | "
            f"{startup} | {first_request} | "
            f"{steady_text} | {cache.get('hits', 0)}/{cache.get('misses', 0)} | "
            f"{repeated.get('worker_process_count', 0)} |"
        )
    lines.extend(
        [
            "",
            "## Repetition indicators",
            "",
            "These are observations, not automatic claims that the repeated work is removable.",
            "",
            "| mode | repetition | cache | HF config loads by stage | "
            "FX cache hit/miss/bypass | graph-memory observations |",
            "| --- | ---: | --- | --- | --- | --- |",
        ]
    )
    for run in runs:
        repeated = run.get("repeat_indicators") or {}
        cache = repeated.get("inductor_fxgraph_cache") or {}
        graph_memory = repeated.get("graph_memory_observations") or []
        graph_text = ", ".join(
            f"stage {item['stage']} {item['event']}={item['memory_gib']} GiB" for item in graph_memory
        )
        lines.append(
            f"| {run['mode']} | {run['repetition']} | {run['cache_state']} | "
            f"`{json.dumps(repeated.get('hf_config_load_count_by_stage', {}), sort_keys=True)}` | "
            f"{cache.get('hits', 0)}/{cache.get('misses', 0)}/{cache.get('bypasses', 0)} | "
            f"{graph_text} |"
        )
    lines.extend(
        [
            "",
            "## Stage-aligned spans",
            "",
            "| mode | repetition | cache | stage | replica | phase | start (s) | end (s) | duration (s) | source |",
            "| --- | ---: | --- | ---: | ---: | --- | ---: | ---: | ---: | --- |",
        ]
    )
    def span_order(item: dict[str, Any]) -> tuple[Any, ...]:
        return item["repetition"], item["mode"], item["cache_state"], item["start_s"], item["stage"] or -1
    for span in sorted(spans, key=span_order):
        lines.append(
            f"| {span['mode']} | {span['repetition']} | {span['cache_state']} | "
            f"{span['stage']} | {span['replica']} | "
            f"{span['phase']} | {span['start_s']:.3f} | {span['end_s']:.3f} | "
            f"{span['duration_s']:.3f} | {span['source']} |"
        )
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

def child_main(args: argparse.Namespace) -> int:
    # Imports live here so the controller itself does not initialize Torch or
    # CUDA and contaminate the measured child processes.
    started = time.perf_counter()
    from vllm_omni.entrypoints.omni import Omni
    imports_s = time.perf_counter() - started
    print(
        "STARTUP_BENCH_EVENT "
        + json.dumps({"event": "python_imports_done", "duration_s": round(imports_s, 6)}, sort_keys=True),
        flush=True,
    )
    overrides = json.loads(args.stage_overrides)
    omni = None
    try:
        engine_started = time.perf_counter()
        omni = Omni(
            model=args.model,
            deploy_config=args.deploy_config,
            stage_overrides=overrides,
            trust_remote_code=True,
            log_stats=True,
            parallel_stage_init=args.parallel_stage_init,
            stage_init_timeout=args.stage_init_timeout,
        )
        ready = time.perf_counter()
        startup_s = ready - started
        engine_init_s = ready - engine_started
        print(
            "STARTUP_BENCH_EVENT "
            + json.dumps({"event": "startup_ready", "duration_s": round(startup_s, 6)}, sort_keys=True),
            flush=True,
        )
        request_setup_s = 0.0
        request_times_s: list[float] = []
        if args.requests:
            setup_started = time.perf_counter()
            # When a file is executed by path, Python puts the file's directory
            # (benchmarks/startup), not the repository root, on sys.path.  The
            # fixed request fixture lives under the repo's namespace-package
            # examples/ tree, so make that import location explicit.
            repo_root = Path(__file__).resolve().parents[2]
            repo_root_str = str(repo_root)
            if repo_root_str not in sys.path:
                sys.path.insert(0, repo_root_str)
            from examples.offline_inference.text_to_speech.qwen3_tts.end2end import get_custom_voice_query
            query = get_custom_voice_query(use_batch_sample=False)
            prompt = query.inputs
            additional = prompt.get("additional_information", {})
            additional["max_new_tokens"] = [args.max_new_tokens]
            request_setup_s = time.perf_counter() - setup_started
            for index in range(args.requests):
                request_started = time.perf_counter()
                # Consume all stage outputs. The benchmark intentionally does
                # not serialize audio, so this is model/runtime latency only.
                omni.generate([prompt], use_tqdm=False)
                request_s = time.perf_counter() - request_started
                request_times_s.append(round(request_s, 6))
                print(
                    "STARTUP_BENCH_EVENT "
                    + json.dumps(
                        {
                            "event": "request_done",
                            "request_index": index,
                            "request_kind": "first" if index == 0 else "steady",
                            "duration_s": round(request_s, 6),
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
        steady_median_s = statistics.median(request_times_s[1:]) if len(request_times_s) > 1 else None
        result = {
            "startup_s": round(startup_s, 6),
            "python_imports_s": round(imports_s, 6),
            "engine_init_s": round(engine_init_s, 6),
            "request_setup_s": round(request_setup_s, 6),
            "request_times_s": request_times_s,
            "steady_median_s": round(steady_median_s, 6) if steady_median_s is not None else None,
            "max_new_tokens": args.max_new_tokens,
            "stage_init_timeout_s": args.stage_init_timeout,
            "model": args.model,
            "stage_overrides": overrides,
            "torch_compile_disable": os.environ.get("TORCH_COMPILE_DISABLE"),
            "torchinductor_cache_dir": os.environ.get("TORCHINDUCTOR_CACHE_DIR"),
            "vllm_cache_root": os.environ.get("VLLM_CACHE_ROOT"),
            "flashinfer_workspace_base": os.environ.get("FLASHINFER_WORKSPACE_BASE"),
            "flashinfer_sampler_env": os.environ.get("VLLM_USE_FLASHINFER_SAMPLER", "unset"),
        }
        print("STARTUP_BENCH_RESULT " + json.dumps(result, sort_keys=True), flush=True)
        return 0
    finally:
        if omni is not None:
            omni.shutdown()

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--deploy-config", default=DEFAULT_DEPLOY_CONFIG)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--modes", nargs="+", choices=tuple(MODES), default=list(MODES))
    parser.add_argument(
        "--repetitions",
        type=int,
        default=1,
        help="cold/warm pairs per mode (default: 1)",
    )
    parser.add_argument("--timeout", type=float, default=1800.0, help="seconds allowed per launch")
    parser.add_argument(
        "--stage-init-timeout",
        type=int,
        default=900,
        help="seconds allowed for each stage to reach READY (passed to Omni; default: 900)",
    )
    parser.add_argument(
        "--requests",
        type=int,
        default=3,
        help="fixed requests after each startup (first request plus steady-state repeats; 0 disables)",
    )
    parser.add_argument("--max-new-tokens", type=int, default=64, help="fixed output-token cap for each request")
    parser.add_argument("--parallel-stage-init", action="store_true")
    parser.add_argument("--keep-going", action="store_true", help="continue after a failed launch")
    parser.add_argument("--expected-gpu", default=None,
                        help="Optional substring required in nvidia-smi GPU information (e.g. GH200 or B300)")
    parser.add_argument("--allow-non-b300", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--_child", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--stage-overrides", default="{}", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.repetitions < 1:
        parser.error("--repetitions must be positive")
    if args.requests < 0:
        parser.error("--requests must be non-negative")
    if args.max_new_tokens < 1:
        parser.error("--max-new-tokens must be positive")
    if args.stage_init_timeout < 1:
        parser.error("--stage-init-timeout must be positive")
    return args

def main() -> int:
    args = parse_args()
    if args._child:
        return child_main(args)
    repo = Path(__file__).resolve().parents[2]
    deploy_config = Path(args.deploy_config)
    if not deploy_config.is_absolute():
        deploy_config = (repo / deploy_config).resolve()
    if not deploy_config.is_file():
        raise SystemExit(f"deploy config not found: {deploy_config}")
    if os.path.isabs(args.model) and not Path(args.model).is_dir():
        raise SystemExit(f"local model directory not found: {args.model}")
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir = Path(args.output_dir or f"startup-results/qwen3-tts-{stamp}").resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    cache_root = output_dir / "inductor-cache"
    cache_root.mkdir()
    hardware = hardware_snapshot(repo)
    (output_dir / "hardware.json").write_text(json.dumps(hardware, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.expected_gpu and args.expected_gpu.lower() not in hardware["nvidia_smi_gpus"].lower():
        raise SystemExit(
            f"GPU preflight failed (expected {args.expected_gpu!r}); "
            f"see {output_dir / 'hardware.json'}."
        )
    metadata = {
        "created_at_utc": utc_now(),
        "model": args.model,
        "deploy_config": str(deploy_config),
        "cold_definition": (
            "Fresh TorchInductor cache per mode/repetition; the shell wrapper, if used, "
            "also provides fresh VLLM_CACHE_ROOT and FLASHINFER_WORKSPACE_BASE per pair. "
            "HF model files, OS page cache, and CUDA driver cache are not cleared."
        ),
        "cache_root": str(cache_root),
        "cache_policy": (
            "delete each mode's TorchInductor cache at the start of every repetition; "
            "cold populates it and the immediately following warm launch reuses it"
        ),
        "inherited_vllm_cache_root": os.environ.get("VLLM_CACHE_ROOT"),
        "inherited_flashinfer_workspace_base": os.environ.get("FLASHINFER_WORKSPACE_BASE"),
        "flashinfer_sampler_env": os.environ.get("VLLM_USE_FLASHINFER_SAMPLER", "unset (vLLM default)"),
        "hf_hub_offline": os.environ.get("HF_HUB_OFFLINE", "unset"),
        "expected_gpu": args.expected_gpu,
        "note_no_cg": (
            "The no-cg mode also sets Stage 1 enforce_eager=True to disable its "
            "internal Code2Wav graph: not a pure CUDA Graph-only ablation."
        ),
        "timestamp_source": "controller receipt of unbuffered merged child stdout/stderr",
        "parallel_stage_init": args.parallel_stage_init,
        "stage_init_timeout_s": args.stage_init_timeout,
        "requests_per_launch": args.requests,
        "request_max_new_tokens": args.max_new_tokens,
        "request_task": "CustomVoice; fixed repository example prompt; batch size 1",
        "repetitions": args.repetitions,
        "execution_order": "repetition-major, then mode order, then cold followed by warm",
        "mode_order": args.modes,
        "modes": {name: MODES[name] for name in args.modes},
        "hardware": hardware,
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    runs: list[dict[str, Any]] = []
    all_spans: list[dict[str, Any]] = []
    script = Path(__file__).resolve()
    for repetition in range(1, args.repetitions + 1):
        for mode in args.modes:
            mode_config = MODES[mode]
            inductor_cache = cache_root / mode
            if inductor_cache.exists():
                shutil.rmtree(inductor_cache)
            inductor_cache.mkdir(parents=True)
            for cache_state in ("cold", "warm"):
                env = os.environ.copy()
                env["PYTHONUNBUFFERED"] = "1"
                env["TORCHINDUCTOR_CACHE_DIR"] = str(inductor_cache)
                # Makes repeated config resolution visible in the raw logs. The
                # controller counts the messages but does not call them waste.
                env["TRANSFORMERS_VERBOSITY"] = "info"
                if mode_config["disable_torch_compile"]:
                    env["TORCH_COMPILE_DISABLE"] = "1"
                else:
                    # Even TORCH_COMPILE_DISABLE=0 disabled compilation in some
                    # PyTorch versions; absence is the only safe enabled value.
                    env.pop("TORCH_COMPILE_DISABLE", None)
                stem = f"{mode}-r{repetition:02d}-{cache_state}"
                command = [
                    sys.executable,
                    str(script),
                    "--_child",
                    "--model",
                    args.model,
                    "--deploy-config",
                    str(deploy_config),
                    "--stage-overrides",
                    json.dumps(mode_config["stage_overrides"], separators=(",", ":")),
                    "--requests",
                    str(args.requests),
                    "--max-new-tokens",
                    str(args.max_new_tokens),
                    "--stage-init-timeout",
                    str(args.stage_init_timeout),
                ]
                if args.parallel_stage_init:
                    command.append("--parallel-stage-init")
                print(
                    f"\n=== repetition {repetition:02d}/{args.repetitions:02d}: "
                    f"{mode} / {cache_state} ===",
                    flush=True,
                )
                run = measured_launch(
                    command=command,
                    env=env,
                    cwd=repo,
                    log_path=output_dir / f"{stem}.log",
                    events_path=output_dir / f"{stem}.events.jsonl",
                    mode=mode,
                    repetition=repetition,
                    cache_state=cache_state,
                    timeout_s=args.timeout,
                )
                runs.append(run)
                events = load_events(Path(run["events"]))
                run["repeat_indicators"] = summarize_events(events)
                all_spans.extend(derive_spans(run, events))
                write_summary(output_dir, metadata, runs, all_spans)
                if run["status"] != "ok":
                    if not args.keep_going:
                        print(f"Launch failed; partial results: {output_dir / 'summary.md'}", file=sys.stderr)
                        return 1
                    if cache_state == "cold":
                        print(
                            "Cold launch failed; skipping its warm partner and continuing: "
                            f"{output_dir / 'summary.md'}",
                            file=sys.stderr,
                        )
                        break
    print(f"\nSummary: {output_dir / 'summary.md'}")
    print(f"Machine-readable: {output_dir / 'summary.json'}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
