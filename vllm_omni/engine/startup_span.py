# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Opt-in startup wall-time spans; one JSONL file per process.

Set VLLM_OMNI_STARTUP_TRACE_DIR to an absolute, run-specific directory.
Monotonic timestamps are comparable between processes on the same host only.
No CUDA synchronization is introduced. Exclusive time excludes only explicit
children on the same thread, not remote workers or uninstrumented operations.
"""

from __future__ import annotations

import functools
import inspect
import json
import os
import socket
import threading
import time
import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path

_profile_context: ContextVar[dict | None] = ContextVar("startup_profile_context", default=None)

_stack: ContextVar[tuple] = ContextVar("startup_spans", default=())


@contextmanager
def startup_span(name, **metadata):
    directory = os.environ.get("VLLM_OMNI_STARTUP_TRACE_DIR")
    if not directory:
        yield
        return
    pid, tid = os.getpid(), threading.get_native_id()
    stack = tuple(s for s in _stack.get() if s["pid"] == pid and s["tid"] == tid)
    span = {"span_id": uuid.uuid4().hex, "pid": pid, "tid": tid, "children": []}
    token = _stack.set((*stack, span))
    wall_start_ns = time.time_ns()
    start_ns = time.perf_counter_ns()
    status = "ok"
    try:
        yield
    except BaseException:
        status = "error"
        raise
    finally:
        end_ns = time.perf_counter_ns()
        _stack.reset(token)
        if stack:
            stack[-1]["children"].append((start_ns, end_ns))
        # Union avoids double subtraction when child intervals overlap.
        covered, edge = 0, start_ns
        for begin, end in sorted(span["children"]):
            begin, end = max(start_ns, begin), min(end_ns, end)
            covered += max(0, end - max(edge, begin))
            edge = max(edge, end)
        record = {
            **metadata,
            "name": name,
            "span_id": span["span_id"],
            "parent_span_id": stack[-1]["span_id"] if stack else None,
            "pid": pid,
            "ppid": os.getppid(),
            "tid": tid,
            "hostname": socket.gethostname(),
            "start_ns": start_ns,
            "end_ns": end_ns,
            "wall_start_ns": wall_start_ns,
            "duration_s": (end_ns - start_ns) / 1e9,
            "inclusive_s": (end_ns - start_ns) / 1e9,
            "exclusive_s": max(0, end_ns - start_ns - covered) / 1e9,
            "status": status,
            "clock": "perf_counter_ns",
        }
        try:
            path = Path(directory)
            path.mkdir(parents=True, exist_ok=True)
            with (path / f"{socket.gethostname()}-{pid}.jsonl").open("a") as stream:
                stream.write(json.dumps(record) + "\n")
        except OSError:
            # Trace storage failure must not replace an engine exception.
            import warnings

            warnings.warn("Could not write startup span", RuntimeWarning, stacklevel=2)


def trace_startup(name, **metadata):
    """Decorate a synchronous startup boundary without changing its signature."""
    def decorate(function):
        signature = inspect.signature(function)

        @functools.wraps(function)
        def wrapped(*args, **kwargs):
            if not os.environ.get("VLLM_OMNI_STARTUP_TRACE_DIR"):
                return function(*args, **kwargs)
            owner = args[0] if args else None
            arguments = signature.bind_partial(*args, **kwargs).arguments
            config = getattr(owner, "vllm_config", None) or kwargs.get("vllm_config")
            model_config = getattr(config, "model_config", None)
            fields = {
                "stage": arguments.get(
                    "stage_id", kwargs.get("omni_stage_id", getattr(model_config, "stage_id", None))
                ),
                "replica": kwargs.get("omni_replica_id", os.environ.get("VLLM_OMNI_REPLICA_ID")),
                "rank": getattr(owner, "rank", None),
                "operation": function.__qualname__,
                **metadata,
            }
            with startup_span(name, **fields):
                if name != "profile_run":
                    return function(*args, **kwargs)
                token = _profile_context.set({key: fields.get(key) for key in ("stage", "replica", "rank")})
                try:
                    return function(*args, **kwargs)
                finally:
                    _profile_context.reset(token)
        return wrapped
    return decorate


def traced_executor(executor_class, stage, replica):
    """Wrap driver boundaries, including dispatch/wait for remote workers."""
    class StartupExecutor(executor_class):
        @trace_startup("executor_init", stage=stage, replica=replica)
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

        @trace_startup("memory_profiling", stage=stage, replica=replica, scope="driver")
        def determine_available_memory(self):
            return super().determine_available_memory()

        @trace_startup("kv_cache_init", stage=stage, replica=replica, scope="driver")
        def initialize_from_config(self, *args, **kwargs):
            return super().initialize_from_config(*args, **kwargs)

        @trace_startup("compile_warmup", stage=stage, replica=replica, scope="driver")
        def compile_or_warm_up_model(self):
            return super().compile_or_warm_up_model()
    return StartupExecutor


def trace_runner_startup(runner):
    """Trace actual runner work invoked by upstream worker warmup methods."""
    if not os.environ.get("VLLM_OMNI_STARTUP_TRACE_DIR"):
        return
    stage = getattr(getattr(runner, "model_config", None), "stage_id", None)
    for method, name in (
        ("profile_run", "profile_run"),
        ("_dummy_run", "model_warmup"),
        ("capture_model", "graph_capture"),
    ):
        original = getattr(runner, method, None)
        if original is not None:
            setattr(runner, method, trace_startup(name, stage=stage, scope="runner")(original))


def trace_core_kv_helpers(core_module, stage, replica):
    """Instrument upstream config helpers in this dedicated engine child only.

    Wrapping the imported callables avoids copying the version-specific KV
    initialization implementation. Missing helpers remain visible as residual.
    """
    for name in ("get_kv_cache_configs", "generate_scheduler_kv_cache_config"):
        original = getattr(core_module, name, None)
        if original is not None and not getattr(original, "_startup_traced", False):
            wrapped = trace_startup("kv_cache_config", stage=stage, replica=replica)(original)
            wrapped._startup_traced = True
            setattr(core_module, name, wrapped)


@contextmanager
def profile_detail_span(name, **metadata):
    """Only emit detailed spans inside the existing traced profile_run scope."""
    context = _profile_context.get()
    if context is None or not os.environ.get("VLLM_OMNI_STARTUP_TRACE_DIR"):
        yield
        return
    with startup_span(name, **{**context, **metadata}):
        yield


def startup_mark(event, **metadata):
    """Cross-process bootstrap milestones, stored separately from interval spans."""
    directory = os.environ.get("VLLM_OMNI_STARTUP_TRACE_DIR")
    if not directory:
        return
    timestamp_ns = time.perf_counter_ns()
    record = {**metadata, "event": event, "timestamp_ns": timestamp_ns,
              "pid": os.getpid(), "ppid": os.getppid(),
              "tid": threading.get_native_id(), "hostname": socket.gethostname(),
              "clock": "perf_counter_ns"}
    path = Path(directory) / "bootstrap"
    path.mkdir(parents=True, exist_ok=True)
    with (path / f"{socket.gethostname()}-{os.getpid()}.jsonl").open("a") as stream:
        stream.write(json.dumps(record) + "\n")
