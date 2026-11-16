# Startup spans

## Next experiment: default / no-CG / eager

Run one cold/warm pair per mode first (six launches). Keep the deploy config,
sampler backend and serial stage initialization consistent across modes. Rank
the largest exclusive residuals by stage and parent before adding more hooks.
Use multiple repetitions later to assess variability; one pair is diagnostic.

Run from the modified checkout on GH200:

```bash
cd /home/lyy/yuyi/vllm-omni
unset VLLM_OMNI_STARTUP_TRACE_DIR
STARTUP_ENV=/home/lyy/vllm-omni-startup/envs/vllm-omni-033 \
RESULTS_BASE=/home/lyy/vllm-omni-startup/results/gh200 \
CACHE_BASE=/home/lyy/vllm-omni-startup/results/gh200/startup-caches \
MODEL_PATH=/home/lyy/models/Qwen3-TTS-12Hz-1.7B-CustomVoice \
GPU_TAG=gh200 EXPECTED_GPU=GH200 \
MODES='default no-cg eager' REPETITIONS=1 REQUESTS=0 \
KEEP_STARTUP_CACHES=1 \
bash benchmarks/startup/run_qwen3_tts_startup_matrix.sh
```

The wrapper now enables tracing automatically under each timestamped batch's
`startup-spans/` directory. Each outer repetition/mode gets its own directory,
then the controller separates cold and warm launches. The command above stores
raw timestamps, logs, event files, summaries and benchmark-controlled caches
under `/home/lyy/vllm-omni-startup/results/gh200`; it does not use `/tmp` for
these artifacts. `KEEP_STARTUP_CACHES=1` retains the compiler caches for inspection.
The wrapper's cache default is also now `$RESULTS_BASE/startup-caches`.
External libraries may still use their own temporary scratch files.

For example, the batch printed as `Results: ...` contains:

```text
qwen3-tts-matrix-gh200-<timestamp>/
  run_settings.txt
  r01-default/
    default-r01-cold.events.jsonl
    default-r01-warm.events.jsonl
    ... logs and summaries ...
  r01-no-cg/
  r01-eager/
  startup-spans/
    r01-default/
      default-r01-cold/<hostname>-<pid>.jsonl
      default-r01-warm/<hostname>-<pid>.jsonl
    r01-no-cg/
    r01-eager/
```

Every engine/worker process inherits the trace directory and writes its own
`hostname-pid.jsonl`. If overriding `VLLM_OMNI_STARTUP_TRACE_DIR`, choose a fresh
absolute directory under `results/gh200`; the wrapper still adds repetition/mode
subdirectories. When invoking the Python benchmark directly instead of the
wrapper, tracing remains opt-in via that environment variable.
The modified checkout must be imported in every process.

Cold means fresh benchmark-controlled vLLM, FlashInfer and TorchInductor caches;
the immediately following warm launch reuses them. HF model files, OS page cache
and CUDA driver cache are not cleared. `no-cg` also sets Stage 1 `enforce_eager`
to disable its inner Code2Wav graph, so it is not a pure CUDA Graph-only ablation.
Keep parallel-stage-init as a separate experiment after this baseline matrix.

## Plot fine-grained timelines

Use the new process-local spans rather than the older log-derived Gantt script:

```bash
cd /home/lyy/yuyi/vllm-omni
BATCH_DIR='/home/lyy/vllm-omni-startup/results/gh200/qwen3-tts-matrix-gh200-<timestamp>'
PYTHON=/home/lyy/vllm-omni-startup/envs/vllm-omni-033/bin/python
"$PYTHON" benchmarks/startup/plot_startup_spans.py "$BATCH_DIR"
```

This exports all six launches and writes `timeline_<mode>_r01_<cache>.png`
and `residuals_<mode>_r01_<cache>.csv` under
`$BATCH_DIR/analysis-fine-grained/`. The plots group Frontend and each Stage/PID into separate sections. Headers show
OS parent PID; the indented tree follows local `parent_span_id` edges. Parents
use thicker/darker bars, children use thinner/lighter bars, and major phases
have pale background blocks. Only important spans show inclusive (`I`) and
exclusive (`E`) numbers; all durations remain available in CSV/Chrome trace. Residual tables are sorted by
exclusive time; interpret each value within its parent context.

Zoom into one mode/cache and time interval, measured from controller launch:

```bash
"$PYTHON" benchmarks/startup/plot_startup_spans.py "$BATCH_DIR" \
  --mode default --cache cold --start 100 --end 260
```

Zoomed images get a time-window suffix. The CSV and Chrome trace exports below
are also generated automatically beside the original timestamp files.

## Export timestamps

Export a single launch for the Gantt plot and a Chrome trace viewer:

```bash
# Set this to the exact Results directory printed by the completed run.
BATCH_DIR='/home/lyy/vllm-omni-startup/results/gh200/qwen3-tts-matrix-gh200-<timestamp>'
for mode in default no-cg eager; do
  for state in cold warm; do
    python3 benchmarks/startup/export_startup_spans.py \
      "$BATCH_DIR/startup-spans/r01-$mode/$mode-r01-$state" \
      --events "$BATCH_DIR/r01-$mode/$mode-r01-$state.events.jsonl"
  done
done
```

The output is `spans.csv` and `trace.json`. `--events` uses the benchmark's
original monotonic launch timestamp as the timeline origin and removes spans
ending after the controller receives `startup_ready`. Without it, the origin
is the earliest span and inference-time runner calls are also included.

## Measurement boundaries

| Boundary | What it measures |
| --- | --- |
| `device_lock_wait` | Stage helper's acquisition path, including bookkeeping; no held phase work |
| `phase_lock_wait` | SH/EX acquisition on the device slice, with mode and device IDs |
| `executor_init` | Executor creation, worker startup, device setup and model loading |
| `model_load` | Worker model construction and weight loading inside executor initialization |
| `memory_profiling` / `profile_run` | Driver dispatch/wait, worker budget measurement, and actual runner profiling |
| `kv_cache_init` / `kv_cache_config` | Core initialization, exact upstream configuration helpers, driver allocation dispatch, worker GPU allocation |
| `compile_warmup` | Driver dispatch and worker compile/warmup, including auxiliary captures |
| `model_warmup` / `graph_capture` | Runner dummy runs and outer graph capture invoked by upstream warmup |

These are eight boundary groups with nested spans. Configuration helper wrapping
is restricted to the engine child and enabled trace sessions. Missing upstream
helpers remain uninstrumented residual; the initialization implementation is
not copied. Model-specific inner capture and JIT can still remain inside dummy
runs or model loading. Existing compilation log durations remain separate
observations: a dummy run may itself trigger compilation, so its duration is
not necessarily pure warmup. Do not subtract a rounded compilation log duration
as though it were an exact nested interval.

Each completed span includes PID, PPID, native TID, hostname, stage/replica/rank
where available, unique span/parent IDs, monotonic start/end nanoseconds,
wall-clock start nanoseconds, status, inclusive seconds and exclusive seconds.
Exceptional exits are recorded and propagated. A killed process may leave no
record for an unfinished span. Identity fields may be null before config setup.

Inclusive is elapsed wall time. Exclusive subtracts the union of explicitly
nested child intervals on the same thread in the same process. It includes
untraced work and waits. Remote worker spans are separate roots; never add
driver and worker inclusive times. Do not sum overlapping stages as startup
wall time. Monotonic clocks align processes on one host, not multiple hosts.
No new GPU synchronization is added; these are host intervals, not kernel times.
Tracing adds file I/O and Python overhead, so compare enabled/disabled runs
before making performance claims. Rank unexplained exclusive durations within
their parent context, then instrument only the largest remaining intervals.

## Profile_run detail instrumentation

This extension is applied to the inspected vLLM 0.33.0 / FlashInfer installation
in `/home/lyy/vllm-omni-startup/envs/vllm-omni-033`, plus the modified Omni checkout.
The reviewed patch, original-file backups and application hashes are retained
under `/home/lyy/vllm-omni-startup/results/gh200/profile-run-proposal/`.
Reinstalling vLLM or FlashInfer can overwrite these local patches. Check source
hashes before reapplying; do not apply to a different version by fuzzy match.
Only the existing traced profile_run scope activates the new nested spans.
Dummy scheduler construction, Omni input preparation/forward, logits, sampling
passes, existing synchronization and cleanup are recorded. FlashInfer sampling
Ninja invocation and dynamic loading are separate nested spans. Ninja may no-op
on a warm cache; artifact_missing is the already-computed pre-build existence
flag, not proof of a cache hit or a compiler invocation. No extra synchronization,
cache lookup, warmup, sampling pass or compilation is requested.

Export with the matching --events JSONL. Every exported span gets mode,
repetition and cache_state, and the Chrome trace category includes mode/cache.
Raw JSONL remains in the existing per-launch result directories. Export format
and plotting commands remain compatible. Input/forward spans are host intervals;
queued GPU work may complete in profile_synchronize. The two profile_sampling
spans distinguish default and warmup_params passes. These startup-only scopes
emit nothing during ordinary inference or later graph-capture warmups.

## Saving code and experiment results

See [experiments/README.md](experiments/README.md) for the prepared snapshot and
the two-commit workflow. Keep installed vLLM/FlashInfer changes as the patches
in `patches/`; committing the Omni checkout alone cannot capture site-packages.
