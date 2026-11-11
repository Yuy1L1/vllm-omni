#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

# Run the validated Qwen3-TTS default/no-CG/eager startup matrix.
#
# Every measured cold -> warm pair gets a brand-new cache root:
#   cold: empty vLLM + FlashInfer + TorchInductor caches
#   warm: reuses exactly the caches populated by the preceding cold launch
#
# Hugging Face model files, OS page cache, and the CUDA driver cache are not
# cleared. The default is one repetition, producing the six-row sanity table.
# Set REPETITIONS=10 only when ten fully cold/warm pairs per mode are desired.

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/../.." && pwd)
STARTUP_ENV=${STARTUP_ENV:-${CONDA_PREFIX:-}}
RESULTS_BASE=${RESULTS_BASE:-$REPO_ROOT/results}
CACHE_BASE=${CACHE_BASE:-${TMPDIR:-/tmp}/vllm-omni-startup-caches}
REPETITIONS=${REPETITIONS:-1}
KEEP_STARTUP_CACHES=${KEEP_STARTUP_CACHES:-0}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
STARTUP_HF_OFFLINE=${STARTUP_HF_OFFLINE:-1}

if [[ -z "$STARTUP_ENV" ]]; then
    echo "Activate the benchmark conda env or set STARTUP_ENV to its absolute path." >&2
    exit 2
fi
if [[ ! -x "$STARTUP_ENV/bin/python" ]]; then
    echo "Python is not executable: $STARTUP_ENV/bin/python" >&2
    exit 2
fi

PYTHON_SITE_PACKAGES=$(
    "$STARTUP_ENV/bin/python" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])'
)
CUDA_PY_ROOT=${CUDA_PY_ROOT:-$PYTHON_SITE_PACKAGES/nvidia/cu13}
CUSPARSELT_INCLUDE=${CUSPARSELT_INCLUDE:-$PYTHON_SITE_PACKAGES/nvidia/cusparselt/include}
CUDA_STUB_DIR=${CUDA_STUB_DIR:-$STARTUP_ENV/targets/x86_64-linux/lib/stubs}

export STARTUP_ENV
export CUDA_VISIBLE_DEVICES
# The Qwen snapshot and remote-code module are already cached locally. Remote
# metadata checks add uncontrolled startup time and can hang before workers
# are spawned, so startup measurements must not access the network.
export HF_HUB_OFFLINE="$STARTUP_HF_OFFLINE"
export TRANSFORMERS_OFFLINE="$STARTUP_HF_OFFLINE"
export HF_HUB_DISABLE_TELEMETRY=1
export PATH="$STARTUP_ENV/bin:$PATH"
export CPATH="$CUDA_PY_ROOT/include:$CUSPARSELT_INCLUDE${CPATH:+:$CPATH}"
export LIBRARY_PATH="$STARTUP_ENV/lib:$CUDA_PY_ROOT/lib:$CUDA_STUB_DIR${LIBRARY_PATH:+:$LIBRARY_PATH}"
export LD_LIBRARY_PATH="$STARTUP_ENV/lib:$CUDA_PY_ROOT/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export FLASHINFER_EXTRA_LDFLAGS="-L$STARTUP_ENV/lib -L$CUDA_PY_ROOT/lib -L$CUDA_STUB_DIR"

if [[ ! "$REPETITIONS" =~ ^[1-9][0-9]*$ ]]; then
    echo "REPETITIONS must be a positive integer, got: $REPETITIONS" >&2
    exit 2
fi
if [[ "$KEEP_STARTUP_CACHES" != 0 && "$KEEP_STARTUP_CACHES" != 1 ]]; then
    echo "KEEP_STARTUP_CACHES must be 0 or 1, got: $KEEP_STARTUP_CACHES" >&2
    exit 2
fi
if [[ "$STARTUP_HF_OFFLINE" != 0 && "$STARTUP_HF_OFFLINE" != 1 ]]; then
    echo "STARTUP_HF_OFFLINE must be 0 or 1, got: $STARTUP_HF_OFFLINE" >&2
    exit 2
fi

cd "$REPO_ROOT"

RESULT_TAG=$(date +%Y%m%d-%H%M%S)
BATCH_RESULT_DIR="$RESULTS_BASE/qwen3-tts-matrix-b300-$RESULT_TAG"
RUN_CACHE_BASE="$CACHE_BASE/qwen3-tts-matrix-b300-$RESULT_TAG"
mkdir -p "$BATCH_RESULT_DIR" "$RUN_CACHE_BASE"

MANIFEST="$BATCH_RESULT_DIR/manifest.tsv"
printf 'repetition\tmode\tstatus\tresult_dir\tcache_root\n' >"$MANIFEST"

current_repetition=0
current_mode=not-started
current_result=not-started
current_cache=not-started
on_error() {
    status=$?
    printf '%s\t%s\tfailed\t%s\t%s\n' \
        "$current_repetition" "$current_mode" "$current_result" "$current_cache" >>"$MANIFEST"
    echo "FAILED: repetition=$current_repetition mode=$current_mode" >&2
    echo "Partial results: $BATCH_RESULT_DIR" >&2
    echo "Failure cache preserved: $current_cache" >&2
    exit "$status"
}
trap on_error ERR

echo "Results: $BATCH_RESULT_DIR"
echo "Temporary startup caches: $RUN_CACHE_BASE"
echo "GPU: physical $CUDA_VISIBLE_DEVICES (logical device 0 inside each child)"
echo "Hugging Face offline mode: $STARTUP_HF_OFFLINE"
echo "Plan: $REPETITIONS repetition(s) x 3 modes x cold/warm"

for ((repetition = 1; repetition <= REPETITIONS; repetition++)); do
    for mode in default no-cg eager; do
        current_repetition=$repetition
        current_mode=$mode
        repetition_label=$(printf 'r%02d' "$repetition")
        current_result="$BATCH_RESULT_DIR/$repetition_label-$mode"
        current_cache="$RUN_CACHE_BASE/$repetition_label-$mode"

        # RUN_CACHE_BASE contains a timestamp and was just created by this
        # script. Each mode/repetition child starts from a distinct empty path.
        if [[ -e "$current_cache" ]]; then
            echo "Refusing to reuse non-empty cold cache path: $current_cache" >&2
            exit 2
        fi
        mkdir -p "$current_cache/vllm" "$current_cache/flashinfer"
        export STARTUP_CACHE_ROOT="$current_cache"
        export VLLM_CACHE_ROOT="$current_cache/vllm"
        export FLASHINFER_WORKSPACE_BASE="$current_cache/flashinfer"

        echo
        echo "=== $repetition_label/$REPETITIONS: $mode cold -> warm ==="
        echo "Fresh vLLM cache: $VLLM_CACHE_ROOT"
        echo "Fresh FlashInfer workspace: $FLASHINFER_WORKSPACE_BASE"

        "$STARTUP_ENV/bin/python" benchmarks/startup/qwen3_tts_startup.py \
            --modes "$mode" \
            --requests 3 \
            --stage-init-timeout 900 \
            --timeout 2400 \
            --output-dir "$current_result"

        printf '%s\t%s\tok\t%s\t%s\n' \
            "$repetition" "$mode" "$current_result" "$current_cache" >>"$MANIFEST"
        if [[ "$KEEP_STARTUP_CACHES" == 0 ]]; then
            rm -rf -- "$current_cache"
            echo "Removed completed pair cache: $current_cache"
        fi
    done
done

trap - ERR

"$STARTUP_ENV/bin/python" - "$BATCH_RESULT_DIR" "$REPETITIONS" <<'PY'
import csv
import json
import statistics
import sys
from pathlib import Path

result_dir = Path(sys.argv[1])
repetitions = int(sys.argv[2])
rows = []
for repetition in range(1, repetitions + 1):
    for mode in ("default", "no-cg", "eager"):
        summary = json.loads((result_dir / f"r{repetition:02d}-{mode}" / "summary.json").read_text())
        for run in summary["runs"]:
            child = run["child_result"]
            request_times = child.get("request_times_s") or []
            rows.append(
                {
                    "repetition": repetition,
                    "mode": mode,
                    "cache": run["cache_state"],
                    "startup_s": float(run["startup_s"]),
                    "first_request_s": float(request_times[0]) if request_times else None,
                    "steady_median_s": child.get("steady_median_s"),
                }
            )

with (result_dir / "startup_matrix.csv").open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)

lines = [
    "# Qwen3-TTS startup matrix",
    "",
    (
        "Cold uses empty vLLM, FlashInfer, and TorchInductor caches. Warm immediately follows cold "
        "and reuses those caches. HF model files, OS page cache, and CUDA driver cache are inherited."
    ),
    "",
    "| Mode | Cache | N | Startup median | First request median | Steady median |",
    "| --- | --- | ---: | ---: | ---: | ---: |",
]
for mode in ("default", "no-cg", "eager"):
    for cache_state in ("cold", "warm"):
        group = [row for row in rows if row["mode"] == mode and row["cache"] == cache_state]
        startup = statistics.median(row["startup_s"] for row in group)
        first = statistics.median(row["first_request_s"] for row in group)
        steady = statistics.median(row["steady_median_s"] for row in group)
        lines.append(
            f"| {mode} | {cache_state} | {len(group)} | {startup:.3f}s | {first:.3f}s | {steady:.3f}s |"
        )
(result_dir / "summary_table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
PY

if [[ "$KEEP_STARTUP_CACHES" == 0 ]]; then
    rm -rf -- "$RUN_CACHE_BASE"
    echo "Removed temporary startup caches: $RUN_CACHE_BASE"
else
    echo "Kept startup caches: $RUN_CACHE_BASE"
fi

echo
cat "$BATCH_RESULT_DIR/summary_table.md"
echo
echo "Completed successfully: $BATCH_RESULT_DIR"
