#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
#
# Qwen3-TTS startup attribution matrix, portable across B300 / GH200.
# Each repetition/mode executes cold then warm in separate processes.
# The cold launch starts with NEW benchmark-owned vLLM, FlashInfer, and
# TorchInductor cache directories; the warm launch reuses them.
# This does NOT flush Hugging Face models, OS page cache, or CUDA driver cache.
#
# Examples:
#   GPU_TAG=gh200 MODEL_PATH=/ssd1/lyy/models/Qwen3-TTS-12Hz-1.7B-CustomVoice \
#   VLLM_USE_FLASHINFER_SAMPLER=0 MODES=default \
#   bash benchmarks/startup/run_qwen3_tts_startup_matrix.sh
#
#   GPU_TAG=b300 REPETITIONS=3 MODES='default no-cg eager' \
#   bash benchmarks/startup/run_qwen3_tts_startup_matrix.sh

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/../.." && pwd)
STARTUP_ENV=${STARTUP_ENV:-${CONDA_PREFIX:-}}
RESULTS_BASE=${RESULTS_BASE:-$REPO_ROOT/results}
CACHE_BASE=${CACHE_BASE:-${TMPDIR:-/tmp}/vllm-omni-startup-caches}
GPU_TAG=${GPU_TAG:-$(hostname -s)}
EXPECTED_GPU=${EXPECTED_GPU:-}  # Optional, e.g. GH200 or B300
MODEL_PATH=${MODEL_PATH:-Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice}
DEPLOY_CONFIG=${DEPLOY_CONFIG:-vllm_omni/deploy/qwen3_tts.yaml}
MODES=${MODES:-'default no-cg eager'}
REPETITIONS=${REPETITIONS:-1}
REQUESTS=${REQUESTS:-3}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-64}
STAGE_INIT_TIMEOUT=${STAGE_INIT_TIMEOUT:-900}
LAUNCH_TIMEOUT=${LAUNCH_TIMEOUT:-2400}
KEEP_STARTUP_CACHES=${KEEP_STARTUP_CACHES:-0}
STARTUP_HF_OFFLINE=${STARTUP_HF_OFFLINE:-1}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

if [[ -z "$STARTUP_ENV" || ! -x "$STARTUP_ENV/bin/python" ]]; then
    echo "Activate the benchmark Conda environment or set STARTUP_ENV to the absolute env path." >&2
    exit 2
fi
if [[ ! "$REPETITIONS" =~ ^[1-9][0-9]*$ ]]; then
    echo "REPETITIONS must be a positive integer: $REPETITIONS" >&2
    exit 2
fi
if [[ ! "$REQUESTS" =~ ^[0-9]+$ || ! "$MAX_NEW_TOKENS" =~ ^[1-9][0-9]*$ ]]; then
    echo "REQUESTS must be nonnegative and MAX_NEW_TOKENS must be positive." >&2
    exit 2
fi
for toggle in KEEP_STARTUP_CACHES STARTUP_HF_OFFLINE; do
    if [[ "${!toggle}" != 0 && "${!toggle}" != 1 ]]; then
        echo "$toggle must be 0 or 1 (got ${!toggle})" >&2
        exit 2
    fi
done
if [[ ! "$GPU_TAG" =~ ^[A-Za-z0-9_.-]+$ ]]; then
    echo "GPU_TAG may only contain letters, numbers, _, ., and -." >&2
    exit 2
fi
read -r -a MODE_ARRAY <<< "$MODES"
if (( ${#MODE_ARRAY[@]} == 0 )); then
    echo "MODES may not be empty." >&2
    exit 2
fi
for mode in "${MODE_ARRAY[@]}"; do
    case "$mode" in
        default|no-cg|eager) ;;
        *) echo "Unknown mode: $mode" >&2; exit 2 ;;
    esac
done

# Only append CUDA include/library paths if they exist on this platform.
# GH200's CPU is aarch64; a hard-coded x86_64 CUDA stub path is wrong.
PYTHON_SITE_PACKAGES=$("$STARTUP_ENV/bin/python" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')
CUDA_PY_ROOT=${CUDA_PY_ROOT:-$PYTHON_SITE_PACKAGES/nvidia/cu13}
CUSPARSELT_INCLUDE=${CUSPARSELT_INCLUDE:-$PYTHON_SITE_PACKAGES/nvidia/cusparselt/include}
if [[ -z "${CUDA_STUB_DIR:-}" ]]; then
    for candidate in \
        "$STARTUP_ENV/targets/$( [[ $(uname -m) == aarch64 ]] && echo sbsa-linux || echo x86_64-linux )/lib/stubs" \
        "$STARTUP_ENV/targets/aarch64-linux/lib/stubs" \
        "$CUDA_PY_ROOT/lib/stubs"; do
        if [[ -d "$candidate" ]]; then
            CUDA_STUB_DIR=$candidate
            break
        fi
    done
fi
prepend_directory_to_env() {
    local var=$1 dir=$2
    [[ -n "$dir" && -d "$dir" ]] || return 0
    export "$var"="$dir${!var:+:${!var}}"
}

export STARTUP_ENV CUDA_VISIBLE_DEVICES
export PATH="$STARTUP_ENV/bin:$PATH"
export HF_HUB_OFFLINE="$STARTUP_HF_OFFLINE"
export TRANSFORMERS_OFFLINE="$STARTUP_HF_OFFLINE"
export HF_HUB_DISABLE_TELEMETRY=1

# prepend_directory_to_env CPATH "$CUDA_PY_ROOT/include"
# prepend_directory_to_env CPATH "$CUSPARSELT_INCLUDE"
unset CPATH
prepend_directory_to_env LIBRARY_PATH "$STARTUP_ENV/lib"
prepend_directory_to_env LIBRARY_PATH "$CUDA_PY_ROOT/lib"
prepend_directory_to_env LD_LIBRARY_PATH "$STARTUP_ENV/lib"
prepend_directory_to_env LD_LIBRARY_PATH "$CUDA_PY_ROOT/lib"

# Additional search paths for the optional FlashInfer sampling JIT build.
# We do not force-enable or disable the FlashInfer sampler here.
FLASHINFER_EXTRA_LDFLAGS=${FLASHINFER_EXTRA_LDFLAGS:-}
for libdir in "$STARTUP_ENV/lib" "$CUDA_PY_ROOT/lib" "${CUDA_STUB_DIR:-}"; do
    if [[ -n "$libdir" && -d "$libdir" ]]; then
        prepend_directory_to_env LIBRARY_PATH "$libdir"
        FLASHINFER_EXTRA_LDFLAGS="${FLASHINFER_EXTRA_LDFLAGS:+$FLASHINFER_EXTRA_LDFLAGS }-L$libdir"
    fi
done
export FLASHINFER_EXTRA_LDFLAGS

cd "$REPO_ROOT"
RESULT_TAG=$(date +%Y%m%d-%H%M%S-%N)
BATCH_RESULT_DIR="$RESULTS_BASE/qwen3-tts-matrix-${GPU_TAG}-$RESULT_TAG"
RUN_CACHE_BASE="$CACHE_BASE/qwen3-tts-matrix-${GPU_TAG}-$RESULT_TAG"
mkdir -p "$BATCH_RESULT_DIR" "$RUN_CACHE_BASE"

MANIFEST="$BATCH_RESULT_DIR/manifest.tsv"
printf 'repetition\tmode\tstatus\tresult_dir\tcache_root\n' > "$MANIFEST"
cat > "$BATCH_RESULT_DIR/run_settings.txt" <<EOF
hostname=$(hostname -s)
gpu_tag=$GPU_TAG
expected_gpu=$EXPECTED_GPU
model=$MODEL_PATH
deploy_config=$DEPLOY_CONFIG
python_env=$STARTUP_ENV
cuda_visible_devices=$CUDA_VISIBLE_DEVICES
flashinfer_sampler=${VLLM_USE_FLASHINFER_SAMPLER:-unset (vLLM default)}
hf_offline=$STARTUP_HF_OFFLINE
modes=$MODES
repetitions=$REPETITIONS
requests=$REQUESTS
max_new_tokens=$MAX_NEW_TOKENS
stage_init_timeout=$STAGE_INIT_TIMEOUT
launch_timeout=$LAUNCH_TIMEOUT
keep_startup_caches=$KEEP_STARTUP_CACHES
cache_policy=fresh vLLM + FlashInfer + TorchInductor for cold; immediate warm reuses them
out_of_scope_cache=Hugging Face models, OS page cache, CUDA driver cache
warning_no_cg=Stage 1 enforce_eager is set; no-cg is not a pure CUDA Graph-only ablation
EOF

echo "Results: $BATCH_RESULT_DIR"
echo "Startup caches: $RUN_CACHE_BASE"
echo "GPU tag: $GPU_TAG; CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "Model: $MODEL_PATH"
echo "FlashInfer sampler: ${VLLM_USE_FLASHINFER_SAMPLER:-unset (vLLM default)}"
echo "Hugging Face offline mode: $STARTUP_HF_OFFLINE"
echo "Plan: $REPETITIONS repetition(s) x ${#MODE_ARRAY[@]} mode(s) x cold/warm"

current_repetition=0
current_mode=not-started
current_result=not-started
current_cache=not-started
on_error() {
    local status=$?
    trap - ERR
    printf '%s\t%s\tfailed\t%s\t%s\n' \
        "$current_repetition" "$current_mode" "$current_result" "$current_cache" >> "$MANIFEST"
    echo "FAILED (exit code $status): repetition=$current_repetition mode=$current_mode" >&2
    echo "Partial results: $BATCH_RESULT_DIR" >&2
    echo "Failure cache preserved: $current_cache" >&2
    exit "$status"
}
trap on_error ERR

for ((repetition = 1; repetition <= REPETITIONS; repetition++)); do
    for mode in "${MODE_ARRAY[@]}"; do
        current_repetition=$repetition
        current_mode=$mode
        repetition_label=$(printf 'r%02d' "$repetition")
        current_result="$BATCH_RESULT_DIR/$repetition_label-$mode"
        current_cache="$RUN_CACHE_BASE/$repetition_label-$mode"

        if [[ -e "$current_cache" ]]; then
            echo "Refusing to reuse existing cold cache path: $current_cache" >&2
            exit 2
        fi
        mkdir -p "$current_cache/vllm" "$current_cache/flashinfer"
        export STARTUP_CACHE_ROOT="$current_cache"
        export VLLM_CACHE_ROOT="$current_cache/vllm"
        export FLASHINFER_WORKSPACE_BASE="$current_cache/flashinfer"
        # TorchInductor cache is created/managed inside qwen3_tts_startup.py.

        echo
        echo "=== $repetition_label / $REPETITIONS | $mode cold -> warm ==="
        cmd=(
            "$STARTUP_ENV/bin/python" benchmarks/startup/qwen3_tts_startup.py
            --model "$MODEL_PATH"
            --deploy-config "$DEPLOY_CONFIG"
            --modes "$mode"
            --requests "$REQUESTS"
            --max-new-tokens "$MAX_NEW_TOKENS"
            --stage-init-timeout "$STAGE_INIT_TIMEOUT"
            --timeout "$LAUNCH_TIMEOUT"
            --output-dir "$current_result"
        )
        if [[ -n "$EXPECTED_GPU" ]]; then
            cmd+=(--expected-gpu "$EXPECTED_GPU")
        fi
        "${cmd[@]}"

        printf '%s\t%s\tok\t%s\t%s\n' \
            "$repetition" "$mode" "$current_result" "$current_cache" >> "$MANIFEST"
        if [[ "$KEEP_STARTUP_CACHES" == 0 ]]; then
            rm -rf -- "$current_cache"
            echo "Removed completed pair cache: $current_cache"
        fi
    done
done
trap - ERR

# Combine all successful cold/warm pairs into a compact matrix summary.
"$STARTUP_ENV/bin/python" - "$BATCH_RESULT_DIR" "$REPETITIONS" "${MODE_ARRAY[@]}" <<'PY'
import csv
import json
import statistics
import sys
from pathlib import Path

result_dir = Path(sys.argv[1])
repetitions = int(sys.argv[2])
modes = sys.argv[3:]
rows = []
for repetition in range(1, repetitions + 1):
    for mode in modes:
        summary = json.loads((result_dir / f"r{repetition:02d}-{mode}" / "summary.json").read_text())
        for run in summary["runs"]:
            child = run.get("child_result") or {}
            request_times = child.get("request_times_s") or []
            rows.append({
                "repetition": repetition,
                "mode": mode,
                "cache": run["cache_state"],
                "sampler": child.get("flashinfer_sampler_env", "unknown"),
                "startup_s": float(run["startup_s"]),
                "first_request_s": float(request_times[0]) if request_times else None,
                "steady_median_s": child.get("steady_median_s"),
            })

with (result_dir / "startup_matrix.csv").open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)

lines = [
    "# Qwen3-TTS startup matrix", "",
    "Cold: fresh benchmark-owned TorchInductor, vLLM, FlashInfer caches; warm: reuse from previous cold.",
    "HF model files, OS page cache, CUDA driver cache are not reset.",
    "Sampler implementation can affect startup and request latency; see run_settings.txt.",
    "The no-cg mode also modifies Stage 1 eager execution; it is not a pure CG-only ablation.", "",
    "| Mode | Cache | N | Startup median | First request median | Steady median |",
    "| --- | --- | ---: | ---: | ---: | ---: |",
]
for mode in modes:
    for cache_state in ("cold", "warm"):
        group = [r for r in rows if r["mode"] == mode and r["cache"] == cache_state]
        def fmt(key):
            values = [float(r[key]) for r in group if r[key] is not None]
            return f"{statistics.median(values):.3f}s" if values else "n/a"
        lines.append(f"| {mode} | {cache_state} | {len(group)} | {fmt('startup_s')} | {fmt('first_request_s')} | {fmt('steady_median_s')} |")
(result_dir / "summary_table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
PY

if [[ "$KEEP_STARTUP_CACHES" == 0 ]]; then
    rm -rf -- "$RUN_CACHE_BASE"
    echo "Removed temporary startup caches: $RUN_CACHE_BASE"
else
    echo "Preserved startup caches: $RUN_CACHE_BASE"
fi

echo
cat "$BATCH_RESULT_DIR/summary_table.md"
echo
printf 'Completed successfully: %s\n' "$BATCH_RESULT_DIR"
