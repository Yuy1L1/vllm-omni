# Qwen3-TTS startup benchmark

`qwen3_tts_startup.py` records controller-monotonic timestamps for startup and
fixed inference requests. `run_qwen3_tts_startup_matrix.sh` runs the
default/no-CG/eager matrix. Every measured pair uses a new cache root: cold
starts with empty vLLM, FlashInfer, and TorchInductor caches; warm immediately
reuses the caches populated by cold.

Hugging Face model files, the OS page cache, and the CUDA driver cache are not
cleared. Different stages may overlap, so their reported durations must not be
summed as startup wall time.

## New-machine setup

Use a vLLM-Omni checkout and environment compatible with the measured commit.
Activate the environment before invoking the wrapper:

```bash
cd /path/to/vllm-omni
conda activate /path/to/vllm-omni-env
export STARTUP_ENV="$CONDA_PREFIX"
```

Download the model before measurement. Do not include network access in a cold
startup launch:

```bash
python -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice')"
```

Choose an idle physical GPU and paths with sufficient local space:

```bash
export CUDA_VISIBLE_DEVICES=0
export RESULTS_BASE=/path/to/results
export CACHE_BASE=/path/to/fast-temporary-cache
./benchmarks/startup/run_qwen3_tts_startup_matrix.sh
```

The wrapper defaults to `STARTUP_HF_OFFLINE=1`. Set it to `0` only for a
deliberate connectivity smoke test, not a reported startup measurement.

One invocation runs one cold/warm pair for each mode and writes
`summary_table.md`, `startup_matrix.csv`, and `manifest.tsv`. To collect ten
fully cold/warm pairs per mode:

```bash
REPETITIONS=10 ./benchmarks/startup/run_qwen3_tts_startup_matrix.sh
```

Completed pair caches are deleted by default. Use `KEEP_STARTUP_CACHES=1` to
retain them. A failed pair preserves its cache and partial logs for diagnosis.

## Plot retained single-run results

`plot_qwen3_tts_single_run.py` produces a stacked startup breakdown from three
single-mode result directories. Run `--help` for the required source paths.
