# Saving startup work without mixing code and generated output

The working branch is `measure_start_up`; `origin` is the personal fork
`Yuy1L1/vllm-omni`, and `upstream` is the vLLM-Omni project. Push this experiment
branch to origin. Keep the existing results directory as the local working store.

The latest complete batch is copied under `gh200/` here (about 4.5 MiB).
It includes raw JSONL, event files/logs, summaries and six hierarchy figures.
No weights, packages or compiler caches are archived. Scoped `.gitignore`
exceptions in this directory allow JSON/JSONL/CSV/log files without changing
repository-wide ignore rules. No `git add -f` is needed.

Use two commits: instrumentation/analysis source first, then the small experiment
snapshot. The external vLLM and FlashInfer patches must be part of the first
commit; otherwise the installed instrumentation cannot be reconstructed.
The snapshot was taken from a dirty working tree, so retain its provenance caveat
in ARCHIVE.md rather than claiming the previous HEAD fully identifies that run.

From the repository root, review and execute:

```bash
git status --short
git diff --stat

# 1. Source, plotting, documentation and installed-package patches.
git add benchmarks/startup/*.py benchmarks/startup/*.sh \
  benchmarks/startup/INSTRUMENTATION.md benchmarks/startup/patches \
  vllm_omni/engine/startup_span.py \
  vllm_omni/engine/stage_engine_core_proc.py \
  vllm_omni/engine/stage_init_utils.py \
  vllm_omni/engine/stage_phase_lock.py \
  vllm_omni/worker/base.py \
  vllm_omni/worker/gpu_ar_worker.py \
  vllm_omni/worker/gpu_generation_worker.py \
  vllm_omni/worker_v2/omni_model_runner.py
git diff --cached --check
git diff --cached --stat
git commit -m "[Misc] Add process-local startup attribution and hierarchy plots"

# 2. Inspect archived logs/metadata (machine names and local paths), then save.
git add benchmarks/startup/experiments
git diff --cached --stat
git commit -m "[Misc] Record GH200 startup instrumentation experiment"

# 3. Publish both commits to the existing experiment branch on your fork.
git push -u origin measure_start_up
```

The agent has prepared files only; it has not staged, committed or pushed them.
For later runs, archive only complete, useful batches and commit each snapshot
separately. Keep the original run_settings, raw timestamps and source hashes.
Avoid force-adding the entire external results tree or committing compiler caches.
