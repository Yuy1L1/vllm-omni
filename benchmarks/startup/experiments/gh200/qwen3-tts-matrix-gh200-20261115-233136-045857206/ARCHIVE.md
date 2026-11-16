# GH200 startup experiment snapshot

Original batch: `/home/lyy/vllm-omni-startup/results/gh200/qwen3-tts-matrix-gh200-20261115-233136-045857206`.
The local raw batch remains the working copy; this directory is a curated Git snapshot.
Included: raw process spans, controller events/logs, run settings, hardware snapshots,
summaries, exported traces/CSVs and the regenerated hierarchy figures.
Excluded: model weights, environment packages and compiler caches.
Snapshot size: 4.51 MiB across 73 files.

## Provenance

Repository HEAD at archival: `64815ca631e66d24620ef1281a42a3843f71d177`.
The working tree had uncommitted instrumentation changes; this HEAD alone does
not identify the experimental code. The archive's hardware.json files retain
the run-time HEAD/status. Analysis hashes identify the current regeneration code,
not proof of an exact clean run-time revision. Commit source and this snapshot
together as two consecutive commits to preserve their relationship.
Installed vLLM/FlashInfer patches and original/patched SHA256 hashes are stored
under `benchmarks/startup/patches/`.
`artifact-manifest.tsv` hashes all copied artifacts.

## Interpretation

Single retained cold/warm pair per mode; this is attribution, not a statistical
speedup claim. Cold resets benchmark-owned TorchInductor/vLLM/FlashInfer caches;
HF files, OS page cache and driver cache are not reset. no-CG also changes Stage 1
eager execution. Process headers show PID/PPID; tree indentation shows recorded
same-process span ancestry, not propagated cross-process span parents.

Logs and hardware metadata include machine names and local paths. Review these
artifacts before publishing the snapshot to a public remote.
