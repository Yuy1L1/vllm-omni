# Qwen3-TTS startup matrix

Cold: fresh benchmark-owned TorchInductor, vLLM, FlashInfer caches; warm: reuse from previous cold.
HF model files, OS page cache, CUDA driver cache are not reset.
Sampler implementation can affect startup and request latency; see run_settings.txt.
The no-cg mode also modifies Stage 1 eager execution; it is not a pure CG-only ablation.

| Mode | Cache | N | Startup median | First request median | Steady median |
| --- | --- | ---: | ---: | ---: | ---: |
| default | cold | 1 | 317.076s | n/a | n/a |
| default | warm | 1 | 140.807s | n/a | n/a |
| no-cg | cold | 1 | 276.748s | n/a | n/a |
| no-cg | warm | 1 | 122.936s | n/a | n/a |
| eager | cold | 1 | 212.912s | n/a | n/a |
| eager | warm | 1 | 119.060s | n/a | n/a |
