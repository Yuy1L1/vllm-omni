# Qwen3-TTS startup matrix

Cold means an empty benchmark-controlled TorchInductor cache; warm reuses the immediately preceding cold launch's caches. vLLM and FlashInfer caches are inherited from the parent (the wrapper isolates them). HF model files, OS page cache, and GPU driver caches are inherited. Cross-stage spans may overlap and must not be added.

## Aggregate across repetitions

| mode | cache | ok/total | startup median (s) | first request median (s) | steady median (s) |
| --- | --- | ---: | ---: | ---: | ---: |
| default | cold | 1/1 | 317.076 |  |  |
| default | warm | 1/1 | 140.807 |  |  |

## Individual launches

| mode | repetition | cache | status | startup (s) | first request (s) | steady median (s) | FX hits/misses | workers |
| --- | ---: | --- | --- | ---: | ---: | ---: | --- | ---: |
| default | 1 | cold | ok | 317.076 |  |  | 0/2 | 2 |
| default | 1 | warm | ok | 140.807 |  |  | 2/0 | 2 |

## Repetition indicators

These are observations, not automatic claims that the repeated work is removable.

| mode | repetition | cache | HF config loads by stage | FX cache hit/miss/bypass | graph-memory observations |
| --- | ---: | --- | --- | --- | --- |
| default | 1 | cold | `{"None": 23}` | 0/2/0 | stage 0 graph_memory_reserved=0.28 GiB |
| default | 1 | warm | `{"None": 23}` | 2/0/0 | stage 0 graph_memory_reserved=0.32 GiB |

## Stage-aligned spans

| mode | repetition | cache | stage | replica | phase | start (s) | end (s) | duration (s) | source |
| --- | ---: | --- | ---: | ---: | --- | ---: | ---: | ---: | --- |
| default | 1 | cold | None | None | python_imports | 0.062 | 14.472 | 14.409 | reported_duration_anchored_to_controller_receive_time |
| default | 1 | cold | None | None | startup_ready | 0.065 | 317.140 | 317.076 | reported_duration_anchored_to_controller_receive_time |
| default | 1 | cold | 0 | 0 | model_load | 61.355 | 116.418 | 55.063 | reported_duration_anchored_to_controller_receive_time |
| default | 1 | cold | 0 | 0 | weights_load | 63.557 | 115.037 | 51.480 | reported_duration_anchored_to_controller_receive_time |
| default | 1 | cold | 0 | 0 | engine_init | 116.784 | 292.744 | 175.960 | reported_duration_anchored_to_controller_receive_time |
| default | 1 | cold | 0 | 0 | graph_capture | 228.130 | 233.130 | 5.000 | reported_duration_anchored_to_controller_receive_time |
| default | 1 | cold | 0 | 0 | model_compile | 233.178 | 256.778 | 23.600 | reported_duration_anchored_to_controller_receive_time |
| default | 1 | cold | 0 | 0 | graph_capture | 282.245 | 288.245 | 6.000 | reported_duration_anchored_to_controller_receive_time |
| default | 1 | cold | 1 | 0 | model_load | 309.462 | 313.641 | 4.179 | reported_duration_anchored_to_controller_receive_time |
| default | 1 | cold | 1 | 0 | weights_load | 310.039 | 312.469 | 2.430 | reported_duration_anchored_to_controller_receive_time |
| default | 1 | cold | 1 | 0 | engine_init | 314.756 | 314.816 | 0.060 | reported_duration_anchored_to_controller_receive_time |
| default | 1 | warm | None | None | startup_ready | 0.054 | 140.861 | 140.807 | reported_duration_anchored_to_controller_receive_time |
| default | 1 | warm | None | None | python_imports | 0.054 | 14.566 | 14.512 | reported_duration_anchored_to_controller_receive_time |
| default | 1 | warm | 0 | 0 | model_load | 33.852 | 91.166 | 57.314 | reported_duration_anchored_to_controller_receive_time |
| default | 1 | warm | 0 | 0 | weights_load | 35.951 | 89.811 | 53.860 | reported_duration_anchored_to_controller_receive_time |
| default | 1 | warm | 0 | 0 | engine_init | 91.519 | 117.139 | 25.620 | reported_duration_anchored_to_controller_receive_time |
| default | 1 | warm | 0 | 0 | graph_capture | 93.916 | 98.916 | 5.000 | reported_duration_anchored_to_controller_receive_time |
| default | 1 | warm | 0 | 0 | model_compile | 98.967 | 102.667 | 3.700 | reported_duration_anchored_to_controller_receive_time |
| default | 1 | warm | 0 | 0 | graph_capture | 107.286 | 113.286 | 6.000 | reported_duration_anchored_to_controller_receive_time |
| default | 1 | warm | 1 | 0 | model_load | 133.411 | 137.573 | 4.162 | reported_duration_anchored_to_controller_receive_time |
| default | 1 | warm | 1 | 0 | weights_load | 133.929 | 136.429 | 2.500 | reported_duration_anchored_to_controller_receive_time |
| default | 1 | warm | 1 | 0 | engine_init | 138.673 | 138.733 | 0.060 | reported_duration_anchored_to_controller_receive_time |
