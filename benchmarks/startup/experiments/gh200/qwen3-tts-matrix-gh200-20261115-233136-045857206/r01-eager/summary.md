# Qwen3-TTS startup matrix

Cold means an empty benchmark-controlled TorchInductor cache; warm reuses the immediately preceding cold launch's caches. vLLM and FlashInfer caches are inherited from the parent (the wrapper isolates them). HF model files, OS page cache, and GPU driver caches are inherited. Cross-stage spans may overlap and must not be added.

## Aggregate across repetitions

| mode | cache | ok/total | startup median (s) | first request median (s) | steady median (s) |
| --- | --- | ---: | ---: | ---: | ---: |
| eager | cold | 1/1 | 212.912 |  |  |
| eager | warm | 1/1 | 119.060 |  |  |

## Individual launches

| mode | repetition | cache | status | startup (s) | first request (s) | steady median (s) | FX hits/misses | workers |
| --- | ---: | --- | --- | ---: | ---: | ---: | --- | ---: |
| eager | 1 | cold | ok | 212.912 |  |  | 0/0 | 2 |
| eager | 1 | warm | ok | 119.060 |  |  | 0/0 | 2 |

## Repetition indicators

These are observations, not automatic claims that the repeated work is removable.

| mode | repetition | cache | HF config loads by stage | FX cache hit/miss/bypass | graph-memory observations |
| --- | ---: | --- | --- | --- | --- |
| eager | 1 | cold | `{"None": 23}` | 0/0/0 |  |
| eager | 1 | warm | `{"None": 23}` | 0/0/0 |  |

## Stage-aligned spans

| mode | repetition | cache | stage | replica | phase | start (s) | end (s) | duration (s) | source |
| --- | ---: | --- | ---: | ---: | --- | ---: | ---: | ---: | --- |
| eager | 1 | cold | None | None | python_imports | 0.052 | 14.025 | 13.973 | reported_duration_anchored_to_controller_receive_time |
| eager | 1 | cold | None | None | startup_ready | 0.052 | 212.964 | 212.912 | reported_duration_anchored_to_controller_receive_time |
| eager | 1 | cold | 0 | 0 | model_load | 60.311 | 117.684 | 57.373 | reported_duration_anchored_to_controller_receive_time |
| eager | 1 | cold | 0 | 0 | weights_load | 62.302 | 116.352 | 54.050 | reported_duration_anchored_to_controller_receive_time |
| eager | 1 | cold | 0 | 0 | engine_init | 118.061 | 191.571 | 73.510 | reported_duration_anchored_to_controller_receive_time |
| eager | 1 | cold | 1 | 0 | model_load | 207.702 | 209.622 | 1.920 | reported_duration_anchored_to_controller_receive_time |
| eager | 1 | cold | 1 | 0 | weights_load | 208.200 | 208.420 | 0.220 | reported_duration_anchored_to_controller_receive_time |
| eager | 1 | cold | 1 | 0 | engine_init | 210.769 | 210.839 | 0.070 | reported_duration_anchored_to_controller_receive_time |
| eager | 1 | warm | None | None | python_imports | 0.059 | 14.479 | 14.420 | reported_duration_anchored_to_controller_receive_time |
| eager | 1 | warm | None | None | startup_ready | 0.059 | 119.119 | 119.060 | reported_duration_anchored_to_controller_receive_time |
| eager | 1 | warm | 0 | 0 | model_load | 33.678 | 93.338 | 59.660 | reported_duration_anchored_to_controller_receive_time |
| eager | 1 | warm | 0 | 0 | weights_load | 35.718 | 91.938 | 56.220 | reported_duration_anchored_to_controller_receive_time |
| eager | 1 | warm | 0 | 0 | engine_init | 93.710 | 98.280 | 4.570 | reported_duration_anchored_to_controller_receive_time |
| eager | 1 | warm | 1 | 0 | model_load | 114.043 | 115.882 | 1.839 | reported_duration_anchored_to_controller_receive_time |
| eager | 1 | warm | 1 | 0 | weights_load | 114.533 | 114.733 | 0.200 | reported_duration_anchored_to_controller_receive_time |
| eager | 1 | warm | 1 | 0 | engine_init | 117.000 | 117.070 | 0.070 | reported_duration_anchored_to_controller_receive_time |
