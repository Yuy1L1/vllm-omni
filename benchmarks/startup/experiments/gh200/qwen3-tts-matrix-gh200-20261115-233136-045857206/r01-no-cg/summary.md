# Qwen3-TTS startup matrix

Cold means an empty benchmark-controlled TorchInductor cache; warm reuses the immediately preceding cold launch's caches. vLLM and FlashInfer caches are inherited from the parent (the wrapper isolates them). HF model files, OS page cache, and GPU driver caches are inherited. Cross-stage spans may overlap and must not be added.

## Aggregate across repetitions

| mode | cache | ok/total | startup median (s) | first request median (s) | steady median (s) |
| --- | --- | ---: | ---: | ---: | ---: |
| no-cg | cold | 1/1 | 276.748 |  |  |
| no-cg | warm | 1/1 | 122.936 |  |  |

## Individual launches

| mode | repetition | cache | status | startup (s) | first request (s) | steady median (s) | FX hits/misses | workers |
| --- | ---: | --- | --- | ---: | ---: | ---: | --- | ---: |
| no-cg | 1 | cold | ok | 276.748 |  |  | 0/0 | 2 |
| no-cg | 1 | warm | ok | 122.936 |  |  | 0/0 | 2 |

## Repetition indicators

These are observations, not automatic claims that the repeated work is removable.

| mode | repetition | cache | HF config loads by stage | FX cache hit/miss/bypass | graph-memory observations |
| --- | ---: | --- | --- | --- | --- |
| no-cg | 1 | cold | `{"None": 23}` | 0/0/0 |  |
| no-cg | 1 | warm | `{"None": 23}` | 0/0/0 |  |

## Stage-aligned spans

| mode | repetition | cache | stage | replica | phase | start (s) | end (s) | duration (s) | source |
| --- | ---: | --- | ---: | ---: | --- | ---: | ---: | ---: | --- |
| no-cg | 1 | cold | None | None | startup_ready | 0.059 | 276.807 | 276.748 | reported_duration_anchored_to_controller_receive_time |
| no-cg | 1 | cold | None | None | python_imports | 0.060 | 14.149 | 14.089 | reported_duration_anchored_to_controller_receive_time |
| no-cg | 1 | cold | 0 | 0 | model_load | 60.093 | 119.368 | 59.275 | reported_duration_anchored_to_controller_receive_time |
| no-cg | 1 | cold | 0 | 0 | weights_load | 62.279 | 117.989 | 55.710 | reported_duration_anchored_to_controller_receive_time |
| no-cg | 1 | cold | 0 | 0 | engine_init | 119.749 | 255.539 | 135.790 | reported_duration_anchored_to_controller_receive_time |
| no-cg | 1 | cold | 1 | 0 | model_load | 271.684 | 273.584 | 1.899 | reported_duration_anchored_to_controller_receive_time |
| no-cg | 1 | cold | 1 | 0 | weights_load | 272.204 | 272.424 | 0.220 | reported_duration_anchored_to_controller_receive_time |
| no-cg | 1 | cold | 1 | 0 | engine_init | 274.680 | 274.740 | 0.060 | reported_duration_anchored_to_controller_receive_time |
| no-cg | 1 | warm | None | None | python_imports | 0.054 | 14.136 | 14.083 | reported_duration_anchored_to_controller_receive_time |
| no-cg | 1 | warm | None | None | startup_ready | 0.054 | 122.990 | 122.936 | reported_duration_anchored_to_controller_receive_time |
| no-cg | 1 | warm | 0 | 0 | model_load | 33.074 | 93.692 | 60.618 | reported_duration_anchored_to_controller_receive_time |
| no-cg | 1 | warm | 0 | 0 | weights_load | 35.157 | 92.367 | 57.210 | reported_duration_anchored_to_controller_receive_time |
| no-cg | 1 | warm | 0 | 0 | engine_init | 94.067 | 101.377 | 7.310 | reported_duration_anchored_to_controller_receive_time |
| no-cg | 1 | warm | 1 | 0 | model_load | 117.549 | 119.484 | 1.934 | reported_duration_anchored_to_controller_receive_time |
| no-cg | 1 | warm | 1 | 0 | weights_load | 118.038 | 118.268 | 0.230 | reported_duration_anchored_to_controller_receive_time |
| no-cg | 1 | warm | 1 | 0 | engine_init | 120.820 | 120.900 | 0.080 | reported_duration_anchored_to_controller_receive_time |
