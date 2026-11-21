#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Sequential startup experiments; persistent results, no compiler-cache clearing."""
import argparse
import csv
import importlib.util
import json
import os
from pathlib import Path
import statistics
import sys

REPO = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('startup_bench', Path(__file__).with_name('qwen3_tts_startup.py'))
bench = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bench)


def launch(output, mode, requests=0, repetition=1, fixed_inference=False):
    output.mkdir(parents=True, exist_ok=False)
    env = dict(os.environ)
    env.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', CUDA_VISIBLE_DEVICES='0', PYTHONPATH=str(REPO),
               VLLM_OMNI_STARTUP_TRACE_DIR=str(output/'spans'), FLASHINFER_DISABLE_VERSION_CHECK='1')
    if bench.MODES[mode]['disable_torch_compile']:
        env['TORCH_COMPILE_DISABLE']='1'
    else:
        env.pop('TORCH_COMPILE_DISABLE',None)
    command=[sys.executable,str(REPO/'benchmarks/startup/qwen3_tts_startup.py'),'--_child',
             '--model','/home/lyy/models/Qwen3-TTS-12Hz-1.7B-CustomVoice',
             '--deploy-config',str(REPO/'vllm_omni/deploy/qwen3_tts.yaml'),
             '--stage-overrides',json.dumps(bench.MODES[mode]['stage_overrides']),
             '--requests',str(requests),'--max-new-tokens','64','--stage-init-timeout','600']
    if fixed_inference: command.append('--fixed-inference')
    result=bench.measured_launch(command=command,env=env,cwd=REPO,log_path=output/'startup.log',
                                events_path=output/'events.jsonl',mode=mode,repetition=repetition,
                                cache_state='existing-cache',timeout_s=1200)
    (output/'result.json').write_text(json.dumps(result,indent=2))
    if result['status']!='ok': raise RuntimeError(f'Failed launch: {output}')
    return result


def summarize_bootstrap(root):
    ready_by_run = {}
    for run in root.glob('r*'):
        events = [json.loads(line) for line in (run/'events.jsonl').read_text().splitlines()]
        origin = next(event['start_ns'] for event in events if event['event']=='run_start')
        for line in (run/'startup.log').read_text().splitlines():
            if 'EngineCore running' not in line: continue
            raw = line.split('] ',1)[1]
            stage, replica = bench.stage_identity(raw)
            if stage is not None:
                relative = float(line.split(' +',1)[1].split('s]',1)[0])
                ready_by_run[(run.name,stage)] = origin + int(relative*1e9)
    records=[]
    for path in root.glob('r*/spans/*/bootstrap/*.jsonl'):
        records.extend({**json.loads(line),'run':path.parents[3].name} for line in path.read_text().splitlines())
    grouped={}
    for record in records:
        grouped.setdefault(record['launch_id'],{})[record['event']]=record
    rows=[]
    for launch_id,events in grouped.items():
        begin=events['parent_start_begin']; entry=events['child_target_entry']
        def seconds(a,b): return (events[b]['timestamp_ns']-events[a]['timestamp_ns'])/1e9
        rows.append({'run':entry['run'],'launch_id':launch_id,'stage':entry['stage'],'pid':entry['pid'],'ppid':entry['ppid'],
                     'start_method':entry['start_method'],
                     'parent_start_call_s':seconds('parent_start_begin','parent_start_return'),
                     'start_to_target_s':seconds('parent_start_begin','child_target_entry'),
                     'engine_module_import_s':(entry['module_import_end_ns']-entry['module_import_begin_ns'])/1e9,
                     'target_to_core_s':seconds('child_target_entry','vllm_engine_core_entry'),
                     'engine_core_s':seconds('vllm_engine_core_entry','vllm_engine_core_exit'),
                     'start_to_constructed_s':seconds('parent_start_begin','stage_constructor_done'),
                     'start_to_ready_receive_s':(ready_by_run.get((entry['run'],entry['stage']),events['stage_constructor_done']['timestamp_ns'])-begin['timestamp_ns'])/1e9,
                     'stage_ready_source':'controller_log_receive'})
    with (root/'bootstrap.csv').open('w') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    summary={str(stage):{key:statistics.median(row[key] for row in rows if row['stage']==stage)
                        for key in ['parent_start_call_s','start_to_target_s','engine_module_import_s','target_to_core_s','engine_core_s']}
             for stage in sorted({row['stage'] for row in rows})}
    (root/'bootstrap-summary.json').write_text(json.dumps(summary,indent=2))
    print('BOOTSTRAP SUMMARY '+json.dumps(summary),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--experiment',choices=['process-bootstrap','warm-inference'],default='process-bootstrap')
    parser.add_argument('--repetitions',type=int,default=3)
    args=parser.parse_args()
    root=args.out;root.mkdir(parents=True,exist_ok=False)
    (root/'hardware.json').write_text(json.dumps(bench.hardware_snapshot(REPO),indent=2))
    if args.experiment=='process-bootstrap':
        for repeat in range(1,args.repetitions+1): launch(root/f'r{repeat:02d}','default',repetition=repeat)
        summarize_bootstrap(root)
    else:
        summaries=[]
        for mode in ('default','no-cg','eager'):
            launch(root/mode,mode,requests=30,fixed_inference=True)
            events=[json.loads(line) for line in (root/mode/'events.jsonl').read_text().splitlines()]
            samples=[e for e in events if e['event']=='request_done' and e['request_index']>=5]
            assert len(samples)==25
            lengths=sorted({e.get('stage0_output_tokens') for e in samples if e.get('stage0_output_tokens') is not None})
            summaries.append({'mode':mode,'N':len(samples),'warmups_excluded':5,
                              'median_latency_s':statistics.median(e['duration_s'] for e in samples),
                              'stage0_token_lengths':lengths,
                              'audio_sample_lengths':sorted({e['audio_samples'] for e in samples}),
                              'median_tokens_per_s':statistics.median(e['tokens_per_s'] for e in samples if e.get('tokens_per_s')) if lengths else None,
                              'all_audio_finite':all(e['audio_finite'] for e in samples)})
            (root/'inference-summary.json').write_text(json.dumps(summaries,indent=2))
        print('INFERENCE SUMMARY '+json.dumps(summaries),flush=True)


if __name__=='__main__': main()
