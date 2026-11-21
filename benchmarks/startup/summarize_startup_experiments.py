#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Summarize fixed-work inference and isolated multiprocessing probes."""
import argparse
import csv
import json
from pathlib import Path
import statistics
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('root',type=Path);parser.add_argument('--date',default='20261116');args=parser.parse_args()
    root=args.root
    inference=root/f'warm-inference-{args.date}'
    methods=root/f'process-method-safety-{args.date}'
    if (inference/'inference-summary.json').exists():
        rows=[];groups=[]
        for mode in ('default','no-cg','eager'):
            path=inference/mode/'events.jsonl'
            if not path.exists():continue
            events=[json.loads(line) for line in path.read_text().splitlines()]
            requests=[e for e in events if e['event']=='request_done']
            samples=[e for e in requests if e['request_index']>=5]
            for e in requests:rows.append({'mode':mode,**e,'sample_role':'warmup' if e['request_index']<5 else 'measured'})
            groups.append((mode,samples))
        fields=sorted({key for row in rows for key in row})
        with (inference/'requests.csv').open('w') as stream:
            writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader();writer.writerows(rows)
        fig,ax=plt.subplots(figsize=(10,5))
        for mode,samples in groups:
            ax.plot([e['request_index'] for e in samples],[e['duration_s'] for e in samples],marker='.',label=mode)
        ax.set_xlabel('Request index (0–4 excluded warmups)');ax.set_ylabel('End-to-end latency (s)')
        ax.set_title('Warm inference: fixed input, seed 42, 64 generated tokens requested')
        ax.legend();ax.grid(alpha=.2);fig.tight_layout();fig.savefig(inference/'inference_latency.png',dpi=180);plt.close(fig)
        hashes={e.get('prompt_sha256') for _,samples in groups for e in samples}
        tokens={e.get('stage0_output_tokens') for _,samples in groups for e in samples}
        audio={e.get('audio_samples') for _,samples in groups for e in samples}
        equivalence={'input_fixture':'get_custom_voice_query(use_batch_sample=False), same source for all modes',
                     'input_hash_comparison_valid':all(e.get('prompt_hash_scope')=='fixture_before_first_request' for _,samples in groups for e in samples),
                     'same_input_hash':(len(hashes)==1 and None not in hashes) if all(e.get('prompt_hash_scope')=='fixture_before_first_request' for _,samples in groups for e in samples) else None,
                     'input_hash_note':'Legacy post-generation hashes include injected request IDs/meta; they are not semantic input fingerprints.',
                     'same_measured_token_count':len(tokens)==1 and None not in tokens,
                     'actual_token_counts':sorted(x for x in tokens if x is not None),
                     'audio_sample_counts':sorted(audio),'same_audio_length':len(audio)==1,
                     'all_finite':all(e['audio_finite'] for _,samples in groups for e in samples),
                     'sample_counts':{mode:len(samples) for mode,samples in groups}}
        (inference/'workload-equivalence.json').write_text(json.dumps(equivalence,indent=2))
        print('WORKLOAD EQUIVALENCE '+json.dumps(equivalence))
    if (methods).exists():
        summary=[]
        for path in sorted((methods).glob('*-parent-cuda-*.json')):
            rows=json.loads(path.read_text())
            summary.append({'scenario':path.stem,'N':len(rows),'correct':all(r['correct'] for r in rows),
                            'no_bad_fork':all(not r['bad_fork'] for r in rows),
                            'graph_correct':all(r['graph_correct'] for r in rows),
                            'child_cuda_clean_before_probe':all(not r['cuda_initialized_before_probe'] for r in rows),
                            **{key:statistics.median(r[key] for r in rows) for key in ['start_to_target_s','torch_import_s','omni_import_s','gpu_probe_s','total_s']},
                            'first_start_to_target_s':rows[0]['start_to_target_s'],
                            'reused_start_to_target_median_s':statistics.median(r['start_to_target_s'] for r in rows[1:]) if len(rows)>1 else None})
        (methods/'process-method-summary.json').write_text(json.dumps(summary,indent=2));print('PROCESS METHOD SUMMARY '+json.dumps(summary))


if __name__=='__main__':main()
