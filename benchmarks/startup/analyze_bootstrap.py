#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Plot process bootstrap intervals and export process-launch Chrome trace flows."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('root',type=Path);args=parser.parse_args()
    fig,ax=plt.subplots(figsize=(13,7));labels=[];index=0
    colors=['#9966aa','#4499bb','#ddaa33','#228855']
    for run in sorted(args.root.glob('r*')):
        event_rows=[json.loads(x) for x in (run/'events.jsonl').read_text().splitlines()]
        origin=next(x['start_ns'] for x in event_rows if x['event']=='run_start')
        grouped={}
        for path in (run/'spans').glob('*/bootstrap/*.jsonl'):
            for line in path.read_text().splitlines():
                row=json.loads(line);grouped.setdefault(row['launch_id'],{})[row['event']]=row
        for marks in sorted(grouped.values(),key=lambda m:m['child_target_entry']['stage']):
            child=marks['child_target_entry'];stage=child['stage']
            intervals=[('parent_start_begin','child_target_entry'),('child_target_entry','vllm_engine_core_entry'),
                       ('vllm_engine_core_entry','vllm_engine_core_exit'),('vllm_engine_core_exit','stage_constructor_done')]
            for color,(start,end) in zip(colors,intervals):
                begin=marks[start]['timestamp_ns'];finish=marks[end]['timestamp_ns']
                ax.barh(index,(finish-begin)/1e9,left=(begin-origin)/1e9,color=color,height=.6)
                if (finish-begin)/1e9>3:
                    ax.text(((finish+begin)/2-origin)/1e9,index,f'{(finish-begin)/1e9:.1f}s',ha='center',va='center',fontsize=9,color='white')
            labels.append(f'{run.name} / Stage {stage} / PID {child["pid"]} ({child["start_method"]})');index+=1
        trace_dirs={path.parent.parent for path in (run/'spans').glob('*/bootstrap/*.jsonl')}
        for trace_dir in trace_dirs:
            subprocess.run([sys.executable,str(Path(__file__).with_name('export_startup_spans.py')),str(trace_dir),'--events',str(run/'events.jsonl')],check=True)
    ax.set_yticks(range(index),labels);ax.invert_yaxis();ax.set_xlabel('Seconds from each controller launch')
    ax.set_title('Process bootstrap vs EngineCore initialization\nBootstrap includes target-preceding imports; intervals are host wall time')
    ax.legend(handles=[Patch(color=c,label=l) for c,l in zip(colors,['start → child target','target → EngineCore','EngineCore init','post-core wiring'])],loc='upper center',bbox_to_anchor=(.5,-.13),ncol=4,fontsize=8)
    ax.set_xlim(left=0)
    ax.grid(axis='x',alpha=.2);fig.tight_layout();fig.savefig(args.root/'bootstrap_timeline.png',dpi=180)


if __name__=='__main__':main()
