#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Isolated spawn/forkserver CUDA correctness probe; never changes Omni's method."""
import argparse
import json
import multiprocessing as mp
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback


def child(queue):
    entry=time.perf_counter_ns()
    result={'pid':os.getpid(),'ppid':os.getppid(),'entry_ns':entry,'method':mp.get_start_method()}
    try:
        import torch
        torch_imported=time.perf_counter_ns()
        result['cuda_initialized_before_omni_import']=torch.cuda.is_initialized()
        import vllm_omni
        imported=time.perf_counter_ns()
        result.update(effective_method_after_import=mp.get_start_method(), torch_version=torch.__version__,
                      torch_import_s=(torch_imported-entry)/1e9,
                      omni_import_s=(imported-torch_imported)/1e9,
                      cuda_initialized_before_probe=torch.cuda.is_initialized(),
                      bad_fork=bool(torch.cuda._is_in_bad_fork()))
        base=(torch.arange(4096,dtype=torch.float32).reshape(64,64)%7)-3
        expected=base@base.T
        gpu=base.to('cuda')
        actual=(gpu@gpu.T).cpu()
        stream=torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            warmup=gpu@gpu.T
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()
        graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured=gpu@gpu.T
        graph.replay()
        graph_actual=captured.cpu()
        torch.cuda.synchronize()
        result.update(graph_correct=bool(torch.allclose(graph_actual,expected,rtol=1e-5,atol=1e-5)),
                      correct=bool(torch.allclose(actual,expected,rtol=1e-5,atol=1e-5)),
                      finite=bool(torch.isfinite(actual).all()),checksum=float(actual.sum()),
                      cuda_initialized_after_probe=torch.cuda.is_initialized(),
                      gpu_probe_s=(time.perf_counter_ns()-imported)/1e9)
    except BaseException:
        result.update(correct=False,error=traceback.format_exc())
    result['end_ns']=time.perf_counter_ns()
    queue.put(result)


def scenario(method,parent_cuda,iterations,out):
    import torch
    if parent_cuda:
        torch.ones(1,device='cuda');torch.cuda.synchronize()
    context=mp.get_context(method)
    rows=[]
    for index in range(iterations):
        queue=context.Queue()
        proc=context.Process(target=child,args=(queue,))
        start=time.perf_counter_ns();proc.start();returned=time.perf_counter_ns()
        try:
            result=queue.get(timeout=180)
        except Exception:
            proc.terminate();proc.join(timeout=10)
            raise
        proc.join(timeout=20)
        if proc.is_alive(): proc.terminate();proc.join(timeout=10)
        queue.close();queue.join_thread()
        result.update(iteration=index,parent_cuda=parent_cuda,parent_cuda_initialized=torch.cuda.is_initialized(),
                      parent_start_s=(returned-start)/1e9,start_to_target_s=(result['entry_ns']-start)/1e9,
                      total_s=(result['end_ns']-start)/1e9,exitcode=proc.exitcode,
                      first_server_start=(method=='forkserver' and index==0))
        rows.append(result)
        out.write_text(json.dumps(rows,indent=2))
        print(json.dumps(result),flush=True)
    if not all(r['correct'] and r['graph_correct'] and r['exitcode']==0 and not r.get('bad_fork')
               and r['effective_method_after_import']==method for r in rows):
        raise RuntimeError('Process-method correctness probe failed')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--method',choices=['spawn','forkserver'])
    parser.add_argument('--parent-cuda',action='store_true')
    parser.add_argument('--iterations',type=int,default=3)
    args=parser.parse_args()
    if args.method:
        scenario(args.method,args.parent_cuda,args.iterations,args.out)
        return
    args.out.mkdir(parents=True,exist_ok=False)
    statuses=[]
    for parent_cuda in (False,True):
        for method in ('spawn','forkserver'):
            name=f'{method}-parent-cuda-{int(parent_cuda)}'
            command=[sys.executable,str(Path(__file__).resolve()),'--out',str(args.out/f'{name}.json'),
                     '--method',method,'--iterations',str(args.iterations)]
            if parent_cuda:command.append('--parent-cuda')
            with (args.out/f'{name}.log').open('w') as log:
                result=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=False,timeout=660)
            statuses.append({'scenario':name,'exitcode':result.returncode})
            (args.out/'scenario-status.json').write_text(json.dumps(statuses,indent=2))
            print('Completed '+name+' exit='+str(result.returncode),flush=True)
    if any(item['exitcode'] for item in statuses):
        raise RuntimeError('One or more isolated scenarios failed; see scenario-status.json')


if __name__=='__main__':main()
