#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Diagnose the first CUDA lazy-init during Omni import in an isolated process."""
import argparse
import json
from pathlib import Path
import time
import traceback


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--out',type=Path,required=True);args=parser.parse_args()
    import torch
    before=torch.cuda.is_initialized()
    original=torch.cuda._lazy_init
    events=[]
    def measured(*positional,**keywords):
        cold=not torch.cuda.is_initialized()
        start=time.perf_counter_ns()
        stack=traceback.format_stack() if cold else None
        try:
            return original(*positional,**keywords)
        finally:
            if cold:
                events.append({'start_ns':start,'end_ns':time.perf_counter_ns(),
                               'initialized_after':torch.cuda.is_initialized(),'stack':stack})
    torch.cuda._lazy_init=measured
    start=time.perf_counter_ns()
    try:
        import vllm_omni
    finally:
        torch.cuda._lazy_init=original
    result={'cuda_initialized_before_omni_import':before,'cuda_initialized_after_omni_import':torch.cuda.is_initialized(),
            'omni_import_s':(time.perf_counter_ns()-start)/1e9,'first_lazy_init_events':events}
    args.out.write_text(json.dumps(result,indent=2))
    print(json.dumps({key:value for key,value in result.items() if key!='first_lazy_init_events'}))


if __name__=='__main__':main()
