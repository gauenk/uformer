
import torch as th

import importlib
import data_hub

from dev_basics import flow
from dev_basics.utils.gpu_mem import GpuMemer,MemIt
from dev_basics.utils.timer import ExpTimer,TimeIt

import pandas as pd
from easydict import EasyDict as edict

def get_pair(cfg):

    # -- load data --
    data,loaders = data_hub.sets.load(cfg)
    indices = data_hub.filter_subseq(data[cfg.dset],cfg.vid_name,
                                     cfg.frame_start,cfg.frame_end)

    # -- unpack --
    imax = 255.
    sample = data[cfg.dset][indices[0]]
    region = sample['region']
    noisy,clean = sample['noisy'][None,],sample['clean'][None,]
    noisy,clean = noisy.to(cfg.device),clean.to(cfg.device)
    noisy,clean = noisy/imax,clean/imax
    return noisy,clean

def run_exp(cfg):

    # -- data --
    noisy,clean = get_pair(cfg)

    # -- network --
    module = importlib.import_module(cfg.python_module)
    model = module.load_model(cfg)

    # -- create timer --
    timer = ExpTimer()
    memer = GpuMemer()

    # -- run flows --
    flows = flow.orun(noisy,cfg.flow)

    # -- run test -
    with th.no_grad():
        with MemIt(memer,"deno"):
            with TimeIt(timer,"deno"):
                model(noisy,flows)

    # -- view report --
    results = {}
    for name,(mem_res,mem_alloc) in memer.items():
        key = "%s_%s" % (name,"mem_res")
        if not(key in results):
            results[key] = []
        results[key].append([mem_res])
        key = "%s_%s" % (name,"mem_alloc")
        if not(key in results):
            results[key] = []
        results[key].append([mem_alloc])
    for name,time in timer.items():
        if not(name in results):
            results[name] = []
        results[name].append(time)

    return results


def base_cfg():
    cfg = edict()

    cfg.python_module = "uformer"

    cfg.dname = "davis"
    cfg.dset = "val"
    cfg.device = "cuda:0"
    cfg.vid_name = "dance-twirl"
    cfg.ntype = "g"
    cfg.sigma = 30
    cfg.nframes = 3
    cfg.frame_start = 0
    cfg.frame_end = 2
    # cfg.isize = "512_512"
    # cfg.isize = "128_128"
    cfg.isize = "256_256"

    cfg.ps = 7
    cfg.stride0 = 4
    cfg.stride1 = 1

    cfg.flow = False

    return cfg

def main():

    cfg = base_cfg()
    cfg.wr = 1
    cfg.kr = 1.
    cfg.k = 25
    N = 3
    fields = {"ws":[8,21,21],"wt":[0,3,3],
              "attn_mode":["window_default","nls_full","nls_refine"]}
    reports = []
    for n in range(N):
        print(n)
        for key,val in fields.items():
            cfg[key] = val[n]
        report = run_exp(cfg)
        reports.append(report)
    reports = pd.DataFrame(reports)
    print(reports)

if __name__ == "__main__":
    main()
