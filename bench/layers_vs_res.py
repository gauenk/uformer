"""

This script benchmarks the forward pass exec time
as the image resolution increases

"""

# -- misc --
import sys,os
import pprint
pp = pprint.PrettyPrinter(indent=4)
from functools import partial

# -- linalg --
import numpy as np
import torch as th
from einops import rearrange,repeat

# -- data mngmnt --
from pathlib import Path
from easydict import EasyDict as edict

# -- caching --
import cache_io

# -- model io --
import uformer

# -- datasets --
import data_hub

# -- benchmarking --
from uformer.utils import timer
from uformer.utils import gpu_mem

# -- profiler --
from torch.profiler import profile, record_function, ProfilerActivity


# -- forward exp --
def run_exp(cfg):

    # -- set device --
    th.cuda.set_device(int(cfg.device.split(":")[1]))

    # -- load model --
    model_cfg = uformer.extract_search(cfg)
    model = uformer.load_model(**model_cfg)
    model = model.to(cfg.device)
    imax = 255.

    # -- load sample --
    data,loaders = data_hub.sets.load(cfg)
    sample = data.tr[0]
    vid = sample['blur'].to(cfg.device)
    print(vid.shape)

    # -- init recording devices --
    etimer = timer.ExpTimer()
    gmem = gpu_mem.GpuRecord()

    # -- burn-in --
    with th.no_grad():
        model(vid)

    # -- profile --
    with profile(activities=[ProfilerActivity.CPU,ProfilerActivity.CUDA], record_shapes=True) as prof:
        with record_function("model_inference"):
            model(vid)
    print(prof.export_chrome_trace("trace.json"))
    print(prof.key_averages(group_by_stack_n=5).table(sort_by="self_cuda_time_total", row_limit=5))


    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

    # -- format times --
    print(etimer)
    print(gmem)
    gmem.sortby(etimer.names) # just to be sure

    # -- format --
    results = edict()
    results.names = etimer.names
    results.times = etimer.timers
    results.mem_times = gmem.mems
    results.mem_tnames = gmem.mems_alloc

    return results

def main():

    # -- (0) start info --
    verbose = True
    pid = os.getpid()
    print("PID: ",pid)

    # -- get cache --
    cache_dir = ".cache_io"
    cache_name = "bench_layers_vs_res"
    cache = cache_io.ExpCache(cache_dir,cache_name)
    # cache.clear()

    # -- get mesh --
    dnames = ["gopro"]
    dset = ["te"]
    vid_names = ["%02d" % x for x in np.arange(0,40)]
    vid_names = vid_names[1:2]


    flow = ["false"]
    ws,wt = [8],[0]
    isizes = ["256_256"]
    stride = [1]
    use_train = ["false"]
    # attn_mode = ["window_refactored"]#,"window_stnls","product_stnls"]
    attn_mode = ["window_stnls"]#,"window_stnls","product_stnls"]
    exp_lists = {"dname":dnames,"vid_name":vid_names,"dset":dset,
                 "flow":flow,"ws":ws,"wt":wt,"attn_mode":attn_mode,
                 "isize":isizes,"stride":stride,"use_train":use_train}
    exps_a = cache_io.mesh_pydicts(exp_lists) # create mesh

    # -- exps version 2 --
    exp_lists['ws'] = [-1]
    exp_lists['wt'] = [-1]
    exp_lists['flow'] = ["false"]
    exp_lists['use_train'] = ["false"]
    exp_lists['stride'] = [1]
    exp_lists['attn_mode'] = ['original']
    exps_b = cache_io.mesh_pydicts(exp_lists) # create mesh
    exps = exps_b# + exps_a

    # -- group with default --
    cfg = uformer.configs.default_cfg()
    cfg.nframes = 1
    cfg.frame_start = 0
    cfg.frame_end = cfg.frame_start + cfg.nframes - 1
    cache_io.append_configs(exps,cfg) # merge the two

    # -- run exps --
    nexps = len(exps)
    for exp_num,exp in enumerate(exps):

        # -- info --
        if verbose:
            print("-="*25+"-")
            print(f"Running experiment number {exp_num+1}/{nexps}")
            print("-="*25+"-")
            pp.pprint(exp)

        # -- logic --
        uuid = cache.get_uuid(exp) # assing ID to each Dict in Meshgrid
        # cache.clear_exp(uuid)
        # if exp.attn_mode == "original":
        #     cache.clear_exp(uuid)
        # if exp.attn_mode == "aug_refactored":
        #     cache.clear_exp(uuid)
        # if exp.attn_mode == "aug_stnls":
        #     cache.clear_exp(uuid)
        # if exp.attn_mode == "product_stnls":
        #     cache.clear_exp(uuid)
        if exp.use_train == "true":
            cache.clear_exp(uuid)
        results = cache.load_exp(exp) # possibly load result
        if results is None: # check if no result
            exp.uuid = uuid
            results = run_exp(exp)
            cache.save_exp(uuid,exp,results) # save to cache

    # -- load results --
    records = cache.load_flat_records(exps)

if __name__ == "__main__":
    main()

