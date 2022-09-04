# -- misc --
import os
import pprint
pp = pprint.PrettyPrinter(indent=4)

# -- vision --
import scipy.io
from PIL import Image

# -- linalg --
import numpy as np
import torch as th
from einops import rearrange,repeat

# -- data mngmnt --
from pathlib import Path
from easydict import EasyDict as edict

# -- data --
import data_hub

# -- optical flow --
# import svnlb

# -- caching results --
import cache_io

# -- network --
import uformer
from uformer import configs
from uformer.utils.misc import optional,rslice_pair
from uformer.utils.metrics import compute_psnrs,compute_ssims
from uformer.utils.model_utils import temporal_chop,expand2square,load_checkpoint

def run_exp(cfg):

    # -- init --
    th.cuda.set_device(int(cfg.device.split(":")[1]))
    configs.set_seed(cfg.seed)

    # -- load model --
    model_cfg = uformer.extract_search(cfg)
    model = uformer.load_model(**model_cfg)
    load_checkpoint(model,cfg.use_train,"")
    imax = 255.

    # -- data --
    data,loaders = data_hub.sets.load(cfg)
    groups = data[cfg.dset].groups
    indices = [i for i,g in enumerate(groups) if cfg.vid_name in g]

    # -- optional filter --
    frame_start = optional(cfg,"frame_start",-1)
    frame_end = optional(cfg,"frame_end",-1)
    if frame_start >= 0 and frame_end >= 0:
        def fbnds(fnums,lb,ub): return (lb <= np.min(fnums)) and (ub >= np.max(fnums))
        indices = [i for i in indices if fbnds(data[cfg.dset].paths['fnums'][groups[i]],
                                               cfg.frame_start,cfg.frame_end)]

    print(indices)
    # -- unpack --
    index = indices[0]
    sample = data[cfg.dset][index]
    noisy,clean = sample['blur'],sample['sharp']
    noisy,clean = noisy.to(cfg.device),clean.to(cfg.device)
    vid_frames,region = sample['fnums'],optional(sample,'region',None)
    fstart = min(vid_frames)
    noisy,clean = rslice_pair(noisy,clean,region)

    # -- psnr --
    noisy_psnrs = compute_psnrs(clean,noisy,div=imax)
    psnrs = compute_psnrs(clean,deno,div=imax)


def main():

    # -- (0) start info --
    verbose = True
    pid = os.getpid()
    print("PID: ",pid)

    # -- get cache --
    cache_dir = ".cache_io"
    cache_name = "inspect_rel_pos"
    cache = cache_io.ExpCache(cache_dir,cache_name)
    # cache.clear()

    # -- get mesh --
    dnames = ["gopro"]
    dset = ["te"]
    vid_names = ["%02d" % x for x in np.arange(0,40)]
    vid_names = vid_names[1:2]

    flow = ["false"]
    ws,wt = [8],[0]
    isizes = ["none"]
    stride = [1]
    use_train = ["false"]
    attn_mode = ["window_refactored","window_dnls","product_dnls"]
    exp_lists = {"dname":dnames,"vid_name":vid_names,"dset":dset,
                 "flow":flow,"ws":ws,"wt":wt,"attn_mode":attn_mode,
                 "isize":isizes,"stride":stride,"use_train":use_train}
    exps_a = cache_io.mesh_pydicts(exp_lists) # create mesh

    exp_lists['use_train'] = ['true']
    exp_lists['attn_mode'] = ['product_dnls']
    exps_b = cache_io.mesh_pydicts(exp_lists) # create mesh
    exps = exps_a + exps_b





    # -- run experiment --
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
        # if exp.attn_mode == "aug_dnls":
        #     cache.clear_exp(uuid)
        # if exp.attn_mode == "product_dnls":
        #     cache.clear_exp(uuid)
        if exp.use_train == "true":
            print(exp)
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
