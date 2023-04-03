# -- misc --
import os,copy
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

def run_exp(_cfg):

    # -- init --
    cfg = copy.deepcopy(_cfg)
    cache_io.exp_strings2bools(cfg)

    # -- init seed/device --
    th.cuda.set_device(int(cfg.device.split(":")[1]))
    configs.set_seed(cfg.seed)

    # -- load model --
    model_cfg = uformer.extract_model_io(cfg)
    model = uformer.load_model(**model_cfg)
    load_checkpoint(model,cfg.use_train,cfg.chkpt_fn)
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

    # -- unpack --
    index = indices[0]
    sample = data[cfg.dset][index]
    noisy,clean = sample['blur'],sample['sharp']
    noisy,clean = noisy.to(cfg.device),clean.to(cfg.device)
    vid_frames,region = sample['fnums'],optional(sample,'region',None)
    fstart = min(vid_frames)
    noisy,clean = rslice_pair(noisy,clean,region)
    tsize,vshape = 2,noisy.shape

    # -- denoise --
    with th.no_grad():
        noisy_sq,mask = expand2square(noisy)
        deno = temporal_chop(noisy_sq/imax,tsize,model)
        deno = th.masked_select(deno,mask.bool()).reshape(*vshape)
        deno = th.clamp(deno,0.,1.)*imax

    # -- psnr --
    noisy_psnrs = compute_psnrs(clean,noisy,div=imax)
    psnrs = compute_psnrs(clean,deno,div=imax)
    print(noisy_psnrs)
    print(psnrs)

    # -- return --
    results = edict()
    results.psnrs = psnrs
    results.noisy_psnrs = psnrs
    return results

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
    chkpt_fn = [""]

    flow = ["false"]
    ws,wt = [8],[0]
    isizes = ["none"]
    stride = [1]
    use_train = ["false"]
    attn_mode = ["window_stnls","window_refactored","window_default","product_stnls"]
    exp_lists = {"dname":dnames,"vid_name":vid_names,"dset":dset,"flow":flow,
                 "ws":ws,"wt":wt,"attn_mode":attn_mode,"isize":isizes,
                 "stride":stride,"use_train":use_train,"chkpt_fn":chkpt_fn}
    exps_a = cache_io.mesh_pydicts(exp_lists) # create mesh

    # -- test trained --
    chkpt_fns = ["993b7b7f-0cbd-48ac-b92a-0dddc3b4ce0e-epoch=13",
                 "993b7b7f-0cbd-48ac-b92a-0dddc3b4ce0e-epoch",
                 "7815163b-842f-4edb-9cf5-21ee7abb1dd6-epoch=34",
                 "7815163b-842f-4edb-9cf5-21ee7abb1dd6",
                 ]
    exp_lists['use_train'] = ['true']
    exp_lists['attn_mode'] = ['product_stnls']
    exp_lists['chkpt_fn'] = chkpt_fns
    exps_b = cache_io.mesh_pydicts(exp_lists) # create mesh
    exps = exps_a + exps_b

    # -- group with default --
    cfg = configs.default_cfg()
    cfg.seed = 123
    # cfg.isize = "256_256"
    cfg.nframes = 1
    cfg.frame_start = 0
    cfg.frame_end = cfg.frame_start + cfg.nframes - 1
    # cfg.isize = "256_256"
    cfg.noise_version = "blur"
    cache_io.append_configs(exps,cfg) # merge the two


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
        # if exp.attn_mode == "window_default":
        #     cache.clear_exp(uuid)
        # if exp.attn_mode == "window_refactored":
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
    print(records[['attn_mode','chkpt_fn','psnrs']])

if __name__ == "__main__":
    main()
