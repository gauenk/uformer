
# -- misc --
import os,copy
import pprint
pp = pprint.PrettyPrinter(indent=4)
from functools import partial

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
import uformer.exps as exps_menu
from uformer import configs
from uformer import lightning
from uformer.utils.misc import optional,rslice_pair,task_keys
from uformer.utils.metrics import compute_psnrs,compute_ssims
from uformer.utils.model_utils import load_checkpoint
from uformer.utils.proc_utils import spatial_chop,temporal_chop,expand2square

def run_exp(_cfg):

    # -- total time --
    timer0 = uformer.utils.timer.ExpTimer()
    timer0.start("total")

    # -- init --
    cfg = copy.deepcopy(_cfg)
    cache_io.exp_strings2bools(cfg)

    # -- init seed/device --
    th.cuda.set_device(int(cfg.device.split(":")[1]))
    configs.set_seed(cfg.seed)

    # -- init results --
    results = edict()
    results.psnrs = []
    results.ssims = []
    results.noisy_psnrs = []
    results.noisy_ssims = []
    results.deno_fns = []
    results.vid_frames = []
    results.vid_name = []
    results.timer_flow = []
    results.timer_deno = []

    # -- load model --
    model_cfg = uformer.extract_model_io(cfg)
    print(model_cfg)
    model = uformer.load_model(**model_cfg)
    substr = optional(cfg,"chkpt","")
    load_checkpoint(model,cfg.use_train,substr)
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

    # -- each subsequence with video name --
    print("indices: ",indices)
    for index in indices:

        # -- clean memory --
        th.cuda.empty_cache()

        # -- unpack --
        sample = data[cfg.dset][index]
        nkey,ckey = task_keys(cfg.task)
        noisy,clean = sample[nkey],sample[ckey]
        noisy,clean = noisy.to(cfg.device),clean.to(cfg.device)
        vid_frames,region = sample['fnums'],optional(sample,'region',None)
        fstart = min(vid_frames)
        noisy,clean = rslice_pair(noisy,clean,region)
        print("[%d] noisy.shape: " % index,noisy.shape)

        # -- create timer --
        timer = uformer.utils.timer.ExpTimer()

        # -- optical flow --
        timer.start("flow")
        if cfg.flow == "true":
            noisy_np = noisy.cpu().numpy()
            flows = svnlb.compute_flow(noisy_np,cfg.sigma)
            flows = edict({k:th.from_numpy(v).to(cfg.device) for k,v in flows.items()})
        else:
            flows = None
        timer.stop("flow")

        # -- get space-time chopping wrapper --
        s_verbose = True
        t_verbose = True
        s_size = cfg.spatial_crop_size
        s_overlap = cfg.spatial_crop_overlap
        t_size = cfg.temporal_crop_size
        t_overlap = cfg.temporal_crop_overlap
        schop_p = partial(spatial_chop,s_size,s_overlap,model,verbose=s_verbose)
        tchop_p = partial(temporal_chop,t_size,t_overlap,schop_p,verbose=t_verbose)
        fwd_fxn = tchop_p # rename

        # -- denoise --
        timer.start("deno")
        with th.no_grad():

            vshape = noisy.shape
            print("noisy.shape: ",noisy.shape)
            noisy_sq,mask = expand2square(noisy,1024.)
            print("noisy_sq.shape: ",noisy_sq.shape)
            deno = fwd_fxn(noisy_sq/imax)
            # deno = tchop_p(noisy_sq/imax)
            print("deno.shape: ",deno.shape)
            deno = th.masked_select(deno,mask.bool()).reshape(*vshape)
            print("deno.shape: ",deno.shape)
            # t = noisy.shape[0]
            # deno = []
            # for ti in range(t):
            #     deno_t = model(noisy[[ti]]/imax)
            #     deno.append(deno_t)
            # deno = th.cat(deno)

            deno = th.clamp(deno,0.,1.)*imax
        timer.stop("deno")

        # -- save example --
        out_dir = Path(cfg.saved_dir) / cfg.dname / cfg.attn_mode / cfg.vid_name
        deno_fns = uformer.utils.io.save_burst(deno,out_dir,"deno",
                                               fstart=fstart,div=1.,fmt="np")
        deno_fns = uformer.utils.io.save_burst(deno,out_dir,"deno",
                                               fstart=fstart,div=1.,fmt="png")
        # uformer.utils.io.save_burst(clean,out_dir,"clean",
        #                             fstart=fstart,div=1.,fmt="np")
        # uformer.utils.io.save_burst(noisy,out_dir,"noisy",
        #                             fstart=fstart,div=1.,fmt="np")

        # -- psnr --
        noisy_psnrs = compute_psnrs(clean,noisy,div=imax)
        psnrs = compute_psnrs(clean,deno,div=imax)
        noisy_ssims = compute_ssims(clean,noisy,div=imax)
        ssims = compute_ssims(clean,deno,div=imax)
        print(noisy_psnrs)
        print(psnrs)

        # -- append results --
        results.noisy_psnrs.append(noisy_psnrs)
        results.psnrs.append(psnrs)
        results.noisy_ssims.append(noisy_ssims)
        results.ssims.append(ssims)
        results.deno_fns.append(deno_fns)
        results.vid_frames.append(vid_frames.numpy())
        results.vid_name.append([cfg.vid_name])
        for name,time in timer.items():
            results[name].append(time)

    # -- compute total time --
    timer0.stop("total")
    ttotal,N = timer0['total'],len(results.psnrs)
    results.timer_total = [ttotal for _ in range(N)]

    return results

def main():

    # -- (0) start info --
    verbose = True
    pid = os.getpid()
    print("PID: ",pid)

    # -- get cache --
    cache_dir = ".cache_io"
    # cache_name = "test_rgb_net"
    cache_name = "davis_bench"
    cache = cache_io.ExpCache(cache_dir,cache_name)
    cache.clear()

    # -- get data mesh --
    dname,dset = ["davis"],["val"]
    vid_names = ["bike-packing"]#,"blackswan","bmx-trees"]
    iexps = {"dname":dname,"vid_name":vid_names,"dset":dset}
    exps = exps_menu.exps_rgb_denoising(iexps,mode="test")
    exps = [exps[0],exps[-1]]

    # -- group with default --
    cfg = configs.default_cfg()
    # cfg.isize = "256_256"
    cfg.task = "rgb_denoise"
    cfg.nframes = 10
    cfg.frame_start = 0
    cfg.frame_end = cfg.frame_start + cfg.nframes - 1
    cfg.noise_version = "rgb_noise"
    cfg.spatial_crop_size = 256
    cfg.spatial_crop_overlap = 0.0
    cfg.temporal_crop_size = 5
    cfg.temporal_crop_overlap = 0/5. # 3 of 5 frames
    print(cfg)
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
        results = cache.load_exp(exp) # possibly load result
        if results is None: # check if no result
            exp.uuid = uuid
            results = run_exp(exp)
            cache.save_exp(uuid,exp,results) # save to cache

    # -- load results --
    records = cache.load_flat_records(exps)
    print(records[['attn_mode','use_train','psnrs','timer_total']])
    print(np.stack(records['psnrs'].to_numpy()).mean(1))

    for attn_mode,mdf in records.groupby("attn_mode"):
        for use_tr,tdf in mdf.groupby("use_train"):
            for vname,vdf in tdf.groupby("vid_name"):
                ssims = np.stack(np.array(vdf['ssims'])).ravel()
                psnrs = np.stack(np.array(vdf['psnrs'])).ravel()
                dtimes = np.stack(np.array(vdf['timer_deno'])).ravel()
                ssims_m = ssims.mean()
                psnrs_m = psnrs.mean()
                dtimes_m = dtimes.mean()
                print(attn_mode,use_tr,vname,psnrs_m,ssims_m,dtimes)

    exit(0)

    for attn_mode,mdf in records.groupby('attn_mode'):
        print(mdf['deno_fns'])
        prepare_sidd(mdf,attn_mode)
    exit(0)
    print(records)
    print(records.filter(like="timer"))
    print(records['psnrs'].mean())
    ssims = np.stack(np.array(records['ssims']))
    psnrs = np.stack(np.array(records['psnrs']))
    print(ssims)
    print(psnrs)
    print(psnrs.shape)
    print(psnrs.mean())
    print(ssims.mean())
    exit(0)

    # -- print by dname,sigma --
    for dname,ddf in records.groupby("dname"):
        # field = "internal_adapt_nsteps"
        field = "adapt_mtype"
        for adapt,adf in ddf.groupby(field):
            adapt_psnrs = np.stack(adf['adapt_psnrs'].to_numpy())
            print("adapt_psnrs.shape: ",adapt_psnrs.shape)
            print(adapt_psnrs)
            for cflow,fdf in adf.groupby("flow"):
                for ws,wsdf in fdf.groupby("ws"):
                    for wt,wtdf in wsdf.groupby("wt"):
                        print("adapt,ws,wt,cflow: ",adapt,ws,wt,cflow)
                        for sigma,sdf in wtdf.groupby("sigma"):
                            ave_psnr,ave_time,num_vids = 0,0,0
                            for vname,vdf in sdf.groupby("vid_name"):
                                print("vdf.psnrs.shape: ",vdf.psnrs.shape)
                                ave_psnr += vdf.psnrs[0].mean()
                                ave_time += vdf['timer_deno'].iloc[0]/len(vdf)
                                num_vids += 1
                            ave_psnr /= num_vids
                            ave_time /= num_vids
                            total_frames = len(sdf)
                            fields = (sigma,ave_psnr,ave_time,total_frames)
                            print("[%d]: %2.3f @ ave %2.2f sec for %d frames" % fields)


if __name__ == "__main__":
    main()
