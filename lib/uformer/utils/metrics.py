import torch as th
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as comp_psnr
from skimage.metrics import structural_similarity as compute_ssim_ski

def ensure_4ndim(clean,deno):
    assert clean.ndim == deno.ndim
    ndim,b = clean.ndim,-1
    if ndim == 5:
        b,t,c,h,w = clean.shape
        clean = clean.view(b*t,c,h,w)
        deno = deno.view(b*t,c,h,w)
    return clean,deno,ndim,b

def original_ndim(ssims,ndim,B):
    if ndim == 5:
        ssims = ssims.reshape((B,-1))
    return ssims

def compute_ssims(clean,deno,div=255.):
    clean,deno,ndim,B = ensure_4ndim(clean,deno)
    nframes = clean.shape[0]
    ssims = []
    for t in range(nframes):
        clean_t = clean[t].cpu().numpy().transpose((1,2,0))/div
        deno_t = deno[t].cpu().numpy().transpose((1,2,0))/div
        ssim_t = compute_ssim_ski(clean_t,deno_t,channel_axis=-1,
                                  data_range=1.)
        ssims.append(ssim_t)
    ssims = np.array(ssims)
    ssims = original_ndim(ssims,ndim,B)
    return ssims

def compute_psnrs(clean,deno,div=255.):
    clean,deno,ndim,B = ensure_4ndim(clean,deno)
    t = clean.shape[0]
    clean = clean.detach().cpu().numpy()
    deno = deno.detach().cpu().numpy()
    # clean_rs = clean.reshape((t,-1))/div
    # deno_rs = deno.reshape((t,-1))/div
    # mse = th.mean((clean_rs - deno_rs)**2,1)
    # psnrs = -10. * th.log10(mse).detach()
    # psnrs = psnrs.cpu().numpy()
    psnrs = []
    t = clean.shape[0]
    for ti in range(t):
        psnr_ti = comp_psnr(clean[ti,:,:,:], deno[ti,:,:,:], data_range=div)
        psnrs.append(psnr_ti)
    psnrs = np.array(psnrs)
    psnrs = original_ndim(psnrs,ndim,B)
    return psnrs

