
# -- misc --
import sys,os,copy
from pathlib import Path

# -- torch --
import torch as th

# -- linalg --
import numpy as np

# -- modules --
from .model import Uformer,Downsample,Upsample

# -- misc imports --
from ..common import optional,select_sigma
from ..utils.model_utils import load_checkpoint_module
from ..utils.model_io import get_pretrained_path
from ..utils.model_utils import load_checkpoint_module,load_checkpoint_qkv

def load_model(*args,**kwargs):

    # -- defaults changed by noise version --
    noise_version = optional(kwargs,'noise_version',"noise")
    default_modulator = True
    default_depth = [1, 2, 8, 8, 2, 8, 8, 2, 1]
    default_state_fn = get_default_state(noise_version)

    # -- get cfg --
    nchnls = optional(kwargs,'nchnls',3)
    input_size = optional(kwargs,'input_size',128)
    device = optional(kwargs,'device','cuda:0')
    depths = optional(kwargs,'input_depth',default_depth)
    nblocks = len(depths)

    # -- other configs --
    embed_dim = optional(kwargs,'embed_dim',32)
    win_size = optional(kwargs,'win_size',8)
    mlp_ratio = optional(kwargs,'mlp_ratio',4)
    qkv_bias = optional(kwargs,'qkv_bias',True)
    token_projection = optional(kwargs,'token_projection','linear')
    token_mlp = optional(kwargs,'token_mlp','leff')
    modulator = optional(kwargs,'modulator',default_modulator)
    cross_modulator = optional(kwargs,'cross_modulator',False)
    dd_in = optional(kwargs,'dd_in',3)
    attn_mode = optional(kwargs,'attn_mode',"window_dnls")
    in_attn_mode = optional(kwargs,'in_attn_mode',attn_mode)

    # -- load configs [files] --
    pretrained_prefix = optional(kwargs,"pretrained_prefix","module.")
    load_pretrained = optional(kwargs,"load_pretrained",True)
    pretrained_path = optional(kwargs,"pretrained_path",default_state_fn)

    # -- load configs --
    reset_qkv = optional(kwargs,"reset_qkv",False)
    attn_reset = optional(kwargs,"attn_reset",False)
    strict_model_load = optional(kwargs,"strict_model_load",True)
    skip_mismatch_model_load = optional(kwargs,"skip_mismatch_model_load",False)

    # -- init model --
    model = Uformer(img_size=input_size, in_chans=nchnls, embed_dim=embed_dim,
                    depths=depths, win_size=win_size, mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias, token_projection=token_projection,
                    token_mlp=token_mlp, downsample=Downsample,
                    upsample=Upsample, modulator=modulator,
                    cross_modulator=cross_modulator,dd_in=dd_in)
    model = model.to(device)

    # -- fill weights --
    if load_pretrained:
        prefix = pretrained_prefix
        state_fn = get_pretrained_path(noise_version,pretrained_path)
        out_attn_mode = attn_mode
        load_checkpoint_qkv(model,state_fn,in_attn_mode,
                            out_attn_mode,embed_dim,prefix=prefix,
                            reset_new=reset_qkv,attn_reset=attn_reset,
                            strict=strict_model_load,
                            skip_mismatch_model_load=skip_mismatch_model_load,
                            nblocks=nblocks)
        # load_checkpoint_module(model,pretrained_path)

    # -- eval mode as default --
    model.eval()

    return model


def get_default_state(noise_version):
    fdir = Path(__file__).absolute().parents[0] / "../../../" # parent of "./lib"
    print("loadig: ",noise_version)
    if noise_version == "noise":
        state_fn = fdir / "weights/Uformer_sidd_B.pth"
    elif noise_version == "blur":
        state_fn = fdir / "weights/Uformer_gopro_B.pth"
    else:
        state_fn = None
        # raise ValueError(f"Uknown noise_version [{noise_version}]")
    if not(state_fn is None):
        assert os.path.isfile(str(state_fn))
        # model_state = th.load(str(state_fn))
    return state_fn
