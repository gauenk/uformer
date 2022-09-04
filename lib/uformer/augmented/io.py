
# -- misc --
import sys,os,copy
from pathlib import Path

# -- torch --
import torch as th

# -- linalg --
import numpy as np

# -- modules --
from .uformer import Uformer

# -- misc imports --
from ..common import optional,select_sigma
from ..utils.model_utils import load_checkpoint_module,load_checkpoint_qkv
from ..utils.model_utils import remove_lightning_load_state

def load_model(*args,**kwargs):

    # -- defaults changed by noise version --
    noise_version = optional(kwargs,'noise_version',"noise")
    if noise_version == "noise":
        default_modulator = True
        default_depth = [1, 2, 8, 8, 2, 8, 8, 2, 1]
        # default_modulator = False
        # default_depth = [2, 2, 2, 2, 2, 2, 2, 2, 2]
    elif noise_version == "blur":
        default_modulator = True
        default_depth = [1, 2, 8, 8, 2, 8, 8, 2, 1]
    else:
        raise ValueError(f"Uknown noise version [{noise_version}]")

    # -- get cfg --
    nchnls = optional(kwargs,'nchnls',3)
    input_size = optional(kwargs,'input_size',128)
    depths = optional(kwargs,'input_depth',default_depth)
    device = optional(kwargs,'device','cuda:0')

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

    # -- relevant configs --
    attn_mode = optional(kwargs,'attn_mode',"window_dnls")
    k = optional(kwargs,'k',-1)
    ps = optional(kwargs,'ps',1)
    pt = optional(kwargs,'pt',1)
    stride0 = optional(kwargs,'stride0',1)
    stride1 = optional(kwargs,'stride1',1)
    ws = optional(kwargs,'ws',8)
    wt = optional(kwargs,'wt',0)
    nbwd = optional(kwargs,'nbwd',1)
    rbwd = optional(kwargs,'rbwd',False)
    exact = optional(kwargs,'exact',False)
    bs = optional(kwargs,'bs',-1)

    # -- init model --
    model = Uformer(img_size=input_size, in_chans=nchnls, embed_dim=embed_dim,
                    depths=depths, win_size=win_size, mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias, token_projection=token_projection,
                    token_mlp=token_mlp,modulator=modulator,
                    cross_modulator=cross_modulator,dd_in=dd_in,
                    attn_mode=attn_mode,ps=ps,pt=pt,ws=ws,wt=wt,k=k,
                    stride0=stride0,stride1=stride1,
                    nbwd=nbwd,rbwd=rbwd,exact=exact,bs=bs)
    model = model.to(device)

    # -- load weights --
    fdir = Path(__file__).absolute().parents[0] / "../../../" # parent of "./lib"
    lit = False
    if noise_version == "noise":
        state_fn = fdir / "weights/Uformer_sidd_B.pth"
        lit = False
    elif noise_version == "blur":
        state_fn = fdir / "weights/Uformer_gopro_B.pth"
    else:
        raise ValueError(f"Uknown noise_version [{noise_version}]")
    assert os.path.isfile(str(state_fn))

    load_pretrained = optional(kwargs,"load_pretrained",True)
    if load_pretrained:
        main_mode,sub_mode = attn_mode.split("_")
        if attn_mode in ["window_default","window_refactored"]:
            load_checkpoint_module(model,state_fn)
        else:
            load_checkpoint_qkv(model,state_fn)

    # -- eval mode as default --
    model.eval()

    return model

