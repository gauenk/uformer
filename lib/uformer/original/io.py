
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


def load_model(*args,**kwargs):

    # -- defaults changed by noise version --
    noise_version = optional(kwargs,'noise_version',"noise")
    default_modulator = True
    default_depth = [1, 2, 8, 8, 2, 8, 8, 2, 1]

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

    # -- init model --
    model = Uformer(img_size=input_size, in_chans=nchnls, embed_dim=embed_dim,
                    depths=depths, win_size=win_size, mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias, token_projection=token_projection,
                    token_mlp=token_mlp, downsample=Downsample,
                    upsample=Upsample, modulator=modulator,
                    cross_modulator=cross_modulator,dd_in=dd_in)
    model = model.to(device)

    # -- load weights --
    # model_sigma = select_sigma(data_sigma)
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

    # -- fill weights --
    load_pretrained = optional(kwargs,"load_pretrained",True)
    if load_pretrained:
        load_checkpoint_module(model,state_fn)

    # -- eval mode as default --
    model.eval()

    return model

