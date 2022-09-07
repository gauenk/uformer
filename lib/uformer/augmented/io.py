
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
from ..common import optional as _optional
from ..utils.model_utils import load_checkpoint_module,load_checkpoint_qkv
from ..utils.model_utils import load_checkpoint_mix_qkv
from ..utils.model_utils import remove_lightning_load_state
from ..utils.model_utils import filter_rel_pos

# -- auto populate fields to extract config --
_fields = []
def optional(pydict,field,default,init):
    if not(field in _fields) and init:
        _fields.append(field)
    return _optional(pydict,field,default)

# -- load model --
def load_model(*args,**kwargs):

    # -- allows for all keys to be aggregated at init --
    init = _optional(kwargs,'__init',False) # purposefully weird key

    # -- defaults changed by noise version --
    noise_version = optional(kwargs,'noise_version',"noise",init)
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
    nchnls = optional(kwargs,'nchnls',3,init)
    input_size = optional(kwargs,'input_size',128,init)
    depths = optional(kwargs,'input_depth',default_depth,init)
    device = optional(kwargs,'device','cuda:0',init)

    # -- other configs --
    embed_dim = optional(kwargs,'embed_dim',32,init)
    win_size = optional(kwargs,'win_size',8,init)
    mlp_ratio = optional(kwargs,'mlp_ratio',4,init)
    qkv_bias = optional(kwargs,'qkv_bias',True,init)
    token_projection = optional(kwargs,'token_projection','linear',init)
    token_mlp = optional(kwargs,'token_mlp','leff',init)
    modulator = optional(kwargs,'modulator',default_modulator,init)
    cross_modulator = optional(kwargs,'cross_modulator',False,init)
    dd_in = optional(kwargs,'dd_in',3,init)

    # -- relevant configs --
    attn_mode = optional(kwargs,'attn_mode',"window_dnls",init)
    k = optional(kwargs,'k',-1,init)
    ps = optional(kwargs,'ps',1,init)
    pt = optional(kwargs,'pt',1,init)
    stride0 = optional(kwargs,'stride0',1,init)
    stride1 = optional(kwargs,'stride1',1,init)
    ws = optional(kwargs,'ws',8,init)
    wt = optional(kwargs,'wt',0,init)
    nbwd = optional(kwargs,'nbwd',1,init)
    rbwd = optional(kwargs,'rbwd',False,init)
    exact = optional(kwargs,'exact',False,init)
    bs = optional(kwargs,'bs',-1,init)
    freeze = optional(kwargs,'freeze',False,init)

    # -- modify network after load --
    filter_by_attn_pre = optional(kwargs,"filter_by_attn_pre",False,init)
    filter_by_attn_post = optional(kwargs,"filter_by_attn_post",False,init)
    load_pretrained = optional(kwargs,"load_pretrained",True,init)

    # -- break here if init --
    if init: return

    # -- init model --
    model = Uformer(img_size=input_size, in_chans=nchnls, embed_dim=embed_dim,
                    depths=depths, win_size=win_size, mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias, token_projection=token_projection,
                    token_mlp=token_mlp,modulator=modulator,
                    cross_modulator=cross_modulator,dd_in=dd_in,
                    attn_mode=attn_mode,ps=ps,pt=pt,ws=ws,wt=wt,k=k,
                    stride0=stride0,stride1=stride1,
                    nbwd=nbwd,rbwd=rbwd,exact=exact,bs=bs,freeze=freeze)
    model = model.to(device)

    # -- apply network filters [before load] --
    if filter_by_attn_pre:
        filter_rel_pos(model,attn_mode)


    # -- load weight --
    if load_pretrained:
        state_fn = pretrained_path(noise_version)
        if "_" in attn_mode:
            main_mode,sub_mode = attn_mode.split("_")
            if attn_mode in ["window_default","window_refactored"]:
                load_checkpoint_module(model,state_fn)
            else:
                load_checkpoint_qkv(model,state_fn)
        elif "-" in attn_mode:
            load_checkpoint_mix_qkv(model,state_fn,attn_mode)

    # -- apply network filters [after load] --
    if filter_by_attn_post:
        filter_rel_pos(model,attn_mode)

    # -- apply dimensions [after load] --
    # if dims_post:
    #     filter_rel_pos(model,attn_mode)

    # -- eval mode as default --
    model.eval()

    return model

# -- run to populate "_fields" --
load_model(__init=True)


def pretrained_path(noise_version):
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
    return state_fn

# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
#
#     Extracting Relevant Fields from Larger Dict
#
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def extract_model_io(cfg):
    # -- used to be manual --
    # fields = ["attn_mode","ws","wt","k","ps","pt","stride0",
    #           "stride1","dil","nbwd","rbwd","exact","bs",
    #           "noise_version","filter_by_attn"]

    # -- auto populated fields --
    fields = _fields
    model_cfg = {}
    for field in fields:
        if field in cfg:
            model_cfg[field] = cfg[field]
    return model_cfg
