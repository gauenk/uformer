
# -- misc --
import sys,os,copy
from pathlib import Path
from functools import partial

# -- torch --
import torch as th

# -- linalg --
import numpy as np

# -- modules --
from .uformer import Uformer

# -- misc imports --
from ..common import optional as _optional
from ..utils.model_utils import load_checkpoint_module,load_checkpoint_qkv
# from ..utils.model_utils import load_checkpoint_mix_qkv
from ..utils.model_utils import remove_lightning_load_state
from ..utils.model_utils import filter_rel_pos#,get_recent_filename
from ..utils.model_io import get_pretrained_path

# -- auto populate fields to extract config --
_fields = []
def optional_full(init,pydict,field,default):
    if not(field in _fields) and init:
        _fields.append(field)
    return _optional(pydict,field,default)

# -- load model --
def load_model(*args,**kwargs):

    # -- allows for all keys to be aggregated at init --
    init = _optional(kwargs,'__init',False) # purposefully weird key
    optional = partial(optional_full,init)

    # -- defaults changed by noise version --
    noise_version = optional(kwargs,'noise_version',"noise")
    default_modulator = True
    default_depth = [1, 2, 8, 8, 2]

    # -- get cfg --
    nchnls = optional(kwargs,'nchnls',3)
    input_size = optional(kwargs,'input_size',128)
    depths = parse_depths(optional(kwargs,'model_depths',default_depth))
    num_heads = parse_heads(optional(kwargs,'num_heads',[1,2,4,8,16]))
    device = optional(kwargs,'device','cuda:0')
    assert len(depths) == len(num_heads),"Must match length."
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
    shift_flag = optional(kwargs,'shift_flag',True)

    # -- relevant configs --
    attn_mode = optional(kwargs,'attn_mode',"window_dnls")
    in_attn_mode = optional(kwargs,'in_attn_mode',attn_mode)
    k = optional(kwargs,'k',-1)
    ps = optional(kwargs,'ps',1)
    pt = optional(kwargs,'pt',1)
    stride0 = optional(kwargs,'stride0',1)
    stride1 = optional(kwargs,'stride1',1)
    ws = optional(kwargs,'ws',8)
    wt = optional(kwargs,'wt',0)
    nbwd = optional(kwargs,'nbwd',1)
    rbwd = optional(kwargs,'rbwd',True)
    exact = optional(kwargs,'exact',False)
    bs = optional(kwargs,'bs',-1)
    freeze = optional(kwargs,'freeze',False)
    input_proj_depth = optional(kwargs,"input_proj_depth",1)
    output_proj_depth = optional(kwargs,"output_proj_depth",1)
    qk_frac = optional(kwargs,'qk_frac',1.)

    # -- modify network after load --
    filter_by_attn_pre = optional(kwargs,"filter_by_attn_pre",False)
    filter_by_attn_post = optional(kwargs,"filter_by_attn_post",False)
    load_pretrained = optional(kwargs,"load_pretrained",True)
    pretrained_path = optional(kwargs,"pretrained_path","")
    pretrained_prefix = optional(kwargs,"pretrained_prefix","module.")
    pretrained_qkv = optional(kwargs,"pretrained_qkv","lin2conv")

    # -- load configs --
    reset_qkv = optional(kwargs,"reset_qkv",False)
    attn_reset = optional(kwargs,"attn_reset",False)
    strict_model_load = optional(kwargs,"strict_model_load",True)
    skip_mismatch_model_load = optional(kwargs,"skip_mismatch_model_load",False)

    # -- break here if init --
    if init: return

    # -- init model --
    model = Uformer(img_size=input_size, in_chans=nchnls,
                    input_proj_depth=input_proj_depth,
                    output_proj_depth=output_proj_depth,
                    depths=depths, num_heads=num_heads,
                    win_size=win_size, mlp_ratio=mlp_ratio,
                    qk_frac=qk_frac, qkv_bias=qkv_bias,
                    token_projection=token_projection,
                    token_mlp=token_mlp,modulator=modulator,
                    cross_modulator=cross_modulator,dd_in=dd_in,
                    attn_mode=attn_mode,ps=ps,pt=pt,ws=ws,wt=wt,k=k,
                    stride0=stride0,stride1=stride1,
                    nbwd=nbwd,rbwd=rbwd,exact=exact,bs=bs,freeze=freeze,
                    embed_dim=embed_dim,shift_flag=shift_flag)
    model = model.to(device)

    # -- apply network filters [before load] --
    if filter_by_attn_pre:
        filter_rel_pos(model,attn_mode)

    # -- load weight --
    if load_pretrained:
        prefix = pretrained_prefix
        #load_checkpoint
        state_fn = get_pretrained_path(noise_version,pretrained_path)
        out_attn_mode = attn_mode
        # if state_fn is None: break
        print("Loading pretrained file: %s" % str(state_fn))
        load_checkpoint_qkv(model,state_fn,in_attn_mode,
                            out_attn_mode,embed_dim,prefix=prefix,
                            reset_new=reset_qkv,attn_reset=attn_reset,
                            strict=strict_model_load,
                            skip_mismatch_model_load=skip_mismatch_model_load,
                            nblocks=nblocks)

    # -- apply network filters [after load] --
    if filter_by_attn_post:
        filter_rel_pos(model,attn_mode)

    # -- eval mode as default --
    model.eval()

    return model


def parse_heads(heads):
    if isinstance(heads,list):
        return heads
    elif isinstance(heads,str):
        heads_l = heads.split("-")
        heads_l = [int(h) for h in heads_l]
        return heads_l
    else:
        raise ValueError(f"Uknown value format for num_heads [{heads}]")

def parse_depths(depths):
    if isinstance(depths,list):# and len(depths) == 5:
        return depths
    elif isinstance(depths,str):
        depths_l = depths.split("-")
        depths_l = [int(d) for d in depths_l]
        return depths_l
    else:
        raise ValueError(f"Uknown value format for depths [{depths}]")

# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
#
#     Extracting Relevant Fields from Larger Dict
#
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def extract_model_io(cfg):
    # -- auto populated fields --
    fields = _fields
    model_cfg = {}
    for field in fields:
        if field in cfg:
            model_cfg[field] = cfg[field]
    return model_cfg

# -- run to populate "_fields" --
load_model(__init=True)


