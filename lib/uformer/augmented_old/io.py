
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

# -- extract config --
from dev_basics.configs import ExtractConfig
econfig = ExtractConfig(__file__)
extract_config = econfig.extract_config

# -- load model --
@econfig.set_init
def load_model(cfg):

    # -- relevant configs --
    econfig.init(cfg)

    arch_cfg = econfig.optional_pairs(cfg,get_arch_pairs())
    search_cfg = econfig.optional_pairs(cfg,get_search_pairs())
    io_cfg = econfig.optional_pairs(cfg,get_io_pairs())

    # -- load configs --
    noise_version = econfig.optional_field(cfg,'noise_version',"noise")
    reset_qkv = econfig.optional_field(cfg,"reset_qkv",False)
    attn_reset = econfig.optional_field(cfg,"attn_reset",False)
    strict_model_load = econfig.optional_field(cfg,"strict_model_load",True)
    skip_mismatch_model_load = econfig.optional_field(cfg,
                                                      "skip_mismatch_model_load",False)
    if econfig.is_init: return

    # -- derived --
    nblocks = len(arch_cfg.depths)
    arch_cfg.shift_flag = "window" in search_cfg.attn_mode
    assert len(arch_cfg.depths) == len(arch_cfg.num_heads),"Must match length."

    # -- init model --
    # print(ws,wt,attn_mode)
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

        prefix = io_cfg.pretrained_prefix
        #load_checkpoint
        state_fn = get_pretrained_path(noise_version,io_cfg.pretrained_path)
        out_attn_mode = search_cfg.attn_mode
        # if state_fn is None: break
        print("Loading pretrained file: %s" % str(state_fn))
        exit(0)
        # load_checkpoint_qkv(model,state_fn,in_attn_mode,
        #                     out_attn_mode,embed_dim,prefix=prefix,
        #                     reset_new=reset_qkv,attn_reset=attn_reset,
        #                     strict=strict_model_load,
        #                     skip_mismatch_model_load=skip_mismatch_model_load,
        #                     nblocks=nblocks)

    # -- apply network filters [after load] --
    if filter_by_attn_post:
        filter_rel_pos(model,attn_mode)

    # -- eval mode as default --
    model.eval()

    return model


def get_arch_pairs():
    default_modulator = True
    default_depth = [2, 2, 2, 2, 2, 2, 2]
    default_heads = [2, 2, 2, 2, 2, 2, 2]
    pairs = {'nchnls':3,'dd_in':3,"input_size":128,
             "depths":default_depth,"num_heads":default_heads,
             "device":"cuda:0","embed_dim":32,
             "win_size":8,"mlp_ratio":4,"qkv_bias":True,
             'token_projection':'linear','token_mlp':'leff',
             'modulator':default_modulator,'cross_modulator':False,
             "input_proj_depth":1,"output_proj_depth":1}
    return pairs

def get_search_pairs():
    pairs = {"attn_mode":"nls_full","in_attn_mode":"nls_full",
             "k":50,"ps":7,"pt":1,"stride0":4,"stride1":1,
             "ws":8,"wt":0,"nbwd":1,"rbwd":False,"exact":False,
             "bs":-1,"freeze":False,"qk_frac":1.,
             "filter_by_attn_pre":False,
             "filter_by_attn_post":False
             }
    return pairs

def get_io_pairs():
    pairs = {"pretrained_load":False,
             "pretrained_path":".",
             "pretrained_prefix":"module.",
             "pretrained_qkv":"lin2conv"}
    return pairs

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



