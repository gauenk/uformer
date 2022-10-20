"""

The experimental meshgrids used
for training and testing in our project.

"""

# -- cache_io for meshgrid --
import cache_io

# -- data mngmnt --
from pathlib import Path
from easydict import EasyDict as edict

def dcat(dict1,dict2):
    if dict2 is None: return
    for key,val in dict2.items():
        dict1[key] = val

def exp_init(iexps = None, mode = "train"):
    if mode == "train":
        return exp_train_init(iexps)
    elif mode == "test":
        return exp_test_init(iexps)
    else:
        raise ValueError(f"Uknown mode [{mode}]")

def exp_train_init(iexps = None):
    isize = ["256_256"]
    expl = {"isize":isize}
    dcat(expl,iexps)
    expl = exp_default_init(expl)
    return expl

def exp_test_init(iexps = None):
    chkpt = [""]
    use_train = ['false']
    expl = {"chkpt":chkpt,"use_train":use_train}
    dcat(expl,iexps)
    expl = exp_default_init(expl)
    return expl

def exp_default_init(iexps = None):
    # -- input checking --
    if iexps is None: iexps = {}
    assert isinstance(iexps,dict),"Must be dict"

    # -- defaults --
    k,ws,ps,wt = [-1],[8],[1],[0]
    filter_by_attn_pre = ["false"]
    filter_by_attn_post = ["false"]
    attn_mode = ["product_dnls"]
    pt,stride0,stride1 = [1],[1],[1]
    dil,nbwd = [1],[1]
    rbwd,exact = ["true"],["false"]
    bs,flow = [-1],['false']
    load_pretrained = ["true"]
    freeze = ["false"]
    pretrained_path = [""]
    pretrained_prefix = ["module."]
    in_attn_mode = ["window_default"] # original attn mode
    embed_dim = [32]
    attn_reset = ['f-f-f-f-f']
    freeze = ['f-f-f-f-f']
    # model_depths = ["1-2-8-8-2-8-8-2-1"]
    model_depths = ["1-2-8-8-2"]
    input_proj_depth = [4]
    output_proj_depth = [1]
    qk_frac = [1.]

    # -- grid --
    exp_lists = {"attn_mode":attn_mode,"ws":ws,"wt":wt,"k":k,"ps":ps,
                 "pt":pt,"stride0":stride0,"stride1":stride1,"dil":dil,
                 "nbwd":nbwd,"rbwd":rbwd,"exact":exact,"bs":bs,'flow':flow,
                 "load_pretrained":load_pretrained,"pretrained_path":pretrained_path,
                 "pretrained_prefix":pretrained_prefix,"in_attn_mode":in_attn_mode,
                 "freeze":freeze,"filter_by_attn_pre":filter_by_attn_pre,
                 "filter_by_attn_post":filter_by_attn_post,"embed_dim":embed_dim,
                 "attn_reset":attn_reset,"in_attn_mode":in_attn_mode,
                 "attn_mode":attn_mode,"attn_reset":attn_reset,"freeze":freeze,
                 "model_depths":model_depths,"input_proj_depth":input_proj_depth,
                 "output_proj_depth":output_proj_depth,"qk_frac":qk_frac,
    }
    # -- apped new values --
    dcat(exp_lists,iexps) # input overwrites defaults
    return exp_lists

def exps_impact_of_replacing_layers(iexps=None):
    expl = exp_init(iexps)
    expl['attn_mode'] = ["pd-w-w-w-w"]
    expl['freeze'] = ["f-f-t-t-t"]
    exps = cache_io.mesh_pydicts(expl)
    expl['attn_mode'] = ["w-w-w-w-pd"]
    expl['freeze'] = ["t-t-t-f-f"]
    exps += cache_io.mesh_pydicts(expl)
    expl['attn_mode'] = ["pd-w-w-w-pd"]
    expl['freeze'] = ["f-f-t-f-f"]
    exps += cache_io.mesh_pydicts(expl)
    expl['attn_mode'] = ["pd-pd-w-pd-pd"]
    expl['freeze'] = ["f-f-f-f-f"]
    exps += cache_io.mesh_pydicts(expl)
    return exps

def exps_compare_attn_modes(iexps=None):
    expl = exp_init(iexps)
    expl['ws'] = [29]
    expl['wt'] = [3]
    expl['ps'] = [7]
    expl['flow'] = ['true']
    expl['attn_mode'] = ['product_dnls','l2_dnls'] # todo: "ca_squeeze" and "channel"
    expl['load_pretrained'] = ['true']
    expl['filter_by_attn_post'] = ['true']
    exps = cache_io.mesh_pydicts(expl)
    return exps

def exps_impact_of_time_search(iexps=None):
    expl = exp_init(iexps)
    expl['filter_by_attn_post'] = ['true']
    expl['ws'] = [29]
    expl['wt'] = [0,1,2,3]
    expl['ps'] = [7]
    expl['flow'] = ['true']
    expl['attn_mode'] = ['product_dnls']
    exps = cache_io.mesh_pydicts(expl)
    expl['wt'] = [1,2,3]
    expl['flow'] = ['false']
    exps += cache_io.mesh_pydicts(expl)
    return exps

def exps_motivate_paper(iexps=None):
    expl = exp_init(iexps)

    # -- standard --
    expl['ws'] = [8]
    expl['wt'] = [0]
    expl['ps'] = [1]
    expl['k'] = [-1]
    expl['flow'] = ['false']
    expl['attn_mode'] = ['window_default']
    expl['filter_by_attn_post'] = ["false"]
    exps = cache_io.mesh_pydicts(expl)

    # -- ours [shifted search space] --
    expl['ws'] = [8]
    expl['wt'] = [0]
    expl['ps'] = [1]
    expl['k'] = [-1]
    expl['flow'] = ['false']
    expl['attn_mode'] = ['product_dnls']
    # expl['attn_mode'] = ['pd-w-w-w-w']
    expl['freeze'] = ["false"]
    expl['shift_flag'] = ["false"]
    expl['filter_by_attn_post'] = ["false"]
    exps += cache_io.mesh_pydicts(expl)

    # -- ours [fully non-local search] --
    # expl['ws'] = [29]
    # expl['wt'] = [3]
    # expl['ps'] = [7]
    # expl['k'] = [64]
    # expl['flow'] = ['false']
    # expl['stride0'] = ["4-1-1-1-1"]
    # expl['stride1'] = ["4-1-1-1-1"]
    # expl['embed_dim_pd'] = [32]
    # expl['embed_dim_w'] = [32]
    # # expl['stride0'] = [4]
    # # expl['stride1'] = [4]
    # # expl['attn_mode'] = ['product_dnls'] # too slow
    # expl['attn_mode'] = ['pd-w-w-w-w']
    # expl['freeze'] = ["f-f-t-t-t"]
    # expl['filter_by_attn_post'] = ["true"]
    # exps += cache_io.mesh_pydicts(expl)


    # -- ours [fully non-local search on edges] --
    expl['ws'] = ["29-29-8-8-8"]
    expl['wt'] = ["3-3-0-0-0"]
    expl['ps'] = ["7-7-1-1-1"]
    expl['k'] = [64]
    expl['flow'] = ['false']
    expl['stride0'] = ["4-4-1-1-1"]
    expl['stride1'] = ["4-4-1-1-1"]
    expl['embed_dim'] = ["9-9-32-32-32"]
    expl['attn_mode'] = ['pd-pd-pd-pd-pd']
    expl['freeze'] = ["false"]
    expl['filter_by_attn_post'] = ["true"]
    exps += cache_io.mesh_pydicts(expl)

    return exps

def exps_train_with_flow(iexps=None):
    # -- ours [fully non-local search] --
    expl = exp_init(iexps)
    expl['k'] = [64]
    expl['ws'] = [29]
    expl['wt'] = [0]
    expl['ps'] = [7]
    expl['flow'] = ['true','false']
    expl['stride0'] = ["4-4-4-1-1"]
    expl['stride1'] = ["4-4-4-1-1"]
    expl['attn_mode'] = ['product_dnls']
    expl['filter_by_attn_post'] = ["true"]
    exps = cache_io.mesh_pydicts(expl)
    return exps

def exps_verify_new_code(iexps=None,mode="train"):
    if mode == "train":
        return exps_verify_new_code_train(iexps)
    elif mode == "test":
        return exps_verify_new_code_test(iexps)
    else:
        raise ValueError("Unable to verify new code.")

def exps_verify_new_code_train(iexps=None):
    expl = exp_init(iexps,"train")

    # -- [exp a] step 0 --
    expl['in_attn_mode'] = ["w-w-w-w-w"]
    expl['attn_mode'] = ["pd-pd-w-w-w"]
    expl['freeze'] = ["f-f-f-t-t"]
    expl['attn_reset'] = ["t-t-f-f-f"]
    exps = cache_io.mesh_pydicts(expl) # create mesh

    # -- [exp b] step 0 --
    expl['in_attn_mode'] = ["w-w-w-w-w"]
    expl['attn_mode'] = ["w-w-w-w-w"]
    expl['freeze'] = ["f-f-f-t-t"]
    expl['attn_reset'] = ["t-t-f-f-f"]
    exps += cache_io.mesh_pydicts(expl) # create mesh

    # -- [exp c] step 0 --
    expl['in_attn_mode'] = ["w-w-w-w-w"]
    expl['attn_mode'] = ["pd-pd-w-w-w"]
    expl['freeze'] = ["f-f-f-t-t"]
    expl['attn_reset'] = ["t-t-f-f-f"]
    expl['embed_dim'] = ["9-9-32-32-32"]
    expl['stride0'] = ['4-4-1-1-1']
    expl['stride1'] = ['1-1-1-1-1']
    expl['ws'] = ["29-29-8-8-8"]
    expl['wt'] = ["2-2-0-0-0"]
    expl['k'] = ["64-64-0-0-0"]
    expl['ps'] = ["7-7-1-1-1"]
    exps += cache_io.mesh_pydicts(expl) # create mesh

    return exps

def exps_verify_new_code_test(iexps=None):

    # -- init --
    expl = exp_init(iexps,"test")
    expl['use_train'] = ['false']

    # -- check different attn modes --
    expl['attn_mode'] = ["window_default"]
    # exps = cache_io.mesh_pydicts(expl) # create mesh

    # -- load trained --
    expl['load_pretrained'] = ["true"]
    expl['use_train'] = ["false"]

    # -- [version 0] --
    expl['in_attn_mode'] = ["w-w-w-w-w"]
    expl['attn_mode'] = ["w-w-w-w-w"]
    expl['pretrained_prefix'] = ["module."]
    expl['pretrained_path'] = [""]
    exps = cache_io.mesh_pydicts(expl) # create mesh

    # -- [version 1] --
    expl['in_attn_mode'] = ["pd-pd-w-w-w"]
    expl['attn_mode'] = ["pd-pd-w-w-w"]
    expl['pretrained_prefix'] = ["net."]
    expl['pretrained_path'] = ["ad209414"]
    exps += cache_io.mesh_pydicts(expl) # create mesh

    # -- [version 2] --
    expl['in_attn_mode'] = ["pd-pd-w-w-w"]
    expl['attn_mode'] = ["pd-pd-w-w-w"]
    expl['embed_dim'] = ["9-9-32-32-32"]
    expl['ws'] = ["29-29-8-8-8"]
    expl['ps'] = ["7-7-1-1-1"]
    expl['wt'] = ["2-2-0-0-0"]
    expl['k'] = ["64-64-0-0-0"]
    # exps += cache_io.mesh_pydicts(expl) # create mesh

    return exps

# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
#
#
#     Run Denoising Experiments
#
#
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def exps_rgb_denoising(iexps=None,mode="train"):
    if mode == "train":
        return exps_rgb_denoising_train(iexps)
    elif mode == "test":
        return exps_rgb_denoising_test(iexps)
    else:
        raise ValueError("Unable to verify new code.")

def exps_rgb_denoising_train(iexps=None):

    # -- init --
    expl = exp_init(iexps,"train")
    expl['freeze'] = ['false']

    # -- [exp a:0] step 0 --
    expl['input_proj_depth'] = [1]
    expl['output_proj_depth'] = [1]
    expl['in_attn_mode'] = ["w-w-w-w-w"]
    expl['attn_mode'] = ["original"]
    expl['attn_reset'] = ["t-t-t-t-t"]
    exps = cache_io.mesh_pydicts(expl) # create mesh

    # -- [exp b:1] step 0 --
    expl['input_proj_depth'] = [4]
    expl['output_proj_depth'] = [4]
    expl['in_attn_mode'] = ["w-w-w-w-w"]
    expl['attn_mode'] = ["pd-pd-w-w-w"]
    expl['attn_reset'] = ["t-t-f-f-f"]
    expl['embed_dim'] = ["9-9-32-32-32"]
    expl['stride0'] = ['4-4-1-1-1']
    expl['stride1'] = ['1-1-1-1-1']
    expl['ws'] = ["29-29-8-8-8"]
    expl['wt'] = ["2-2-0-0-0"]
    expl['k'] = ["64-64-0-0-0"]
    expl['ps'] = ["7-7-1-1-1"]
    exps += cache_io.mesh_pydicts(expl) # create mesh

    # -- [exp c:2] step 0 -- ["slim"]
    expl['in_attn_mode'] = ["w-w-w"]
    expl['num_heads'] = ['1-2-4']
    expl['attn_mode'] = ["pd-pd-pd"]
    expl['attn_reset'] = ["t-t-t"]
    expl['embed_dim'] = ["9-9-9"]
    expl['stride0'] = ['4-2-1']
    expl['stride1'] = ['1-1-1']
    expl['ws'] = ["29-15-9"]
    expl['wt'] = ["0-0-0"]
    expl['k'] = [64]
    expl['ps'] = [7]
    expl['model_depths'] = ["2-4-4"]
    exps += cache_io.mesh_pydicts(expl) # create mesh

    # -- [exp d:3] step 0 -- ["skinny"]
    expl['model_depths'] = ["2-2-4"]
    exps += cache_io.mesh_pydicts(expl) # create mesh

    # -- [exp e:4] step 0 -- ["mid"]
    expl['in_attn_mode'] = ["w-w-w-w"]
    expl['num_heads'] = ['1-2-4-8']
    expl['attn_mode'] = ["pd-pd-pd-pd"]
    expl['attn_reset'] = ["t-t-t-t"]
    expl['embed_dim'] = ["9-9-9-9"]
    expl['stride0'] = ['4-4-2-1']
    expl['stride1'] = ['1-1-1-1']
    expl['ws'] = ["29-15-9-9"]
    expl['wt'] = ["0-0-0-0"]
    expl['k'] = [64]
    expl['ps'] = [7]
    expl['model_depths'] = ["2-2-2-2"]
    exps += cache_io.mesh_pydicts(expl) # create mesh

    # -- [exp f:5] -- ["qk_frac"]
    expl['in_attn_mode'] = ["w-w-w"]
    expl['num_heads'] = ['1-2-4']
    expl['attn_mode'] = ["pd-pd-pd"]
    expl['attn_reset'] = ["t-t-t"]
    expl['embed_dim'] = ["32-32-32"]
    expl['stride0'] = ['4-2-1']
    expl['stride1'] = ['1-1-1']
    expl['ws'] = ["29-15-9"]
    expl['wt'] = ["0-0-0"]
    expl['k'] = [64]
    expl['ps'] = [7]
    expl['model_depths'] = ["2-4-4"]
    expl['qk_frac'] = [.25]
    exps += cache_io.mesh_pydicts(expl) # create mesh

    # -- [exp f:6] -- ["qk_frac" modded ]
    expl['in_attn_mode'] = ["w-w-w"]
    expl['num_heads'] = ['1-2-4']
    expl['attn_mode'] = ["pd-pd-pd"]
    expl['attn_reset'] = ["t-t-t"]
    expl['embed_dim'] = ["24-24-24"]
    expl['stride0'] = ['4-2-1']
    expl['stride1'] = ['1-1-1']
    expl['ws'] = ["25-15-9"]
    expl['wt'] = ["0-0-0"]
    expl['k'] = [64]
    expl['ps'] = [7]
    expl['model_depths'] = ["2-4-4"]
    expl['qk_frac'] = [.25]
    exps += cache_io.mesh_pydicts(expl) # create mesh

    # -- [exp g:7] --
    expl['num_heads'] = ['1-2-4-8-16']
    # expl['attn_mode'] = ["product_attn"]
    expl['attn_mode'] = ["pd-w-w-w-w"]
    expl['attn_reset'] = ["true"]
    expl['embed_dim'] = ['24-32-32-32-32']
    expl['qk_frac'] = [0.25]
    expl['stride0'] = ['4-2-1-1-1']
    expl['stride1'] = ['1-1-1-1-1']
    expl['ws'] = ["29-15-9-9-9"]
    expl['wt'] = ["0-0-0-0-0"]
    expl['k'] = [64]
    expl['ps'] = ['7-5-3-3-3']
    expl['model_depths'] = ["1-2-8-8-2"]
    expl['pretrained_path'] = [""]
    expl['input_proj_depth'] = [1]
    expl['output_proj_depth'] = [1]
    expl['qk_frac'] = ['.25-1-1-1-1']
    expl['rbwd'] = ['false']
    exps += cache_io.mesh_pydicts(expl) # create mesh

    return exps

def exps_rgb_denoising_skinny_10_20(iexps=None):
    # -- init --
    expl = exp_init(iexps,"test")
    expl['use_train'] = ['false']
    del expl['freeze']# = ['false']
    expl['num_heads'] = ['1-2-4']
    expl['attn_mode'] = ["pd-pd-pd"]
    expl['attn_reset'] = ["t-t-t"]
    expl['embed_dim'] = ["9-9-9"]
    expl['stride0'] = ['4-2-1']
    expl['stride1'] = ['1-1-1']
    expl['ws'] = ["29-15-9"]
    expl['wt'] = ["0-0-0"]
    expl['k'] = [64]
    expl['ps'] = [7]
    expl['model_depths'] = ["2-4-4"]
    expl['pretrained_path'] = ["output/checkpoints/a40d6c5f-d612-42fe-9ecf-de0d93ab28ba-epoch=116.ckpt"]
    expl['input_proj_depth'] = [4]
    # expl['pretrained_path'] = ["ad209414"]
    # expl['in_attn_mode'] = ["w-w-w-w-w"]
    # expl['attn_mode'] = ["pd-pd-w-w-w"]
    # model_depths = ["1-2-8-8-2"]
    exps = cache_io.mesh_pydicts(expl) # create mesh
    expl['wt'] = ['3-0-0']
    exps += cache_io.mesh_pydicts(expl) # create mesh
    expl['wt'] = ['3-3-0']
    exps += cache_io.mesh_pydicts(expl) # create mesh
    expl['wt'] = ['3-3-3']
    exps += cache_io.mesh_pydicts(expl) # create mesh

    return exps

def exps_rgb_denoising_qkfrac_10_20(iexps=None):
    # -- init --
    expl = exp_init(iexps,"test")
    expl['use_train'] = ['false']
    del expl['freeze']# = ['false']
    expl['in_attn_mode'] = ["w-w-w"]
    expl['num_heads'] = ['1-2-4']
    expl['attn_mode'] = ["pd-pd-pd"]
    expl['attn_reset'] = ["t-t-t"]
    expl['embed_dim'] = ["32-32-32"]
    expl['stride0'] = ['4-2-1']
    expl['stride1'] = ['1-1-1']
    expl['ws'] = ["29-15-9"]
    expl['wt'] = ["0-0-0"]
    expl['k'] = [64]
    expl['ps'] = [7]
    expl['model_depths'] = ["2-4-4"]
    expl['qk_frac'] = [.25]
    expl['pretrained_path'] = ["output/checkpoints/b9024811-080a-4909-9458-af0d42c6c5f2-epoch=20.ckpt"]
    expl['input_proj_depth'] = [4]
    # expl['pretrained_path'] = ["ad209414"]
    # expl['in_attn_mode'] = ["w-w-w-w-w"]
    # expl['attn_mode'] = ["pd-pd-w-w-w"]
    # model_depths = ["1-2-8-8-2"]
    exps = cache_io.mesh_pydicts(expl) # create mesh
    expl['wt'] = ['3-0-0']
    exps += cache_io.mesh_pydicts(expl) # create mesh
    expl['wt'] = ['3-3-0']
    exps += cache_io.mesh_pydicts(expl) # create mesh
    expl['wt'] = ['3-3-3']
    exps += cache_io.mesh_pydicts(expl) # create mesh

    return exps

def exps_rgb_denoising_test(iexps=None):

    # -- init --
    expl = exp_init(iexps,"test")
    expl['use_train'] = ['false']
    expl['attn_reset'] = ["false"]
    expl['pretrained_prefix'] = ["net."]

    # -- [exp a] step 0 --
    expl['pretrained_path'] = [""]
    expl['in_attn_mode'] = ["original"]
    expl['attn_mode'] = ["original"]
    expl['model_depths'] = ["none"]
    exps = cache_io.mesh_pydicts(expl) # create mesh

    # -- [exp b] step 0 --
    del expl['freeze']# = ['false']
    expl['num_heads'] = ['1-2-4']
    expl['attn_mode'] = ["pd-pd-pd"]
    expl['attn_reset'] = ["t-t-t"]
    expl['embed_dim'] = ["9-9-9"]
    expl['stride0'] = ['4-2-1']
    expl['stride1'] = ['1-1-1']
    expl['ws'] = ["29-15-9"]
    expl['wt'] = ["0-0-0"]
    expl['k'] = [64]
    expl['ps'] = [7]
    expl['model_depths'] = ["2-4-4"]
    expl['pretrained_path'] = ["a40d6c5f-d612-42"]
    expl['input_proj_depth'] = [4]
    exps += cache_io.mesh_pydicts(expl) # create mesh

    # -- [exp c] step 0 --
    expl['input_proj_depth'] = [1]
    expl['pretrained_path'] = ["f69d3cc4-986f-4f55-b0b9-b5f90b3b5957-epoch=58-val_loss=1.03e-03.ckpt"]
    expl['in_attn_mode'] = ["w-w-w-w-w"]
    expl['attn_mode'] = ["pd-pd-pd-pd-pd"]
    expl['attn_reset'] = ["t-t-t-t-t"]
    expl['embed_dim'] = ["9-9-9-9-9"]
    expl['stride0'] = ['4-4-2-2-2']
    expl['stride1'] = ['1-1-1-1-1']
    expl['ws'] = ["29-15-11-9-9"]
    expl['wt'] = ["2-2-2-0-0"]
    expl['k'] = ["64-64-64-64-64"]
    expl['ps'] = ["7-7-7-5-3"]
    expl['model_depths'] = ["2-2-2-2-2-2-2-2-2"]
    exps += cache_io.mesh_pydicts(expl) # create mesh

    # -- [exp d] --
    expl['pretrained_path'] = ["f69d3cc4-986f-4f55-b0b9-b5f90b3b5957-epoch=58-val_loss=1.03e-03.ckpt"]
    expl['in_attn_mode'] = ["pd-pd-w-w-w"]
    expl['attn_mode'] = ["pd-pd-w-w-w"]
    expl['embed_dim'] = ["9-9-32-32-32"]
    expl['stride0'] = ['4-4-1-1-1']
    expl['stride1'] = ['1-1-1-1-1']
    expl['ws'] = ["29-29-8-8-8"]
    expl['wt'] = ["2-2-0-0-0"]
    expl['k'] = ["64-64-0-0-0"]
    expl['ps'] = ["7-7-1-1-1"]
    exps += cache_io.mesh_pydicts(expl) # create mesh

    return exps

# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
#
#
#       Run Deraining Experiments
#
#
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def exps_deraining(iexps=None,mode="train"):
    if mode == "train":
        return exps_deraining_train(iexps)
    elif mode == "test":
        return exps_deraining_test(iexps)
    else:
        raise ValueError("Unable to verify new code.")

def exps_deraining_train(iexps=None):

    # -- init --
    expl = exp_init(iexps,"train")
    expl['freeze'] = ['false']
    expl['input_proj_depth'] = [4]

    # -- [exp e] step 0 --
    expl['in_attn_mode'] = ["w-w-w"]
    expl['attn_mode'] = ["pd-pd-pd"]
    expl['attn_reset'] = ["true"]
    expl['embed_dim'] = [9]
    expl['stride0'] = ['4-2-1']
    expl['stride1'] = ['1-1-1']
    expl['ws'] = ["29-15-9"]
    expl['wt'] = ["0-0-0"]
    expl['k'] = ["64-64-64"]
    expl['ps'] = ["7-5-3"]
    expl['model_depths'] = ["2-2-2"]
    expl['num_heads'] = ["1-2-4"]
    exps = cache_io.mesh_pydicts(expl) # create mesh

    return exps

def exps_deraining_test(iexps=None):

    # -- init --
    expl = exp_init(iexps,"test")
    expl['use_train'] = ['false']
    expl['attn_reset'] = ["f-f-f-f-f"]
    expl['pretrained_prefix'] = ["net."]

    # -- [exp a] step 0 --
    expl['pretrained_path'] = ["98918264-f802-44da-9255-8788f37cb0fc-epoch=60-val_loss=9.69e-04.ckpt"]
    expl['in_attn_mode'] = ["w-w-w-w-w"]
    expl['attn_mode'] = ["w-w-w-w-w"]
    exps = cache_io.mesh_pydicts(expl) # create mesh


    # -- [exp b] step 0 --
    expl['pretrained_path'] = ["ad209414"]
    expl['in_attn_mode'] = ["w-w-w-w-w"]
    expl['attn_mode'] = ["pd-pd-w-w-w"]
    model_depths = ["1-2-8-8-2"]
    exps += cache_io.mesh_pydicts(expl) # create mesh

    # -- [exp c] step 0 --
    expl['pretrained_path'] = ["f69d3cc4-986f-4f55-b0b9-b5f90b3b5957-epoch=58-val_loss=1.03e-03.ckpt"]
    expl['in_attn_mode'] = ["w-w-w-w-w"]
    expl['attn_mode'] = ["pd-pd-pd-pd-pd"]
    expl['attn_reset'] = ["t-t-t-t-t"]
    expl['embed_dim'] = ["9-9-9-9-9"]
    expl['stride0'] = ['4-4-2-2-2']
    expl['stride1'] = ['1-1-1-1-1']
    expl['ws'] = ["29-15-11-9-9"]
    expl['wt'] = ["2-2-2-0-0"]
    expl['k'] = ["64-64-64-64-64"]
    expl['ps'] = ["7-7-7-5-3"]
    expl['model_depths'] = ["2-2-2-2-2-2-2-2-2"]
    exps += cache_io.mesh_pydicts(expl) # create mesh

    # -- [exp e] --
    expl['pretrained_path'] = ["f69d3cc4-986f-4f55-b0b9-b5f90b3b5957-epoch=58-val_loss=1.03e-03.ckpt"]
    expl['in_attn_mode'] = ["pd-pd-w-w-w"]
    expl['attn_mode'] = ["pd-pd-w-w-w"]
    expl['embed_dim'] = ["9-9-32-32-32"]
    expl['stride0'] = ['4-4-1-1-1']
    expl['stride1'] = ['1-1-1-1-1']
    expl['ws'] = ["29-29-8-8-8"]
    expl['wt'] = ["2-2-0-0-0"]
    expl['k'] = ["64-64-0-0-0"]
    expl['ps'] = ["7-7-1-1-1"]
    exps += cache_io.mesh_pydicts(expl) # create mesh

    return exps

# --- final mesh --
def get_exp_mesh(iexps=None):
    exps = exps_impact_of_replacing_layers(iexps)
    exps += exps_compare_attn_modes(iexps)
    exps += exps_impact_of_time_search(iexps)
    exps += exps_motivate_paper(iexps)
    return exps


