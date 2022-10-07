"""

Compute a grid of flops values for different model configs

"""

# -- python imports --
import os,copy
import pprint
from pathlib import Path
from easydict import EasyDict as edict
pp = pprint.PrettyPrinter(indent=4)

# - caching --
import cache_io

# -- uformer imports --
import uformer
import uformer.configs as configs
from uformer import exps as exps_menu

# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
#
#    Primary Logic to Computer Flops
#
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def compute_flops(_cfg):

    # -- set-up --
    cfg = copy.deepcopy(_cfg)
    cache_io.exp_strings2bools(cfg)
    configs.set_seed(cfg.seed)
    root = (Path(__file__).parents[0] / ".." ).absolute()

    # -- load model --
    model_cfg = uformer.extract_model_io(cfg)
    model = uformer.load_model(**model_cfg)

    # -- compute flops --
    H,W = [int(x) for x in cfg.isize.split("_")]
    flops = model.flops(H,W)

    # -- compute params --
    params = count_params(model)

    # -- format results --
    results = edict()
    results.flops = flops
    results.params = params

    return results

# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
#
#    Experimental Configuration Grids
#
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-


def create_grids_depth3():
    expl = exps_menu.exp_test_init()
    expl['load_pretrained'] = ["false"]
    expl['freeze'] = ["false"]
    expl['in_attn_mode'] = ["w-w-w"]
    expl['attn_mode'] = ["pd-pd-pd"]
    expl['attn_reset'] = ["f-f-f"]
    expl['embed_dim'] = ["3-6-9","32-32-32"]
    expl['stride0'] = ["4-2-1"]
    expl['stride1'] = ["1-1-1"]
    expl['ws'] = ["29-15-9"]
    expl['wt'] = ["0-0-0"]
    expl['k'] = ["64-64-64"]
    expl['ps'] = ["7-5-3"]
    expl['model_depths'] = ["2-2-2"]
    expl['num_heads'] = ["1-2-4"]
    exps = cache_io.mesh_pydicts(expl) # create mesh

    expl['attn_mode'] = ["w-w-w"]
    expl['attn_reset'] = ["f-f-f"]
    expl['embed_dim'] = ["32-32-32"]
    expl['stride0'] = [1]
    expl['stride1'] = [1]
    expl['ws'] = [8]
    expl['wt'] = [0]
    expl['k'] = [64]
    expl['ps'] = [8]
    expl['model_depths'] = ["2-4-8"]
    expl['num_heads'] = ["1-2-4"]
    exps += cache_io.mesh_pydicts(expl) # create mesh
    return exps

def create_grids_depth4():
    pass

def create_grids():
    exps = create_grids_depth3()
    # exps += create_grids_depth4()
    return exps

# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
#
#             Main Function
#
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def main():

    # -- print os pid --
    print("PID: ",os.getpid())

    # -- init --
    verbose = True
    cache_dir = ".cache_io"
    cache_name = "compute_flops"
    cache = cache_io.ExpCache(cache_dir,cache_name)
    cache.clear()

    # -- get experimental configs --
    cfg = configs.default_train_cfg()
    cfg.seed = 234
    cfg.isize = "512_512"
    # cfg.isize = "256_256"
    exps = create_grids()
    cache_io.append_configs(exps,cfg) # merge the two

    # -- launch each experiment --
    for exp_num,exp in enumerate(exps):

        # -- info --
        if verbose:
            print("-="*25+"-")
            print(f"Running experiment number {exp_num+1}/{len(exps)}")
            print("-="*25+"-")
            pp.pprint(exp)

        # -- check if loaded --
        uuid = cache.get_uuid(exp) # assing ID to each Dict in Meshgrid
        results = cache.load_exp(exp) # possibly load result

        # -- run experiment --
        if results is None: # check if no result
            exp.uuid = uuid
            results = compute_flops(exp)
            cache.save_exp(uuid,exp,results) # save to cache

    # -- results --
    records = cache.load_flat_records(exps)
    vals = ['flops','params','attn_mode','embed_dim','model_depths']
    vals += ['stride0','stride1']
    print(records[vals])


if __name__ == "__main__":
    main()
