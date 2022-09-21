
# -- misc --
import os,math,tqdm
import pprint,copy
pp = pprint.PrettyPrinter(indent=4)

# -- linalg --
import numpy as np
import torch as th
from einops import rearrange,repeat

# -- data mngmnt --
from pathlib import Path
from easydict import EasyDict as edict

# -- data --
import data_hub

# -- caching results --
import cache_io

# -- network --
import uformer
from uformer import lightning
import uformer.configs as configs
from uformer import exps as exps_menu
import uformer.utils.gpu_mem as gpu_mem
from uformer.utils.timer import ExpTimer
from uformer.utils.metrics import compute_psnrs,compute_ssims
from uformer.lightning import UformerLit,MetricsCallback
from uformer.utils.misc import rslice,write_pickle,read_pickle,optional

# -- lightning module --
import torch
import pytorch_lightning as pl
from pytorch_lightning import Callback
from pytorch_lightning.loggers import CSVLogger
# from pytorch_lightning.callbacks import RichProgressBar
from pytorch_lightning.callbacks import ModelCheckpoint,StochasticWeightAveraging
from pytorch_lightning.utilities.distributed import rank_zero_only

def launch_training(_cfg):

    # -=-=-=-=-=-=-=-=-
    #
    #     Init Exp
    #
    # -=-=-=-=-=-=-=-=-

    # -- set-up --
    cfg = copy.deepcopy(_cfg)
    cache_io.exp_strings2bools(cfg)
    configs.set_seed(cfg.seed)
    root = Path(__file__).parents[0].absolute()

    # -- create timer --
    timer = ExpTimer()

    # -- init log dir --
    log_dir = root / "output/log/" / str(cfg.uuid)
    print("Log Dir: ",log_dir)
    if not log_dir.exists():
        log_dir.mkdir(parents=True)
    log_subdirs = ["train"]
    for sub in log_subdirs:
        log_subdir = log_dir / sub
        if not log_subdir.exists(): log_subdir.mkdir()

    # -- prepare save directory for pickles --
    save_dir = root / "output/training/" / cfg.uuid
    if not save_dir.exists():
        save_dir.mkdir(parents=True)

    # -- network --
    print(cfg)
    model_cfg = uformer.extract_model_io(cfg)
    print(model_cfg)
    model = UformerLit(model_cfg,flow=cfg.flow,isize=cfg.isize,
                       batch_size=cfg.batch_size_tr,lr_init=cfg.lr_init,
                       weight_decay=cfg.weight_decay,nepochs=cfg.nepochs,
                       warmup_epochs=cfg.warmup_epochs)

    # -- load dataset with testing mods isizes --
    # model.isize = None
    cfg_clone = copy.deepcopy(cfg)
    # cfg_clone.isize = None
    # cfg_clone.cropmode = "center"
    cfg_clone.nsamples_val = cfg.nsamples_at_testing
    data,loaders = data_hub.sets.load(cfg_clone)

    # -- init validation performance --
    init_val_report = MetricsCallback()
    logger = CSVLogger(log_dir,name="init_val_te",flush_logs_every_n_steps=1)
    trainer = pl.Trainer(gpus=1,precision=32,limit_train_batches=1.,
                         max_epochs=3,log_every_n_steps=1,
                         callbacks=[init_val_report],logger=logger)
    timer.start("init_val_te")
    # trainer.test(model, loaders.te)
    timer.stop("init_val_te")
    init_val_results = init_val_report.metrics
    print("--- Init Validation Results ---")
    print(init_val_results)
    init_val_res_fn = save_dir / "init_val.pkl"
    write_pickle(init_val_res_fn,init_val_results)
    print(timer)

    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
    #
    #          Training
    #
    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

    # -- reset model --
    model.isize = cfg.isize

    # -- data --
    data,loaders = data_hub.sets.load(cfg)
    print(cfg.uuid)
    print("Num Training Vids: ",len(data.tr))
    print("Log Dir: ",log_dir)

    # -- pytorch_lightning training --
    logger = CSVLogger(log_dir,name="train",flush_logs_every_n_steps=1)
    chkpt_fn = cfg.uuid + "-{epoch:02d}-{val_loss:2.2e}"
    checkpoint_callback = ModelCheckpoint(monitor="val_loss",save_top_k=10,mode="min",
                                          dirpath=cfg.checkpoint_dir,filename=chkpt_fn)
    chkpt_fn = cfg.uuid + "-{epoch:02d}"
    cc_recent = ModelCheckpoint(monitor="epoch",save_top_k=10,mode="max",
                                dirpath=cfg.checkpoint_dir,filename=chkpt_fn)
    # swa_callback = StochasticWeightAveraging(swa_lrs=1e-4)
    trainer = pl.Trainer(accelerator="gpu",devices=2,precision=32,
                         accumulate_grad_batches=3,
                         limit_train_batches=.5,limit_val_batches=5,
                         max_epochs=cfg.nepochs-1,log_every_n_steps=1,
                         logger=logger,gradient_clip_val=0.0,
                         callbacks=[checkpoint_callback,cc_recent],
                         strategy="ddp_find_unused_parameters_false")
    timer.start("train")
    # ckpt_path=None
    ckpt_path = "output/checkpoints/6c0e6401-1dc2-49ff-a1f7-bb712eb55082-epoch=16.ckpt"
    trainer.fit(model, loaders.tr, loaders.val, ckpt_path=ckpt_path)
    timer.stop("train")
    best_model_path = checkpoint_callback.best_model_path


    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
    #
    #       Validation Testing
    #
    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

    # -- reload dataset with no isizes --
    # model.isize = None
    # cfg_clone = copy.deepcopy(cfg)
    # cfg_clone.isize = None
    # cfg_clone.cropmode = "center"
    cfg_clone.nsamples_tr = cfg.nsamples_at_testing
    cfg_clone.nsamples_val = cfg.nsamples_at_testing
    data,loaders = data_hub.sets.load(cfg_clone)

    # -- training performance --
    tr_report = MetricsCallback()
    logger = CSVLogger(log_dir,name="train_te",flush_logs_every_n_steps=1)
    trainer = pl.Trainer(gpus=1,precision=32,limit_train_batches=1.,
                         max_epochs=1,log_every_n_steps=1,
                         callbacks=[tr_report],logger=logger)
    timer.start("train_te")
    trainer.test(model, loaders.tr)
    timer.stop("train_te")
    tr_results = tr_report.metrics
    tr_res_fn = save_dir / "train.pkl"
    write_pickle(tr_res_fn,tr_results)

    # -- validation performance --
    val_report = MetricsCallback()
    logger = CSVLogger(log_dir,name="val_te",flush_logs_every_n_steps=1)
    trainer = pl.Trainer(gpus=1,precision=32,limit_train_batches=1.,
                         max_epochs=1,log_every_n_steps=1,
                         callbacks=[val_report],logger=logger)
    timer.start("val_te")
    trainer.test(model, loaders.val)
    timer.stop("val_te")
    val_results = val_report.metrics
    print("--- Tuned Validation Results ---")
    print(val_results)
    val_res_fn = save_dir / "val.pkl"
    write_pickle(val_res_fn,val_results)

    # -- report --
    results = edict()
    results.best_model_path = best_model_path
    results.init_val_results_fn = init_val_res_fn
    results.train_results_fn = tr_res_fn
    results.val_results_fn = val_res_fn
    results.train_time = timer["train"]
    results.test_train_time = timer["train_te"]
    results.test_val_time = timer["val_te"]
    results.test_init_val_time = timer["init_val_te"]

    return results

def main():

    # -- print os pid --
    print("PID: ",os.getpid())

    # -- init --
    verbose = True
    cache_dir = ".cache_io"
    cache_name = "train_gopro"
    cache = cache_io.ExpCache(cache_dir,cache_name)
    cache.clear()

    # -- search info --
    # exps = exps_menu.get_exp_mesh()
    # exps = exps_menu.exps_motivate_paper()
    exps = exps_menu.exps_verify_new_code(mode="train")

    # -- group with default --
    cfg = configs.default_train_cfg()
    cfg.seed = 234

    # -- num to train --
    cfg.nsamples_tr = 0
    cfg.nsamples_val = 0
    cfg.nframes = 1

    # -- trainig --
    cfg.batch_size_tr = 5
    cfg.lr_init = 0.0002/1.
    cfg.weight_decay = 0.02
    cfg.nepochs = 250
    cfg.warmup_epochs = 0
    cfg.noise_version="blur" # fixed.
    # cfg.pretrained_path = "./output/checkpoints/0dffce4c-326a-4152-a2de-5517c53d0ac8-epoch=87.ckpt"
    # cfg.pretrained_path = "./output/checkpoints/aeeaf451-53ae-4923-9dba-a93f6c12b6f3-epoch=33.ckpt"

    # -- [running 0] --
    run0 = False
    if run0:
        cfg.pretrained_path = "./output/checkpoints/3b6d0601-2b39-41c5-b60c-49b9d75e5c6a-epoch=107.ckpt"
        cfg.pretrained_prefix = "net."
        cfg.in_attn_mode = "pd-pd-w-w-w" # the loaded attn mode
        cfg.attn_mode = "pd-pd-w-w-w"
        cfg.reset_qkv = False

    # -- [running 1] --
    run1 = True
    if run1:
        cfg.load_pretrained = "false"
        cfg.pretrained_path = "./output/checkpoints/3b6d0601-2b39-41c5-b60c-49b9d75e5c6a-epoch=107.ckpt"
        cfg.pretrained_prefix = "net."
        cfg.in_attn_mode = "pd-pd-pd-pd-pd" # the loaded attn mode
        cfg.attn_mode = "pd-pd-pd-pd-pd"
        cfg.reset_qkv = True

    # -- pick an exp --
    exps = [exps[2]] # run0
    # exps = [exps[3]] # run1
    nexps = len(exps)

    # cfg.reset_qkv = True
    cfg.load_pretrained = "false"

    # -- mix --
    cache_io.append_configs(exps,cfg) # merge the two

    # -- launch each experiment --
    for exp_num,exp in enumerate(exps):

        # -- info --
        if verbose:
            print("-="*25+"-")
            print(f"Running experiment number {exp_num+1}/{nexps}")
            print("-="*25+"-")
            pp.pprint(exp)

        # -- check if loaded --
        uuid = cache.get_uuid(exp) # assing ID to each Dict in Meshgrid
        # cache.clear_exp(uuid)
        results = cache.load_exp(exp) # possibly load result

        # -- possibly continue from current epochs --
        # todo:

        # -- run experiment --
        if results is None: # check if no result
            exp.uuid = uuid
            results = launch_training(exp)
            cache.save_exp(uuid,exp,results) # save to cache

    # -- results --
    records = cache.load_flat_records(exps)
    print(records.columns)
    print(records['uuid'])
    print(records['best_model_path'].iloc[0])
    print(records['best_model_path'].iloc[1])


    # -- load res --
    uuids = list(records['uuid'].to_numpy())
    cas = list(records['ca_fwd'].to_numpy())
    fns = list(records['init_val_results_fn'].to_numpy())
    res_a = read_pickle(fns[0])
    res_b = read_pickle(fns[1])
    print(uuids)
    print(cas)
    print(res_a['test_psnr'])
    print(res_a['test_index'])
    print(res_b['test_psnr'])
    print(res_b['test_index'])

    fns = list(records['val_results_fn'].to_numpy())
    res_a = read_pickle(fns[0])
    res_b = read_pickle(fns[1])
    print(uuids,cas,fns)
    print(res_a['test_psnr'])
    print(res_a['test_index'])
    print(res_b['test_psnr'])
    print(res_b['test_index'])



# def find_records(path,uuid):
#     files = []
#     for fn in path.iterdir():
#         if uuid in fn:
#             files.append(fn)
#     for fn in files:
#         fn.



if __name__ == "__main__":
    main()
