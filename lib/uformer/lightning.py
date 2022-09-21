

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

# -- optical flow --
# import svnlb

# -- caching results --
import cache_io

# -- network --
import uformer
import uformer.utils.io as io
import uformer.configs as configs
import uformer.utils.gpu_mem as gpu_mem
from uformer.utils.timer import ExpTimer
from uformer.utils.metrics import compute_psnrs,compute_ssims
from uformer.utils.misc import rslice,write_pickle,read_pickle
from uformer.utils.model_utils import filter_rel_pos
from uformer.utils.model_utils import reset_product_attn_mods
# from uformer.utils.model_utils import qkv_convert_state

# -- learning --
from uformer.warmup_scheduler import GradualWarmupScheduler

# -- generic logging --
import logging
logging.basicConfig()

# -- lightning module --
import torch
import pytorch_lightning as pl
from pytorch_lightning import Callback
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.utilities.distributed import rank_zero_only

# -- local --
from .common import optional

def grab_grad(model):
    for param in model.parameters():
        if hasattr(param,'weight'):
            print(param.weight.grad)
            exit(0)

class UformerLit(pl.LightningModule):

    def __init__(self,model_cfg,flow=True,isize=None,batch_size=32,lr_init=0.0002,
                 weight_decay=0.02,nepochs=250,warmup_epochs=3,scheduler="default"):
        super().__init__()

        # -- meta params --
        self.flow = flow
        self.isize = isize

        # -- learning --
        self.batch_size = batch_size
        self.lr_init = lr_init
        self.weight_decay = weight_decay
        self.nepochs = nepochs
        self.warmup_epochs = warmup_epochs
        self.scheduler = scheduler

        # -- load model --
        self.net = uformer.load_model(**model_cfg)
        self.net.train()
        self.net._apply_freeze()

        # -- set logger --
        self.gen_loger = logging.getLogger('lightning')
        self.gen_loger.setLevel("NOTSET")
        self.attn_mode = model_cfg['attn_mode']

    def forward(self,vid,clamp=False):

        # -- pick if prod --
        use_prod = not("_" in self.attn_mode)
        if not(use_prod):
            main,sub = self.attn_mode.split("_")
            use_prod = main == "product"

        # -- fwd pass --
        if use_prod:
            return self.forward_product(vid,clamp=clamp)
        else:
            return self.forward_default(vid,clamp=clamp)

    def forward_product(self,vid,clamp=False):
        flows = self._get_flow(vid)
        deno = self.net(vid,flows=flows)
        if clamp:
            deno = th.clamp(deno,0.,1.)
        return deno

    def forward_default(self,vid,clamp=False):
        flows = self._get_flow(vid)
        # model = self._model[0]
        # model.model = self.net
        model = self.net
        if self.isize is None:
            deno = self.net(vid)
        else:
            deno = self.net(vid,flows=flows)
        if clamp:
            deno = th.clamp(deno,0.,1.)
        return deno

    def _get_flow(self,vid):
        if self.flow == True:
            noisy_np = vid.cpu().numpy()
            if noisy_np.shape[1] == 1:
                noisy_np = np.repeat(noisy_np,3,axis=1)
            flows = svnlb.compute_flow(noisy_np,30.)
            flows = edict({k:th.from_numpy(v).to(self.device) for k,v in flows.items()})
        else:
            t,c,h,w = vid.shape
            zflows = th.zeros((t,2,h,w)).to(self.device)
            flows = edict()
            flows.fflow,flows.bflow = zflows,zflows
        return flows

    def get_default_optim(self):
        optim = th.optim.AdamW(self.parameters(),
                               lr=self.lr_init, betas=(0.9, 0.999),
                               eps=1e-8, weight_decay=self.weight_decay)
        warmup_epochs = self.warmup_epochs
        scheduler_cosine = th.optim.lr_scheduler.CosineAnnealingLR(optim,
                                self.nepochs-warmup_epochs, eta_min=1e-6)
        if warmup_epochs > 0:
            scheduler = GradualWarmupScheduler(optim, multiplier=1,
                                               total_epoch=warmup_epochs,
                                               after_scheduler=scheduler_cosine)
        else:
            scheduler = scheduler_cosine
        return optim, scheduler

    def get_steplr_optim(self):
        optim = th.optim.AdamW(self.parameters(),
                               lr=self.lr_init, betas=(0.9, 0.999),
                               eps=1e-8, weight_decay=self.weight_decay)
        step_size = 5
        scheduler = th.optim.lr_scheduler.StepLR(optim, step_size, gamma=0.5,
                                                 last_epoch=-1)
        return optim, scheduler

    def configure_optimizers(self):
        if self.scheduler == "default":
            optim,scheduler = get_default_optim()
        elif self.scheduler == "step_lr":
            optim,scheduler = get_steplr_optim()
        else:
            raise ValueError("Uknown scheduler.")
        return [optim], [scheduler]

    def training_step(self, batch, batch_idx):

        # -- each sample in batch --
        loss = 0 # init @ zero
        nbatch = len(batch['blur'])
        denos,cleans = [],[]
        for i in range(nbatch):
            # th.cuda.empty_cache()
            deno_i,clean_i,loss_i = self.training_step_i(batch, i)
            loss += loss_i
            denos.append(deno_i)
            cleans.append(clean_i)
        loss = loss / nbatch

        # -- append --
        denos = th.stack(denos)
        cleans = th.stack(cleans)
        # grab_grad(self.net)

        # -- log --
        self.log("train_loss", loss.item(), on_step=True,
                 on_epoch=False,batch_size=self.batch_size)

        # -- terminal log --
        psnr = np.mean(compute_psnrs(denos,cleans,div=1.)).item()
        self.gen_loger.info("train_psnr: %2.2f" % psnr)
        # print("train_psnr: %2.2f" % val_psnr)
        self.log("train_psnr", psnr, on_step=True,
                 on_epoch=False, batch_size=self.batch_size)

        # -- scheduler step --
        # sch = self.lr_schedulers()
        # sch.step()

        return loss

    def training_step_i(self, batch, i):

        # -- unpack batch
        noisy = batch['blur'][i]/255.
        clean = batch['sharp'][i]/255.
        region = batch['region'][i]
        # print("noisy.shape: ",noisy.shape)

        # -- get data --
        noisy = rslice(noisy,region)
        clean = rslice(clean,region)

        # -- foward --
        deno = self.forward(noisy,False)

        # -- save a few --
        # io.save_burst(deno,"./output/","deno")
        # io.save_burst(noisy,"./output/","noisy")
        # io.save_burst(clean,"./output/","clean")
        # exit(0)

        # -- report loss --
        eps = 1e-3
        diff = th.sqrt((clean - deno)**2 + eps**2)
        loss = th.mean(diff)

        return deno.detach(),clean,loss


    def validation_step(self, batch, batch_idx):

        # -- denoise --
        noisy,clean = batch['blur'][0]/255.,batch['sharp'][0]/255.
        region = batch['region'][0]
        noisy = rslice(noisy,region)
        clean = rslice(clean,region)

        # -- forward --
        gpu_mem.print_peak_gpu_stats(False,"val",reset=True)
        with th.no_grad():
            deno = self.forward(noisy,True)
        mem_res,mem_alloc = gpu_mem.print_peak_gpu_stats(False,"val",reset=True)

        # -- loss --
        loss = th.mean((clean - deno)**2)
        _loss = loss.item()

        # -- report --
        self.log("val_loss", _loss, on_step=False,
                 on_epoch=True,batch_size=1, sync_dist=True)
        self.log("val_mem_res", mem_res, on_step=False,
                 on_epoch=True,batch_size=1, sync_dist=True)
        self.log("val_mem_alloc", mem_alloc, on_step=False,
                 on_epoch=True,batch_size=1, sync_dist=True)


        # -- terminal log --
        val_psnr = np.mean(compute_psnrs(deno,clean,div=1.)).item()
        self.gen_loger.info("val_psnr: %2.2f" % val_psnr)

    def test_step(self, batch, batch_nb):

        # -- denoise --
        index,region = batch['index'][0],batch['region'][0]
        noisy,clean = batch['blur'][0]/255.,batch['sharp'][0]/255.
        noisy = rslice(noisy,region)
        clean = rslice(clean,region)

        # -- forward --
        gpu_mem.print_peak_gpu_stats(False,"test",reset=True)
        with th.no_grad():
            deno = self.forward(noisy,True)
        mem_res,mem_alloc = gpu_mem.print_peak_gpu_stats(False,"test",reset=True)

        # -- compare --
        loss = th.mean((clean - deno)**2)
        psnr = np.mean(compute_psnrs(deno,clean,div=1.)).item()
        ssim = np.mean(compute_ssims(deno,clean,div=1.)).item()

        # -- terminal log --
        self.log("psnr", psnr, on_step=True, on_epoch=False, batch_size=1)
        self.log("ssim", ssim, on_step=True, on_epoch=False, batch_size=1)
        self.log("index",  int(index.item()),on_step=True,on_epoch=False,batch_size=1)
        self.log("mem_res",  mem_res, on_step=True, on_epoch=False, batch_size=1)
        self.log("mem_alloc",  mem_alloc, on_step=True, on_epoch=False, batch_size=1)
        self.gen_loger.info("te_psnr: %2.2f" % psnr)

        # -- log --
        results = edict()
        results.test_loss = loss.item()
        results.test_psnr = psnr
        results.test_ssim = ssim
        results.test_mem_alloc = mem_alloc
        results.test_mem_res = mem_res
        results.test_index = index.cpu().numpy().item()
        return results

class MetricsCallback(Callback):
    """PyTorch Lightning metric callback."""

    def __init__(self):
        super().__init__()
        self.metrics = {}

    def _accumulate_results(self,each_me):
        for key,val in each_me.items():
            if not(key in self.metrics):
                self.metrics[key] = []
            if hasattr(val,"ndim"):
                ndim = val.ndim
                val = val.cpu().numpy().item()
            self.metrics[key].append(val)

    @rank_zero_only
    def log_metrics(self, metrics, step):
        # metrics is a dictionary of metric names and values
        # your code to record metrics goes here
        print("logging metrics: ",metrics,step)

    def on_train_epoch_end(self, trainer, pl_module):
        each_me = copy.deepcopy(trainer.callback_metrics)
        self._accumulate_results(each_me)

    def on_validation_epoch_end(self, trainer, pl_module):
        each_me = copy.deepcopy(trainer.callback_metrics)
        self._accumulate_results(each_me)

    def on_test_epoch_end(self, trainer, pl_module):
        each_me = copy.deepcopy(trainer.callback_metrics)
        self._accumulate_results(each_me)

    def on_train_batch_end(self, trainer, pl_module, outs,
                           batch, batch_idx, dl_idx):
        each_me = copy.deepcopy(trainer.callback_metrics)
        self._accumulate_results(each_me)


    def on_validation_batch_end(self, trainer, pl_module, outs,
                                batch, batch_idx, dl_idx):
        each_me = copy.deepcopy(trainer.callback_metrics)
        self._accumulate_results(each_me)

    def on_test_batch_end(self, trainer, pl_module, outs,
                          batch, batch_idx, dl_idx):
        self._accumulate_results(outs)


def remove_lightning_load_state(state):
    names = list(state.keys())
    for name in names:
        name_new = name.split(".")[1:]
        name_new = ".".join(name_new)
        state[name_new] = state[name]
        del state[name]
