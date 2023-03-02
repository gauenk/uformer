
import torch as th
from einops import rearrange,repeat
from ..utils.proc_utils import expand2square


def unfold(x,ws):
    B, H, W, C = x.shape
    x = x.view(B, H // ws, ws, W // ws, ws, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, ws, ws,)
    return windows

def fold(windows,ws,B,H,W):
    x = windows.view(B, H // ws, W // ws, ws, ws, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x

def windows_to_qkv(windows,nheads):
    L,N,C = windows.shape
    q = rearrange(windows,'l n (c h) -> l h n c',h=nheads)
    return q,q,q

class NLSearch():

    def __init__(self,k=7, ps=7, ws=8, nheads=1,
                 stride0=4, stride1=1):
        self.k = k
        self.ps = ps
        self.ws = ws
        self.nheads = nheads
        self.stride0 = stride0
        self.stride1 = stride1

    def __call__(self,vid,foo=0,bar=0):
        B,T,C,H,W = vid.shape
        vid = expand2square(vid,self.ws)[0]
        vid = rearrange(vid,'b t c h w -> (b t) h w c')
        windows = unfold(vid,self.ws)
        q,k,v = windows_to_qkv(windows,self.nheads)
        attn = (q @ k.transpose(-2, -1))
        inds = th.zeros_like(attn).type(th.int32)
        return attn,inds

    def flops(self,B,C,H,W):

        #
        # -- init --
        #
        ws = self.ws
        assert (H % ws == 0) and (W % ws == 0)
        N = ws**2
        nW = H*W/N

        #
        # -- compute --
        #

        # attn = (q @ k.transpose(-2, -1))
        nflops = nW * self.nheads * N * (C // self.nheads) * N
        #  x = (attn @ v)
        nflops += nW * self.nheads * N * N * (C // self.nheads)
        nflops = B * nflops
        return nflops
