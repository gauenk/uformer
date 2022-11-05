
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


class NLSearch():

    def __init__(self,k=7, ps=7, ws=10, nheads=1,
                 stride0=4, stride1=1):
        self.k = k
        self.ps = ps
        self.ws = ws
        self.nheads = nheads
        self.stride0 = stride0
        self.stride1 = stride1

    def __call__(self,vid,foo=0,bar=0):
        B,T,C,H,W = vid.shape
        print("vid.shape: ",vid.shape)
        vid = expand2square(vid,self.ws)[0]
        print("vid.shape: ",vid.shape)
        vid = rearrange(vid,'b t c h w -> (b t) h w c')
        print("vid.shape: ",vid.shape)
        windows = unfold(vid,self.ws)
        q,k,v = windows,windows,windows
        attn = (q @ k.transpose(-2, -1))
        inds = th.zeros_like(attn).type(th.int32)
        print("attn.shape: ",attn.shape)
        return attn,inds
