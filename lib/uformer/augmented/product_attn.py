# -- torch network deps --
import torch as th
import torch.nn as nn
from einops import rearrange,repeat

# -- extra deps --
from timm.models.layers import trunc_normal_

# -- project deps --
from .proj import ConvProjection,LinearProjection,ConvProjectionNoReshape

# -- dnls --
import dnls

class ProductAttention(nn.Module):
    def __init__(self, dim, win_size,num_heads, token_projection='linear',
                 qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.,
                 ps=1,pt=1,k=-1,ws=8,wt=0,dil=1,stride0=1,stride1=1,
                 nbwd=1,rbwd=False,exact=False,bs=-1):

        super().__init__()

        # -- search info --
        self.k = k
        self.ps = ps
        self.pt = pt
        self.ws = ws
        self.wt = wt
        self.dil = dil
        self.stride0 = stride0
        self.stride1 = stride1
        self.nbwd = nbwd
        self.rbwd = rbwd
        self.exact = exact
        self.bs = bs

        # -- attn info --
        self.dim = dim
        self.win_size = win_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            th.zeros((2 * win_size[0] - 1) * (2 * win_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = th.arange(self.win_size[0]) # [0,...,Wh-1]
        coords_w = th.arange(self.win_size[1]) # [0,...,Ww-1]
        coords = th.stack(th.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = th.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.win_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.win_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.win_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        # print(self.relative_position_bias_table.shape,relative_position_index.shape)
        self.register_buffer("relative_position_index", relative_position_index)
        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.qkv = ConvProjectionNoReshape(dim,num_heads,dim//num_heads,bias=qkv_bias)

        self.token_projection = token_projection
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.softmax = nn.Softmax(dim=-1)

    def get_weights(self,module):
        weights = []
        for name,mod in module.named_parameters():
            flat = mod.data.ravel()
            weights.append(flat)
        weights = th.cat(weights,0)
        return weights

    def forward(self, vid, attn_kv=None, mask=None):

        # -- unpack --
        vid = rearrange(vid,'t h w c -> t c h w')
        T, C, H, W = vid.shape
        # print("vid.shape: ",vid.shape)
        # print("vid[min,max]: ",vid.min().item(),vid.max().item())
        qkv_weights = self.get_weights(self.qkv)
        # print("qkv_weights[min,max]: ",
        #       qkv_weights.min().item(),qkv_weights.max().item())
        # print("qkv_weights[nan]: ",th.any(th.isnan(qkv_weights)))

        # -- init --
        mask = None
        rel_pos = self.get_rel_pos()
        search,wpsum,fold = self.init_dnls(vid.shape,vid.device,self.num_heads)

        # -- qkv --
        q_vid, k_vid, v_vid = self.qkv(vid,attn_kv)
        q_vid = q_vid * self.scale
        #print("q_vid.shape:",q_vid.shape,q_vid.shape[1]//self.num_heads,self.num_heads)

        # -- attn map --
        stride0 = self.stride0
        stride1 = self.stride1
        ntotal = T*((H-1)//stride0+1)*((W-1)//stride0+1)
        # print(ntotal,T,H,W,stride0)
        # print("q_vid[min,max]: ",q_vid.min().item(),q_vid.max().item())
        # print("k_vid[min,max]: ",k_vid.min().item(),k_vid.max().item())
        # print(ntotal,q_vid.shape,k_vid.shape)
        dists,inds = search(q_vid,0,ntotal,k_vid)

        # -- debug --
        # any_nan = th.any(th.isnan(dists))
        # print(dists)
        # print("nans [0]: ",any_nan)
        # if any_nan: exit(0)
        # print("dists[min,max]: ",dists.min().item(),dists.max().item())

        if search.ws != 8: # don't match
            dists = self.softmax(dists)
        else:
            dists = search.window_attn_mod(dists,rel_pos,mask,vid.shape)
        # -- debug --
        # any_nan = th.any(th.isnan(dists))
        # print("nans [0.5]: ",any_nan)
        # if any_nan: exit(0)
        # print("[softmax] dists[min,max]: ",dists.min().item(),dists.max().item())

        dists = self.attn_drop(dists)

        # -- debug --
        # any_nan = th.any(th.isnan(dists))
        # print("nans [1]: ",any_nan)
        # if any_nan: exit(0)

        # -- prod with "v" --
        dists = dists.contiguous()
        x = wpsum(v_vid,dists,inds)
        ps = x.shape[-1]
        x = rearrange(x,'(o n) h c ph pw -> (o ph pw) n (h c)',o=ntotal)

        # -- debug --
        # any_nan = th.any(th.isnan(x))
        # print("nans [2]: ",any_nan)
        # if any_nan: exit(0)

        # -- proj --
        x = self.proj(x)
        x = self.proj_drop(x)

        # -- prepare for folding --
        x = rearrange(x,'(o ph pw) n c -> (o n) 1 1 c ph pw',ph=ps,pw=ps)
        x = x.contiguous()
        # print("x[min,max]: ",x.min().item(),x.max().item())

        # -- fold --
        fold(x,0)

        # -- unpack --
        any_zero = th.any(th.abs(fold.zvid)<1e-10)
        any_fold_nan = th.any(th.isnan(fold.vid))
        vid = fold.vid / fold.zvid
        vid = rearrange(vid,'t c h w -> t h w c')

        # -- debug --
        any_nan = th.any(th.isnan(vid))
        if any_nan:
            print("found a nan!: ",any_nan,any_zero,any_fold_nan)
            exit(0)

        return vid

    def get_rel_pos(self):
        if self.relative_position_bias_table is None:
            return None
        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)].view(
                self.win_size[0] * self.win_size[1],
                self.win_size[0] * self.win_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        # print(relative_position_bias.shape)
        relative_position_bias = relative_position_bias.\
            permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        # ratio = attn.size(-1)//relative_position_bias.size(-1)
        # relative_position_bias = repeat(relative_position_bias,
        #                                 'nH l c -> nH l (c d)', d = ratio)
        return relative_position_bias


    def init_dnls(self,vshape,device,nheads):

        # -- unpack params --
        k       = self.k
        if k == 0: k = -1
        ps      = self.ps
        pt      = self.pt
        ws      = self.ws
        wt      = self.wt
        dil     = self.dil
        stride0 = self.stride0
        stride1 = self.stride1
        nbwd    = self.nbwd
        rbwd    = self.rbwd
        exact   = self.exact
        # print("k,ps,ws,wt: ",k,ps,ws,wt)
        # print(rbwd,nbwd)

        # -- fixed --
        fflow,bflow = None,None
        use_k = k > 0
        reflect_bounds = True
        use_search_abs = False
        only_full = False
        use_adj = False
        full_ws = True
        stype = "prod_with_heads"
        # stype = "window"
        # print(k,ps,pt,ws,wt,dil,stride0,stride1,nbwd,rbwd,exact)
        # exit(0)

        # -- init --
        search = dnls.search.init(stype,fflow, bflow, k,
                                  ps, pt, ws, wt, nheads,
                                  chnls=-1,dilation=dil,
                                  stride0=stride0,stride1=stride1,
                                  reflect_bounds=reflect_bounds,
                                  use_k=use_k,use_adj=use_adj,
                                  search_abs=use_search_abs,full_ws=full_ws,
                                  h0_off=0,w0_off=0,h1_off=0,w1_off=0,
                                  exact=exact,nbwd=nbwd,rbwd=rbwd)
        wpsum = dnls.reducers.WeightedPatchSumHeads(ps, pt, h_off=0, w_off=0,
                                                    dilation=dil,
                                                    reflect_bounds=reflect_bounds,
                                                    adj=0, exact=exact)
        fold = dnls.iFoldz(vshape,None,stride=stride0,dilation=dil,
                           adj=0,only_full=only_full,
                           use_reflect=reflect_bounds,device=device)
        return search,wpsum,fold

    def extra_repr(self) -> str:
        return f'dim={self.dim}, win_size={self.win_size}, num_heads={self.num_heads}'

    def flops(self, H, W):
        # calculate flops for 1 window with token length of N
        # print(N, self.dim)
        flops = 0
        N = self.win_size[0]*self.win_size[1]
        nW = H*W/N
        # qkv = self.qkv(x)
        # flops += N * self.dim * 3 * self.dim
        flops += self.qkv.flops(H*W, H*W)

        # attn = (q @ k.transpose(-2, -1))

        flops += nW * self.num_heads * N * (self.dim // self.num_heads) * N
        #  x = (attn @ v)
        flops += nW * self.num_heads * N * N * (self.dim // self.num_heads)

        # x = self.proj(x)
        flops += nW * N * self.dim * self.dim
        print("W-MSA:{%.2f}"%(flops/1e9))
        return flops
