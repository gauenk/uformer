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
from dnls.utils.inds import get_nums_hw

class WindowAttentionRefactored(nn.Module):
    def __init__(self, dim, win_size,num_heads, token_projection='linear',
                 qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
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
        self.register_buffer("relative_position_index", relative_position_index)
        trunc_normal_(self.relative_position_bias_table, std=.02)

        if token_projection =='conv':
            self.qkv = ConvProjection(dim,num_heads,dim//num_heads,bias=qkv_bias)
        elif token_projection =='linear':
            self.qkv = LinearProjection(dim,num_heads,dim//num_heads,bias=qkv_bias)
        else:
            raise Exception("Projection error!")
        # self.qkv = ConvProjectionNoReshape(dim,num_heads,dim//num_heads,bias=qkv_bias)


        self.token_projection = token_projection
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, vid, attn_kv=None, mask=None):

        # -- unpack --
        vid = rearrange(vid,'t h w c -> t c h w').contiguous()
        T, C, H, W = vid.shape

        # -- init --
        rel_pos = self.get_rel_pos()
        search,wpsum,fold = self.init_dnls(vid.shape,vid.device,self.num_heads)

        # -- qkv --
        q_vid, k_vid, v_vid = self.qkv_videos(vid,attn_kv)
        # q_vid, k_vid, v_vid = self.qkv(vid,attn_kv)
        # q_vid = q_vid * self.scale

        # -- attn map --
        ntotal = T*H*W
        dists,inds = search(q_vid,0,ntotal,k_vid)
        dists = search.window_attn_mod(dists,rel_pos,mask,vid.shape)
        dists = self.attn_drop(dists)

        # -- prod with "v" --
        x = wpsum(v_vid,dists,inds)
        x = rearrange(x,'(o n) h c 1 1 -> o n (h c)',o=ntotal)

        # -- proj --
        x = self.proj(x)
        x = self.proj_drop(x)

        # -- prepare for folding --
        x = rearrange(x,'o n c -> (o n) 1 1 c 1 1')
        x = x.contiguous()

        # -- fold --
        fold(x,0)

        # -- unpack --
        vid = fold.vid
        vid = rearrange(vid,'t c h w -> t h w c')

        # attn = (q @ k.transpose(-2, -1))
        # rel_pos = self.get_rel_pos()
        # attn = attn + rel_pos.unsqueeze(0)
        # attn = self.modify_attn(attn,mask)
        # attn = self.attn_drop(attn)

        # x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        # x = self.proj(x)
        # x = self.proj_drop(x)

        return vid

    def modify_attn(self,attn,mask):
        if mask is not None:
            nW = mask.shape[0]
            mask = repeat(mask, 'nW m n -> nW m (n d)',d = ratio)
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N*ratio) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N*ratio)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)
        return attn

    def get_rel_pos(self):
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
        fflow,bflow = None,None
        k = -1
        ps_search = 1
        pt = 1
        ws = 8
        wt = 0
        dil = 1
        stride = 1
        use_k = k > 0
        reflect_bounds = True
        use_search_abs = False
        only_full = False
        exact = True
        search = dnls.search.init("window",fflow, bflow, k,
                                  ps_search, pt, ws, wt, nheads,
                                  chnls=-1,dilation=dil,
                                  stride0=stride,stride1=stride,
                                  reflect_bounds=reflect_bounds,
                                  use_k=use_k,use_adj=False,full_ws=True,
                                  search_abs=use_search_abs,
                                  h0_off=0,w0_off=0,h1_off=0,w1_off=0,
                                  exact=exact)
        wpsum = dnls.reducers.WeightedPatchSumHeads(ps_search, pt, h_off=0, w_off=0,
                                                    dilation=dil,
                                                    reflect_bounds=reflect_bounds,
                                                    adj=0, exact=exact)
        fold = dnls.iFoldz(vshape,None,stride=stride,dilation=dil,
                           adj=0,only_full=only_full,
                           use_reflect=reflect_bounds,device=device)
        return search,wpsum,fold

    def qkv_videos(self,x,attn_kv):
        """
        This function makes checking gradients super easy.
        """

        # -- init --
        t,c,h,w = x.shape
        ps = 8
        dil = 1
        adj = ps//2
        stride = 8
        vshape = x.shape
        only_full = True
        unfold = dnls.iUnfold(ps,None,stride=stride,dilation=dil,
                              adj=adj,only_full=only_full,border="reflect")
        qfold = dnls.iFold(vshape,None,stride=stride,dilation=dil,
                           adj=adj,only_full=only_full,
                           use_reflect=True,device=x.device)
        kfold = dnls.iFold(vshape,None,stride=stride,dilation=dil,
                           adj=adj,only_full=only_full,
                           use_reflect=True,device=x.device)
        vfold = dnls.iFold(vshape,None,stride=stride,dilation=dil,
                           adj=adj,only_full=only_full,
                           use_reflect=True,device=x.device)
        # -- unfold --
        nh = (h-1)//stride+1
        nw = (w-1)//stride+1
        ntotal = t*nh*nw
        patches = unfold(x,0,ntotal)
        patches = rearrange(patches,'n 1 1 c ph pw -> n (ph pw) c')

        # -- transform --
        q, k, v = self.qkv(patches,attn_kv)
        q = q * self.scale

        # -- reshape --
        q = rearrange(q,'n h (ph pw) c -> n 1 1 (h c) ph pw',ph=ps,pw=ps)
        k = rearrange(k,'n h (ph pw) c -> n 1 1 (h c) ph pw',ph=ps,pw=ps)
        v = rearrange(v,'n h (ph pw) c -> n 1 1 (h c) ph pw',ph=ps,pw=ps)

        # -- contig --
        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()

        # -- folding --
        qfold(q,0)
        kfold(k,0)
        vfold(v,0)

        # -- unpack --
        q_vid = qfold.vid
        k_vid = kfold.vid
        v_vid = vfold.vid

        return q_vid,k_vid,v_vid

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
