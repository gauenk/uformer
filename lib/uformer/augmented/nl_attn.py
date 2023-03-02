# -- torch network deps --
import torch as th
import torch.nn as nn
from einops import rearrange,repeat

# -- extra deps --
from timm.models.layers import trunc_normal_

# -- project deps --
from .proj import ConvProjection,LinearProjection,ConvProjectionNoReshape

# # -- local --
# from .state import update_state,run_state_search

# -- benchmarking --
from dev_basics.utils.timer import ExpTimer,ExpTimerList

# -- neighborhood attn --
# import nat

# -- dnls --
import dnls

class NonLocalAttention(nn.Module):
    def __init__(self, dim, win_size,num_heads, token_projection='linear',
                 qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.,
                 ps=1,pt=1,k=-1,ws=8,wt=0,dil=1,stride0=1,stride1=1,
                 nbwd=1,rbwd=False,exact=False,bs=-1,qk_frac=1.,wr=1,kr=1.,
                 update_dists=False,search_fxn="dnls",use_state_update=False):

        super().__init__()

        # -- search info --
        self.k = k
        self.ps = ps
        self.pt = pt
        self.ws = ws
        self.wt = wt
        self.wr = wr
        self.kr = kr
        self.dil = dil
        self.stride0 = stride0
        self.stride1 = stride1
        self.nbwd = nbwd
        self.rbwd = rbwd
        self.exact = exact
        self.bs = bs
        self.use_state_update = use_state_update
        # self.update_dists = False #update_dists
        self.search_fxn = search_fxn

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
        self.qkv = ConvProjectionNoReshape(dim,num_heads,dim//num_heads,
                                           qk_frac,bias=qkv_bias)

        self.token_projection = token_projection
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)

        # -- init search --
        self.search_name = search_fxn
        self.search = self.init_search(search_fxn)
        self.wpsum = self.init_wpsum()
        # self.fold = self.init_fold()

        # -- timers --
        self.use_timer = False
        self.times = ExpTimerList(self.use_timer)
        self.timer = None

    def get_weights(self,module):
        weights = []
        for name,mod in module.named_parameters():
            flat = mod.data.ravel()
            weights.append(flat)
        weights = th.cat(weights,0)
        return weights

    def get_qkv(self,vid):

        # -- compute --
        B, T, C, H, W = vid.shape
        vid = vid.view(B*T,C,H,W)
        q_vid, k_vid, v_vid = self.qkv(vid,None)
        q_vid = q_vid * self.scale

        # -- reshape --
        q_vid = q_vid.view(B,T,-1,H,W)
        k_vid = k_vid.view(B,T,-1,H,W)
        v_vid = v_vid.view(B,T,-1,H,W)

        return q_vid,k_vid,v_vid

    def get_qkv_patches(self,vid):
        # -- unfold --
        q_vid,k_vid,v_vid = self.get_qkv(vid)
        q_patches = unfold(q_vid)
        k_patches = unfold(k_vid)
        v_patches = unfold(v_vid)
        return q_patches,k_patches,v_patches

    def run_softmax(self,dists,mask,vshape):
        dists = self.softmax(dists)
        # if self.search.ws != 8: # don't match
        #     dists = self.softmax(dists)
        # else:
        #     rel_pos = self.get_rel_pos()
        #     dists = self.search.window_attn_mod(dists,rel_pos,mask,vid.shape)
        dists = self.attn_drop(dists)
        dists = dists.contiguous()
        return dists

    def run_aggregation(self,v_vid,dists,inds):
        B, T, _, H, W = v_vid.shape
        stride0 = self.stride0
        ntotal = T*((H-1)//stride0+1)*((W-1)//stride0+1)
        patches = self.wpsum(v_vid,dists,inds)
        ps = patches.shape[-1]
        shape_str = 'b h (o n) c ph pw -> (b o ph pw) n (h c)'
        patches = rearrange(patches,shape_str,o=ntotal)
        return patches

    def run_fold(self,patches,vshape):

        # -- init folding --
        B,ps = vshape[0],self.ps
        fold = self.init_fold(vshape,patches.device)

        # -- reshape for folding --
        shape_str = '(b o ph pw) n c -> b (o n) 1 1 c ph pw'
        patches = rearrange(patches,shape_str,b=B,ph=ps,pw=ps)
        patches = patches.contiguous()

        # -- fold --
        fold(patches,0)

        # -- unpack --
        vid = fold.vid / fold.zvid

        # -- debug --
        any_nan = th.any(th.isnan(vid))
        if any_nan:
            any_fold_nan = th.any(th.isnan(fold.vid))
            any_zero = th.any(th.abs(fold.zvid)<1e-10)
            print("found a nan!: ",any_nan,any_zero,any_fold_nan)
            exit(0)
        return vid

    def forward(self, vid, mask=None, flows=None, state=None):

        # -- update state --
        # if state is None: state = ["None","none"]
        # print([state[0] is None, state[1] is None],self.search,self.search_name)

        # -- init timer --
        self.timer = ExpTimer(self.use_timer)

        # -- unpack --
        # self.wpsum_patches = None

        # -- init --
        # self.search.update_flow(vid,flows)

        # -- qkv --
        q_vid,k_vid,v_vid = self.get_qkv(vid)

        # -- run search --
        dists,inds = self.run_search(q_vid,k_vid,flows,state)

        # -- softmax --
        dists = self.run_softmax(dists,mask,vid.shape)

        # -- aggregate --
        patches = self.run_aggregation(v_vid,dists,inds)

        # -- post-process --
        patches = self.proj(patches)
        patches = self.proj_drop(patches)

        # -- fold --
        vid = self.run_fold(patches,vid.shape)
        # print("vid.shape: ",vid.shape)

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

    # def run_search_patches(self,q_patches,qstart,ntotal,k_patches,state):
    #     dists,inds = None,None
    #     return dists,inds

    # def run_search(self,q_vid,k_vid,state):
    #     if state is None:
    #         # -- dnls search --
    #         B, T, _, H, W = q_vid.shape
    #         qstart,stride0 = 0,self.stride0
    #         ntotal = T*((H-1)//stride0+1)*((W-1)//stride0+1)
    #         dists,inds = self.search(q_vid,qstart,ntotal,k_vid)
    #     else:
    #         # -- streaming search --
    #         dists,inds = run_state_search(q_vid,qstart,ntotal,k_vid,state)
    #         update_state(state,dists,inds)
    #     return dists,inds

    def init_search(self,search_fxn):
        if search_fxn in ["refine"]:
            search = self.init_refine()
        else:
            search = self.init_nls()
        return search

    def run_search(self,q_vid,k_vid,flows,state):
        self.timer.sync_start("search")
        if self.search_name == "refine":
            inds_p = self.inds_rs1(state[0])
            dists,inds = self.search(q_vid,k_vid,inds_p)
        elif self.search_name == "rand_inds":
            dists,inds = self.search(q_vid,k_vid)
        else:
            dists,inds = self.search(q_vid,k_vid,flows.fflow,flows.bflow)
        self.update_state(state,dists,inds,q_vid.shape)
        self.timer.sync_stop("search")
        return dists,inds

    def update_state(self,state,dists,inds,vshape):
        if not(self.use_state_update): return
        T,C,H,W = vshape[-4:]
        nH = (H-1)//self.stride0+1
        nW = (W-1)//self.stride0+1
        # state[0] = state[1]
        state[0] = self.inds_rs0(inds.detach(),nH,nW) # forward state

    def inds_rs0(self,inds,nH,nW):
        if not(inds.ndim == 5): return inds
        rshape = 'b h (T nH nW) k tr -> T nH nW b h k tr'
        inds = rearrange(inds,rshape,nH=nH,nW=nW)
        return inds

    def inds_rs1(self,inds):
        if not(inds.ndim == 7): return inds
        rshape = 'T nH nW b h k tr -> b h (T nH nW) k tr'
        inds = rearrange(inds,rshape)
        return inds

    def init_nls(self):

        # # -- unpack params --
        k       = self.k
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
        nheads  = self.num_heads
        reflect_bounds = True
        use_adj = False
        full_ws = True

        # print(k,ps,pt,ws,wt,dil,stride0,stride1,nheads)
        NonLocalSearch = dnls.search.NonLocalSearch
        search = NonLocalSearch(ps, pt, ws, wt, nheads,
                                dist_type="prod",dilation=dil,
                                reflect_bounds=reflect_bounds,
                                use_adj=use_adj,full_ws=full_ws,
                                stride0=stride0,stride1=stride1,
                                nbwd=nbwd,rbwd=rbwd,exact=exact)
        return search
        
    def init_refine(self):

        # # -- unpack params --
        k       = self.k
        ps      = self.ps
        pt      = self.pt
        ws      = self.ws
        wt      = self.wt
        wr      = self.wr
        kr      = self.kr
        dil     = self.dil
        stride0 = self.stride0
        stride1 = self.stride1
        nbwd    = self.nbwd
        rbwd    = self.rbwd
        exact   = self.exact
        nheads  = self.num_heads
        reflect_bounds = True
        use_adj = False
        full_ws = True

        RefineSearch = dnls.search.RefineSearch
        search = RefineSearch(ws, ps, k, wr, kr, nheads,
                              dist_type="prod",dilation=dil,
                              reflect_bounds=reflect_bounds,
                              use_adj=use_adj,full_ws=full_ws,
                              stride0=stride0,stride1=stride1,
                              nbwd=nbwd,rbwd=rbwd,exact=exact)
        return search

    def init_nat(self):

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
        nheads  = self.num_heads

        # -- fixed --
        use_k = k > 0
        reflect_bounds = True
        use_search_abs = False
        only_full = False
        use_adj = False
        full_ws = True
        stype = "prod_with_heads"

        # -- init --
        search = nat

        return search

    def init_wpsum(self):

        # -- unpack params --
        ps      = self.ps
        pt      = self.pt
        dil     = self.dil

        # -- fixed --
        exact = False
        reflect_bounds = True

        # -- init --
        wpsum = dnls.reducers.WeightedPatchSumHeads(ps, pt, h_off=0, w_off=0,
                                                    dilation=dil,
                                                    reflect_bounds=reflect_bounds,
                                                    adj=0, exact=exact)
        return wpsum

    def init_fold(self,vshape,device):
        dil     = self.dil
        stride0 = self.stride0
        only_full = False
        reflect_bounds = True
        fold = dnls.iFoldz(vshape,None,stride=stride0,dilation=dil,
                           adj=0,only_full=only_full,
                           use_reflect=reflect_bounds,device=device)
        return fold

    def extra_repr(self) -> str:
        return f'dim={self.dim}, win_size={self.win_size}, num_heads={self.num_heads}'

    def flops(self, H, W):

        # -- init flops --
        flops = 0

        # -- num of reference points --
        nrefs = ((H-1)//self.stride0+1) * ((W-1)//self.stride0+1)

        # -- convolution flops --
        flops += self.qkv.flops(H,W)
        # print("product: ",self.qkv.flops(H,W))


        # -- non-local search --
        C = self.qkv.to_q.out_channels
        vshape = (1,C,H,W)
        flops += self.search.flops(1,C,H,W)
        # print(vshape)
        # print("search flops: ",self.search.flops(1,C,H,W))

        # -- weighted patch sum --
        k = self.search.k
        nheads = self.num_heads
        chnls_per_head = C//nheads
        flops += self.wpsum.flops(nrefs,chnls_per_head,nheads,k)
        # print("wpsum flops: ",self.wpsum.flops(nrefs,chnls_per_head,nheads,k))

        # -- projection --
        flops += nrefs * self.dim * self.dim

        # -- fold --
        ps = self.ps
        flops += nrefs * ps * ps
        # print(flops)

        return flops

