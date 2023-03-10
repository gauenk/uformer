
# -- torch network deps --
import torch as th
import torch.nn as nn
from einops import rearrange,repeat
from functools import partial

# -- extra deps --
from timm.models.layers import trunc_normal_

# -- project deps --
from .proj import InputProj,InputProjSeq,OutputProj,OutputProjSeq
from .basic_uformer import BasicUformerLayer
# from .basic_uformer import create_basic_enc_layer,create_basic_dec_layer
# from .basic_uformer import create_basic_conv_layer
from .scaling import Downsample,Upsample
# from .parse import fields2blocks
from ..utils.model_utils import rescale_flows
# from ..utils.model_utils import expand_embed_dims

class Uformer(nn.Module):
    def __init__(self, arch, blocks, blocklists):
        super().__init__()

        # -- init --
        self.num_enc_layers = len(depths)-1
        self.num_dec_layers = len(depths)-1
        self.patch_norm = patch_norm
        self.mlp_ratio = mlp_ratio
        self.token_projection = token_projection
        self.mlp = token_mlp
        self.win_size =win_size
        self.reso = img_size
        self.pos_drop = nn.Dropout(p=drop_rate)
        self.dd_in = dd_in

        # stochastic depth
        enc_dpr = [x.item() for x in th.linspace(0, drop_path_rate,
                                                    sum(depths[:self.num_enc_layers]))]
        conv_dpr = [drop_path_rate]*depths[-1]
        dec_dpr = enc_dpr[::-1]
        # print(enc_dpr)

        # -- reflect depths --
        depths_ref = depths + depths[:-1][::-1]
        num_heads_ref = num_heads + num_heads[0:][::-1]
        # print(depths_ref)

        # -- input/output --
        nhead0 = blocklists[0].nheads
        self.input_proj = InputProjSeq(depth=arch_cfg.input_proj_depth,
                                       in_channel=arch_cfg.dd_in,
                                       out_channel=arch_cfg.embed_dim*nhead0,
                                       kernel_size=3, stride=1,
                                       act_layer=nn.LeakyReLU)
        self.output_proj = OutputProj(in_channel=2*arch_cfg.embed_dim*nhead0,
                                      out_channel=arch_cfg.in_chans,
                                      kernel_size=3,stride=1)

        # -- init --
        start,stop = 0,0

        # -- encoder layers --
        enc_list = []
        for enc_i in range(num_encs):

            # -- init --
            start = stop
            stop = start + blocklists[enc_i].depth
            blocklist_i = blocklists[enc_i]
            blocks_i = [blocks[i] for i in range(start,stop)]
            enc_layer = BlockList("enc",blocklist_i,blocks_i)
            down_layer = Downsample(scales[enc_i].in_dim,scales[enc_i].out_dim)
            setattr(self,"encoderlayer_%d" % enc_i,enc_layer)
            setattr(self,"dowsample_%d" % enc_i,down_layer)

            # -- add to list --
            paired_layer = [enc_layer,down_layer]
            enc_list.append(paired_layer)

        self.enc_list = enc_list

        # -- center --
        start = stop
        stop = start + blocklists[num_encs].depth
        blocklist_i = blocklists[num_encs]
        blocks_i = [blocks[i] for i in range(start,stop)]
        setattr(self,"conv",BlockList("conv",blocklist_i,blocks_i))

        # -- decoder --
        dec_list = []
        for dec_i in range(num_encs+1,2*num_encs+1):

            # -- init --
            start = stop
            stop = start + blocklists[dec_i].depth
            blocklist_i = blocklists[dec_i]
            blocks_i = [blocks[i] for i in range(start,stop)]
            up_layer = Upsample(scales[dec_i].in_dim,scales[dec_i].out_dim)
            dec_layer = BlockList("dec",blocklist_i,blocks_i)
            setattr(self,"upsample_%d" % dec_i,up_layer)
            setattr(self,"decoderlayer_%d" % dec_i,dec_layer)

            # -- add to list --
            paired_layer = [up_layer,dec_layer]
            dec_list.append(paired_layer)

        self.dec_list = dec_list

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @th.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @th.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def forward(self, x, flows=None, states=None):

        mask=None
        # -- Input Projection --
        b,t,c,h,w = x.shape
        y = self.input_proj(x)
        y = self.pos_drop(y)
        # print("y.shape: ",y.shape)
        z = y
        num_encs = self.num_enc_layers

        # -- init states --
        if states is None:
            # states = [None for _ in range(2*num_encs+1)]
            states = [None,None]

        # -- enc --
        encs = []
        for i,(enc,down) in enumerate(self.enc_list):
            _h,_w = h//(2**i),w//(2**i)
            flows_i = rescale_flows(flows,_h,_w)
            z = enc(z,_h,_w,mask=mask,flows=flows_i,state=states)
            encs.append(z)
            z = down(z)

        # -- middle --
        mod = 2**num_encs
        _h,_w = h//mod,w//mod
        flows_i = rescale_flows(flows,_h,_w)
        z = self.conv(z,_h,_w,mask=mask,flows=flows_i,state=states)
        del flows_i

        # -- dec --
        for i,(up,dec) in enumerate(self.dec_list):
            i_rev = (num_encs-1)-i
            _h,_w = h//(2**(i_rev)),w//(2**(i_rev))
            flows_i = rescale_flows(flows,_h,_w)
            z = up(z)
            z = th.cat([z,encs[i_rev]],-3)
            # print("z.shape: ",z.shape)
            z = dec(z,_h,_w,mask=mask,flows=flows_i,state=states)

        # -- Output Projection --
        y = self.output_proj(z)

        # -- residual connection --
        out = x + y if self.dd_in == 3 else y

        return out

    @property
    def max_batch_size(self):
        return -1

    def flops(self,h,w):

        # -- init flops --
        flops = 0

        # -- Input Projection --
        flops += self.input_proj.flops(h,w)
        num_encs = self.num_enc_layers

        # -- enc --
        encs = []
        for i,(enc,down) in enumerate(self.enc_list):
            _h,_w = h//(2**i),w//(2**i)
            flops += enc.flops(_h,_w)
            flops += down.flops(_h,_w)

        # -- middle --
        mod = 2**num_encs
        _h,_w = h//mod,w//mod
        flops += self.conv.flops(_h,_w)

        # -- dec --
        for i,(up,dec) in enumerate(self.dec_list):
            i_rev = num_encs-1-i
            _h,_w = h//(2**(i_rev)),w//(2**(i_rev))
            flops += up.flops(_h,_w)
            flops += dec.flops(_h,_w)

        # -- Output Projection --
        flops += self.output_proj.flops(h,w)

        return flops
