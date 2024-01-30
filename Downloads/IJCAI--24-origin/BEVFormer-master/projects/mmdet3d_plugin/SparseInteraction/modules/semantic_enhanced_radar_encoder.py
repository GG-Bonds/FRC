import math
import numpy as np
from einops import rearrange

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import normal_

from mmcv.utils import ext_loader, TORCH_VERSION, digit_version
from mmcv.cnn import xavier_init, Linear
from mmcv.cnn.bricks.registry import  ATTENTION
from mmcv.cnn.bricks.transformer import build_attention
from mmdet.models.utils.builder import TRANSFORMER
from mmcv.runner.base_module import BaseModule


ext_module = ext_loader.load_ext(
    '_ext', ['ms_deform_attn_backward', 'ms_deform_attn_forward'])


def pos2embed(pos, num_pos_feats=128, temperature=10000):
    scale = 2 * math.pi
    pos = pos * scale
    dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=pos.device)
    dim_t = 2 * (dim_t // 2) / num_pos_feats + 1
    pos_x = pos[..., 0] / dim_t
    pos_y = pos[..., 1] / dim_t
    pos_x = torch.stack((pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()), dim=-1).flatten(-2)
    pos_y = torch.stack((pos_y[..., 0::2].sin(), pos_y[..., 1::2].cos()), dim=-1).flatten(-2)
    posemb = torch.cat((pos_y, pos_x), dim=-1)
    return posemb


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x

def linear_relu_ln(embed_dims, in_loops, out_loops, input_dims=None):
    if input_dims is None:
        input_dims = embed_dims
    layers = []
    for _ in range(out_loops):
        for _ in range(in_loops):
            layers.append(Linear(input_dims, embed_dims))
            layers.append(nn.ReLU(inplace=True))
            input_dims = embed_dims
        layers.append(nn.LayerNorm(embed_dims))
    return layers


@TRANSFORMER.register_module()
class Semantic_Enhanced_Radar_Encoder(BaseModule):
    """Implements the DETR transformer.
    Following the official DETR implementation, this module copy-paste
    from torch.nn.Transformer with modifications:
        * positional encodings are passed in MultiheadAttention
        * extra LN at the end of encoder is removed
        * decoder returns a stack of activations from all decoding layers
    See `paper: End-to-End Object Detection with Transformers
    <https://arxiv.org/pdf/2005.12872>`_ for details.
    Args:
        encoder (`mmcv.ConfigDict` | Dict): Config of
            TransformerEncoder. Defaults to None.
        decoder ((`mmcv.ConfigDict` | Dict)): Config of
            TransformerDecoder. Defaults to None
        init_cfg (obj:`mmcv.ConfigDict`): The Config for initialization.
            Defaults to None.
    """

    def __init__(self, 
            in_channels_radar=64,
            hidden_dim=256,
            num_feature_levels=1,
            num_cams=6,
            use_cams_embeds=True,
            pc_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
            layers=1,
            num_points_in_pillar=4,
            att_layer=dict(type='RadarCrossCameraAttention',
                        hidden_dim=256,
                        heads=8,
                        dim_head= 32,
                        deformable_attention=dict(
                                type='MSDeformableAttention3D',
                                embed_dims=256,
                                num_points=8,
                                num_levels=1),
                        ),
            init_cfg=None,
            only_cross=False,
            readd=False,
            cross_pe=False,
            spread=False,
            **kwargs,
    ):

        super(Semantic_Enhanced_Radar_Encoder, self).__init__(init_cfg=init_cfg)
        self.hidden_dim=hidden_dim
        self.num_feature_levels = num_feature_levels
        self.num_cams = num_cams
        self.use_cams_embeds = use_cams_embeds
        self.pc_range = pc_range
        self.layers = layers
        self.num_points_in_pillar = num_points_in_pillar
        self.only_cross = only_cross

        self.radar_linear = nn.Linear(in_channels_radar, self.hidden_dim)
        self.out_linear = nn.Linear(self.hidden_dim, in_channels_radar)

        self.level_embeds = nn.Parameter(torch.Tensor(
            self.num_feature_levels, self.hidden_dim))
        self.cams_embeds = nn.Parameter(
            torch.Tensor(self.num_cams, self.hidden_dim))
        
        if self.only_cross == False:
            self.radar_embed = MLP(hidden_dim, hidden_dim, hidden_dim, 2)

        self.sere_layers = nn.ModuleList([ATTENTION.build(att_layer) for _ in range(layers)])
        
        self.readd = readd

        self.cross_pe = cross_pe
        if cross_pe == True:
            self.cross_pe_embeds = nn.Sequential(*linear_relu_ln(hidden_dim, 1, 2, num_points_in_pillar))

        self.spread = spread
        if self.spread is True:
            self.spread_net = nn.Sequential(
            nn.Conv2d(in_channels_radar, self.hidden_dim, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(self.hidden_dim, in_channels_radar, kernel_size=3, stride=1, padding=1, bias=False),
        )

        self.init_weights()

    def init_weights(self):
        # follow the official DETR to init parameters
        for m in self.modules():
            if hasattr(m, 'weight') and m.weight.dim() > 1:
                xavier_init(m, distribution='uniform')
        xavier_init(self.radar_linear, distribution='uniform', bias=0.)
        xavier_init(self.out_linear, distribution='uniform', bias=0.)
        if self.only_cross == False:
            xavier_init(self.radar_embed, distribution='uniform', bias=0.)

        if self.cross_pe == True:
            xavier_init(self.cross_pe_embeds, distribution='uniform', bias=0.)
        if self.spread == True:
            xavier_init(self.spread_net, distribution='uniform', bias=0.)

        normal_(self.level_embeds)
        normal_(self.cams_embeds)
        
        self._is_init = True


    def get_reference_points(self, H, W, Z=8, num_points_in_pillar=1, dim='3d', bs=1, device='cuda', dtype=torch.float):
        """Get the reference points used in SCA and TSA.
        Args:
            H, W: spatial shape of bev.
            Z: hight of pillar.
            D: sample D points uniformly from each pillar.
            device (obj:`device`): The device where
                reference_points should be.
        Returns:
            Tensor: reference points used in decoder, has \
                shape (bs, num_keys, num_levels, 2).
        """
        # reference points in 3D space, used in spatial cross-attention (SCA)
        if dim == '3d':
            zs = torch.linspace(0.5, Z - 0.5, num_points_in_pillar, dtype=dtype,
                                device=device).view(-1, 1, 1).expand(num_points_in_pillar, H, W) / Z
            xs = torch.linspace(0.5, W - 0.5, W, dtype=dtype,
                                device=device).view(1, 1, W).expand(num_points_in_pillar, H, W) / W
            ys = torch.linspace(0.5, H - 0.5, H, dtype=dtype,
                                device=device).view(1, H, 1).expand(num_points_in_pillar, H, W) / H
            ref_3d = torch.stack((xs, ys, zs), -1)
            ref_3d = ref_3d.permute(0, 3, 1, 2).flatten(2).permute(0, 2, 1)
            ref_3d = ref_3d[None].repeat(bs, 1, 1, 1)
            return ref_3d

        # reference points on 2D bev plane, used in temporal self-attention (TSA).
        elif dim == '2d':
            ref_y, ref_x = torch.meshgrid(
                torch.linspace(
                    0.5, H - 0.5, H, dtype=dtype, device=device),
                torch.linspace(
                    0.5, W - 0.5, W, dtype=dtype, device=device)
            )
            ref_y = ref_y.reshape(-1)[None] / H
            ref_x = ref_x.reshape(-1)[None] / W
            ref_2d = torch.stack((ref_x, ref_y), -1)
            ref_2d = ref_2d.repeat(bs, 1, 1).unsqueeze(2)
            return ref_2d
    

    def point_sampling(self, reference_points, pc_range,  img_metas):
        lidar2img = []
        for img_meta in img_metas:
            lidar2img.append(img_meta['lidar2img'])
        lidar2img = np.asarray(lidar2img)
        lidar2img = reference_points.new_tensor(lidar2img)  # (B, N, 4, 4)
        reference_points = reference_points.clone()
        reference_points[..., 0:1] = reference_points[..., 0:1] * \
            (pc_range[3] - pc_range[0]) + pc_range[0]
        reference_points[..., 1:2] = reference_points[..., 1:2] * \
            (pc_range[4] - pc_range[1]) + pc_range[1]
        reference_points[..., 2:3] = reference_points[..., 2:3] * \
            (pc_range[5] - pc_range[2]) + pc_range[2]

        reference_points = torch.cat(
            (reference_points, torch.ones_like(reference_points[..., :1])), -1)

        reference_points = reference_points.permute(1, 0, 2, 3)
        D, B, num_query = reference_points.size()[:3]
        num_cam = lidar2img.size(1)

        reference_points = reference_points.view(
            D, B, 1, num_query, 4).repeat(1, 1, num_cam, 1, 1).unsqueeze(-1)

        lidar2img = lidar2img.view(
            1, B, num_cam, 1, 4, 4).repeat(D, 1, 1, num_query, 1, 1)

        reference_points_cam = torch.matmul(lidar2img.to(torch.float32),
                                            reference_points.to(torch.float32)).squeeze(-1)
        eps = 1e-5

        bev_mask = (reference_points_cam[..., 2:3] > eps)
        reference_points_cam[..., 0:2] = reference_points_cam[..., 0:2] / torch.maximum(
            reference_points_cam[..., 2:3], torch.ones_like(reference_points_cam[..., 2:3]) * eps)

        reference_points_cam[..., 0] /= img_metas[0]['img_shape'][0][1]
        reference_points_cam[..., 1] /= img_metas[0]['img_shape'][0][0]     # [4, 1, 6, 2500, 2]
        bev_mask = (bev_mask & (reference_points_cam[..., 1:2] > 0.0)           
                    & (reference_points_cam[..., 1:2] < 1.0)
                    & (reference_points_cam[..., 0:1] < 1.0)
                    & (reference_points_cam[..., 0:1] > 0.0)
                    )

        if digit_version(TORCH_VERSION) >= digit_version('1.8'):
            bev_mask = torch.nan_to_num(bev_mask)
        else:
            bev_mask = bev_mask.new_tensor(
                np.nan_to_num(bev_mask.cpu().numpy()))

        reference_points_cam = reference_points_cam.permute(2, 1, 3, 0, 4)
        bev_mask = bev_mask.permute(2, 1, 3, 0, 4).squeeze(-1)


        return reference_points_cam, bev_mask

    def forward(self, mlvl_feats, mlvl_radar_feats, img_metas, **kwargs):

        feat_flatten = []
        spatial_shapes = []
        for lvl, feat in enumerate(mlvl_feats):
            bs, c, h, w = feat.shape
            feat = feat.view(bs // 6, 6, c, h, w)
            spatial_shape = (h, w)
            feat = feat.flatten(3).permute(1, 0, 3, 2)
            if self.use_cams_embeds:
                feat = feat + self.cams_embeds[:, None, None, :].to(feat.dtype)
            feat = feat + self.level_embeds[None,
                                            None, lvl:lvl + 1, :].to(feat.dtype)
            spatial_shapes.append(spatial_shape)
            feat_flatten.append(feat)

        feat_flatten = torch.cat(feat_flatten, 2)
   
        spatial_shapes = torch.as_tensor(
            spatial_shapes, dtype=torch.long, device=mlvl_feats[0].device)
        level_start_index = torch.cat((spatial_shapes.new_zeros(
            (1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))

        radar_feats = mlvl_radar_feats[0]
        if self.spread is True:
            radar_feats_spread = self.spread_net(radar_feats)
            radar_feats = radar_feats + radar_feats_spread
            
        bs, _, bev_h, bev_w = radar_feats.shape
        radar_query = rearrange(radar_feats, 'b c h w -> b (h w) c')
        ref_3d = self.get_reference_points(
            bev_h, bev_w, self.pc_range[5]-self.pc_range[2], self.num_points_in_pillar, dim='3d', bs=bs,  device=radar_feats.device, dtype=radar_feats.dtype)
        ref_2d = self.get_reference_points(
            bev_h, bev_w, dim='2d', bs=bs, device=radar_feats.device, dtype=radar_feats.dtype)
        
        if self.only_cross == False:
            radar_pe = self.radar_embed(pos2embed(ref_2d, self.hidden_dim // 2))   
        else:
            radar_pe = None
        
        reference_points_cam, bev_mask = self.point_sampling(
            ref_3d, self.pc_range, img_metas)
        radar_mask = (radar_query.sum(-1) != 0)     # bs h*w
        bev_mask = (bev_mask & radar_mask[None, :, :, None])
        radar_query = self.radar_linear(radar_query)

        if self.cross_pe:
            cross_pe = reference_points_cam[..., :3]
            cross_pe = cross_pe[..., 2:3]
            cross_pe = self.cross_pe_embeds(cross_pe.flatten(-2, -1))
        else:
            cross_pe = None
        reference_points_cam = reference_points_cam[..., :2]

        for sere in self.sere_layers:
            radar_query = sere(radar_query, radar_pe, feat_flatten, bev_mask, radar_mask, spatial_shapes, level_start_index, reference_points_cam, cross_pe)
        
        out = self.out_linear(radar_query)
        out = out.view(bs, bev_h, bev_w, -1).permute(0, 3, 1, 2)

        if self.readd is True:
            out = out + radar_feats

        bs, c, h, w = out.shape
        semantic_enhanced_radar = out.masked_fill_(~radar_mask.view(bs, 1, h, w), 0.0)

        return semantic_enhanced_radar, radar_mask


@ATTENTION.register_module()
class Semantic_Enhanced_Radar_Layer(BaseModule):
    """A wrapper for ``torch.nn.MultiheadAttention``.
    This module implements MultiheadAttention with identity connection,
    and positional encoding  is also passed as input.
    Args:
    """

    def __init__(self,
                hidden_dim=64,
                deformable_attention=dict(
                                type='MSDeformableAttention3D',
                                embed_dims=256,
                                num_points=8,
                                num_levels=1),
                only_cross=False,
                **kwargs
                ):
        super(Semantic_Enhanced_Radar_Layer, self).__init__()

        self.only_cross = only_cross
        self.hidden_dim = hidden_dim
        self.radar_cross_atten = RadarCrossAttention(deformable_attention=deformable_attention)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)
    
        self.init_weight()

    def init_weight(self):
        """Default initialization for Parameters of Module."""
        xavier_init(self.output_proj, distribution='uniform', bias=0.)

    def forward(self,
                radar_query,
                camera_value,
                bev_mask,
                spatial_shapes=None,
                level_start_index=None,
                reference_points_cam=None,
                cross_pe=None,
                ):
        bs, _, _ = radar_query.shape
        radar_query = self.radar_cross_atten(
                    radar_query,
                    camera_value,
                    camera_value,
                    reference_points_cam=reference_points_cam,
                    spatial_shapes=spatial_shapes,
                    level_start_index=level_start_index,
                    bev_mask=bev_mask,
                    cross_pe=cross_pe,
                    )
        out = self.norm1(radar_query)
        return self.output_proj(out)
    
    
class RadarCrossAttention(BaseModule):
    def __init__(self,
                 embed_dims=256,
                 num_cams=6,
                 pc_range=None,
                 dropout=0.1,
                 init_cfg=None,
                 batch_first=False,
                 deformable_attention=dict(
                     type='MSDeformableAttention3D',
                     embed_dims=256,
                     num_levels=1),
                 **kwargs
                 ):
        super(RadarCrossAttention, self).__init__(init_cfg)

        self.init_cfg = init_cfg
        self.dropout = nn.Dropout(dropout)
        self.pc_range = pc_range
        self.fp16_enabled = False
        self.deformable_attention = build_attention(deformable_attention)
        self.embed_dims = embed_dims
        self.num_cams = num_cams
        self.output_proj = nn.Linear(embed_dims, embed_dims)
        self.batch_first = batch_first
        self.init_weight()

    def init_weight(self):
        """Default initialization for Parameters of Module."""
        xavier_init(self.output_proj, distribution='uniform', bias=0.)

    def forward(self,
                query,
                key,
                value,
                residual=None,
                query_pos=None,
                spatial_shapes=None,
                reference_points_cam=None,
                bev_mask=None,
                level_start_index=None,
                cross_pe=None,
                **kwargs):
        """Forward Function of Detr3DCrossAtten.
        Args:
            query (Tensor): Query of Transformer with shape
                (num_query, bs, embed_dims).
            key (Tensor): The key tensor with shape
                `(num_key, bs, embed_dims)`.
            value (Tensor): The value tensor with shape
                `(num_key, bs, embed_dims)`. (B, N, C, H, W)
            residual (Tensor): The tensor used for addition, with the
                same shape as `x`. Default None. If None, `x` will be used.
            query_pos (Tensor): The positional encoding for `query`.
                Default: None.
            key_pos (Tensor): The positional encoding for  `key`. Default
                None.
            reference_points (Tensor):  The normalized reference
                points with shape (bs, num_query, 4),
                all elements is range in [0, 1], top-left (0,0),
                bottom-right (1, 1), including padding area.
                or (N, Length_{query}, num_levels, 4), add
                additional two dimensions is (w, h) to
                form reference boxes.
            key_padding_mask (Tensor): ByteTensor for `query`, with
                shape [bs, num_key].
            spatial_shapes (Tensor): Spatial shape of features in
                different level. With shape  (num_levels, 2),
                last dimension represent (h, w).
            level_start_index (Tensor): The start index of each level.
                A tensor has shape (num_levels) and can be represented
                as [0, h_0*w_0, h_0*w_0+h_1*w_1, ...].
        Returns:
             Tensor: forwarded results with shape [num_query, bs, embed_dims].
        """

        if key is None:
            key = query
        if value is None:
            value = key

        if residual is None:
            inp_residual = query
            slots = torch.zeros_like(query)
        if query_pos is not None:
            query = query + query_pos
        bs, num_query, _ = query.size()
        n, bs, _, D, _ = reference_points_cam.shape        # n, b, Q, D, 2
        for b in range(bs):
            for nc in range(n):

                index_query_per_img = bev_mask[nc, b].sum(-1).nonzero().squeeze(-1)
                if index_query_per_img.shape[0] == 0:
                    continue
                query_ = query[b, index_query_per_img].unsqueeze(0)
                reference_points_cam_ = reference_points_cam[nc, b, index_query_per_img].unsqueeze(0)
                if cross_pe is not None:
                    cross_pe_ = cross_pe[nc, b, index_query_per_img].unsqueeze(0)
                else:
                    cross_pe_ = None
                out = self.deformable_attention(query=query_, key=key[nc, b].unsqueeze(0), value=value[nc, b].unsqueeze(0),
                                    reference_points=reference_points_cam_, spatial_shapes=spatial_shapes, query_pos=cross_pe_,
                                    level_start_index=level_start_index)
            
                slots[b, index_query_per_img] += out.squeeze(0)

        count = bev_mask.sum(-1) > 0
        count = count.permute(1, 2, 0).sum(-1)
        count = torch.clamp(count, min=1.0)
        slots = slots / count[..., None]
        slots = self.output_proj(slots)

        return self.dropout(slots) + inp_residual