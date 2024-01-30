from mmcv.ops.multi_scale_deform_attn import multi_scale_deformable_attn_pytorch
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import xavier_init, constant_init
from mmcv.cnn.bricks.registry import (ATTENTION,
                                      TRANSFORMER_LAYER,
                                      TRANSFORMER_LAYER_SEQUENCE)
from mmcv.cnn.bricks.transformer import build_attention
import math
from mmcv.runner import force_fp32, auto_fp16

from mmcv.runner.base_module import BaseModule, ModuleList, Sequential

from mmcv.utils import ext_loader
from .multi_scale_deformable_attn_function import MultiScaleDeformableAttnFunction_fp32, \
    MultiScaleDeformableAttnFunction_fp16
ext_module = ext_loader.load_ext(
    '_ext', ['ms_deform_attn_backward', 'ms_deform_attn_forward'])

@ATTENTION.register_module()
class MultiModal_CrossAttention(BaseModule):
    """An attention module used in BEVFormer.
    Args:
        embed_dims (int): The embedding dimension of Attention.
            Default: 256.
        num_cams (int): The number of cameras
        dropout (float): A Dropout layer on `inp_residual`.
            Default: 0..
        init_cfg (obj:`mmcv.ConfigDict`): The Config for initialization.
            Default: None.
        deformable_attention: (dict): The config for the deformable attention used in SCA.
    """

    def __init__(self,
                 embed_dims=256,
                 num_cams=6,
                 pc_range=None,
                 dropout=0.1,
                 init_cfg=None,
                 batch_first=False,
                 use_radar=True,
                 radar_num_levels=1,
                 radar_num_points=4,
                 radar_dims=64,
                 radar_num_heads=8,
                 im2col_step=64,
                 deformable_attention=dict(
                     type='MSDeformableAttention3D',
                     embed_dims=256,
                     num_levels=4),
                 **kwargs
                 ):
        super(MultiModal_CrossAttention, self).__init__(init_cfg)

        self.init_cfg = init_cfg
        self.dropout = nn.Dropout(dropout)
        self.pc_range = pc_range
        self.fp16_enabled = False
        self.deformable_attention = build_attention(deformable_attention)
        self.embed_dims = embed_dims
        self.num_cams = num_cams
        self.output_proj = nn.Linear(embed_dims, embed_dims)
        self.batch_first = batch_first

        self.use_radar = use_radar
        self.im2col_step=im2col_step
        if self.use_radar:
          
            self.radar_num_heads = radar_num_heads
            self.radar_num_levels = radar_num_levels
            self.radar_num_points = radar_num_points

            self.rad_sampling_offsets = nn.Linear(
                embed_dims, radar_num_heads * radar_num_points * 2)
            self.rad_attention_weights = nn.Linear(embed_dims,
                                            radar_num_heads * radar_num_points)
            self.rad_value_proj = nn.Linear(radar_dims, radar_dims)
            self.rad_output_proj = nn.Linear(radar_dims, radar_dims)

            self.modality_fusion_layer = nn.Sequential(
                nn.Linear(embed_dims + radar_dims, embed_dims),
                nn.LayerNorm(embed_dims),
                nn.ReLU(inplace=False),
                nn.Linear(embed_dims, embed_dims),
                nn.LayerNorm(embed_dims),
            )

        self.init_weight()


    def init_weight(self):
        """Default initialization for Parameters of Module."""
        xavier_init(self.output_proj, distribution='uniform', bias=0.)
        if self.use_radar:
            constant_init(self.rad_sampling_offsets, 0.)
            thetas = torch.arange(
            self.radar_num_heads,
            dtype=torch.float32) * (2.0 * math.pi / self.radar_num_heads)
            grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
            grid_init = (grid_init /
                     grid_init.abs().max(-1, keepdim=True)[0]).view(
            self.radar_num_heads, 1, 1,
            2).repeat(1, self.radar_num_levels, self.radar_num_points, 1)
            for i in range(self.radar_num_points):
                grid_init[:, :, i, :] *= i + 1
            constant_init(self.rad_sampling_offsets, 0.)
            self.rad_sampling_offsets.bias.data = grid_init.reshape(-1)
            constant_init(self.rad_attention_weights, val=0., bias=0.)
            xavier_init(self.rad_value_proj, distribution='uniform', bias=0.)
            xavier_init(self.rad_output_proj, distribution='uniform', bias=0.)
            xavier_init(self.modality_fusion_layer, distribution='uniform', bias=0.)

    
    @force_fp32(apply_to=('query', 'key', 'value', 'query_pos', 'reference_points_cam'))
    def forward(self,
                query,
                key,
                value,
                residual=None,
                query_pos=None,
                key_padding_mask=None,
                reference_points=None,
                spatial_shapes=None,
                reference_points_cam=None,
                bev_mask=None,
                level_start_index=None,
                flag='encoder',
                ref_radar=None,
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
        
        if kwargs['radar_mask_embeds'] is not None:
            query = query + kwargs['radar_mask_embeds']

        bs, num_query, _ = query.size()
        D = reference_points_cam.size(3)
        indexes = []
        for i, mask_per_img in enumerate(bev_mask):
            index_query_per_img = mask_per_img[0].sum(-1).nonzero().squeeze(-1)
            indexes.append(index_query_per_img)
        max_len = max([len(each) for each in indexes])

        # each camera only interacts with its corresponding BEV queries. This step can  greatly save GPU memory.
        queries_rebatch = query.new_zeros(
            [bs, self.num_cams, max_len, self.embed_dims])
        reference_points_rebatch = reference_points_cam.new_zeros(
            [bs, self.num_cams, max_len, D, 2])
        for j in range(bs):
            for i, reference_points_per_img in enumerate(reference_points_cam):   
                index_query_per_img = indexes[i]
                queries_rebatch[j, i, :len(index_query_per_img)] = query[j, index_query_per_img]
                reference_points_rebatch[j, i, :len(index_query_per_img)] = reference_points_per_img[j, index_query_per_img]

        num_cams, l, bs, embed_dims = key.shape

        key = key.permute(2, 0, 1, 3).reshape(
            bs * self.num_cams, l, self.embed_dims)
        value = value.permute(2, 0, 1, 3).reshape(
            bs * self.num_cams, l, self.embed_dims)
        queries = self.deformable_attention(query=queries_rebatch.view(bs*self.num_cams, max_len, self.embed_dims), key=key, value=value,
                                            reference_points=reference_points_rebatch.view(bs*self.num_cams, max_len, D, 2), spatial_shapes=spatial_shapes,
                                            level_start_index=level_start_index).view(bs, self.num_cams, max_len, self.embed_dims)
        for j in range(bs):
            for i, index_query_per_img in enumerate(indexes):
                slots[j, index_query_per_img] += queries[j, i, :len(index_query_per_img)]
        
        
        if self.use_radar:
            radar_spatial_shapes = kwargs['radar_spatial_shapes']
            radar_key_padding_mask = kwargs['radar_key_padding_mask']
            value = kwargs['radar_feats']
            bs, num_value, _ = value.shape

            value_mask = (value.sum(-1) != 0)

            assert (radar_spatial_shapes[:, 0] * radar_spatial_shapes[:, 1]).sum() == num_value

            value = self.rad_value_proj(value)
            out = torch.zeros_like(value)
          
            if radar_key_padding_mask is not None:      
                value = value.masked_fill(radar_key_padding_mask[..., None], 0.0)
            value = value.view(bs, num_value, self.radar_num_heads, -1)
            
            for b in range(bs):
                valid_index = value_mask[b].nonzero().squeeze(-1)       # just cross valid radar pixel.
                query_rebatch = query[b:b+1, valid_index]
                value_rebatch = value[b].unsqueeze(0)
                num_query = query_rebatch.shape[1]

                sampling_offsets = self.rad_sampling_offsets(query_rebatch).view(
                    1, num_query, self.radar_num_heads, self.radar_num_levels, self.radar_num_points, 2)
                attention_weights = self.rad_attention_weights(query_rebatch).view(
                    1, num_query, self.radar_num_heads, self.radar_num_levels * self.radar_num_points)
                attention_weights = attention_weights.softmax(-1)

                attention_weights = attention_weights.view(1, num_query,
                                                        self.radar_num_heads,
                                                        self.radar_num_levels,
                                                        self.radar_num_points)
                ref_points = ref_radar[b:b+1, valid_index]
                ref_points = ref_points[..., :2]
                if ref_points.shape[-1] == 2:
                    offset_normalizer = torch.stack(
                        [radar_spatial_shapes[..., 1], radar_spatial_shapes[..., 0]], -1)
                
                    sampling_locations = ref_points[:, :, None, :, None, :] \
                        + sampling_offsets \
                        / offset_normalizer[None, None, None, :, None, :]
                else:
                    raise ValueError(
                        f'Last dim of reference_points must be'
                        f' 2, but get {reference_points.shape[-1]} instead.')
                if torch.cuda.is_available() and value.is_cuda:
                    if value.dtype == torch.float16:
                        MultiScaleDeformableAttnFunction = MultiScaleDeformableAttnFunction_fp32
                    else:
                        MultiScaleDeformableAttnFunction = MultiScaleDeformableAttnFunction_fp32
                    output = MultiScaleDeformableAttnFunction.apply(
                        value_rebatch, radar_spatial_shapes, kwargs['radar_level_start_index'], sampling_locations,
                        attention_weights, self.im2col_step)
                else:
                    output = multi_scale_deformable_attn_pytorch(
                        value_rebatch, radar_spatial_shapes, sampling_locations, attention_weights)
                    
                out[b:b+1, valid_index] += output

            radar_output = self.rad_output_proj(out)

        count = bev_mask.sum(-1) > 0
        count = count.permute(1, 2, 0).sum(-1)
        count = torch.clamp(count, min=1.0)
        slots = slots / count[..., None]
        slots = self.output_proj(slots)

        if self.use_radar:
            slots = self.modality_fusion_layer(torch.cat([radar_output, slots],dim=-1))

        return self.dropout(slots) + inp_residual
