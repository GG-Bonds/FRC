import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from mmcv.cnn import xavier_init
from mmcv.runner.base_module import BaseModule
from mmdet.models import HEADS


@HEADS.register_module()
class Gen_Foreground_Prior_Queries(BaseModule):
    r"""
    Args:
        in_channels (int): Channels of input feature.
        context_channels (int): Channels of transformed feature.
    """
    def __init__(
        self,
        th_priori=None,
        dynamic=True,
        nms_kernel_size=None,
        embed_dims=256,
    ):
        super(Gen_Foreground_Prior_Queries, self).__init__()
        self.th_priori = th_priori
        self.dynamic = dynamic
        self.nms_kernel_size = nms_kernel_size

        self.pos_priori_encoder = nn.Sequential(
            nn.Linear(embed_dims*3//2, embed_dims),
            nn.ReLU(),
            nn.Linear(embed_dims, embed_dims),
        )

        self.sem_priori_encoder = nn.Sequential(
                nn.Linear(embed_dims, embed_dims),
                nn.ReLU(),
                nn.Linear(embed_dims, embed_dims),
            )
        
        self.hight_net = nn.Sequential(
                nn.Linear(embed_dims, embed_dims // 4),
                nn.ReLU(),
                nn.Linear(embed_dims // 4, 1),
            )
        self.init_weights()

    def init_weights(self):
        """Initialize the weights."""
        xavier_init(self.hight_net, distribution='uniform', bias=0.)
        xavier_init(self.sem_priori_encoder, distribution='uniform', bias=0.)
        xavier_init(self.pos_priori_encoder, distribution='uniform', bias=0.)
   

    def forward(self, reference_points, query_pos, query, bev_embed, bev_h, bev_w, **kwargs):
        
        foreground_radar_mask_logit = kwargs['foreground_radar_mask_logit'].detach().sigmoid()
        foreground_radar_mask = kwargs['foreground_radar_mask'] 
        
        if self.nms_kernel_size is not None:        # NMS
            padding = self.nms_kernel_size // 2
            local_max = torch.zeros_like(foreground_radar_mask_logit)
            local_max_inner = F.max_pool2d(foreground_radar_mask_logit, kernel_size=self.nms_kernel_size, stride=1, padding=0)
            local_max[:, :, padding:(-padding), padding:(-padding)] = local_max_inner
            foreground_radar_mask_logit = foreground_radar_mask_logit * (foreground_radar_mask_logit == local_max)  # [BS, num_class, H, W]
        
        if self.th_priori is not None:         # Remove low-quality foreground priori.
            foreground_priori_mask = foreground_radar_mask_logit > self.th_priori
            foreground_radar_mask = foreground_radar_mask & foreground_priori_mask
        
        foreground_radar_mask_logit = foreground_radar_mask_logit.view(1, foreground_radar_mask_logit.shape[1], -1)  # [BS, num_class, H*W]

        num_proposals = self.topK
        if self.dynamic is True:        # valid radar pix may be smaller than topK.
            num_proposals = min(num_proposals, foreground_radar_mask.sum())

        top_proposals = foreground_radar_mask_logit.reshape(1, -1).argsort(dim=-1, descending=True)[..., :num_proposals]  # [BS, num_proposals]
        top_proposals_index = top_proposals % foreground_radar_mask_logit.shape[-1]  
        
        bev_priori_embedding = bev_embed.gather(index=top_proposals_index[:, None, :].permute(0, 2, 1).expand(-1, -1, bev_embed.shape[-1]), dim=1)
        bev_priori_embedding = bev_priori_embedding.detach()
        sem_priori_embedding = self.sem_priori_encoder(bev_priori_embedding)
        
        bev_grid = self.get_bev_grid(bev_h, bev_w, bs=1, device=bev_embed.device).squeeze(1)
        ref_priori = bev_grid.gather(index=top_proposals_index[:, None, :].permute(0, 2, 1).expand(-1, -1, bev_grid.shape[-1]), dim=1)
        ref_priori[:, :, 2:3] = self.hight_net(bev_priori_embedding).sigmoid()
        pos_priori_embedding = self.pos_priori_encoder(pos2posemb3d(ref_priori))

        reference_points = torch.cat([reference_points, ref_priori], dim=1)
        query_pos = torch.cat([query_pos, pos_priori_embedding], dim=1)
        query = torch.cat([query, sem_priori_embedding], dim=1)

        return reference_points, query_pos, query


    def get_bev_grid(self, H, W, Z=1, num_points_in_pillar=1, bs=1, device='cuda', dtype=torch.float):
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
    

def pos2posemb3d(pos, num_pos_feats=128, temperature=10000):
    scale = 2 * math.pi
    pos = pos * scale
    dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=pos.device)
    dim_t = temperature ** (2 * torch.div(dim_t, 2, rounding_mode='floor') / num_pos_feats)
    pos_x = pos[..., 0, None] / dim_t
    pos_y = pos[..., 1, None] / dim_t
    pos_z = pos[..., 2, None] / dim_t
    pos_x = torch.stack((pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()), dim=-1).flatten(-2)
    pos_y = torch.stack((pos_y[..., 0::2].sin(), pos_y[..., 1::2].cos()), dim=-1).flatten(-2)
    pos_z = torch.stack((pos_z[..., 0::2].sin(), pos_z[..., 1::2].cos()), dim=-1).flatten(-2)
    posemb = torch.cat((pos_y, pos_x, pos_z), dim=-1)
    return posemb