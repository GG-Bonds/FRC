import torch
import torch.nn as nn

from mmdet.core import (multi_apply, multi_apply )
from mmdet.models import HEADS
from mmdet.models.utils import build_transformer
from mmdet3d.models.builder import build_head
from mmcv.runner.base_module import BaseModule


@HEADS.register_module()
class Noisy_Radar_Filter(BaseModule):
    def __init__(self,
                 semantic_enhanced_radar_encoder,
                 foreground_radar_mask=None,
                 use_gt_mask=True,
                 **kwargs):
        
        super(Noisy_Radar_Filter, self).__init__()
        self.semantic_enhanced_radar_encoder = build_transformer(semantic_enhanced_radar_encoder)
        self.foreground_radar_mask = build_head(foreground_radar_mask)
        self.use_gt_mask = use_gt_mask


    def forward(self, mlvl_feats, mlvl_radar, gt_bev_mask=None, img_metas=None):
        results = dict()

        semantic_enhanced_radar, radar_mask = self.semantic_enhanced_radar_encoder(mlvl_feats, mlvl_radar, img_metas)
        foreground_radar_mask, foreground_radar_mask_logit = self.foreground_radar_mask(semantic_enhanced_radar) 

        if gt_bev_mask is not None and self.use_gt_mask is True:
            gt_bev_mask = torch.stack(gt_bev_mask, dim=0).to(semantic_enhanced_radar.device)
            if gt_bev_mask.dim() == 3:
                gt_bev_mask = gt_bev_mask.unsqueeze(1)

            gt_bev_mask = gt_bev_mask & radar_mask   
            if gt_bev_mask.dim() == 5:
                gt_bev_mask = gt_bev_mask.squeeze(0)
            
            gt_bev_mask = (gt_bev_mask[:, 0:1, :, :] != 0)
            semantic_enhanced_radar = gt_bev_mask | semantic_enhanced_radar
            results['gt_bev_mask'] = gt_bev_mask
        
        results['radar_mask'] = radar_mask
        results['radar_feats'] = semantic_enhanced_radar
        results['foreground_radar_mask_logit'] = foreground_radar_mask_logit
        results['foreground_radar_mask'] = foreground_radar_mask & radar_mask 

        return results