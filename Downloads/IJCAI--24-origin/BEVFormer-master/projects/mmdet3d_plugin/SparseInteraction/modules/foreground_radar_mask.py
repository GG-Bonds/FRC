# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F

from mmcv.cnn import Linear
from mmcv.runner.base_module import BaseModule

from mmdet.models import HEADS
from mmdet.models.losses import weight_reduce_loss


def py_sigmoid_focal_loss(pred,
                          target,
                          weight=None,
                          gamma=2.0,
                          alpha=0.25,
                          reduction='mean',
                          avg_factor=None,
                          focal_loss_pos_weight=1.0,
                          ):
    """PyTorch version of `Focal Loss <https://arxiv.org/abs/1708.02002>`_.

    Args:
        pred (torch.Tensor): The prediction with shape (N, C), C is the
            number of classes
        target (torch.Tensor): The learning label of the prediction.
        weight (torch.Tensor, optional): Sample-wise loss weight.
        gamma (float, optional): The gamma for calculating the modulating
            factor. Defaults to 2.0.
        alpha (float, optional): A balanced form for Focal Loss.
            Defaults to 0.25.
        reduction (str, optional): The method used to reduce the loss into
            a scalar. Defaults to 'mean'.
        avg_factor (int, optional): Average factor that is used to average
            the loss. Defaults to None.
    """
    pred_sigmoid = pred.sigmoid()
    target = target.type_as(pred)
    pt = (1 - pred_sigmoid) * target  + pred_sigmoid * (1 - target)
    focal_weight = (alpha * target* focal_loss_pos_weight + (1 - alpha) *
                    (1 - target)) * pt.pow(gamma)
    loss = F.binary_cross_entropy_with_logits(
        pred, target, reduction='none') * focal_weight
    if weight is not None:
        if weight.shape != loss.shape:
            if weight.size(0) == loss.size(0):
                # For most cases, weight is of shape (num_priors, ),
                #  which means it does not have the second axis num_class
                weight = weight.view(-1, 1)
            else:
                # Sometimes, weight per anchor per class is also needed. e.g.
                #  in FSAF. But it may be flattened of shape
                #  (num_priors x num_class, ), while loss is still of shape
                #  (num_priors, num_class).
                assert weight.numel() == loss.numel()
                weight = weight.view(loss.size(0), -1)
        assert weight.ndim == loss.ndim
    loss = weight_reduce_loss(loss, weight, reduction, avg_factor)
    return loss


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


@HEADS.register_module()
class Foreground_Radar_Mask(BaseModule):
    r"""
    Args:
        in_channels (int): Channels of input feature.
        context_channels (int): Channels of transformed feature.
    """

    def __init__(
        self,
        in_channels=512,
        mask_thre = 0.01,
        ce_loss_weight=1.0,
        focal_loss_weight=5.0,
        focal_loss_pos_weight=4.0,
        loss_seg_type=dict(
            ce=True,
            focal=False,
        ),
    ):
        super(Foreground_Radar_Mask, self).__init__()
        self.loss_seg_type = loss_seg_type
        
        self.focal_loss_weight = focal_loss_weight
        self.focal_loss_pos_weight = focal_loss_pos_weight
        
        self.mask_net = nn.Sequential(
            nn.Conv2d(in_channels, in_channels//2, kernel_size=1, padding=0, stride=1),
            nn.BatchNorm2d(in_channels//2),
            nn.ReLU(),
            nn.Conv2d(in_channels//2, 1, kernel_size=1, padding=0, stride=1),
        )

        self.ce_loss = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([ce_loss_weight]), reduction='none')  # From lss
        
        self.mask_thre = mask_thre
        self.ce_loss_weight = ce_loss_weight
   

    def forward(self, input):
        foreground_radar_mask_logit = self.mask_net(input)
        foreground_radar_mask = foreground_radar_mask_logit.sigmoid() > self.mask_thre
        return foreground_radar_mask, foreground_radar_mask_logit
    
    
    def get_bev_mask_loss(self, gt_bev_mask, pred_foreground_radar_mask, radar_mask):
        
        foreground_radar_mask, foreground_radar_mask_logit = pred_foreground_radar_mask

        foreground_radar_mask = foreground_radar_mask.view(-1)
        foreground_radar_mask_logit = foreground_radar_mask_logit.view(-1)
        gt_bev_mask = gt_bev_mask.view(-1)
        
        gt_foreground_radar_mask = gt_bev_mask[foreground_radar_mask].to(torch.float).unsqueeze(0)
        foreground_radar_mask_logit = foreground_radar_mask_logit[radar_mask].unsqueeze(0)
        
        loss_seg = dict()
        if self.loss_seg_type['ce'] == True:
            loss_seg['mask_ce_loss'] = self.ce_loss(foreground_radar_mask_logit, gt_foreground_radar_mask) * self.ce_loss_weight
            loss_seg['mask_ce_loss'] = loss_seg['mask_ce_loss'].mean()

        if self.loss_seg_type['focal'] == True:
            loss_seg['mask_focal_loss'] = py_sigmoid_focal_loss(foreground_radar_mask_logit,gt_foreground_radar_mask, focal_loss_pos_weight=self.focal_loss_pos_weight) * self.focal_loss_weight

        return loss_seg  
