# ---------------------------------------------
# Copyright (c) OpenMMLab. All rights reserved.
# ---------------------------------------------
#  Modified by Zhiqi Li
# ---------------------------------------------

import torch
from mmcv.runner import force_fp32, auto_fp16
from mmdet.models import DETECTORS
from mmdet3d.core import bbox3d2result
from mmdet3d.models.detectors.mvx_two_stage import MVXTwoStageDetector
from projects.mmdet3d_plugin.models.utils.grid_mask import GridMask
import time
import copy
import numpy as np
from projects.mmdet3d_plugin.models.utils.bricks import run_time
from mmdet3d.ops import Voxelization
from mmdet3d.models import builder
from mmdet3d.models.builder import build_head
import torch.nn.functional as F
from mmdet.models.utils import build_transformer

@DETECTORS.register_module()
class SparseInteraction(MVXTwoStageDetector):
    """SparseInteraction.
    Args:
        video_test_mode (bool): Decide whether to use temporal information during inference.
    """

    def __init__(self,
                 use_grid_mask=False,
                 pts_voxel_layer=None,
                 pts_voxel_encoder=None,
                 pts_middle_encoder=None,
                 pts_fusion_layer=None,
                 img_backbone=None,
                 pts_backbone=None,
                 img_neck=None,
                 pts_neck=None,
                 pts_bbox_head=None,
                 img_roi_head=None,
                 img_rpn_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 video_test_mode=False,


                 radar_voxel_layer=None,
                 radar_voxel_encoder=None,
                 radar_middle_encoder=None,

                 noisy_radar_filter=None,
                 frpn=None,
                 ):

        super(SparseInteraction,
              self).__init__(pts_voxel_layer, pts_voxel_encoder,
                             pts_middle_encoder, pts_fusion_layer,
                             img_backbone, pts_backbone, img_neck, pts_neck,
                             pts_bbox_head, img_roi_head, img_rpn_head,
                             train_cfg, test_cfg, pretrained)
        self.grid_mask = GridMask(
            True, True, rotate=1, offset=False, ratio=0.5, mode=1, prob=0.7)
        self.use_grid_mask = use_grid_mask
        self.fp16_enabled = False
        # temporal
        self.video_test_mode = video_test_mode
        self.prev_frame_info = {
            'prev_bev': None,
            'scene_token': None,
            'prev_pos': 0,
            'prev_angle': 0,
        }

        if pts_bbox_head is None:
            self.pts_bbox_head = None

        if radar_voxel_layer is not None:
            self.radar_voxel_layer = Voxelization(**radar_voxel_layer)
        else:
            self.radar_voxel_layer = None

        if radar_voxel_encoder is not None:
            self.radar_voxel_encoder = builder.build_voxel_encoder(
                radar_voxel_encoder)
        else:
            self.radar_voxel_encoder = None
        
        if radar_middle_encoder is not None:
            self.radar_middle_encoder = builder.build_middle_encoder(
                radar_middle_encoder)
        else:
            self.radar_middle_encoder = None
        
        if noisy_radar_filter is not None:
            self.noisy_radar_filter = build_head(noisy_radar_filter)
        else:
            self.noisy_radar_filter = None


    def with_specific_component(self, component_name):
        try:
            return getattr(self, component_name) is not None
        except:
            return False


    @torch.no_grad()
    @force_fp32()
    def radar_voxelize(self, points):
        """Apply dynamic voxelization to points.

        Args:
            points (list[torch.Tensor]): Points of each sample.

        Returns:
            tuple[torch.Tensor]: Concatenated points, number of points
                per voxel, and coordinates.
        """
        voxels, coors, num_points = [], [], []
        for res in points:
            res_voxels, res_coors, res_num_points = self.radar_voxel_layer(res)
            voxels.append(res_voxels)
            coors.append(res_coors)
            num_points.append(res_num_points)
        voxels = torch.cat(voxels, dim=0)
        num_points = torch.cat(num_points, dim=0)
        coors_batch = []
        for i, coor in enumerate(coors):
            coor_pad = F.pad(coor, (1, 0), mode='constant', value=i)
            coors_batch.append(coor_pad)
        coors_batch = torch.cat(coors_batch, dim=0)
        return voxels, num_points, coors_batch

    def extract_radar_feat(self, radar=None):
        if self.radar_voxel_layer is None or radar is None:
            return None
        voxels, num_points, coors = self.radar_voxelize(radar)
        voxel_features = self.radar_voxel_encoder(voxels, num_points, coors,
                                                )
        batch_size = coors[-1, 0] + 1
        x = self.radar_middle_encoder(voxel_features, coors, batch_size)

        return [x]

    def extract_img_feat(self, img, img_metas, len_queue=None):
        """Extract features of images."""
        B = img.size(0)
        if img is not None:
            if img.dim() == 5 and img.size(0) == 1:
                img.squeeze_()
            elif img.dim() == 5 and img.size(0) > 1:
                B, N, C, H, W = img.size()
                img = img.reshape(B * N, C, H, W)
            if self.use_grid_mask:
                img = self.grid_mask(img)

            img_feats = self.img_backbone(img)
            if isinstance(img_feats, dict):
                img_feats = list(img_feats.values())
        else:
            return None

        if self.with_img_neck:
            img_feats = self.img_neck(img_feats)

        if not isinstance(img_feats, tuple):
            img_feats = [img_feats]

        return_map = {
            'img_feats': img_feats,
        }
        return return_map


    @auto_fp16(apply_to=('img'))
    def extract_feat(self, img, img_metas=None, len_queue=None, radar=None, gt_bev_mask=None, **kwargs):
        """Extract features from images and points."""
        B = img.shape[0]
        results = self.extract_img_feat(img, img_metas, len_queue=len_queue)

        results['radar_feats'] = self.extract_radar_feat(radar)

        
        if self.with_specific_component('noisy_radar_filter'):
            results.update(self.noisy_radar_filter(results['img_feats'], results['radar_feats'], gt_bev_mask, img_metas))

        else:
            results['foreground_radar_mask_logit'] = None
            results['foreground_radar_mask'] = None
            results['radar_mask'] = None
            results['gt_bev_mask'] = None

        img_feats_reshaped = []
        for img_feat in results['img_feats']:
            BN, C, H, W = img_feat.size()
            if len_queue is not None:
                img_feats_reshaped.append(img_feat.view(int(B/len_queue), len_queue, int(BN / B), C, H, W))
            else:
                img_feats_reshaped.append(img_feat.view(B, int(BN / B), C, H, W))
        results['img_feats'] = img_feats_reshaped
    
    
        if len_queue is not None and results['radar_feats'] is not None:
            BN, C, H, W = results['radar_feats'][0].shape
            results['radar_feats'] = [results['radar_feats'][0].view(int(BN/len_queue), len_queue, C, H, W)]
            if results['foreground_radar_mask'] is not None:
                results['foreground_radar_mask'] = [results['foreground_radar_mask'].view(int(BN/len_queue), len_queue, 1, H, W)]
                results['foreground_radar_mask_logit'] = [results['foreground_radar_mask_logit'].view(int(BN/len_queue), len_queue, -1, H, W)]

        return results


    def forward_pts_train(self,
                          pts_feats,
                          gt_bboxes_3d,
                          gt_labels_3d,
                          img_metas,
                          prev_bev=None,
                          radar_feats=None,
                          pre_bev_mask=None,
                          predict_bev_logit=None,
                          ):
        """Forward function'
        Args:
            pts_feats (list[torch.Tensor]): Features of point cloud branch
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`]): Ground truth
                boxes for each sample.
            gt_labels_3d (list[torch.Tensor]): Ground truth labels for
                boxes of each sampole
            img_metas (list[dict]): Meta information of samples.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                boxes to be ignored. Defaults to None.
            prev_bev (torch.Tensor, optional): BEV features of previous frame.
        Returns:
            dict: Losses of each branch.
        """
        outs = self.pts_bbox_head(
            pts_feats, img_metas, prev_bev, radar_feats=radar_feats, predict_bev_mask=pre_bev_mask, predict_bev_logit=predict_bev_logit, gt_bboxes_3d=gt_bboxes_3d)
        loss_inputs = [gt_bboxes_3d, gt_labels_3d, outs]
        losses = self.pts_bbox_head.loss(*loss_inputs, img_metas=img_metas)
        return losses

    def forward_dummy(self, img):
        dummy_metas = None
        return self.forward_test(img=img, img_metas=[[dummy_metas]])

    def forward(self, return_loss=True, **kwargs):
        """Calls either forward_train or forward_test depending on whether
        return_loss=True.
        Note this setting will change the expected inputs. When
        `return_loss=True`, img and img_metas are single-nested (i.e.
        torch.Tensor and list[dict]), and when `resturn_loss=False`, img and
        img_metas should be double nested (i.e.  list[torch.Tensor],
        list[list[dict]]), with the outer list indicating test time
        augmentations.
        """
        if return_loss:
            return self.forward_train(**kwargs)
        else:
            return self.forward_test(**kwargs)
    
    def obtain_history_bev(self, imgs_queue, img_metas_list, radar=None, **kwargs):
        """Obtain history BEV features iteratively. To save GPU memory, gradients are not calculated.
            radar: B*len_queu, 
        """
        self.eval()

        with torch.no_grad():
            prev_bev = None
            bs, len_queue, num_cams, C, H, W = imgs_queue.shape
            imgs_queue = imgs_queue.reshape(bs*len_queue, num_cams, C, H, W)

            img_metas_list_ = []
            gt_bev_mask_list = []
            
            if 'gt_bev_mask' not in kwargs.keys():
                gt_bev_mask_list = None
            else:
                gt_bev_mask_list = []

            for queue_i in range(len_queue):
                for b in range(bs):
                    img_metas_list_.append(img_metas_list[b][queue_i])
                    if gt_bev_mask_list is not None:
                        gt_bev_mask_list.append(kwargs['gt_bev_mask'][b][queue_i])

            results = self.extract_feat(img=imgs_queue, img_metas=img_metas_list_, len_queue=len_queue, radar=radar, gt_bev_mask=gt_bev_mask_list)
            img_feats_list = results['img_feats']
            radar_feats_list = results['radar_feats']
            foreground_radar_mask = results['foreground_radar_mask']
            foreground_radar_mask_logit = results['foreground_radar_mask_logit']
            
            for i in range(len_queue):
                img_metas = [each[i] for each in img_metas_list]
                if not img_metas[0]['prev_bev_exists']:
                    prev_bev = None
                img_feats = [each_scale[:, i] for each_scale in img_feats_list]
                radar_feats = [each_scale[:, i] for each_scale in radar_feats_list] if radar_feats_list is not None else None
                
                predict_bev_mask = [each_scale[:, i] for each_scale in foreground_radar_mask] if foreground_radar_mask is not None else None
                predict_bev_mask = torch.cat(predict_bev_mask, dim=0) if foreground_radar_mask is not None else None
                
                predict_bev_logit = [each_scale[:, i] for each_scale in foreground_radar_mask_logit] if foreground_radar_mask_logit is not None else None
                predict_bev_logit = torch.cat(predict_bev_logit, dim=0) if foreground_radar_mask_logit is not None else None
                              
                prev_bev = self.pts_bbox_head(
                    img_feats, img_metas, prev_bev, only_bev=True, radar_feats=radar_feats, predict_bev_mask=predict_bev_mask, predict_bev_logit=predict_bev_logit)
            
            self.train()
            return prev_bev


    @auto_fp16(apply_to=('img', 'points'))
    def forward_train(self,
                      points=None,
                      img_metas=None,
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
                      img=None,
                      proposals=None,
                      gt_bboxes_ignore=None,
                      radar=None,
                      **kwargs,
                      ):
        """Forward training function.
        Args:
            points (list[torch.Tensor], optional): Points of each sample.
                Defaults to None.
            img_metas (list[dict], optional): Meta information of each sample.
                Defaults to None.
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`], optional):
                Ground truth 3D boxes. Defaults to None.
            gt_labels_3d (list[torch.Tensor], optional): Ground truth labels
                of 3D boxes. Defaults to None.
            gt_labels (list[torch.Tensor], optional): Ground truth labels
                of 2D boxes in images. Defaults to None.
            gt_bboxes (list[torch.Tensor], optional): Ground truth 2D boxes in
                images. Defaults to None.
            img (torch.Tensor optional): Images of each sample with shape
                (N, C, H, W). Defaults to None.
            proposals ([list[torch.Tensor], optional): Predicted proposals
                used for training Fast RCNN. Defaults to None.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                2D boxes in images to be ignored. Defaults to None.
        Returns:
            dict: Losses of different branches.
        """

        len_queue = img.size(1)
        prev_img = img[:, :-1, ...]
        img = img[:, -1, ...]
        if radar is not None:
            B = len(radar)
            prev_radar = []
            cur_radar = []
            for b in range(B):
                cur_radar.append(radar[b][-1])
            
            for queue_i in range(len_queue - 1):
                for b in range(B):
                    prev_radar.append(radar[b][queue_i])
            radar = cur_radar
        else:
            prev_radar = None
            
        # do not use temporal information
        if not self.video_test_mode:
            prev_bev = None
        else:
            prev_img_metas = copy.deepcopy(img_metas)
            prev_bev = self.obtain_history_bev(prev_img, prev_img_metas, prev_radar, **kwargs)

        img_metas = [each[len_queue-1] for each in img_metas]
        if 'gt_bev_mask' in kwargs.keys():
            gt_bev_mask = [each[len_queue-1] for each in kwargs['gt_bev_mask']]
        else:
            gt_bev_mask = None
        if not img_metas[0]['prev_bev_exists']:
            prev_bev = None
        
        results = self.extract_feat(img=img, img_metas=img_metas, radar=radar, gt_bev_mask=gt_bev_mask)
        losses = dict()
        
        if self.with_specific_component('noisy_radar_filter'):
            losses_mask = self.noisy_radar_filter.foreground_radar_mask.get_bev_mask_loss(gt_bev_mask, results['foreground_radar_mask_logit'], results['foreground_radar_mask'])
            losses.update(losses_mask)

        if self.with_specific_component('pts_bbox_head'):
            losses_pts = self.forward_pts_train(results['img_feats'], 
                                                gt_bboxes_3d,
                                                gt_labels_3d, 
                                                img_metas,
                                                gt_bboxes_ignore, 
                                                prev_bev,
                                                results['radar_feats'], 
                                                pre_bev_mask=results['bev_mask'], 
                                                predict_bev_logit=results['bev_logit'])
            losses.update(losses_pts)

        return losses

    def forward_test(self, img_metas, img=None, **kwargs):
        for var, name in [(img_metas, 'img_metas')]:
            if not isinstance(var, list):
                raise TypeError('{} must be a list, but got {}'.format(
                    name, type(var)))
        img = [img] if img is None else img

        if img_metas[0][0]['scene_token'] != self.prev_frame_info['scene_token']:
            # the first sample of each scene is truncated
            self.prev_frame_info['prev_bev'] = None
        # update idx
        self.prev_frame_info['scene_token'] = img_metas[0][0]['scene_token']

        # do not use temporal information
        if not self.video_test_mode:
            self.prev_frame_info['prev_bev'] = None

        # Get the delta of ego position and angle between two timestamps.
        tmp_pos = copy.deepcopy(img_metas[0][0]['can_bus'][:3])
        tmp_angle = copy.deepcopy(img_metas[0][0]['can_bus'][-1])
        if self.prev_frame_info['prev_bev'] is not None:
            img_metas[0][0]['can_bus'][:3] -= self.prev_frame_info['prev_pos']
            img_metas[0][0]['can_bus'][-1] -= self.prev_frame_info['prev_angle']
        else:
            img_metas[0][0]['can_bus'][-1] = 0
            img_metas[0][0]['can_bus'][:3] = 0

        new_prev_bev, bbox_results = self.simple_test(
            img_metas[0], img[0], prev_bev=self.prev_frame_info['prev_bev'], **kwargs)
        # During inference, we save the BEV features and ego motion of each timestamp.
        self.prev_frame_info['prev_pos'] = tmp_pos
        self.prev_frame_info['prev_angle'] = tmp_angle
        self.prev_frame_info['prev_bev'] = new_prev_bev
        return bbox_results

    def simple_test_pts(self, x, img_metas, prev_bev=None, rescale=False, radar_feats=None, pre_bev_mask=None, predict_bev_logit=None, gt_bev_mask=None,):
        """Test function"""
        outs = self.pts_bbox_head(x, img_metas, prev_bev=prev_bev, radar_feats=radar_feats, predict_bev_mask=pre_bev_mask, predict_bev_logit=predict_bev_logit, gt_bev_mask=gt_bev_mask)

        bbox_list = self.pts_bbox_head.get_bboxes(
            outs, img_metas, rescale=rescale)
        bbox_results = [
            bbox3d2result(bboxes, scores, labels)
            for bboxes, scores, labels in bbox_list
        ]
        return outs['bev_embed'], bbox_results

    def simple_test(self, img_metas, img=None, prev_bev=None, rescale=False, radar=None, **kwargs):
        """Test function without augmentaiton."""
        results = self.extract_feat(img=img, img_metas=img_metas, radar=radar if radar is None else radar[0], **kwargs)
    
        bbox_list = [dict() for i in range(len(img_metas))]
        
        new_prev_bev, bbox_pts = self.simple_test_pts(
            results['img_feats'], img_metas, prev_bev, rescale=rescale, radar_feats=results['radar_feats'], pre_bev_mask=results['bev_mask'], predict_bev_logit=results['bev_logit'], gt_bev_mask=results['gt_bev_mask'])
        for result_dict, pts_bbox in zip(bbox_list, bbox_pts):
            result_dict['pts_bbox'] = pts_bbox
        return new_prev_bev, bbox_list
