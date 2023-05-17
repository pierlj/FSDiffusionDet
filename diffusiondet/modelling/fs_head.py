import copy
import math

import numpy as np
import torch
from torch import nn, Tensor
import torch.nn.functional as F

from detectron2.modeling.poolers import ROIPooler
from detectron2.structures import Boxes


_DEFAULT_SCALE_CLAMP = math.log(100000.0 / 16)

from .head import DynamicHead, RCNNHead

class ClassDynamicConv(nn.Module):

    def __init__(self, cfg):
        super().__init__()

        self.hidden_dim = cfg.MODEL.DiffusionDet.HIDDEN_DIM
        self.dim_dynamic = cfg.MODEL.DiffusionDet.DIM_DYNAMIC
        self.num_dynamic = cfg.MODEL.DiffusionDet.NUM_DYNAMIC
        self.num_params = self.hidden_dim * self.dim_dynamic
        self.dynamic_layer = nn.Linear(self.hidden_dim, self.num_dynamic * self.num_params)

        self.norm1 = nn.LayerNorm(self.dim_dynamic)
        self.norm2 = nn.LayerNorm(self.hidden_dim)

        self.activation = nn.ReLU(inplace=True)

        pooler_resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        num_output = self.hidden_dim * pooler_resolution ** 2
        self.out_layer = nn.Linear(num_output, self.hidden_dim)
        self.norm3 = nn.LayerNorm(self.hidden_dim)

    def forward(self, pro_features, class_roi_features):
        '''
        pro_features: (N_classes,  N * nr_boxes, self.d_model)
        roi_features: (N_classes, 49, N * nr_boxes, self.d_model)
        '''
        features = class_roi_features#.permute(1, 0, 2)
        N_c, _, d = pro_features.shape
        parameters = self.dynamic_layer(pro_features).unsqueeze(2)

        param1 = parameters[..., :self.num_params].view(N_c, -1, self.hidden_dim, self.dim_dynamic)
        param2 = parameters[..., self.num_params:].view(N_c, -1, self.dim_dynamic, self.hidden_dim)

        features = torch.einsum('ijkl,iklm->ijkm', features, param1)
        # features = torch.bmm(features, param1)
        features = self.norm1(features)
        features = self.activation(features)

        features = torch.einsum('ijkl,iklm->ijkm', features, param2)
        # features = torch.bmm(features, param2)
        features = self.norm2(features)
        features = self.activation(features)

        features = features.permute(0,2,1,3)
        features = features.flatten(2)

        features = self.out_layer(features)
        features = self.norm3(features)
        features = self.activation(features)

        return features


class SupportDynamicConv(nn.Module):

    def __init__(self, cfg, support_interact=True):
        super().__init__()

        self.hidden_dim = cfg.MODEL.DiffusionDet.HIDDEN_DIM
        self.dim_dynamic = cfg.MODEL.DiffusionDet.DIM_DYNAMIC
        self.num_dynamic = cfg.MODEL.DiffusionDet.NUM_DYNAMIC
        self.num_params = self.hidden_dim * self.dim_dynamic
        self.dynamic_layer = nn.Linear(self.hidden_dim, self.num_dynamic * self.num_params)

        self.norm1 = nn.LayerNorm(self.dim_dynamic)
        self.norm2 = nn.LayerNorm(self.hidden_dim)

        self.activation = nn.ReLU(inplace=True)

        # pooler_resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        # num_output = self.hidden_dim * pooler_resolution ** 2
        # self.out_layer = nn.Linear(num_output, self.hidden_dim)
        # self.norm3 = nn.LayerNorm(self.hidden_dim)

    # def forward(self, support_features, roi_features):
    #     '''
    #     support_features: (C, 49, 1, self.d_model)
    #     roi_features: (49, N * nr_boxes, self.d_model)
    #     '''
    #     features = roi_features.unsqueeze(0)
    #     C, pooler_res, _, _ = support_features.shape
    #     parameters = self.dynamic_layer(support_features)#.permute(0, 2, 1, 3)

    #     param1 = parameters[..., :self.num_params].view(C, pooler_res, 1, self.hidden_dim, self.dim_dynamic)
    #     param2 = parameters[..., self.num_params:].view(C, pooler_res, 1, self.dim_dynamic, self.hidden_dim)

    #     features = torch.einsum('rjtl,ijklm->ijtm', features, param1)
    #     # features = torch.bmm(features, param1)
    #     features = self.norm1(features)
    #     features = self.activation(features)

    #     features = torch.einsum('rjtl,ijklm->ijtm', features, param2)
    #     # features = torch.bmm(features, param2)
    #     features = self.norm2(features)
    #     features = self.activation(features)

    #     return features

    def forward(self, support_features, roi_features):
        '''
        support_features: (C, 49, 1, self.d_model)
        roi_features: (C, 49, N * nr_boxes, self.d_model)
        '''
        features = roi_features
        C, pooler_res, _, d = support_features.shape
        
        support_representation = support_features.mean(dim=1).view(C, d)

        class_specific_features = torch.einsum('cijk,ck->cijk', features, support_representation)

        return class_specific_features


class FSDynamicHead(DynamicHead):

    def __init__(self, cfg, roi_input_shape, model_ref=None, head_class=None):
        self.model_ref = model_ref
        nn.Module.__init__(self)

        # Build RoI.
        box_pooler = self._init_box_pooler(cfg, roi_input_shape)
        self.box_pooler = box_pooler
        
        # Build heads.
        num_classes = cfg.MODEL.DiffusionDet.NUM_CLASSES
        d_model = cfg.MODEL.DiffusionDet.HIDDEN_DIM
        dim_feedforward = cfg.MODEL.DiffusionDet.DIM_FEEDFORWARD
        nhead = cfg.MODEL.DiffusionDet.NHEADS
        dropout = cfg.MODEL.DiffusionDet.DROPOUT
        activation = cfg.MODEL.DiffusionDet.ACTIVATION
        num_heads = cfg.MODEL.DiffusionDet.NUM_HEADS


        model_ref = self.model_ref
        rcnn_head = FSRCNNHead(cfg, d_model, num_classes, dim_feedforward, nhead, dropout, activation, model_ref=model_ref)
        

        self.head_series = _get_clones(rcnn_head, num_heads)
        self.num_heads = num_heads
        self.return_intermediate = cfg.MODEL.DiffusionDet.DEEP_SUPERVISION

        # Gaussian random feature embedding layer for time
        self.d_model = d_model
        time_dim = d_model * 4
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(d_model),
            nn.Linear(d_model, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim),
        )

        # Init parameters.
        self.use_focal = cfg.MODEL.DiffusionDet.USE_FOCAL
        self.use_fed_loss = cfg.MODEL.DiffusionDet.USE_FED_LOSS
        self.num_classes = num_classes
        if self.use_focal or self.use_fed_loss:
            prior_prob = cfg.MODEL.DiffusionDet.PRIOR_PROB
            self.bias_value = -math.log((1 - prior_prob) / prior_prob)
        self._reset_parameters()
    
    def forward(self, features, init_bboxes, t, init_features):
        if self.training:
            n_classes = len(self.model_ref().iteration_classes)
        else:
            n_classes = len(self.model_ref().selected_classes)
            
        init_bboxes = init_bboxes.repeat(1, n_classes,1)
        return super().forward(features, init_bboxes, t, init_features)

class FSRCNNHead(RCNNHead):

    def __init__(self, cfg, d_model, num_classes, *args, model_ref=None, **kwargs):
        super().__init__(cfg, d_model, 1, *args, **kwargs) # only 1 class as prediction are done separately

        self.model_ref = model_ref
        self.num_classes = num_classes

        self.support_attention = SupportDynamicConv(cfg)
        self.inst_interact = ClassDynamicConv(cfg)


    def forward(self, features, bboxes, pro_features, pooler, time_emb):
        """
        :param bboxes: (N, nr_boxes, 4)
        :param pro_features: (N, nr_boxes, d_model)
        """

        N, nr_boxes = bboxes.shape[:2]
        pooler_resolution = self.cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION ** 2
        
        # roi_feature.
        proposal_boxes = list()
        for b in range(N):
            proposal_boxes.append(Boxes(bboxes[b]))
        roi_features = pooler(features, proposal_boxes)


        # support attention
        support_features = self.model_ref()._support_features
        if self.training:
            iteration_classes = self.model_ref().iteration_classes
        else:
            iteration_classes = self.model_ref().selected_classes


        n_classes = len(iteration_classes)
        n_boxes_per_class = nr_boxes // n_classes

        support_features = torch.stack([feat.mean(dim=0, keepdim=True).view(1, self.d_model, pooler_resolution).permute(2,0,1) 
                                            for c, feat in support_features.items()
                                            if c in iteration_classes])
        
        roi_features = roi_features.view(N, n_classes, n_boxes_per_class, self.d_model, -1).permute(1,0,2,3,4)
        roi_features = roi_features.reshape(n_classes, N*n_boxes_per_class, self.d_model, -1).permute(0,3,1,2)

        class_specific_features = self.support_attention(support_features, roi_features) #Nc, 49, N*Nb, d

        if pro_features is None:
            pro_features = roi_features.mean(1, keepdim=True)
            # pro_features = pro_features.repeat(n_classes, 1, 1)

        # self_att.
        # pro_features = pro_features.view(N, nr_boxes, self.d_model).permute(1, 0, 2)
        # pro_features2 = self.self_attn(pro_features, pro_features, value=pro_features)[0]
        # pro_features = pro_features + self.dropout1(pro_features2)
        # pro_features = self.norm1(pro_features)
        

        

        # inst_interact.
        pro_features = pro_features.view(n_classes, n_boxes_per_class, N, self.d_model)\
                                   .permute(0, 2, 1, 3).reshape(n_classes, N * n_boxes_per_class, self.d_model)
        pro_features2 = self.inst_interact(pro_features, class_specific_features)
        pro_features = pro_features + self.dropout2(pro_features2)
        obj_features = self.norm2(pro_features)

        # obj_feature.
        obj_features2 = self.linear2(self.dropout(self.activation(self.linear1(obj_features))))
        obj_features = obj_features + self.dropout3(obj_features2)
        obj_features = self.norm3(obj_features) # Nc, N*Nb, d
        
        fc_feature = obj_features#.permute(0, 2, 1).reshape(n_classes, N * n_boxes_per_class, -1)

        scale_shift = self.block_time_mlp(time_emb)
        scale_shift = torch.repeat_interleave(scale_shift, n_boxes_per_class, dim=0)
        scale, shift = scale_shift[None].chunk(2, dim=-1)
        fc_feature = fc_feature * (scale + 1) + shift

        cls_feature = fc_feature.clone()
        reg_feature = fc_feature.clone()
        for cls_layer in self.cls_module:
            cls_feature = cls_layer(cls_feature)
        for reg_layer in self.reg_module:
            reg_feature = reg_layer(reg_feature)
        class_logits = self.class_logits(cls_feature)
        bboxes_deltas = self.bboxes_delta(reg_feature)


        bboxes_deltas = bboxes_deltas.view(n_classes, N, n_boxes_per_class, 4)\
                                     .permute(1, 0, 2, 3).flatten(end_dim=2)#.flatten(1)
        pred_bboxes = self.apply_deltas(bboxes_deltas, bboxes.view(-1, 4))

        # obj_features = torch.gather(obj_features.permute(1,0,2), 1, 
        #                     classes_labels.repeat(1,1,256))[:,0]
        class_logits = class_logits.view(n_classes, N, n_boxes_per_class).permute(1,2,0)
        pred_bboxes = pred_bboxes.view(N, n_classes * n_boxes_per_class, 4)
        return class_logits, pred_bboxes, obj_features