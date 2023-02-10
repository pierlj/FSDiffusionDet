import copy
import logging

import torch
import torch.nn.functional as F
from torch import nn

from detectron2.layers import ShapeSpec
from detectron2.utils.registry import Registry
from detectron2.modeling.poolers import ROIPooler
from detectron2.modeling import META_ARCH_REGISTRY, build_backbone, detector_postprocess
from detectron2.modeling.backbone.build import BACKBONE_REGISTRY
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.structures import Boxes, ImageList, Instances


from ..data.fs_dataloading import *
from ..data.utils import filter_class_table


class SupportExtractor(nn.Module):
    def __init__(self, cfg, device, mode='build_resnet_fpn_backbone', fpn=False, resnet_depth=50, *args, **kwargs):
        """
        Extractor objects that computes features from support images and annotations.
        
        mode can take values 'identical', 'build_resnet_fpn_backbone', 'build_swintransformer_fpn_backbone'
        """
        super().__init__(*args, **kwargs)

        self.logger = logging.getLogger(__name__)

        if 'resnet' in mode:
            cfg = cfg.clone()
            cfg.merge_from_list(['MODEL.RESNETS.DEPTH', resnet_depth])

        self.cfg = cfg
        self.device = device
        self.k_shot = cfg.FEWSHOT.K_SHOT
        
        if mode == 'identical' and backbone is not None:
            self.extractor = backbone 
        else:
            input_shape = ShapeSpec(channels=len(cfg.MODEL.PIXEL_MEAN))

            self.extractor = BACKBONE_REGISTRY.get(mode)(cfg, input_shape)
        
        if 'resnet' in mode:
            self.checkpointer = DetectionCheckpointer(self.extractor, cfg.OUTPUT_DIR, save_to_disk=False)
            self.checkpointer.load(cfg.FEWSHOT.SUPPORT_EXTRACTOR.WEIGHT)

            # for name, child in self.extractor.bottom_up.named_children():
            #     if 'res' in module_name:
            #         self.extractor.bottom_up._modules['ext_' + module_name] = self.extractor.bottom_up._modules.pop(module_name)
                    # setattr(self.extractor.bottom_up, 'ext_' + attr,
                    #         getattr(self.extractor.bottom_up, attr))
                    # delattr(self.extractor.bottom_up, attr)
        self.size_divisibility = self.extractor.size_divisibility

        pixel_mean = torch.Tensor(cfg.MODEL.PIXEL_MEAN).to(self.device).view(3, 1, 1)
        pixel_std = torch.Tensor(cfg.MODEL.PIXEL_STD).to(self.device).view(3, 1, 1)
        self.normalizer = lambda x: (x - pixel_mean) / pixel_std

        self.pooler = self._init_box_pooler(cfg, self.extractor.output_shape())
        
        

    def __call__(self, selected_classes, dataset, dataset_metadata):
        self.logger.info('Compute support features for classes: {}'.format(selected_classes))
        support_features_dict = {c:[] for c in selected_classes}
        dataloader = self.get_dataloader(selected_classes, dataset, dataset_metadata)
        with torch.no_grad():
            for data in dataloader:
                image_batch, image_whwh = self.preprocess_image(data)
                image_batch = image_batch.to(self.device)
                support_features = self.extractor(image_batch.tensor)

                support_bbox = [d['instances'].gt_boxes.to(self.device) for d in data]
                pooled_support_features = self.pooler(list(support_features.values())[:-1], support_bbox)
                del support_features

                for idx_batch in range(len(data)):
                    # print(data[idx_batch]['class_sampled'], data[idx_batch]['image_id'], data[idx_batch]['instances'].gt_boxes)
                    support_features_dict[data[idx_batch]['class_sampled']].append(pooled_support_features[idx_batch])

            for class_idx, class_features in support_features_dict.items():
                support_features_dict[class_idx] = torch.stack(class_features)
            # for class_idx, class_features in support_features_dict.items():
            #     class_dict = {}
            #     for feat_name in class_features[0]:
            #         feat_list = []
            #         for feat_idx in range(len(class_features)):
            #             feat_list.append(support_features_dict[class_idx][feat_idx][feat_name])
            #         class_dict[feat_name] = torch.cat(feat_list)
            #     support_features_dict[class_idx] = class_dict
        return support_features_dict, (dataset_metadata.base_classes, dataset_metadata.novel_classes)
    

    def get_dataloader(self, selected_classes, dataset, dataset_metadata, remap_labels=False):

        sampler = SupportClassSampler(self.cfg, 
                                      dataset_metadata, 
                                      selected_classes, 
                                      n_query=self.k_shot, 
                                      base_support=self.cfg.FEWSHOT.BASE_SUPPORT)
        
        class_repartition = {'base': dataset_metadata.base_classes,
                             'novel': dataset_metadata.novel_classes}
        mapper = SupportClassMapper(selected_classes, 
                            dataset_metadata.thing_dataset_id_to_contiguous_id,
                            class_repartition, 
                            self.cfg.FEWSHOT.BASE_SUPPORT,
                            self.cfg, 
                            is_train=False, 
                            remap_labels=remap_labels,
                            log=False)
        
        dataset = SupportDataset(dataset)

        dataloader = FilteredDataLoader(self.cfg, 
                                        dataset, 
                                        mapper, 
                                        sampler, 
                                        dataset_metadata, 
                                        is_eval=True, 
                                        is_support=True, 
                                        forced_bs=8)
        return dataloader
    
    def preprocess_image(self, support_input):
        """
        Normalize, pad and batch the input images.
        """
        images = [self.normalizer(x["image"].to(self.device)) for x in support_input]
        images = ImageList.from_tensors(images, self.size_divisibility)

        images_whwh = list()
        for bi in support_input:
            h, w = bi["image"].shape[-2:]
            images_whwh.append(torch.tensor([w, h, w, h], dtype=torch.float32, device=self.device))
        images_whwh = torch.stack(images_whwh)

        return images, images_whwh
    

    @staticmethod
    def _init_box_pooler(cfg, input_shape):

        in_features = cfg.MODEL.ROI_HEADS.IN_FEATURES
        pooler_resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        pooler_scales = tuple(1.0 / input_shape[k].stride for k in in_features)
        sampling_ratio = cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        pooler_type = cfg.MODEL.ROI_BOX_HEAD.POOLER_TYPE

        # If StandardROIHeads is applied on multiple feature maps (as in FPN),
        # then we share the same predictors and therefore the channel counts must be the same
        in_channels = [input_shape[f].channels for f in in_features]
        # Check all channel counts are equal
        assert len(set(in_channels)) == 1, in_channels

        box_pooler = ROIPooler(
            output_size=pooler_resolution,
            scales=pooler_scales,
            sampling_ratio=sampling_ratio,
            pooler_type=pooler_type,
        )
        return box_pooler


@BACKBONE_REGISTRY.register()
def build_custom_extractor(cfg, input_shape):
    extractor = CustomExtractor(cfg, input_shape)
    return extractor


class CustomExtractor(nn.Module):
    def __init__(self, cfg, input_shape):
        super().__init__()

