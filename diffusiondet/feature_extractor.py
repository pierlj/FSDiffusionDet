import copy

import torch
import torch.nn.functional as F
from torch import nn

from detectron2.layers import ShapeSpec
from detectron2.utils.registry import Registry
from detectron2.modeling import META_ARCH_REGISTRY, build_backbone, detector_postprocess
from detectron2.modeling.backbone.build import BACKBONE_REGISTRY
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.structures import Boxes, ImageList, Instances


from .data.fs_dataloading import *
from .data.utils import filter_class_table


class SupportExtractor():
    def __init__(self, cfg, mode='build_resnet_fpn_backbone', fpn=False, resnet_depth=50, *args, **kwargs):
        """
        Extractor objects that computes features from support images and annotations.
        
        mode can take values 'identical', 'build_resnet_fpn_backbone', 'build_swintransformer_fpn_backbone'
        """
        super().__init__(*args, **kwargs)

        if 'resnet' in mode:
            cfg = cfg.clone()
            cfg.merge_from_list(['MODEL.RESNETS.DEPTH', resnet_depth])

        self.cfg = cfg
        self.k_shot = cfg.FEWSHOT.K_SHOT
        
        if mode == 'identical' and backbone is not None:
            self.extractor = backbone 
        else:
            input_shape = ShapeSpec(channels=len(cfg.MODEL.PIXEL_MEAN))

            self.extractor = BACKBONE_REGISTRY.get(mode)(cfg, input_shape)
        
        if 'resnet' in mode:
            self.checkpointer = DetectionCheckpointer(self.extractor, cfg.OUTPUT_DIR, save_to_disk=False)
            self.checkpointer.load(cfg.FEWSHOT.SUPPORT_EXTRACTOR.WEIGHT)

        self.size_divisibility = self.extractor.size_divisibility
        self.device = next(self.extractor.parameters()).device

        pixel_mean = torch.Tensor(cfg.MODEL.PIXEL_MEAN).to(self.device).view(3, 1, 1)
        pixel_std = torch.Tensor(cfg.MODEL.PIXEL_STD).to(self.device).view(3, 1, 1)
        self.normalizer = lambda x: (x - pixel_mean) / pixel_std

        
        

    def __call__(self, selected_classes, dataset, dataset_metadata):
        support_features_dict = {c:[] for c in selected_classes}
        dataloader = self.get_dataloader(selected_classes, dataset, dataset_metadata)
        for data in dataloader:
            image_batch, image_whwh = self.preprocess_image(data)
            support_features = self.extractor(image_batch.tensor)
            for idx_batch in range(len(data)):
                feature_dict = {}
                for feat_name, feat in support_features.items():
                    feature_dict[feat_name] = feat[idx_batch][None] # keep 1xDxHxW dimension
                support_features_dict[data[idx_batch]['class_sampled']].append(feature_dict)
        for class_idx, class_features in support_features_dict.items():
            class_dict = {}
            for feat_name in class_features[0]:
                feat_list = []
                for feat_idx in range(len(class_features)):
                    feat_list.append(support_features_dict[class_idx][feat_idx][feat_name])
                class_dict[feat_name] = torch.cat(feat_list)
            support_features_dict[class_idx] = class_dict
        return support_features_dict
    

    def get_dataloader(self, selected_classes, dataset, dataset_metadata, remap_labels=False):

        sampler = SupportClassSampler(self.cfg, dataset_metadata, selected_classes, n_query=self.k_shot, is_support=True)
        mapper = ClassMapper(selected_classes, 
                            dataset_metadata.thing_dataset_id_to_contiguous_id,
                            self.cfg, 
                            is_train=True, 
                            remap_labels=remap_labels)
        
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


@BACKBONE_REGISTRY.register()
def build_custom_extractor(cfg, input_shape):
    extractor = CustomExtractor(cfg, input_shape)
    return extractor


class CustomExtractor(nn.Module):
    def __init__(self, cfg, input_shape):
        super().__init__()

