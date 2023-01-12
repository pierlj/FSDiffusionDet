import copy

import torch
import torch.nn.functional as F
from torch import nn
from fvcore.common.checkpoint import Checkpointer

from detectron2.layers import ShapeSpec
from detectron2.utils.registry import Registry
from detectron2.modeling import META_ARCH_REGISTRY, build_backbone, detector_postprocess

from .data.fs_dataloading import ClassMapper, ClassSampler, FilteredDataLoader
from .data.utils import filter_class_table

BACKBONE_REGISTRY = Registry("BACKBONE")

class SupportExtractor(nn.Module):
    def __init__(self, cfg, dataset_metadata, mode='build_resnet_fpn_backbone', fpn=False, resnet_depth=50, *args, **kwargs):
        """
        Extractor objects that computes features from support images and annotations.
        
        mode can take values 'identical', 'build_resnet_fpn_backbone', 'build_swintransformer_fpn_backbone'
        """
        super().__init__(*args, **kwargs)

        if 'resnet' in mode:
            cfg.merge_from_list(['MODEL.RESNET.DEPTH', resnet_depth])

        self.cfg = cfg
        self.k_shot = cfg.FEWSHOT.K_SHOT
        self.dataset_metadata = copy.deepcopy(dataset_metadata)
        
        if mode == 'identical' and backbone is not None:
            self.extractor = backbone 
        else:
            if input_shape is None:
                input_shape = ShapeSpec(channels=len(cfg.MODEL.PIXEL_MEAN))

            self.extractor = BACKBONE_REGISTRY.get(mode)(cfg, input_shape)
        
        if 'resnet' in mode:
            self.checkpointer = Checkpointer(self.extractor, cfg.OUTPUT_DIR, False)
            self.checkpointer.load(cfg.SUPPORT_EXTRACTOR.WEIGHT)
        

    def forward(self, selected_classes):

        dataloader = self.get_dataloader(selected_classes)

        support_features = None
        return support_features
    
    def get_dataloader(self, selected_classes):
        sampler = ClassSampler(self.cfg, self.dataset_metadata, selected_classes, n_query=self.k_shot, is_support=True)
        mapper = ClassMapper(selected_classes, 
                            self.dataset_metadata.thing_dataset_id_to_contiguous_id,
                            self.cfg, 
                            is_train=True, 
                            remap_labels=remap_labels)

        dataloader = FilteredDataLoader(self.cfg, self.dataset, mapper, sampler, self.dataset_metadata, is_eval=True)


@BACKBONE_REGISTRY.register()
def build_custom_extractor(cfg, input_shape):
    extractor = CustomExtractor(cfg, input_shape)
    return extractor


class CustomExtractor(nn.Module):
    def __init__(self, cfg, input_shape):
        super().__init__()

