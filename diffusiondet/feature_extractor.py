import torch
import torch.nn.functional as F
from torch import nn
from fvcore.common.checkpoint import Checkpointer

from detectron2.layers import ShapeSpec
from detectron2.utils.registry import Registry
from detectron2.modeling import META_ARCH_REGISTRY, build_backbone, detector_postprocess

BACKBONE_REGISTRY = Registry("BACKBONE")

class SupportExtractor(nn.Module):
    def __init__(self, cfg, mode='build_resnet_fpn_backbone', fpn=False, resnet_depth=50, *args, **kwargs):
        """
        Extractor objects that computes features from support images and annotations.
        
        mode can take values 'identical', 'build_resnet_fpn_backbone', 'build_swintransformer_fpn_backbone'
        """
        super().__init__(*args, **kwargs)

        if 'resnet' in mode:
            cfg.merge_from_list(['MODEL.RESNET.DEPTH', resnet_depth])

        self.cfg = cfg
        
        

        if mode == 'identical' and backbone is not None:
            self.extractor = backbone 
        else:
            if input_shape is None:
                input_shape = ShapeSpec(channels=len(cfg.MODEL.PIXEL_MEAN))

            self.extractor = BACKBONE_REGISTRY.get(mode)(cfg, input_shape)
        
        if 'resnet' in mode:
            self.checkpointer = Checkpointer(self.extractor, cfg.OUTPUT_DIR, False)
            self.checkpointer.load(cfg.SUPPORT_EXTRACTOR.WEIGHT)
        

    def forward(self, support_data):



        support_features = None
        return support_features


@BACKBONE_REGISTRY.register()
def build_custom_extractor(cfg, input_shape):
    extractor = CustomExtractor(cfg, input_shape)
    return extractor


class CustomExtractor(nn.Module):
    def __init__(self, cfg, input_shape):
        super().__init__()

