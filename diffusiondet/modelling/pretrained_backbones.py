import clip
import torch
import numpy as np
import torch.nn as nn

from detectron2.modeling.backbone.build import BACKBONE_REGISTRY

from detectron2.config import get_cfg
from detectron2.modeling.backbone.fpn import build_resnet_fpn_backbone, FPN, build_resnet_backbone, LastLevelMaxPool
from detectron2.modeling.backbone import Backbone
from detectron2.modeling import ResNet
from detectron2.layers import ShapeSpec

class ResNetD2Format(ResNet):
    def __init__(self, backbone, stem_length, out_features=['res2', 'res3', 'res4', 'res5'], freeze_at=0):
        Backbone.__init__(self)
        
        backbone_modules = [m for m in backbone.children()]
        self.stem = nn.Sequential(*backbone_modules[:stem_length])
        stages = backbone_modules[stem_length:]
        
        self._out_features = out_features
        self.num_classes = None
        
        current_stride = 4
        self._out_feature_strides = {"stem": current_stride}
        self._out_feature_channels = {"stem": self.stem[-4].out_channels}

        self.stage_names, self.stages = [], []

        if out_features is not None:
            # Avoid keeping unused layers in this module. They consume extra memory
            # and may cause allreduce to fail
            num_stages = max(
                [{"res2": 1, "res3": 2, "res4": 3, "res5": 4}.get(f, 0) for f in out_features]
            )
            stages = stages[:num_stages]
        for i, blocks in enumerate(stages):
            assert len(blocks) > 0, len(blocks)

            name = "res" + str(i + 2)
            stage = nn.Sequential(*blocks)

            self.add_module(name, stage)
            self.stage_names.append(name)
            self.stages.append(stage)

            self._out_feature_strides[name] = current_stride = int(
                current_stride * np.prod([k.stride for k in blocks])
            )
            self._out_feature_channels[name] = curr_channels = blocks[-1].conv3.out_channels
        self.stage_names = tuple(self.stage_names)  # Make it static for scripting

        if out_features is None:
            out_features = [name]
        self._out_features = out_features
        assert len(self._out_features)
        children = [x[0] for x in self.named_children()]
        for out_feature in self._out_features:
            assert out_feature in children, "Available children: {}".format(", ".join(children))
        self.freeze(freeze_at)
    

@BACKBONE_REGISTRY.register()
def build_resnet_dino_fpn_backbone(cfg, input_shape: ShapeSpec):
    dino_bb = torch.hub.load('facebookresearch/dino:main', 'dino_resnet50')
    bottom_up = ResNetD2Format(dino_bb, 4, out_features=cfg.MODEL.FPN.IN_FEATURES)

    in_features = cfg.MODEL.FPN.IN_FEATURES
    out_channels = cfg.MODEL.FPN.OUT_CHANNELS
    backbone = FPN(bottom_up=bottom_up, 
          in_features=in_features,
          out_channels=out_channels,
          norm='',
          top_block=LastLevelMaxPool(),
          fuse_type=cfg.MODEL.FPN.FUSE_TYPE)
    return backbone

@BACKBONE_REGISTRY.register()
def build_resnet_clip_fpn_backbone(cfg, input_shape: ShapeSpec):
    model, preprocess = clip.load('RN50', 'cpu')
    clip_bb = model.visual
    bottom_up = ResNetD2Format(clip_bb, 10, out_features=cfg.MODEL.FPN.IN_FEATURES)

    in_features = cfg.MODEL.FPN.IN_FEATURES
    out_channels = cfg.MODEL.FPN.OUT_CHANNELS
    backbone = FPN(bottom_up=bottom_up, 
          in_features=in_features,
          out_channels=out_channels,
          norm='',
          top_block=LastLevelMaxPool(),
          fuse_type=cfg.MODEL.FPN.FUSE_TYPE)
    return backbone
