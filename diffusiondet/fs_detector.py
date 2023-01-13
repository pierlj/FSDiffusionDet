from detectron2.modeling import META_ARCH_REGISTRY

from .detector import DiffusionDet
from .feature_extractor import SupportExtractor

@META_ARCH_REGISTRY.register()
class FSDiffusionDet(DiffusionDet):

    def __init__(self, cfg, *args, **kwargs):
        super().__init__(cfg, *args, **kwargs)

        self.support_extractor = SupportExtractor(cfg)
        self._support_features = None
    
    def compute_support_features(self, selected_classes, dataset, dataset_metadata):
        self._support_features = self.support_extractor(selected_classes, dataset, dataset_metadata)
