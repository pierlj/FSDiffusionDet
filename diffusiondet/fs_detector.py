from detectron2.modeling import META_ARCH_REGISTRY

from .detector import DiffusionDet
from .feature_extractor import SupportExtractor

@META_ARCH_REGISTRY.register()
class FSDiffusionDet(DiffusionDet):

    def __init__(self, cfg, *args, **kwargs):
        super().__init__(cfg, *args, **kwargs)
        self._support_features = None
        self.cfg = cfg

    def build_support_extractor(self):
        self.support_extractor = SupportExtractor(self.cfg, self.device).to(self.device)
    
    def compute_support_features(self, selected_classes, dataset, dataset_metadata):
        self._support_features = self.support_extractor(selected_classes, dataset, dataset_metadata)
