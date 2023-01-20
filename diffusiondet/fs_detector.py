import torch
from detectron2.modeling import META_ARCH_REGISTRY

from .detector import DiffusionDet
from .feature_extractor import SupportExtractor

@META_ARCH_REGISTRY.register()
class FSDiffusionDet(DiffusionDet):

    def __init__(self, cfg, *args, **kwargs):
        super().__init__(cfg, *args, **kwargs)
        self._support_features = None
        self.cfg = cfg
        self.selected_classes = None

    def build_support_extractor(self):
        self.support_extractor = SupportExtractor(self.cfg, self.device).to(self.device)
    
    def compute_support_features(self, selected_classes, dataset, dataset_metadata):
        self._support_features = self.support_extractor(selected_classes, dataset, dataset_metadata)

    def inference(self, box_cls, box_pred, image_sizes):
        results = super().inference(box_cls, box_pred, image_sizes)
        if self.selected_classes is not None:
            if not isinstance(self.selected_classes, torch.Tensor):
                self.selected_classes = torch.tensor(self.selected_classes, 
                                                    device=results[0].pred_classes.device)
            for idx, r in enumerate(results):
                r.pred_classes = self.selected_classes[r.pred_classes]
                results[idx] = r
        return results
