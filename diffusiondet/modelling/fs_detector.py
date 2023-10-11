import torch
from collections import OrderedDict
from detectron2.layers import batched_nms
from detectron2.modeling import META_ARCH_REGISTRY, detector_postprocess

from collections import namedtuple


from .detector import DiffusionDet
from .feature_extractor import SupportExtractor
from ..util.box_ops import box_cxcywh_to_xyxy, box_xyxy_to_cxcywh

ModelPrediction = namedtuple('ModelPrediction', ['pred_noise', 'pred_x_start'])

@META_ARCH_REGISTRY.register()
class FSDiffusionDet(DiffusionDet):

    def __init__(self, cfg, *args, **kwargs):
        self.selected_classes = None
        self._support_features = None
        self._num_classes = None
        self.cfg = cfg
        super().__init__(cfg, *args, **kwargs)
        
        self.selected_classes = None
        self.base_prediction = True

        

    def build_support_extractor(self):
        self.support_extractor = SupportExtractor(self.cfg, 
                                                self.device, 
                                                mode='identical', 
                                                backbone=self.backbone).to(self.device)
    
    def compute_support_features(self, selected_classes, dataset, dataset_metadata):
        self._support_features, self._dataset_classes = self.support_extractor(selected_classes, dataset, dataset_metadata)

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

    @property
    def num_classes(self):
        if self.selected_classes is not None and self.cfg.TRAIN_MODE == 'support_attention':
            return len(self.selected_classes)
        elif self._num_classes is not None:
            return self._num_classes
        else:
            return self.cfg.MODEL.DiffusionDet.NUM_CLASSES
    
    def ddim_sample(self, batched_inputs, backbone_feats, images_whwh, images, clip_denoised=True, do_postprocess=True):
        batch = images_whwh.shape[0]
        num_boxes = self.num_proposals if self.cfg.TRAIN_MODE != 'support_attention'\
                                     else self.num_proposals * len(self.selected_classes)
        shape = (batch, self.num_proposals, 4)
        total_timesteps, sampling_timesteps, eta, objective = self.num_timesteps, self.sampling_timesteps, self.ddim_sampling_eta, self.objective

        # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = torch.linspace(-1, total_timesteps - 1, steps=sampling_timesteps + 1)
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:]))  # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]

        img = torch.randn(shape, device=self.device)

        ensemble_score, ensemble_label, ensemble_coord = [], [], []
        x_start = None
        for time, time_next in time_pairs:
            time_cond = torch.full((batch,), time, device=self.device, dtype=torch.long)
            self_cond = x_start if self.self_condition else None

            preds, outputs_class, outputs_coord = self.model_predictions(backbone_feats, images_whwh, img, time_cond,
                                                                         self_cond, clip_x_start=clip_denoised)

            pred_noise, x_start = preds.pred_noise, preds.pred_x_start

            if self.box_renewal:  # filter
                score_per_image, box_per_image = outputs_class[-1][0], outputs_coord[-1][0]
                threshold = 0.5
                score_per_image = torch.sigmoid(score_per_image)
                if self.cfg.TRAIN_MODE != 'support_attention':
                    value, _ = torch.max(score_per_image, -1, keepdim=False)
                else:
                    value = score_per_image.permute(1, 0).flatten()

                keep_idx = value > threshold
                num_remain = torch.sum(keep_idx)

                pred_noise = pred_noise[:, keep_idx, :]
                x_start = x_start[:, keep_idx, :]
                if self.cfg.TRAIN_MODE == 'support_attention':
                    img = img.repeat(1, len(self.selected_classes), 1)
                img = img[:, keep_idx, :]
            if time_next < 0:
                img = x_start
                continue

            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]

            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()

            noise = torch.randn_like(img)

            img = x_start * alpha_next.sqrt() + \
                  c * pred_noise + \
                  sigma * noise

            if self.box_renewal:  # filter
                # replenish with randn boxes
                img = torch.cat((img, torch.randn(1, self.num_proposals - num_remain, 4, device=img.device)), dim=1)
            if self.use_ensemble and self.sampling_timesteps > 1:
                box_pred_per_image, scores_per_image, labels_per_image = self.inference(outputs_class[-1],
                                                                                        outputs_coord[-1],
                                                                                        images.image_sizes)
                ensemble_score.append(scores_per_image)
                ensemble_label.append(labels_per_image)
                ensemble_coord.append(box_pred_per_image)

        if self.use_ensemble and self.sampling_timesteps > 1:
            box_pred_per_image = torch.cat(ensemble_coord, dim=0)
            scores_per_image = torch.cat(ensemble_score, dim=0)
            labels_per_image = torch.cat(ensemble_label, dim=0)
            if self.use_nms:
                keep = batched_nms(box_pred_per_image, scores_per_image, labels_per_image, 0.5)
                box_pred_per_image = box_pred_per_image[keep]
                scores_per_image = scores_per_image[keep]
                labels_per_image = labels_per_image[keep]

            result = Instances(images.image_sizes[0])
            result.pred_boxes = Boxes(box_pred_per_image)
            result.scores = scores_per_image
            result.pred_classes = labels_per_image
            results = [result]
        else:
            output = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1]}
            box_cls = output["pred_logits"]
            box_pred = output["pred_boxes"]
            results = self.inference(box_cls, box_pred, images.image_sizes)
        if do_postprocess:
            processed_results = []
            for results_per_image, input_per_image, image_size in zip(results, batched_inputs, images.image_sizes):
                height = input_per_image.get("height", image_size[0])
                width = input_per_image.get("width", image_size[1])
                # height = image_size[0]
                # width = image_size[1]
                r = detector_postprocess(results_per_image, height, width)
                processed_results.append({"instances": r})
            return processed_results
    
    def model_predictions(self, backbone_feats, images_whwh, x, t, x_self_cond=None, clip_x_start=False):
        x_boxes = torch.clamp(x, min=-1 * self.scale, max=self.scale)
        x_boxes = ((x_boxes / self.scale) + 1) / 2
        x_boxes = box_cxcywh_to_xyxy(x_boxes)
        x_boxes = x_boxes * images_whwh[:, None, :]
        outputs_class, outputs_coord = self.head(backbone_feats, x_boxes, t, None)

        x_start = outputs_coord[-1]  # (batch, num_proposals, 4) predict boxes: absolute coordinates (x1, y1, x2, y2)
        x_start = x_start / images_whwh[:, None, :]
        x_start = box_xyxy_to_cxcywh(x_start)
        x_start = (x_start * 2 - 1.) * self.scale
        x_start = torch.clamp(x_start, min=-1 * self.scale, max=self.scale)


        if self.cfg.TRAIN_MODE == 'support_attention':
            x = x.repeat(1, len(self.selected_classes), 1)
        pred_noise = self.predict_noise_from_start(x, t, x_start)

        return ModelPrediction(pred_noise, x_start), outputs_class, outputs_coord
