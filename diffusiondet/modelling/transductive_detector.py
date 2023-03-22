import numpy as np
from scipy import sparse
from sklearn.neighbors import NearestNeighbors

import torch
from collections import OrderedDict
from detectron2.layers import batched_nms
from detectron2.modeling import META_ARCH_REGISTRY, detector_postprocess

from collections import namedtuple


from .detector import DiffusionDet
from ..util.box_ops import box_cxcywh_to_xyxy, box_xyxy_to_cxcywh
from ..util.laplacian_shot import bound_update

ModelPrediction = namedtuple('ModelPrediction', ['pred_noise', 'pred_x_start'])

@META_ARCH_REGISTRY.register()
class TDiffusionDet(DiffusionDet):

    def __init__(self, cfg, *args, **kwargs):
        self.selected_classes = None
        self._support_features = None
        self._num_classes = None
        self.cfg = cfg
        super().__init__(cfg, *args, **kwargs)

        self._support_embeddings =  None

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
        if self.selected_classes is not None:
            return len(self.selected_classes)
        elif self._num_classes is not None:
            return self._num_classes
        else:
            return self.cfg.MODEL.DiffusionDet.NUM_CLASSES
    
    def ddim_sample(self, batched_inputs, backbone_feats, images_whwh, images, clip_denoised=True, do_postprocess=True):
        batch = images_whwh.shape[0]
        num_boxes = self.num_proposals
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

            preds, _, outputs_coord, output_embeddings = self.model_predictions(backbone_feats, images_whwh, img, time_cond,
                                                                         self_cond, clip_x_start=clip_denoised)

            outputs_class = self.transductive_inference(output_embeddings)



            pred_noise, x_start = preds.pred_noise, preds.pred_x_start

            if self.box_renewal:  # filter
                score_per_image, box_per_image = outputs_class[-1][0], outputs_coord[-1][0]
                threshold = 0.5
                score_per_image = torch.sigmoid(score_per_image)
                value, _ = torch.max(score_per_image, -1, keepdim=False)

                keep_idx = value > threshold
                num_remain = torch.sum(keep_idx)

                pred_noise = pred_noise[:, keep_idx, :]
                x_start = x_start[:, keep_idx, :]

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
                r = detector_postprocess(results_per_image, height, width)
                processed_results.append({"instances": r})
            return processed_results
    
    def model_predictions(self, backbone_feats, images_whwh, x, t, x_self_cond=None, clip_x_start=False):
        x_boxes = torch.clamp(x, min=-1 * self.scale, max=self.scale)
        x_boxes = ((x_boxes / self.scale) + 1) / 2
        x_boxes = box_cxcywh_to_xyxy(x_boxes)
        x_boxes = x_boxes * images_whwh[:, None, :]
        outputs_class, outputs_coord, output_embeddings = self.head(backbone_feats, x_boxes, t, None)

        x_start = outputs_coord[-1]  # (batch, num_proposals, 4) predict boxes: absolute coordinates (x1, y1, x2, y2)
        x_start = x_start / images_whwh[:, None, :]
        x_start = box_xyxy_to_cxcywh(x_start)
        x_start = (x_start * 2 - 1.) * self.scale
        x_start = torch.clamp(x_start, min=-1 * self.scale, max=self.scale)

        pred_noise = self.predict_noise_from_start(x, t, x_start)

        return ModelPrediction(pred_noise, x_start), outputs_class, outputs_coord, output_embeddings

    def forward(self, batched_inputs, do_postprocess=True):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper` .
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:

                * image: Tensor, image in (C, H, W) format.
                * instances: Instances

                Other information that's included in the original dicts, such as:

                * "height", "width" (int): the output resolution of the model, used in inference.
                  See :meth:`postprocess` for details.
        """
        images, images_whwh = self.preprocess_image(batched_inputs)
        if isinstance(images, (list, torch.Tensor)):
            images = nested_tensor_from_tensor_list(images)

        # Feature Extraction.
        src = self.backbone(images.tensor)
        features = list()
        for f in self.in_features:
            feature = src[f]
            features.append(feature)

        # Prepare Proposals.
        if not self.training:
            results = self.ddim_sample(batched_inputs, features, images_whwh, images)
            return results

        if self.training:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
            targets, x_boxes, noises, t = self.prepare_targets(gt_instances)
            t = t.squeeze(-1)
            x_boxes = x_boxes * images_whwh[:, None, :]

            outputs_class, outputs_coord, output_embs = self.head(features, x_boxes, t, None)
            output = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1]}

            if self.deep_supervision:
                output['aux_outputs'] = [{'pred_logits': a, 'pred_boxes': b}
                                         for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]

            loss_dict = self.criterion(output, targets)
            weight_dict = self.criterion.weight_dict
            for k in loss_dict.keys():
                if k in weight_dict:
                    loss_dict[k] *= weight_dict[k]
            return loss_dict

    def transductive_inference(self, query_embeddings):
        support_embeddings, support_labels = self.extract_support_embeddings()
        
        k_shot = self.cfg.FEWSHOT.K_SHOT
        n_ways = len(self.selected_classes)
        # compute prototype as mean support embedding per class
        support_embeddings = torch.stack([emb[-1, 0, 0] for emb in support_embeddings], dim=0)
        support_labels = torch.cat([torch.cat(lab, dim=0) for lab in support_labels], dim=0)
        
        support_labels, sorted_indices = torch.sort(support_labels)
        support_embeddings = support_embeddings[sorted_indices]

        support_embeddings = support_embeddings.reshape(n_ways, k_shot, -1).mean(dim=1)
        support_labels = support_labels.reshape(n_ways, k_shot)[:,0]

    
        query_embeddings = query_embeddings[-1]
        B, n_boxes, d = query_embeddings.shape
        n_classes, d = support_embeddings.shape
        query_embeddings = query_embeddings.flatten(end_dim=1) # B*Nboxes x d

        # center and normalize query and support embeddings
        # TODO change center to use train mean (all classes and all images)
        mean_embedding = support_embeddings.mean(dim=0)
        # mean_embedding = query_embeddings.mean(dim=0)
        # mean_embedding = get_train_mean()


        support_embeddings = support_embeddings - mean_embedding
        support_embeddings = support_embeddings / support_embeddings.norm(p=2, dim=1, keepdim=True)

        # query_embeddings = query_embeddings - query_embeddings.mean(dim=0) 
        query_embeddings = query_embeddings - mean_embedding
        query_embeddings = query_embeddings / query_embeddings.norm(p=2, dim=1, keepdim=True)

        #add support rectification here

        substract = support_embeddings[:, None, :] - query_embeddings
        distance = substract.norm(p=2, dim=-1).cpu().numpy()

        W = self.create_affinity(query_embeddings.cpu().numpy(), 10)
        lmd = 1.0
        logits = bound_update(distance.transpose() ** 2, W, lmd)  
        logits =  torch.from_numpy(logits).to(substract.device)
        logits = torch.randn_like(logits)
        return logits.reshape(1, B, n_boxes, n_classes)


    def create_affinity(self, X, knn):
        N, D = X.shape
        # print('Compute Affinity ')
        nbrs = NearestNeighbors(n_neighbors=knn).fit(X)
        dist, knnind = nbrs.kneighbors(X)

        row = np.repeat(range(N), knn - 1)
        col = knnind[:, 1:].flatten()
        data = np.ones(X.shape[0] * (knn - 1))
        W = sparse.csc_matrix((data, (row, col)), shape=(N, N), dtype=np.float)

        return W

    
    def extract_support_embeddings(self):
        if self._support_embeddings is None:
            batched_embeddings = []
            batched_labels = []
            for batched_inputs in self.support_loader:
                images, images_whwh = self.preprocess_image(batched_inputs)
                if isinstance(images, (list, torch.Tensor)):
                    images = nested_tensor_from_tensor_list(images)

                # Feature Extraction.
                src = self.backbone(images.tensor)
                features = list()
                for f in self.in_features:
                    feature = src[f]
                    features.append(feature)

                gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
                targets, x_boxes, noises, _ = self.prepare_targets(gt_instances)
                support_labels = [inst.gt_classes for inst in gt_instances]
                support_boxes = [inst.gt_boxes.tensor[None] for inst in gt_instances]
                
                t = torch.zeros(1, device=x_boxes.device)
                
                for idx_support, boxes in enumerate(support_boxes):
                    feat_support = [feat[idx_support][None] for feat in features]
                    
                    _, _, output_embs = self.head(feat_support, boxes, t, None, is_support=True)
                    batched_embeddings.append(output_embs)
                batched_labels.append(support_labels)
            self._support_embeddings = (batched_embeddings, batched_labels)
            return batched_embeddings, batched_labels
        else: 
            return self._support_embeddings
            