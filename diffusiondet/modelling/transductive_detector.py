import os

import numpy as np
from scipy import sparse
from sklearn.neighbors import NearestNeighbors

import torch
import torch.nn.functional as F
import torchvision
from collections import OrderedDict
from detectron2.layers import batched_nms
from detectron2.modeling import META_ARCH_REGISTRY, detector_postprocess
from detectron2.structures import Boxes, ImageList, Instances


from collections import namedtuple

from ..train.transductive_trainer import TransductiveTrainer
from .detector import DiffusionDet
from ..util.box_ops import box_cxcywh_to_xyxy, box_xyxy_to_cxcywh
from ..util.laplacian_shot import *
from ..util.misc import save_pickle, load_pickle
from ..util.plot_utils import plot_tsne, save_tsne_data

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
        self._train_mean = None
        self.save_dir = None

    def inference(self, box_cls, box_pred, image_sizes):
        results = []

        if self.selected_classes is not None:
            if not isinstance(self.selected_classes, torch.Tensor):
                self.selected_classes = torch.tensor(self.selected_classes, 
                                                    device=box_cls[0].device)
        for cls_per_img, pred_per_img, sizes_per_img in zip(box_cls, box_pred, image_sizes):
            n_boxes = pred_per_img.shape[0]

            scores_per_image = torch.sigmoid(cls_per_img)
            # scores_per_image = cls_per_img
            labels = torch.arange(self.num_classes, device=self.device). \
                unsqueeze(0).repeat(n_boxes, 1).flatten(0, 1)

            result = Instances(sizes_per_img)
            scores_per_image, topk_indices = scores_per_image.flatten(0, 1).topk(n_boxes, sorted=False)
            labels_per_image = labels[topk_indices]
            pred_per_img = pred_per_img.view(-1, 1, 4).repeat(1, self.num_classes, 1).view(-1, 4)
            pred_per_img = pred_per_img[topk_indices]


            if self.use_nms:
                keep = batched_nms(pred_per_img, scores_per_image, labels_per_image, 0.5)
                pred_per_img = pred_per_img[keep]
                scores_per_image = scores_per_image[keep]
                labels_per_image = labels_per_image[keep]

            result.pred_boxes = Boxes(pred_per_img)
            result.scores = scores_per_image
            if self.selected_classes is not None:
                result.pred_classes = self.selected_classes[labels_per_image]
            else:
                result.pred_classes = labels_per_image
            results.append(result)

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

        # gt_boxes = [torch.stack([torch.tensor(ann['bbox']) for ann in b['instances']]).to(self.device) for b in batched_inputs]
        # gt_boxes = [torch.cat([bbox[:, :2], bbox[:, :2] + bbox[:, 2:]], dim=-1) for bbox in gt_boxes]
        # gt_labels = [torch.stack([torch.tensor(ann['category_id']) for ann in b['instances']]).to(self.device) for b in batched_inputs]
        gt_boxes = [b['instances'].gt_boxes.tensor.to(self.device) for b in batched_inputs]
        gt_labels = [b['instances'].gt_classes.to(self.device) for b in batched_inputs]
        
        ensemble_score, ensemble_label, ensemble_coord = [], [], []
        x_start = None
        for time, time_next in time_pairs:
            time_cond = torch.full((batch,), time, device=self.device, dtype=torch.long)
            self_cond = x_start if self.self_condition else None

            preds, outputs_class, outputs_coord, outputs_embeddings = self.model_predictions(backbone_feats, images_whwh, img, time_cond,
                                                                         self_cond, clip_x_start=clip_denoised)

            # gt_labels = 
            
            # outputs_class = self.transductive_inference(output_embeddings)

            # pred_noise, x_start = preds.pred_noise, preds.pred_x_start
            batch_idx = torch.arange(outputs_class.shape[1], device=self.device).repeat_interleave(outputs_class.shape[2]) 
            scores = outputs_class[-1].flatten(end_dim=1)
            coords = outputs_coord[-1].flatten(end_dim=1)
            
            embeddings = outputs_embeddings[-1].flatten(end_dim=1)
            img = img.flatten(end_dim=1)

            threshold = 0.05
            scores = torch.sigmoid(scores)
            value, _ = torch.max(scores, -1, keepdim=False)

            keep_idx = value > threshold
            num_remain = torch.sum(keep_idx)

            # scores = scores[keep_idx,:]
            coords = coords[keep_idx,:]
            img = img[keep_idx, :]
            batch_idx = batch_idx[keep_idx]


            outputs_class = outputs_class[-1].flatten(end_dim=1)
            outputs_class = outputs_class[keep_idx,:]
            predicted_labels = outputs_class.argmax(dim=-1)
        
        n_box_per_img = [len((batch_idx == i).nonzero(as_tuple=True)[0]) for i in range(outputs_coord.shape[1])]

        ft_logits = torch.split(outputs_class, n_box_per_img)
        ft_boxes = torch.split(coords, n_box_per_img)
        
        transductive_logits = self.transductive_inference(embeddings[keep_idx, :], outputs_class, coords)
        transductive_logits = torch.split(torch.log(transductive_logits/ (1 - transductive_logits)), n_box_per_img)
        # transductive_logits = torch.split(transductive_logits, n_box_per_img)


        embeddings_split = torch.split(embeddings[keep_idx, :], n_box_per_img)
        # matched_labels = self.match_labels_with_gt(ft_boxes, gt_boxes, gt_labels)
        matched_labels = self.match_gt_with_labels(ft_boxes, gt_boxes, gt_labels)
        
        # compute true labels with matching to check perf upper bound
        # classes_tensor = torch.tensor(self.selected_classes, device=self.device)
        # keeps = [(m[None] == classes_tensor[:, None]).sum(dim=0).nonzero().flatten() for m in matched_labels]
        # matched_labels = [(m[k][None, :] == classes_tensor[:, None]).nonzero()[:, 0] for m, k in zip(matched_labels, keeps)]
        # true_logits = [torch.nn.functional.one_hot(m, num_classes=3) for m in matched_labels]
 

        # Save embedding + images in folder to visualize 
        save_tsne_data(embeddings_split, ft_logits, ft_boxes, batched_inputs, n_box_per_img,
                        matched_labels, transductive_logits, 
                        self.selected_classes, self.save_dir)
        # plot_tsne(self.tsne, embeddings[keep_idx, :], labels=torch.cat(matched_labels))

        # output = {'pred_logits': ft_logits, 'pred_boxes': ft_boxes}
        output = {'pred_logits': transductive_logits, 'pred_boxes': ft_boxes}
        # true_boxes = [b[k] for b, k in zip(ft_boxes, keeps)]
        # output = {'pred_logits': true_logits, 'pred_boxes': true_boxes}
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
    
    def match_labels_with_gt(self, pred_boxes, gt_boxes, gt_labels):
        matched_labels = []
        for (preds, gts, labels) in zip(pred_boxes, gt_boxes, gt_labels):
            ious = torchvision.ops.box_iou(preds, gts)
            iou_val, match_idx = torch.max(ious, dim=-1)
            match_idx = labels[match_idx]
            match_idx[iou_val < 0.1] = -1
            matched_labels.append(match_idx)
        return matched_labels

    def match_gt_with_labels(self, pred_boxes, gt_boxes, gt_labels):
        matched_labels = []
        for (preds, gts, labels) in zip(pred_boxes, gt_boxes, gt_labels):
            if preds.shape[0] > 0:
                ious = torchvision.ops.box_iou(preds, gts)
                iou_val, gt_index_per_box = torch.max(ious, dim=1)
                labels_per_box = labels[gt_index_per_box]
                iou_val, box_label_per_gt = torch.max(ious, dim=0) # index best boxes for each gt
                labels_matched = box_label_per_gt[gt_index_per_box]
                match_idx = torch.where(labels_matched == torch.arange(labels_matched.shape[0], device=labels_matched.device),
                                            labels[gt_index_per_box], torch.ones_like(gt_index_per_box) * -1)
                # match_idx[iou_val < 0.1] = -1
                matched_labels.append(match_idx)
            else:
                matched_labels.append(torch.tensor([], device=preds.device))
        return matched_labels
    
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

    def transductive_inference(self, query_embeddings, ft_labels=None, ft_boxes=None):
        # train_mean = self.extract_train_mean()
        support_embeddings, support_labels = self.extract_support_embeddings()
        
        k_shot = self.cfg.FEWSHOT.K_SHOT
        n_ways = len(self.selected_classes)
        # compute prototype as mean support embedding per class
        support_embeddings = torch.stack([emb[-1, 0, 0] for emb in support_embeddings], dim=0)
        support_labels = torch.cat([torch.cat(lab, dim=0) for lab in support_labels], dim=0)
        
        support_labels, sorted_indices = torch.sort(support_labels)
        support_embeddings = support_embeddings[sorted_indices]

        support_aggregation = False
        if support_aggregation:
            support_embeddings = support_embeddings.reshape(n_ways, k_shot, -1).mean(dim=1)
            support_labels = support_labels.reshape(n_ways, k_shot)[:,0]

    
        n_boxes, d = query_embeddings.shape
        n_classes, d = support_embeddings.shape

        # center and normalize query and support embeddings
        # TODO change center to use train mean (all classes and all images)
        mean_embedding = support_embeddings.mean(dim=0)
        # mean_embedding = query_embeddings.mean(dim=0)
        # mean_embedding = get_train_mean()
        

        # support_embeddings = support_embeddings - train_mean[None]
        support_embeddings = support_embeddings - mean_embedding
        support_embeddings = support_embeddings / support_embeddings.norm(p=2, dim=1, keepdim=True)

        # query_embeddings = query_embeddings - train_mean[None]
        query_embeddings = query_embeddings - query_embeddings.mean(dim=0) 
        query_embeddings = query_embeddings / query_embeddings.norm(p=2, dim=1, keepdim=True)

        

        #add support rectification here
        support_rect = False
        if support_rect:
            shift = support_embeddings.mean(dim=0) - query_embeddings.mean(dim=0)
            query_embeddings += shift[None]
            all_embeddings = torch.cat([support_embeddings, query_embeddings], dim=0)
            cos_sim = F.cosine_similarity(all_embeddings[:, None, :], support_embeddings[None, :, :], dim=2)
            predict = torch.argmin(1 - cos_sim, dim=1)
            W = F.softmax(1 * cos_sim, dim=1)
            support_embeddings = torch.cat([
                (W[predict==i, i].unsqueeze(1) * all_embeddings[predict==i]).mean(0, keepdim=True) for i in predict.unique()
            ])

        substract = support_embeddings[:, None, :] - query_embeddings
        distance = substract.norm(p=2, dim=-1).cpu().numpy()

        W = self.create_affinity(query_embeddings.cpu().numpy(), 3)
        lmd = 1.0
        eta = 0.0
        # logits = bound_update(distance.transpose() ** 2, W, lmd)  
        if ft_labels is not None and support_aggregation:
            logits = bound_update_mod(distance.transpose(), W, lmd, torch.sigmoid(ft_labels).cpu().numpy(), eta) 
        elif ft_labels is not None:
            repeated_ft_scores = torch.sigmoid(ft_labels)[:, :, None].repeat(1, 1, k_shot).flatten(start_dim=1).cpu().numpy()
            # logits = bound_update_mod(distance.transpose()**2, W, lmd, repeated_ft_scores, eta) 
            logits, cluster_means = bound_update_kmeans(support_embeddings, query_embeddings, W, lmd, repeated_ft_scores, eta) 
        logits =  torch.from_numpy(logits).to(substract.device)
        save_pickle(os.path.join(self.save_dir, 'visualization/tmp', 'support_cluster_means.pkl'), cluster_means)

        if not support_aggregation:
            logits = logits.reshape(-1, n_ways, k_shot).max(dim=-1)[0]
        # logits = - torch.from_numpy(distance).to(substract.device)

        t_labels = torch.argmax(logits, dim=-1)
        boxes_index = torch.arange(ft_boxes.shape[0], device=self.device)
        ft_boxes_per_class = [ft_boxes[t_labels == c] for c in range(logits.shape[-1])]
        ft_boxes_idx = [boxes_index[t_labels == c] for c in range(logits.shape[-1])]

        scores_all = torch.zeros_like(ft_labels[:,0])
        for boxes, indices in zip(ft_boxes_per_class, ft_boxes_idx):
            hw = boxes[:, 2:] - boxes[:, :2]
            keep = (hw.prod(dim=1) != 0)
            h_over_w = hw[:, 1] / hw[:, 0]
            # geom_features = torch.cat([hw, h_over_w[:, None]], dim=-1)
            geom_features = torch.cat([h_over_w[:, None]], dim=-1)
            geom_features = torch.nan_to_num(geom_features, 0.0)
            geom_features[geom_features == float('inf')] = 0
            mean = geom_features[keep].mean(dim=0)
            std = geom_features[keep].std(dim=0)
            score = 1 - torch.erf(torch.abs(geom_features - mean)/(std * math.sqrt(2)))
            scores_all[indices[keep]] = score[keep].prod(dim=1)
        # logits = logits[:, [2,1,0]]
        return logits
        # return logits * torch.sigmoid(ft_labels)
        # return  scores_all[:, None].repeat(1,3)


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
            save_pickle(os.path.join(self.save_dir, 'visualization/tmp', 'support_embeddings.pkl'), 
                        (batched_embeddings, batched_labels))
            return batched_embeddings, batched_labels
        else: 
            return self._support_embeddings
    
    def extract_train_mean(self):
        if self._train_mean is None:
            if not os.path.isfile(self.save_dir + '/train_mean.plk'):
                # use build_test_loader to avoid augmentations
                train_loader = TransductiveTrainer.build_train_mean_dataloader(self.cfg, 
                                            self.base_classes, 
                                            [self.cfg.DATASETS.TRAIN[0]])
                batched_embeddings = []
                batched_labels = []
                for idx, batched_inputs in enumerate(train_loader):
                    print('Computing train mean: {:.2f}%'.format(idx / len(train_loader) * 100))
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
                        batched_embeddings.append(output_embs.cpu()) # copy features to cpu to prevent memory overflow
                    batched_labels.append(support_labels)
                    # if idx > 10:
                    #     break
                
                batched_embeddings = torch.cat([b[-1,0,:, :] for b in batched_embeddings])
                self._train_mean = batched_embeddings.mean(dim=0).to(self.device)

                save_pickle(self.save_dir + '/train_mean.plk', self._train_mean)
            else:
                self._train_mean = load_pickle(self.save_dir + '/train_mean.plk').to(self.device)
            

            return self._train_mean
        else: 
            return self._train_mean