import os
import datetime
import logging
import time
import json
import itertools
import copy
from collections import OrderedDict, abc
from contextlib import ExitStack, contextmanager
from typing import List, Union
import torch
from torch import nn
import numpy as np
from tabulate import tabulate


from pycocotools.coco import COCO
from pycocosiou.cocoeval import COCOeval

import detectron2.utils.comm as comm
from detectron2.evaluation import inference_context
from detectron2.utils.comm import get_world_size, is_main_process
from detectron2.utils.logger import log_every_n_seconds, create_small_table
from detectron2.utils.file_io import PathManager



from ..eval.coco_evaluation import COCOEvaluator

try:
    from .fast_eval_api import COCOeval_opt
except ImportError:
    COCOeval_opt = COCOeval


class FSEvaluator(COCOEvaluator):
    def __init__(self, selected_classes, dataset_name, *args, name='Base classes eval', metric_save_path=None, **kwargs):
        super().__init__(dataset_name, *args, **kwargs)
        self.selected_classes = selected_classes
        self.dataset_name = dataset_name
        self.name = name
        self.metric_save_path = metric_save_path

    
    def inference_on_dataset(self, model, data_loader, validation=False):              
        """
            Run model on the data_loader and evaluate the metrics with evaluator.
            Also benchmark the inference speed of `model.__call__` accurately.
            The model will be used in eval mode.

            Args:
                model (callable): a callable which takes an object from
                    `data_loader` and returns some outputs.

                    If it's an nn.Module, it will be temporarily set to `eval` mode.
                    If you wish to evaluate a model in `training` mode instead, you can
                    wrap the given model and override its behavior of `.eval()` and `.train()`.
                data_loader: an iterable object with a length.
                    The elements it generates will be the inputs to the model.
                evaluator: the evaluator(s) to run. Use `None` if you only want to benchmark,
                    but don't want to do any evaluation.

            Returns:
                The return value of `evaluator.evaluate()`
            """
        num_devices = get_world_size()
        logger = logging.getLogger(__name__)
        logger.info('Starting {}: {}'.format(self.name, self.selected_classes))

        total = len(data_loader)  # inference data loader must have a fixed length


        self.reset()

        num_warmup = min(5, total - 1)
        start_time = time.perf_counter()
        total_data_time = 0
        total_compute_time = 0
        total_eval_time = 0
        with ExitStack() as stack:
            if isinstance(model, nn.Module):
                stack.enter_context(inference_context(model))
            stack.enter_context(torch.no_grad())

            start_data_time = time.perf_counter()
            for idx, inputs in enumerate(data_loader):
                total_data_time += time.perf_counter() - start_data_time
                if idx == num_warmup:
                    start_time = time.perf_counter()
                    total_data_time = 0
                    total_compute_time = 0
                    total_eval_time = 0

                start_compute_time = time.perf_counter()

                # model inference
                outputs = model(inputs)

                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                total_compute_time += time.perf_counter() - start_compute_time

                start_eval_time = time.perf_counter()

                # evaluator process
                self.process(inputs, outputs)

                total_eval_time += time.perf_counter() - start_eval_time

                iters_after_start = idx + 1 - num_warmup * int(idx >= num_warmup)
                data_seconds_per_iter = total_data_time / iters_after_start
                compute_seconds_per_iter = total_compute_time / iters_after_start
                eval_seconds_per_iter = total_eval_time / iters_after_start
                total_seconds_per_iter = (time.perf_counter() - start_time) / iters_after_start
                if idx >= num_warmup * 2 or compute_seconds_per_iter > 5:
                    eta = datetime.timedelta(seconds=int(total_seconds_per_iter * (total - idx - 1)))
                    log_every_n_seconds(
                        logging.INFO,
                        (
                            f"Inference done {idx + 1}/{total}. "
                            f"Dataloading: {data_seconds_per_iter:.4f} s/iter. "
                            f"Inference: {compute_seconds_per_iter:.4f} s/iter. "
                            f"Eval: {eval_seconds_per_iter:.4f} s/iter. "
                            f"Total: {total_seconds_per_iter:.4f} s/iter. "
                            f"ETA={eta}"
                        ),
                        n=5,
                    )
                start_data_time = time.perf_counter()

        # Measure the time only for this worker (before the synchronization barrier)
        total_time = time.perf_counter() - start_time
        total_time_str = str(datetime.timedelta(seconds=total_time))
        # NOTE this format is parsed by grep
        logger.info(
            "Total inference time: {} ({:.6f} s / iter per device, on {} devices)".format(
                total_time_str, total_time / (total - num_warmup), num_devices
            )
        )
        total_compute_time_str = str(datetime.timedelta(seconds=int(total_compute_time)))
        logger.info(
            "Total inference pure compute time: {} ({:.6f} s / iter per device, on {} devices)".format(
                total_compute_time_str, total_compute_time / (total - num_warmup), num_devices
            )
        )

        img_ids = data_loader.sampler.selected_indices.int().tolist()
        # results = self.evaluate(cat_ids=self.selected_classes, img_ids=img_ids)
        results = self.evaluate(cat_ids=self.selected_classes)

        # # Run separately evaluation on each class 
        # if not validation:
        #     results_per_class = {}
        #     for c in self.selected_classes:
        #         results_per_class[c] = self.evaluate(cat_ids=[c])
        # An evaluator may return None when not in main process.
        # Replace it by an empty dict instead to make it easier for downstream code to handle
        if results is None:
            results = {}
        return results
    
    def evaluate(self, img_ids=None, cat_ids=None):
        """
        Args:
            img_ids: a list of image IDs to evaluate on. Default to None for the whole dataset
        """
        if self._distributed:
            comm.synchronize()
            predictions = comm.gather(self._predictions, dst=0)
            predictions = list(itertools.chain(*predictions))

            if not comm.is_main_process():
                return {}
        else:
            predictions = self._predictions

        if len(predictions) == 0:
            self._logger.warning("[COCOEvaluator] Did not receive valid predictions.")
            return {}

        if self._output_dir:
            PathManager.mkdirs(self._output_dir)
            file_path = os.path.join(self._output_dir, "instances_predictions.pth")
            with PathManager.open(file_path, "wb") as f:
                torch.save(predictions, f)

        self._results = OrderedDict()
        if "proposals" in predictions[0]:
            self._eval_box_proposals(predictions)
        if "instances" in predictions[0]:
            self._eval_predictions(predictions, img_ids=img_ids, cat_ids=cat_ids)
        return copy.deepcopy(self._results)
    
    def _eval_predictions(self, predictions, img_ids=None, cat_ids=None):
        """
        Evaluate predictions. Fill self._results with the metrics of the tasks.
        """
        self._logger.info("Preparing results for COCO format ...")
        coco_results = list(itertools.chain(*[x["instances"] for x in predictions]))
        tasks = self._tasks or self._tasks_from_predictions(coco_results)

        # unmap the category ids for COCO
        if hasattr(self._metadata, "thing_dataset_id_to_contiguous_id"):
            dataset_id_to_contiguous_id = self._metadata.thing_dataset_id_to_contiguous_id
            all_contiguous_ids = list(dataset_id_to_contiguous_id.values())
            num_classes = len(all_contiguous_ids)
            assert min(all_contiguous_ids) == 0 and max(all_contiguous_ids) == num_classes - 1

            reverse_id_mapping = {v: k for k, v in dataset_id_to_contiguous_id.items()}
            if cat_ids is not None:
                cat_ids = [reverse_id_mapping[c] for c in cat_ids]

            for result in coco_results:
                category_id = result["category_id"]
                assert category_id < num_classes, (
                    f"A prediction has class={category_id}, "
                    f"but the dataset only has {num_classes} classes and "
                    f"predicted class id should be in [0, {num_classes - 1}]."
                )
                result["category_id"] = reverse_id_mapping[category_id]

        if self._output_dir:
            file_path = os.path.join(self._output_dir, "coco_instances_results.json")
            self._logger.info("Saving results to {}".format(file_path))
            with PathManager.open(file_path, "w") as f:
                f.write(json.dumps(coco_results))
                f.flush()

        if not self._do_evaluation:
            self._logger.info("Annotations are not available for evaluation.")
            return

        self._logger.info(
            "Evaluating predictions with {} COCO API...".format(
                "unofficial" if self._use_fast_impl else "official"
            )
        )
        for task in sorted(tasks):
            assert task in {"bbox", "segm", "keypoints"}, f"Got unknown task: {task}!"
            coco_eval = (
                self._evaluate_predictions_on_coco(
                    self._coco_api,
                    coco_results,
                    task,
                    kpt_oks_sigmas=self._kpt_oks_sigmas,
                    use_fast_impl=self._use_fast_impl,
                    img_ids=img_ids,
                    # img_ids=[1537,1030,521,522,1553,1027,1543,21,1048,546,1025,389,1657,24,152],
                    cat_ids=cat_ids,
                    max_dets_per_image=self._max_dets_per_image,
                )
                if len(coco_results) > 0
                else None  # cocoapi does not handle empty results very well
            )
            class_names = [self._metadata.get("thing_classes")[idx] for idx in cat_ids]
            res = self._derive_coco_results(
                coco_eval, task, class_names=class_names
            )
            self._results[task] = res

    def _evaluate_predictions_on_coco(
        self,
        coco_gt,
        coco_results,
        iou_type,
        kpt_oks_sigmas=None,
        use_fast_impl=True,
        img_ids=None,
        cat_ids=None,
        max_dets_per_image=None,
    ):
        """
        Evaluate the coco results using COCOEval API.
        """
        assert len(coco_results) > 0

        if iou_type == "segm":
            coco_results = copy.deepcopy(coco_results)
            # When evaluating mask AP, if the results contain bbox, cocoapi will
            # use the box area as the area of the instance, instead of the mask area.
            # This leads to a different definition of small/medium/large.
            # We remove the bbox field to let mask AP use mask area.
            for c in coco_results:
                c.pop("bbox", None)

        coco_dt = coco_gt.loadRes(coco_results)
        coco_eval = (COCOeval_opt if use_fast_impl else COCOeval)(coco_gt, coco_dt, iou_type)
        # For COCO, the default max_dets_per_image is [1, 10, 100].
        if max_dets_per_image is None:
            max_dets_per_image = [1, 10, 100]  # Default from COCOEval
        else:
            assert (
                len(max_dets_per_image) >= 3
            ), "COCOeval requires maxDets (and max_dets_per_image) to have length at least 3"
            # In the case that user supplies a custom input for max_dets_per_image,
            # apply COCOevalMaxDets to evaluate AP with the custom input.
            if max_dets_per_image[2] != 100:
                coco_eval = COCOevalMaxDets(coco_gt, coco_dt, iou_type)
        if iou_type != "keypoints":
            coco_eval.params.maxDets = max_dets_per_image

        if img_ids is not None:
            coco_eval.params.imgIds = img_ids
        
        if cat_ids is not None:
            coco_eval.params.catIds = cat_ids

        if iou_type == "keypoints":
            # Use the COCO default keypoint OKS sigmas unless overrides are specified
            if kpt_oks_sigmas:
                assert hasattr(coco_eval.params, "kpt_oks_sigmas"), "pycocotools is too old!"
                coco_eval.params.kpt_oks_sigmas = np.array(kpt_oks_sigmas)
            # COCOAPI requires every detection and every gt to have keypoints, so
            # we just take the first entry from both
            num_keypoints_dt = len(coco_results[0]["keypoints"]) // 3
            num_keypoints_gt = len(next(iter(coco_gt.anns.values()))["keypoints"]) // 3
            num_keypoints_oks = len(coco_eval.params.kpt_oks_sigmas)
            assert num_keypoints_oks == num_keypoints_dt == num_keypoints_gt, (
                f"[COCOEvaluator] Prediction contain {num_keypoints_dt} keypoints. "
                f"Ground truth contains {num_keypoints_gt} keypoints. "
                f"The length of cfg.TEST.KEYPOINT_OKS_SIGMAS is {num_keypoints_oks}. "
                "They have to agree with each other. For meaning of OKS, please refer to "
                "http://cocodataset.org/#keypoints-eval."
            )

        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        return coco_eval
    
    def _derive_coco_results(self, coco_eval, iou_type, class_names=None):
        """
        Derive the desired score numbers from summarized COCOeval.

        Args:
            coco_eval (None or COCOEval): None represents no predictions from model.
            iou_type (str):
            class_names (None or list[str]): if provided, will use it to predict
                per-category AP.

        Returns:
            a dict of {metric name: score}
        """

        metrics = {
            "bbox": ["AP", "AP50", "AP75", "APs", "APm", "APl"],
            "segm": ["AP", "AP50", "AP75", "APs", "APm", "APl"],
            "keypoints": ["AP", "AP50", "AP75", "APm", "APl"],
        }[iou_type]

        if coco_eval is None:
            self._logger.warn("No predictions from the model!")
            return {metric: float("nan") for metric in metrics}

        # the standard metrics
        results = {
            metric: float(coco_eval.stats[idx] * 100 if coco_eval.stats[idx] >= 0 else "nan")
            for idx, metric in enumerate(metrics)
        }
        self._logger.info(
            "Evaluation results for {}: \n".format(iou_type) + create_small_table(results)
        )
        if not np.isfinite(sum(results.values())):
            self._logger.info("Some metrics cannot be computed and is shown as NaN.")

        if class_names is None or len(class_names) <= 1:
            return results
        # Compute per-category AP
        # from https://github.com/facebookresearch/Detectron/blob/a6a835f5b8208c45d0dce217ce9bbda915f44df7/detectron/datasets/json_dataset_evaluator.py#L222-L252 # noqa
        precisions = coco_eval.eval["precision"]
        self.write_results(coco_eval.eval)
        
        # precision has dims (iou, recall, cls, area range, max dets)
        assert len(class_names) == precisions.shape[2]

        results_per_category = []
        for idx, name in enumerate(class_names):
            # area range index 0: all area ranges
            # max dets index -1: typically 100 per image
            precision = precisions[:, :, idx, 0, -1]
            precision = precision[precision > -1]
            ap = np.mean(precision) if precision.size else float("nan")
            results_per_category.append(("{}".format(name), float(ap * 100)))

        # tabulate it
        N_COLS = min(6, len(results_per_category) * 2)
        results_flatten = list(itertools.chain(*results_per_category))
        results_2d = itertools.zip_longest(*[results_flatten[i::N_COLS] for i in range(N_COLS)])
        table = tabulate(
            results_2d,
            tablefmt="pipe",
            floatfmt=".3f",
            headers=["category", "AP"] * (N_COLS // 2),
            numalign="left",
        )
        self._logger.info("Per-category {} AP: \n".format(iou_type) + table)

        results.update({"AP-" + name: ap for name, ap in results_per_category})
        return results

    def write_results(self, coco_res):
        if self.metric_save_path is not None:
            serialized_res = {}
            for k, v in coco_res.items():
                if k != 'params':
                    if isinstance(v, np.ndarray):
                        serialized_res[k] = v.tolist()
                    else:
                        serialized_res[k] = v
            with open(self.metric_save_path, 'w') as f:
                json.dump(serialized_res, f)







    
class EpisodicEvaluator(FSEvaluator):
    def __init__(self,):
        super().__init__()

    def inference_on_dataset(self):
        raise NotImplementedError()