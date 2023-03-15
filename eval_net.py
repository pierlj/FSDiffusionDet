# ==========================================
# Modified by Shoufa Chen
# ===========================================
# Modified by Peize Sun, Rufeng Zhang
# Contact: {sunpeize, cxrfzhang}@foxmail.com
#
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
DiffusionDet Training Script.

This script is a simplified version of the training script in detectron2/tools.
"""

import os
import logging

import torch

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.engine import default_argument_parser, default_setup, launch
from detectron2.evaluation import verify_results
from detectron2.modeling import build_model
from detectron2.utils.logger import setup_logger

from diffusiondet import DiffusionDetDatasetMapper, add_diffusiondet_config, \
    DiffusionDetWithTTA, add_additional_config, add_fs_config, create_unique_output_path
from diffusiondet.util.model_ema import add_model_ema_configs, may_get_ema_checkpointer, EMADetectionCheckpointer

from diffusiondet.train import DiffusionTrainer, FineTuningTrainer, TransductiveTrainer
from diffusiondet.data import register_dataset, LOCAL_CATALOG, get_datasets
from diffusiondet.eval.fs_evaluator import FSEvaluator

def setup(args, model_dir):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    add_diffusiondet_config(cfg)
    add_model_ema_configs(cfg)
    add_fs_config(cfg)
    add_additional_config(cfg)

    cfg.merge_from_file(os.path.join(model_dir, 'config.yaml'))
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def select_trainer(cfg):
    if cfg.TRAIN_MODE == 'regular':
        return DiffusionTrainer
    elif cfg.TRAIN_MODE in ['simplefs', 'support_attention'] :
        return FineTuningTrainer
    elif cfg.TRAIN_MODE == 'transductive':
        return TransductiveTrainer

def main(args):
    registered = False
    logger = setup_logger(name='diffusiondet', abbrev_name='diffdet')
    if args.model_path.endswith('.pth'):
        model_dir = '/'.join(args.model_path.split('/')[:-1])
        model_paths = [(args.base_eval, args.model_path)]
    else:
        model_dir = args.model_path
        model_paths = [
                       (False, os.path.join(model_dir, 'model_finetuned_final.pth')),
                       (True, os.path.join(model_dir, 'model_base_final.pth'))]

    os.rename(os.path.join(model_dir, 'last_checkpoint'), os.path.join(model_dir, '_last_checkpoint'))
    for base_eval, path in model_paths:

        with open(os.path.join(model_dir, 'last_checkpoint'), 'w') as f:
            f.write(path.split('/')[-1])

        cfg = setup(args, model_dir)
        if not registered:
            logger.info('Registering dataset from LOCAL CATALOG with key: {}'.format(cfg.DATASETS.TEST[0].split('_')[0]))
            register_dataset(LOCAL_CATALOG[cfg.DATASETS.TEST[0].split('_')[0]])
            registered = True

        Trainer = select_trainer(cfg)


        model = Trainer.build_model(cfg, is_finetuned=not base_eval)
        kwargs = may_get_ema_checkpointer(cfg, model)
        if cfg.MODEL_EMA.ENABLED:
            EMADetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR, **kwargs).resume_or_load(cfg.MODEL.WEIGHTS,
                                                                                                resume=True)
        else:
            DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR, **kwargs).resume_or_load(cfg.MODEL.WEIGHTS,
                                                                                            resume=True)
        
        dataset_name = cfg.DATASETS.TEST
        _, dataset_metadata = get_datasets(dataset_name, cfg)
        selected_classes = dataset_metadata.base_classes if base_eval else dataset_metadata.novel_classes
        model.selected_classes = None if base_eval else selected_classes
        output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        metric_save_path = os.path.join(cfg.OUTPUT_DIR, 'base_classes_metrics.json' if base_eval else \
                                                                'novel_classes_metrics.json')
        evaluator_name ='Base classes evaluation' if base_eval else 'Novel classes evaluation'
        logger.info('Test')
        evaluators = [FSEvaluator(selected_classes, dataset_name[0], cfg, True, output_folder, 
                                                            name=evaluator_name, 
                                                            metric_save_path=metric_save_path),]        

        res = Trainer.eval(cfg, model, evaluators, validation=False)
    os.rename(os.path.join(model_dir, '_last_checkpoint'), os.path.join(model_dir, 'last_checkpoint'))


if __name__ == "__main__":
    args_parser = default_argument_parser()
    args_parser.add_argument("--model-path", default=None, type=str, help='Path to model to evaluate.')
    args_parser.add_argument("--base-eval", default=False, type=bool, help='Wether to evaluate on base classes only or not.')
    args = args_parser.parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
