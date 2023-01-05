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

import torch

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.engine import default_argument_parser, default_setup, launch
from detectron2.evaluation import verify_results
from detectron2.modeling import build_model

from diffusiondet import DiffusionDetDatasetMapper, add_diffusiondet_config, \
    DiffusionDetWithTTA, add_additional_config, add_fs_config, create_unique_output_path
from diffusiondet.util.model_ema import add_model_ema_configs, may_get_ema_checkpointer, EMADetectionCheckpointer

from diffusiondet.train import DiffusionTrainer, FineTuningTrainer
from diffusiondet.data import register_dataset, LOCAL_CATALOG


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    add_diffusiondet_config(cfg)
    add_model_ema_configs(cfg)
    add_fs_config(cfg)
    add_additional_config(cfg)

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.OUTPUT_DIR = create_unique_output_path(cfg.OUTPUT_DIR) if not args.resume else cfg.OUTPUT_DIR
    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def select_trainer(cfg):
    if cfg.TRAIN_MODE == 'regular':
        return DiffusionTrainer
    elif cfg.TRAIN_MODE == 'simplefs':
        return FineTuningTrainer

def main(args):
    cfg = setup(args)
    print('Registering dataset from LOCAL CATALOG with key: {}'.format(cfg.DATASETS.TRAIN[0].split('_')[0]))
    register_dataset(LOCAL_CATALOG[cfg.DATASETS.TRAIN[0].split('_')[0]])

    Trainer = select_trainer(cfg)

    if args.eval_only:
        model = Trainer.build_model(cfg)
        kwargs = may_get_ema_checkpointer(cfg, model)
        if cfg.MODEL_EMA.ENABLED:
            EMADetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR, **kwargs).resume_or_load(cfg.MODEL.WEIGHTS,
                                                                                              resume=args.resume)
        else:
            DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR, **kwargs).resume_or_load(cfg.MODEL.WEIGHTS,
                                                                                           resume=args.resume)
        res = Trainer.ema_test(cfg, model)
        if cfg.TEST.AUG.ENABLED:
            res.update(Trainer.test_with_TTA(cfg, model))
        if comm.is_main_process():
            verify_results(cfg, res)
        return res

    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    return trainer.launch_training()


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
