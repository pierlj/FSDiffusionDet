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
import jstyleson
from datetime import datetime

import torch
import logging

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.engine import default_argument_parser, default_setup, launch
from detectron2.evaluation import verify_results
from detectron2.modeling import build_model

from diffusiondet import DiffusionDetDatasetMapper, add_diffusiondet_config, \
    DiffusionDetWithTTA, add_additional_config, add_fs_config, create_unique_output_path
from diffusiondet.util.model_ema import add_model_ema_configs, may_get_ema_checkpointer, EMADetectionCheckpointer

from diffusiondet.train import DiffusionTrainer, FineTuningTrainer, TransductiveTrainer
from diffusiondet.data import register_dataset, LOCAL_CATALOG

def setup(args, cfg_file, study_name):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    add_diffusiondet_config(cfg)
    add_model_ema_configs(cfg)
    add_fs_config(cfg)
    add_additional_config(cfg)

    cfg.merge_from_file(cfg_file)
    cfg.merge_from_list(args.opts)
    cfg.OUTPUT_DIR = create_unique_output_path(cfg.OUTPUT_DIR, study_folder=study_name)
    
    return cfg


def select_trainer(cfg):
    if cfg.TRAIN_MODE == 'regular':
        return DiffusionTrainer
    elif cfg.TRAIN_MODE in ['simplefs', 'support_attention'] :
        return FineTuningTrainer
    elif cfg.TRAIN_MODE == 'transductive':
        return TransductiveTrainer

def main(args):
    
    study_dict = build_cfg_list_from_exp_file(args.config_file)

    for study_name, cfg_list in study_dict.items():
        study_name = study_name + '_' + datetime.now().strftime("%d-%m-%Y_%H:%M:%S")
        for study_cfg in cfg_list:
            cfg_file = study_cfg[1]
            study_cfg = study_cfg[2:]
            cfg = setup(args, cfg_file, study_name)
            cfg.merge_from_list(study_cfg)
            cfg.freeze()

            logging.getLogger('detectron2').handlers.clear()
            logging.getLogger('fvcore').handlers.clear()
            default_setup(cfg, args)

            register_dataset(LOCAL_CATALOG[cfg.DATASETS.TRAIN[0].split('_')[0]])

            Trainer = select_trainer(cfg)

            trainer = Trainer(cfg)
            trainer.resume_or_load(resume=False)
            trainer.launch_training()


def build_cfg_list_from_exp_file(study_file):
    with open(study_file, 'r') as f:
        study_json = jstyleson.load(f)
    
    study_names = study_json['names']
    study_dict = {}
    if len(study_names) == 1:
        study_names = study_names * len(study_json['studies'])
    
    for study_name, study in zip(study_names, study_json['studies']):
        seed_list = study_json['seed'] if 'seed' in study_json else [None]
        for seed in seed_list:
            n_values = [len(values) if isinstance(values, list) else 1 for param, values in study.items()]
            n_exp = max(n_values)
            assert all([v == n_exp or v == 1 for v in n_values]), 'Inside one study, the number of different value for a parameter must be either 1 or n_exp'
            study_dict[study_name] = []
            for i in range(n_exp):
                exp_list = []
                for param, values in study.items():
                    exp_list.append(param)
                    if isinstance(values, list):
                        exp_list.append(values[i])
                    else:
                        exp_list.append(values)
                    if seed is not None:
                        exp_list.append("SEED")
                        exp_list.append(seed)
                study_dict[study_name].append(exp_list)
    return study_dict



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
