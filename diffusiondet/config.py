import os
from datetime import datetime

from detectron2.config import CfgNode as CN


def add_diffusiondet_config(cfg):
    """
    Add config for DiffusionDet
    """
    cfg.MODEL.DiffusionDet = CN()
    cfg.MODEL.DiffusionDet.NUM_CLASSES = 80
    cfg.MODEL.DiffusionDet.NUM_PROPOSALS = 300

    # RCNN Head.
    cfg.MODEL.DiffusionDet.NHEADS = 8
    cfg.MODEL.DiffusionDet.DROPOUT = 0.0
    cfg.MODEL.DiffusionDet.DIM_FEEDFORWARD = 2048
    cfg.MODEL.DiffusionDet.ACTIVATION = 'relu'
    cfg.MODEL.DiffusionDet.HIDDEN_DIM = 256
    cfg.MODEL.DiffusionDet.NUM_CLS = 1
    cfg.MODEL.DiffusionDet.NUM_REG = 3
    cfg.MODEL.DiffusionDet.NUM_HEADS = 6

    # Dynamic Conv.
    cfg.MODEL.DiffusionDet.NUM_DYNAMIC = 2
    cfg.MODEL.DiffusionDet.DIM_DYNAMIC = 64

    # Loss.
    cfg.MODEL.DiffusionDet.CLASS_WEIGHT = 2.0
    cfg.MODEL.DiffusionDet.GIOU_WEIGHT = 2.0
    cfg.MODEL.DiffusionDet.L1_WEIGHT = 5.0
    cfg.MODEL.DiffusionDet.DEEP_SUPERVISION = True
    cfg.MODEL.DiffusionDet.NO_OBJECT_WEIGHT = 0.1

    # Focal Loss.
    cfg.MODEL.DiffusionDet.USE_FOCAL = True
    cfg.MODEL.DiffusionDet.USE_FED_LOSS = False
    cfg.MODEL.DiffusionDet.ALPHA = 0.25
    cfg.MODEL.DiffusionDet.GAMMA = 2.0
    cfg.MODEL.DiffusionDet.PRIOR_PROB = 0.01

    # Dynamic K
    cfg.MODEL.DiffusionDet.OTA_K = 5

    # Diffusion
    cfg.MODEL.DiffusionDet.SNR_SCALE = 2.0
    cfg.MODEL.DiffusionDet.SAMPLE_STEP = 1

    # Inference
    cfg.MODEL.DiffusionDet.USE_NMS = True

    # Swin Backbones
    cfg.MODEL.SWIN = CN()
    cfg.MODEL.SWIN.SIZE = 'B'  # 'T', 'S', 'B'
    cfg.MODEL.SWIN.USE_CHECKPOINT = False
    cfg.MODEL.SWIN.OUT_FEATURES = (0, 1, 2, 3)  # modify

    # Optimizer.
    cfg.SOLVER.OPTIMIZER = "ADAMW"
    cfg.SOLVER.BACKBONE_MULTIPLIER = 1.0

    # TTA.
    cfg.TEST.AUG.MIN_SIZES = (400, 500, 600, 640, 700, 900, 1000, 1100, 1200, 1300, 1400, 1800, 800)
    cfg.TEST.AUG.CVPODS_TTA = True
    cfg.TEST.AUG.SCALE_FILTER = True
    cfg.TEST.AUG.SCALE_RANGES = ([96, 10000], [96, 10000], 
                                 [64, 10000], [64, 10000],
                                 [64, 10000], [0, 10000],
                                 [0, 10000], [0, 256],
                                 [0, 256], [0, 192],
                                 [0, 192], [0, 96],
                                 [0, 10000])

def add_fs_config(cfg):
    cfg.FEWSHOT = CN()
    cfg.FEWSHOT.SPLIT_METHOD = 'deterministic'
    cfg.FEWSHOT.N_CLASSES_TEST = 3
    cfg.FEWSHOT.N_WAYS_TEST = 3
    cfg.FEWSHOT.N_WAYS_TRAIN = 3
    cfg.FEWSHOT.K_SHOT = 10
    cfg.FEWSHOT.BASE_SUPPORT = 'rng' # 'rng' for random base example at each computation, 'same' for fixed support

    cfg.FINETUNE = CN()
    cfg.FINETUNE.MAX_ITER = 100
    cfg.FINETUNE.NOVEL_ONLY = True
    cfg.FINETUNE.CROSS_DOMAIN = False

    cfg.FEWSHOT.ATTENTION = CN()
    cfg.FEWSHOT.ATTENTION.EXTRACT_EVERY = 1
    cfg.FEWSHOT.ATTENTION.MAX_NUM_CLASSES = 5

    cfg.FEWSHOT.SUPPORT_EXTRACTOR = CN()
    cfg.FEWSHOT.SUPPORT_EXTRACTOR.WEIGHT = "detectron2://ImageNetPretrained/torchvision/R-50.pkl"


    cfg.FINETUNE.MODEL_FREEZING = CN()
    cfg.FINETUNE.MODEL_FREEZING.BACKBONE_AT = 0
    cfg.FINETUNE.MODEL_FREEZING.BACKBONE_MODE = 'all'
    cfg.FINETUNE.MODEL_FREEZING.MODULES = ['backbone']
    cfg.FINETUNE.MODEL_FREEZING.HEAD_ALL = False

    


def add_additional_config(cfg):
    cfg.DATASETS.VAL = ()
    cfg.TRAIN_MODE = 'regular'
    cfg.PREVENT_WEIGHTS_LOADING = False

    cfg.MODEL.DiffusionDet.REG_LOSS_TYPE = 'giou' # 'siou' or 'giou'
    cfg.MODEL.DiffusionDet.REG_COST_TYPE = 'giou' # 'siou' or 'giou'

def create_unique_output_path(ouput_dir, study_folder=None):
    dt_string = datetime.now().strftime("%d-%m-%Y_%H:%M:%S")
    if study_folder is None:
        output_dir = os.path.join(ouput_dir, dt_string)
    else:
        output_dir = os.path.join(ouput_dir, study_folder, dt_string)
    return output_dir