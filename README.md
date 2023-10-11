# Few-Shot DiffusionDet: Improving Few-Shot and Cross-Domain Object Detection on Aerial Images with a Diffusion-Based Detector

Few-Shot DiffusionDet is an adaptation of [DiffusionDet](https://arxiv.org/abs/2211.09788) in the few-shot regime. It provides sensible improvements on aerial few-shot object detection. The few-shot training strategy can be summarized as follows:

1. Train the DiffusionDet in a regular manner on the base classes.
2. Replace the last classification layer with a randomly initialized layer matching the number of novel classes.
3. Freeze the model (partly)
4. Reset the optimizer and learning rate scheduler.
5. Train the model on the novel classes with only the few available examples.

## Getting started 
As Few-Shot DiffusionDet is based on DiffusionDet, it requires Python ≥ 3.6, PyTorch ≥ 1.9.0 and corresponding Torchvision, and OpenCV. A few other libraries are required and listed in `requirements.txt`, which can be installed through: 
```
pip install -r requirements.txt
```

## Data preparation and loading 
Datasets must be registered inside the `diffusiondet/data/local_catalog.py` file. This file should contain a dictionnary for each registered dataset as follows:
```
dota_dict = {'root_path': '/path/to/datasets/DOTA/coco_format/',
                'name': 'DOTA',
                'splits': ['train', 'test', 'val'],
                'class_split_file': '/path/to/datasets/DOTA/coco_format/classes_split.txt'}
```

Splits should match the folder names containing the images for each data split. It should also match the name of the annotation files, as in the COCO format (`annotations_%SPLIT%.json`).  

`class_split_file` refers to simple text file containing two lines that splits the classes of the dataset in two groups: base classes (train_classes) and novel classes (test_classes). This file is mandatory in the few-shot regime.

```
train_classes:1,2,4,6,7,8,9,10,11,12,13,14,16
test_classes:3,5,15
```

## Configuration
Several configuration parameters have been added to DiffusionDet to deal with the Few-Shot Regime:

| Parameter name | Description |
| -------------  | ----------  |
|TRAIN_MODE |  `'regular'` for regular training, `'simplefs'` for few-shot training (2 stages).|
|FEWSHOT.K_SHOT| Number of shots used during fine-tuning on novel classes.| 
|FEWSHOT.N_CLASSES_TEST| Number of novel classes.|
|FEWSHOT.SPLIT_METHOD| Select how base and novel classes should be split before training. Can be set to `deterministic` (use class_split file), `rng` or `all_novel` (for cross-domain)| 
|FINETUNE.CROSS_DOMAIN| Number of novel classes.|
|FINETUNE.NOVEL_ONLY| Specify if fine-tuning should be done on the novel classes only or all classes.|
|FINETUNE.CROSS_DOMAIN| Specify a distinct behavior for Cross-Domain (no base-training).|
|FINETUNE.N_CLASSES_TEST| Number of novel classes.|
| FINETUNE.MODEL_FREEZING.BACKBONE_AT | Freeze backbone up to stage `i`|
| FINETUNE.MODEL_FREEZING.BACKBONE_MODE | Specify which weights should be frozen `'all'` for all parameters, `'norm'` and `'bias'` for all except norm and bias parameters.|
| FINETUNE.MODEL_FREEZING.MODULES | Specify which part of the model can be frozen, `'backbone'` or `'head'`.|
| FINETUNE.MODEL_FREEZING.HEAD_ALL | Freeze either all the detection head but the last layer or train all the head.|



## Training

#### Single experiment
Training FSDiffusionDet can be launched using `train_net.py`:
```
python train_net.py --num-gpus 8 \
    --config-file configs/finetuning/diffdet.dota.res50.yaml
```

#### Batched studies
Batched experiments can also be started with the script `launch_experiments.py`. A *study* gather several experiments and can be specified in study config file as in `configs/studies/study_example.json`. It can be launched similarly: 
```
python launch_experiment.py --num-gpus 8 \
    --config-file configs/studies/study_example.json
```

The study file contains a dictionnary with two entries: `names` and `studies`. Both should contain lists of the same size. `names` contains the list of the study's names, while `studies` contains the study parameters as a dictionnary, for instance: 

```
{
        "CFG_FILE": "configs/finetuning/diffdet.dota.res50.yaml",
        "MODEL.DiffusionDet.NUM_PROPOSALS" :[200, 250]
}
```
Each parameter should be either a list of values or a single values. If more than one parameter is specified as a list, the different list should have the same size and it will define the number of experiments that will be launched. Experiment `n` will use the parameter values `n` inside the respective list for the list-specified params and the single value for the other parameters. 

Distinct folders will be created for each study with several subfolders for each experiment within the study. 



## Application to Cross-Domain Few-Shot Object Detection
To apply FSDiffusionDet to Cross-Domain scenario, three parameters must be set correctly:
- `FEWSHOT.SPLIT_METHOD = 'all_novel'`
- `FINETUNE.CROSS_DOMAIN = True`
- ``SOLVER.MAX_ITER = 0`

The Cross-Domain training is done as a fine-tuning only, on all classes of the dataset. A base-trained model should be specified with `MODEL.WEIGHTS`, this model should have been trained in a regular fashion on a different beforehand. 







