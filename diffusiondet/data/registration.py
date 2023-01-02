import os

from detectron2.data.datasets import register_coco_instances
from detectron2.data import DatasetCatalog, MetadataCatalog, get_detection_dataset_dicts


def create_class_table(dataset, classes):
    class_table = {c:[] for c in classes}
    for idx, img_meta in enumerate(dataset):
        for annot in img_meta['annotations']:
            class_table[annot['category_id']].append(idx)
    
    for c, indx_list in class_table.items():
        class_table[c] = list(set(indx_list))
    return class_table
                
    
def read_split_file(path):
    base_c, novel_c = [], []
    with open(path, 'r') as f:
        lines = f.readlines()
        assert len(lines)== 2, 'Wrong classes split file format, should be: \ntrain_classes:1,2,3 \n test_classes:4,5,6'
        base_c = list(map(lambda x: int(x), lines[0][:-1].split(':')[-1].split(',')))
        novel_c = list(map(lambda x: int(x), lines[1][:-1].split(':')[-1].split(',')))
        # assert len(base_c) >= self.cfg.FEWSHOT.N_WAYS_TRAIN, 'N_WAYS_TRAIN too large for number of training classes defined in file'
        # assert len(novel_c) >= self.cfg.FEWSHOT.N_WAYS_TEST, 'N_WAYS_TEST too large for number of training classes defined in file'

        base_c.sort()
        novel_c.sort()
    return base_c, novel_c

def register_dataset(dataset_dict):
    for split in dataset_dict['splits']:
        dataset_name =  dataset_dict['name'] + '_' + split
        
        base_c, novel_c = read_split_file(os.path.join(dataset_dict['root_path'], dataset_dict['class_split_file']))
        
        register_coco_instances(dataset_name, {'base_classes': base_c,
                                              'novel_classes': novel_c}, 
                                os.path.join(dataset_dict['root_path'], 'annotations', 'instances_{}.json'.format(split)), 
                                os.path.join(dataset_dict['root_path'], split))

def get_dataset_from_config(cfg):
    dataset = get_detection_dataset_dicts(
            cfg.DATASETS.TRAIN,
            filter_empty=cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS,
            min_keypoints=cfg.MODEL.ROI_KEYPOINT_HEAD.MIN_KEYPOINTS_PER_IMAGE
            if cfg.MODEL.KEYPOINT_ON else 0,
            proposal_files=cfg.DATASETS.PROPOSAL_FILES_TRAIN if cfg.MODEL.LOAD_PROPOSALS else None,
        )
    
    metadata = None
    # if datasets are concatenated metadata cannot be easily combined
    # TO DO: write function that merge metadata of two datasets
    if len(cfg.DATASETS.TRAIN) == 1:
        metadata = MetadataCatalog.get(cfg.DATASETS.TRAIN[0])
        metadata.class_table = create_class_table(dataset,
                                                list(metadata.thing_dataset_id_to_contiguous_id.values()))
    return dataset, metadata