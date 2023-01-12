import math
import copy
import torch
from torch.utils.data import Sampler

from detectron2.structures import Instances
from detectron2.data import build_detection_train_loader, build_detection_test_loader
from detectron2.data.samplers import InferenceSampler
import detectron2.data.transforms as T

from .dataset_mapper import DiffusionDetDatasetMapper
from .utils import filter_class_table


class ClassSampler(Sampler):
    """
    Wraps a RandomSampler, sample a specific number
    of image to build a query set, i.e. a set that 
    contains a fixed number of images of each selected
    class.
    """

    def __init__(self, cfg, dataset_metadata, selected_classes, n_query=None, shuffle=True, is_train=True, is_support=False, seed=3):
        self.dataset_metadata = dataset_metadata
        self.selected_classes = selected_classes
        self.n_query = n_query
        self.slice = slice(0, n_query)
        self.shuffle = shuffle
        self.is_train = is_train
        self.seed = seed

        self.class_table = copy.deepcopy(dataset_metadata.class_table)
        if is_train and not is_support:
            self.class_table = filter_class_table(self.class_table, cfg.FEWSHOT.K_SHOT, self.dataset_metadata.novel_classes)
        elif is_support:
            self.class_table = filter_class_table(self.class_table, cfg.FEWSHOT.K_SHOT, self.dataset_metadata.classes)

    def __iter__(self):
        table = self.class_table
        selected_indices = []
        for c in self.selected_classes:
            if isinstance(c, torch.Tensor):
                class_id = int(c.item())
            else:
                class_id = int(c)
            keep = torch.randperm(len(table[class_id]))[self.slice]
            selected_indices = selected_indices + [table[class_id][k] for k in keep]
        selected_indices = torch.Tensor(selected_indices)
        # Retrieve indices inside dataset from img ids
        if self.shuffle:
            shuffle = torch.randperm(selected_indices.shape[0])
            yield from selected_indices[shuffle].long().tolist()
        else:
            yield from selected_indices.long().tolist()

    def __len__(self):
        length = 0
        for c in self.selected_classes:
            if isinstance(c, torch.Tensor):
                cls = int(c.item())
            else:
                cls = int(c)
            table = self.class_table
            if self.n_query is not None:
                length += min(
                    self.n_query,
                    len(table[cls]))
            else:
                length += len(table[cls])

        return length
    


def filter_instances(instances, keep):
    fields = instances.get_fields()
    new_instances = Instances(instances.image_size,
                            **{k: v[keep] for k,v in fields.items()})
    # new_instances.old_instances = instances
    return new_instances
     

class FilteredDataLoader():
    """
    Wrapper around the dataloader class from pytorch created with detectron2's 
    building function. 

    Two methods are available to change the annotation class filter and the allowed dataset pool of images. 
     
    """
    def __init__(self, cfg, dataset, mapper, sampler, dataset_metadata, is_eval=False):
        self.mapper = mapper
        self.dataset = dataset
        self.sampler = sampler
        self.dataset_metadata = dataset_metadata
        self.is_eval = is_eval
        if not is_eval:
            self.dataloader = build_detection_train_loader(cfg,
                    mapper=mapper,
                    dataset=dataset,
                    sampler=sampler)
            self.keep_annotations_from_classes = mapper.selected_classes
            self.draw_images_from_classes = sampler.selected_classes
        else:
            self.sampler = InferenceSampler(len(dataset))
            self.dataloader = build_detection_test_loader(
                    mapper=mapper,
                    dataset=dataset,
                    sampler=sampler,
                    num_workers=cfg.DATALOADER.NUM_WORKERS,)
        
        
    
    def __iter__(self):
        yield next(iter(self.dataloader))
        
    def __len__(self):
        return len(self.dataloader)
    
    def change_mapper_classes(self, selected_classes):
        self.keep_annotations_from_classes = selected_classes
        self.mapper.selected_classes = torch.tensor(selected_classes)

    
    def change_sampler_classes(self, selected_classes):
        self.draw_images_from_classes = selected_classes
        self.sampler.selected_classes = torch.tensor(selected_classes)

    def change_sampler_mapper_classes(self, selected_classe):
        self.change_mapper_classes(selected_classes)
        self.change_sampler_classes(selected_classes)

class ClassMapper(DiffusionDetDatasetMapper):
    """
    Dataset Mapper extension to filter out annotations whose labels do not belong
    into selected_classes list. 
    """
    def __init__(self, selected_classes, contiguous_mapping, *args, remap_labels=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.selected_classes = torch.tensor(selected_classes)
        self.contiguous_mapping = contiguous_mapping
        self.remap_labels = remap_labels
        
    def __call__(self, dataset_dict):
        dataset_dict = copy.deepcopy(super().__call__(dataset_dict))
        instances = dataset_dict['instances']
        labels = instances.get_fields()['gt_classes']

        
        keep = torch.where(labels.unsqueeze(-1) == self.selected_classes.unsqueeze(0))[0]
        dataset_dict['old_instances'] = copy.deepcopy(instances)
        instances = filter_instances(instances, keep)
        if self.remap_labels:
            instances.get_fields()['gt_classes']
            labels = torch.where(self.selected_classes[None] == labels[:,None])[1]
            instances.set('gt_classes', labels)

        dataset_dict['instances'] = instances
        
        return dataset_dict