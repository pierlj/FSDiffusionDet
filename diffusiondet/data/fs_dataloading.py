import math
import random
import copy
import torch
from torch.utils.data import Sampler
import torch.utils.data as data
import logging


from detectron2.structures import Instances
from detectron2.data import build_detection_train_loader, build_detection_test_loader
from detectron2.data.samplers import InferenceSampler
from detectron2.data.build import trivial_batch_collator
from detectron2.utils.serialize import PicklableWrapper

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

    def __init__(self, cfg, dataset_metadata, selected_classes, n_query=None, shuffle=True, is_train=True, seed=3):
        self.cfg = cfg
        self.dataset_metadata = dataset_metadata
        self.selected_classes = selected_classes
        self.n_query = n_query
        self.slice = slice(0, n_query)
        self.shuffle = shuffle
        self.is_train = is_train
        self.seed = seed

        self.class_table = copy.deepcopy(dataset_metadata.class_table)
        self.filter_table()
    
    def filter_table(self):
        if self.is_train:
            self.class_table = filter_class_table(self.class_table, self.cfg.FEWSHOT.K_SHOT, self.dataset_metadata.novel_classes)

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
        if not self.is_train:
            selected_indices = torch.unique(selected_indices) 
            # important to prevent box redundancy and low performance 
        # Retrieve indices inside dataset from img ids
        self.selected_indices = selected_indices
        if self.shuffle:
            shuffle = torch.randperm(selected_indices.shape[0])
            yield from selected_indices[shuffle].long().tolist()
        else:
            yield from selected_indices.long().tolist()

    def __len__(self):
        length = 0
        all_indices = []
        for c in self.selected_classes:
            if isinstance(c, torch.Tensor):
                c = int(c.item())
            else:
                c = int(c)
            table = self.class_table
            all_indices = all_indices + table[c]
            if self.n_query is not None and self.n_query > 0:
                length += min(
                    self.n_query,
                    len(table[c]))
            else:
                length += len(table[c])
        if not self.is_train:
            length = torch.tensor(all_indices).unique().shape[0]
        return length


class SupportClassSampler(ClassSampler):
    """
    Wraps a RandomSampler, sample a specific number
    of image to build a query set, i.e. a set that 
    contains a fixed number of images of each selected
    class.
    """

    def __init__(self, *args, base_support=None, **kwargs):
        self.base_support = base_support
        super().__init__(*args, **kwargs)
        

    def filter_table(self):
        if self.base_support == 'same':
            self.class_table = filter_class_table(self.class_table, self.cfg.FEWSHOT.K_SHOT, self.selected_classes)
        else:
            self.class_table = filter_class_table(self.class_table, self.cfg.FEWSHOT.K_SHOT, self.dataset_metadata.novel_classes)

    def __iter__(self):
        table = self.class_table
        selected_indices = []
        for c in self.selected_classes:
            if isinstance(c, torch.Tensor):
                class_id = int(c.item())
            else:
                class_id = int(c)
            keep = torch.randperm(len(table[class_id]))[self.slice]
            selected_indices = selected_indices + [(table[class_id][k], class_id) for k in keep]
        selected_indices = torch.Tensor(selected_indices)
        # Retrieve indices inside dataset from img ids
        if self.shuffle:
            shuffle = torch.randperm(selected_indices.shape[0])
            yield from selected_indices[shuffle].long().tolist()
        else:
            yield from selected_indices.long().tolist()



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
    def __init__(self, cfg, dataset, mapper, sampler, dataset_metadata, is_eval=False, is_support=False, forced_bs=None):
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

            self.dataloader.batch_size = min(len(self.dataloader.dataset),
                                        self.dataloader.batch_size)

        else:
            self.sampler = InferenceSampler(len(dataset))
            batch_size = 1
            if forced_bs is not None:
                batch_size = forced_bs
            if not is_support:
                self.dataloader = build_detection_test_loader(
                        mapper=mapper,
                        dataset=dataset,
                        sampler=sampler,
                        batch_size=batch_size,
                        num_workers=cfg.DATALOADER.NUM_WORKERS,)
            else:
                self.dataset = SupportMapDataset(dataset, mapper)
                self.dataloader = data.DataLoader(
                                self.dataset,
                                batch_size=batch_size,
                                sampler=sampler,
                                drop_last=False,
                                num_workers=cfg.DATALOADER.NUM_WORKERS,
                                collate_fn=trivial_batch_collator,
                            )
            
        
        
    
    def __iter__(self):
        return iter(self.dataloader)
        
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

class SupportClassMapper(DiffusionDetDatasetMapper):
    """
    Dataset Mapper extension to filter out annotations whose labels do not belong
    into selected_classes list. 
    """
    def __init__(self, selected_classes, contiguous_mapping, class_repartition, base_support, *args, remap_labels=False, seed=1234, **kwargs):
        super().__init__(*args, **kwargs)
        self.selected_classes = torch.tensor(selected_classes)
        self.contiguous_mapping = contiguous_mapping
        self.class_repartition = class_repartition
        self.remap_labels = remap_labels
        self.seed = seed
        self.base_support = base_support
        self.rng = torch.Generator()
        
    def __call__(self, dataset_dict):
        dataset_dict = copy.deepcopy(super().__call__(dataset_dict))
        instances = dataset_dict['instances']
        labels = instances.gt_classes
        labels_allowed = torch.tensor(dataset_dict['class_sampled'])
        
        if dataset_dict['class_sampled'] in self.class_repartition['novel'] or self.base_support == 'same':
            self.rng.manual_seed(self.seed)
        keep = torch.where(labels.unsqueeze(-1) == labels_allowed.unsqueeze(0))[0]
        keep = keep[torch.randint(keep.shape[0], (1,), generator=self.rng)]
        dataset_dict['old_instances'] = copy.deepcopy(instances)
        instances = filter_instances(instances, keep)
        if self.remap_labels:
            instances.get_fields()['gt_classes']
            labels = torch.where(self.selected_classes[None] == labels[:,None])[1]
            instances.set('gt_classes', labels)

        dataset_dict['instances'] = instances
        
        return dataset_dict


class SupportDataset(list):
    def __getitem__(self, idx):
        item = super().__getitem__(idx[0])
        item['class_sampled'] = idx[1]
        return item

class SupportMapDataset(data.Dataset):
    """
    Map a function over the elements in a dataset.
    """

    def __init__(self, dataset, map_func):
        """
        Args:
            dataset: a dataset where map function is applied. Can be either
                map-style or iterable dataset. When given an iterable dataset,
                the returned object will also be an iterable dataset.
            map_func: a callable which maps the element in dataset. map_func can
                return None to skip the data (e.g. in case of errors).
                How None is handled depends on the style of `dataset`.
                If `dataset` is map-style, it randomly tries other elements.
                If `dataset` is iterable, it skips the data and tries the next.
        """
        self._dataset = dataset
        self._map_func = PicklableWrapper(map_func)  # wrap so that a lambda will work

        self._rng = random.Random(42)
        self._fallback_candidates = set(range(len(dataset)))

    def __new__(cls, dataset, map_func):
        return super().__new__(cls)

    def __getnewargs__(self):
        return self._dataset, self._map_func

    def __len__(self):
        return len(self._dataset)

    def __getitem__(self, idx):
        cur_idx = idx
        data = self._map_func(self._dataset[cur_idx])
        return data