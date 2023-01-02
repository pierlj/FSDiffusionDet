import math
import copy
import torch
from torch.utils.data import Sampler

from detectron2.structures import Instances
from detectron2.data import DatasetMapper, build_detection_train_loader
import detectron2.data.transforms as T


class ClassSampler(Sampler):
    """
    Wraps a RandomSampler, sample a specific number
    of image to build a query set, i.e. a set that 
    contains a fixed number of images of each selected
    class.
    """

    def __init__(self, dataset_metadata, selected_classes, n_query=None, shuffle=True, rng=None):
        self.dataset_metadata = dataset_metadata
        self.selected_classes = selected_classes
        self.n_query = n_query
        self.slice = slice(0, n_query)
        self.shuffle = shuffle
        self.rng = rng


    def __iter__(self):
        table = self.dataset_metadata.class_table
        selected_indices = []
        for c in self.selected_classes:
            if isinstance(c, torch.Tensor):
                class_id = int(c.item())
            else:
                class_id = int(c)
            keep = torch.randperm(len(table[class_id]), generator=self.rng)[self.slice]
            selected_indices = selected_indices + [table[class_id][k] for k in keep]
        selected_indices = torch.Tensor(selected_indices)

        # Retrieve indices inside dataset from img ids
        if self.shuffle:
            shuffle = torch.randperm(selected_indices.shape[0], generator=self.rng)
            yield from selected_indices[shuffle].long().tolist()
        else:
            yield from selected_indices.long().tolist()

    def __len__(self):
        length = 0
        for c in self.selected_classes:
            cls = int(c.item())
            table = self.dataset_metadata.class_table
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
    def __init__(self, cfg, dataset, mapper, sampler):
        self.mapper = mapper
        self.dataset = dataset
        self.sampler = sampler
        self.dataloader = build_detection_train_loader(cfg, 
                mapper=mapper,
                dataset=dataset,
                sampler=sampler)
    
    def __iter__(self):
        yield next(iter(self.dataloader))
        
    def __len__(self):
        return len(self.dataloader)
    
    def change_mapper_classes(self, selected_classes):
        # self.dataloader.dataset.dataset.dataset._map_func._obj.selected_classes = selected_classes
        self.mapper.selected_classes = torch.tensor(selected_classes)

    
    def change_sampler_classes(self, selected_classes):
        # self.dataloader.dataset.dataset.sampler.selected_classes = selected_classes
        self.sampler.selected_classes = torch.tensor(selected_classes)


class ClassMapper(DatasetMapper):
    def __init__(self, selected_classes, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.selected_classes = torch.tensor(selected_classes)
        
    def __call__(self, dataset_dict):
        dataset_dict = copy.deepcopy(super().__call__(dataset_dict))
        instances = dataset_dict['instances']
        labels = instances.get_fields()['gt_classes']
        
        keep = torch.where(labels.unsqueeze(-1) == self.selected_classes.unsqueeze(0))[0]
        dataset_dict['old_instances'] = copy.deepcopy(instances)
        instances = filter_instances(instances, keep)
        dataset_dict['instances'] = instances
        
        return dataset_dict