import torch

from detectron2.structures import Instances


def filter_class_table(class_table, k_shots, classes, rng=None, seed=1234):
    for c in classes:
        if rng is None:
            rng = torch.Generator()
        rng.manual_seed(seed)
        perm = torch.randperm(len(class_table[c]), generator=rng)
        class_table[c] = torch.tensor(class_table[c])[perm][:k_shots].tolist()
    return class_table

def filter_instances(instances, keep):
    fields = instances.get_fields()
    new_instances = Instances(instances.image_size,
                            **{k: v[keep] for k,v in fields.items()})
    # new_instances.old_instances = instances
    return new_instances