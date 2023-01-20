import torch

def filter_class_table(class_table, k_shots, classes, rng=None, seed=1234):
    for c in classes:
        if rng is None:
            rng = torch.Generator()
        rng.manual_seed(seed)
        perm = torch.randperm(len(class_table[c]), generator=rng)
        class_table[c] = torch.tensor(class_table[c])[perm][:k_shots].tolist()
    return class_table