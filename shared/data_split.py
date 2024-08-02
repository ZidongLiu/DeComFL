import torch
import contextlib
import numpy as np
from torch.utils.data.dataset import Subset


def get_noniid_split_indexes(labels: list[int], num_split):
    if num_split == 1:
        return [list(range(len(labels)))]
        
    label_idx_dict = {}
    for i, label in enumerate(labels):
        if label in label_idx_dict:
            label_idx_dict[label].append(i)
        else:
            label_idx_dict[label] = [i]
    
    unique_labels = list(label_idx_dict.keys())
    
    label_split_indexes = {}
    split_label_idx = [[] for _ in range(num_split)]
    
    for label, label_idxes in label_idx_dict.items():
        num_label = len(label_idxes)
        if num_label < num_split:
            label_split = list(range(num_label)) + [num_label for _ in range(num_split + 1 - num_label)]
        else:
            label_split = [0] + list(np.sort(np.random.choice(range(1, num_label), num_split - 1, replace=False))) + [num_label]
        for i in range(num_split):
            split_label_idx[i].extend(label_idxes[label_split[i]:label_split[i+1]])

    return [sorted(v) for v in split_label_idx]


@contextlib.contextmanager
def temp_np_seed(seed):
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)


def non_iid_split(dataset, labels, num_split, random_seed):
    if num_split == 1:
        return [dataset]
    with temp_np_seed(random_seed):
        split_indexes = get_noniid_split_indexes(labels, num_split)

    return [
        Subset(dataset, split_indexes[i])
        for i in range(num_split)
    ]
