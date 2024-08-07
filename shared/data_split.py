import torch
import math
import contextlib
import numpy as np
from torch.utils.data.dataset import Subset


def get_dirichlet_split_indexes(labels: list[int], num_split, alpha):
    n_data = len(labels)
    if num_split == 1:
        return [list(range(n_data))]
        
    label_indices_dict = {}
    for i, label in enumerate(labels):
        if label in label_indices_dict:
            label_indices_dict[label].append(i)
        else:
            label_indices_dict[label] = [i]
    
    unique_labels = list(label_indices_dict.keys())
    n_labels = len(unique_labels)
    if not sorted(unique_labels) == list(range(n_labels)):
        raise ValueError(
            "Please re-map the labels of dataset into "
            "0 to num_classes-1 integers."
        )

    data_class_prob = np.array([len(label_indices_dict[i]) for i in range(n_labels)]) / n_data
    # This is a matrix with dimension "num_split * num_classes"
    full_class_prob = np.random.dirichlet(alpha * data_class_prob, num_split)

    split_label_idx = [[] for _ in range(num_split)]
    
    for label, label_indices in label_indices_dict.items():
        cum_prob = np.cumsum(full_class_prob[:, label])
        normalized_cum_prob = (cum_prob / cum_prob[-1]).tolist()
        num_label_indices = len(label_indices)
        for rank in range(num_split):
            start_prob = 0 if rank == 0 else normalized_cum_prob[rank - 1]
            end_prob = normalized_cum_prob[rank]
            my_index_start = math.floor(start_prob * num_label_indices)
            my_index_end = math.floor(end_prob * num_label_indices)
            split_label_idx[rank].extend(label_indices[my_index_start:my_index_end])

    return [sorted(v) for v in split_label_idx]


@contextlib.contextmanager
def temp_np_seed(seed):
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)


def dirichlet_split(dataset, labels, num_split, alpha, random_seed):
    if num_split == 1:
        return [dataset]

    with temp_np_seed(random_seed):
        split_indexes = get_dirichlet_split_indexes(labels, num_split, alpha)

    return [
        Subset(dataset, split_indexes[i])
        for i in range(num_split)
    ]
