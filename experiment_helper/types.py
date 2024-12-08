from enum import Enum


class LargeModel(Enum):
    opt_125m = "opt-125m"
    opt_350m = "opt-350m"
    opt_1p3b = "opt-1.3b"
    opt_2p7b = "opt-2.7b"
    opt_6p7b = "opt-6.7b"
    opt_13b = "opt-13b"
    opt_30b = "opt-30b"


class ModelDtype(Enum):
    float32 = "float32"
    float16 = "float16"
    bfloat16 = "bfloat16"


class Dataset(Enum):
    # TODO: split 2 different type of dataset
    mnist = "mnist"
    cifar10 = "cifar10"
    fashion = "fashion"
    shakespeare = "shakespeare"

    # language classification
    sst2 = "sst2"
    rte = "rte"
    multirc = "multirc"
    cb = "cb"
    wic = "wic"
    wsc = "wsc"
    boolq = "boolq"

    # language generation
    squad = "squad"
    drop = "drop"
    xsum = "xsum"


class RandomGradEstimateMethod(Enum):
    rge_central = "rge-central"
    rge_forward = "rge-forward"
