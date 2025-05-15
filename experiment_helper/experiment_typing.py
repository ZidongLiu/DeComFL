from enum import Enum


class LargeModel(Enum):
    opt_125m = "opt-125m"
    opt_350m = "opt-350m"
    opt_1p3b = "opt-1.3b"
    opt_2p7b = "opt-2.7b"
    opt_6p7b = "opt-6.7b"
    opt_13b = "opt-13b"
    opt_30b = "opt-30b"
    deepseek_qwen_1p5b = "deepseek-qwen-1.5b"


class ModelDtype(Enum):
    float32 = "float32"
    float16 = "float16"
    bfloat16 = "bfloat16"
