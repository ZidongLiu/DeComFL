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
    qwen3_0p6b = "qwen3-0.6b"
    qwen3_1p7b = "qwen3-1.7b"
    qwen3_4b = "qwen3-4b"
    qwen3_8b = "qwen3-8b"
    llama3p2_1b = "llama3.2-1b"
    llama3p2_3b = "llama3.2-3b"
    gemma3_270m = "gemma-3-270m"
    gemma3_1b = "gemma-3-1b"
    gemma3_4b = "gemma-3-4b"
    smollm3_3b = "smollm3-3b"


class ModelDtype(Enum):
    float32 = "float32"
    float16 = "float16"
    bfloat16 = "bfloat16"
