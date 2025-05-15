import torch
from transformers import AutoModelForCausalLM

hf_model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
torch_dtype = torch.float32

model = AutoModelForCausalLM.from_pretrained(hf_model_name, torch_dtype=torch_dtype)
