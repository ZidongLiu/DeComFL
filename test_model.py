# -*- coding: utf-8 -*-
"""
Created on Sun Oct 12 19:31:14 2025

@author: Zidong
"""

from transformers import AutoModelForCausalLM
import torch


def get_trainable_model_parameters(
    model: torch.nn.Module,
):
    for param in model.parameters():
        if param.requires_grad:
            yield param


hf_model_name = "facebook/opt-125m"
model = AutoModelForCausalLM.from_pretrained(hf_model_name)

trainable_model_parameters = get_trainable_model_parameters(model)

optimizer = torch.optim.SGD(
    trainable_model_parameters,
    lr=1e-4,
    weight_decay=1e-5,
    momentum=0.1,
)
