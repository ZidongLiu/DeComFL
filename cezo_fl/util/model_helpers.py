# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 20:58:19 2024

@author: Zidong
"""

from os import path
from typing import Iterator

import torch
import torch.optim as optim
from peft import PeftModel
from transformers.models.opt.modeling_opt import OPTForCausalLM

from cezo_fl.util.language_utils import LLMBatchInput


def get_current_datetime_str():
    from datetime import datetime

    now = datetime.now()
    year, month, day, hour, minute, second = (
        now.year,
        now.month,
        now.day,
        now.hour,
        now.minute,
        now.second,
    )
    return f"{year}-{month}-{day}-{hour}-{minute}-{second}"


@torch.no_grad()
def eval_network_and_get_loss(params_dict, network, x, y, loss_func):
    state_dict_backup = network.state_dict()
    network.load_state_dict(params_dict, strict=False)
    loss = loss_func(network(x), y).detach().item()
    network.load_state_dict(state_dict_backup)
    return loss


def save_model_and_optimizer(optimizer, model, model_path, model_prefix):
    save_path = path.join(
        path.dirname(path.dirname(__file__)),
        f"models/{model_prefix}-{get_current_datetime_str()}.pt",
    )
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        save_path,
    )


def load_model_and_optimizer(optimizer, model, model_path):
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    model.eval()


def get_model_and_optimizer(model, optimizer=None, checkpoint=None):
    if optimizer is None:
        optimizer = optim.SGD(get_trainable_model_parameters(model), lr=0.01, momentum=0)

    if checkpoint:
        load_model_and_optimizer(optimizer, model, checkpoint)

    return model, optimizer


def get_trainable_model_parameters(
    model: torch.nn.Module,
) -> Iterator[torch.nn.Parameter]:
    for param in model.parameters():
        if param.requires_grad:
            yield param


def model_forward(
    model: OPTForCausalLM | PeftModel | torch.nn.Module, batch_inputs: torch.Tensor | LLMBatchInput
):
    if isinstance(model, (OPTForCausalLM, PeftModel)):
        return model(input_ids=batch_inputs.input_ids, attention_mask=batch_inputs.attention_mask)
    elif isinstance(model, torch.nn.Module):
        return model(batch_inputs)
    else:
        raise Exception("This model type is not supported")
