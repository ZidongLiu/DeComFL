# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 20:58:19 2024

@author: Zidong
"""

from os import path
from typing import Iterator, TypeAlias

import torch
from peft import PeftModel
from transformers.modeling_utils import PreTrainedModel

from cezo_fl.util.language_utils import LLMBatchInput

LanguageModel: TypeAlias = PreTrainedModel | PeftModel
AllModel: TypeAlias = torch.nn.Module | LanguageModel


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


def get_trainable_model_parameters(
    model: torch.nn.Module,
) -> Iterator[torch.nn.Parameter]:
    for param in model.parameters():
        if param.requires_grad:
            yield param


def model_forward(model: AllModel, batch_inputs: torch.Tensor | LLMBatchInput):
    if isinstance(model, (PreTrainedModel, PeftModel)):
        assert isinstance(batch_inputs, LLMBatchInput)
        return model(input_ids=batch_inputs.input_ids, attention_mask=batch_inputs.attention_mask)
    elif isinstance(model, torch.nn.Module):
        assert isinstance(batch_inputs, torch.Tensor)
        return model(batch_inputs)
    else:
        raise Exception("This model type is not supported")


def model_generate(model: LanguageModel, batch_inputs: LLMBatchInput, generation_kwargs: dict):
    if "max_new_tokens" in generation_kwargs:
        generation_kwargs = generation_kwargs.copy()
        assert "max_length" in generation_kwargs  # both should be specified.
        # Dynamic adjust the max_new_tokens according to input length
        generation_kwargs["max_new_tokens"] = min(
            generation_kwargs["max_new_tokens"],
            generation_kwargs["max_length"] - batch_inputs.input_ids.size(1),
        )
        del generation_kwargs["max_length"]
    return model.generate(
        batch_inputs.input_ids,  # attention_mask is not needed for generation model.
        **generation_kwargs,
    )
