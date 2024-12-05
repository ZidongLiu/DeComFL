from typing import Literal, Callable
import torch
import torch.nn as nn

from functools import partial
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM


from cezo_fl.util import model_helpers
from cezo_fl.util.model_helpers import AllModel
from cezo_fl.models.cnn_fashion import CNN_FMNIST
from cezo_fl.models.cnn_mnist import CNN_MNIST
from cezo_fl.models.lenet import LeNet
from cezo_fl.models.lstm import CharLSTM
from cezo_fl.random_gradient_estimator import RandomGradientEstimator as RGE
from cezo_fl.util.language_utils import (
    LM_TEMPLATE_MAP,
    SUPPORTED_LLM,
    get_lm_loss,
    get_hf_tokenizer,
)
from cezo_fl.util.metrics import accuracy

from dataclasses import dataclass


def get_torch_dtype(args_model_dtype: Literal["float32", "float16", "bfloat16"]) -> torch.dtype:
    return {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[args_model_dtype]


def get_model_and_optimizer(
    dataset: str,
    model_dtype: Literal["float32", "float16", "bfloat16"],
    lr: float,
    momentum: float,
    large_model: str | None,
    lora: bool = False,
    lora_alpha: int = 16,
    lora_r: int = 8,
) -> tuple[AllModel, torch.optim.SGD]:
    torch_dtype = get_torch_dtype(model_dtype)
    model: AllModel
    if dataset == "mnist":
        model = CNN_MNIST().to(torch_dtype)
        optimizer = torch.optim.SGD(
            model_helpers.get_trainable_model_parameters(model),
            lr=lr,
            weight_decay=1e-5,
            momentum=momentum,
        )
    elif dataset == "cifar10":
        model = LeNet().to(torch_dtype)
        optimizer = torch.optim.SGD(
            model_helpers.get_trainable_model_parameters(model),
            lr=lr,
            weight_decay=5e-4,
            momentum=momentum,
        )
    elif dataset == "fashion":
        model = CNN_FMNIST().to(torch_dtype)
        optimizer = torch.optim.SGD(
            model_helpers.get_trainable_model_parameters(model),
            lr=lr,
            weight_decay=1e-5,
            momentum=momentum,
        )
    elif dataset == "shakespeare":
        model = CharLSTM().to(torch_dtype)
        optimizer = torch.optim.SGD(
            model_helpers.get_trainable_model_parameters(model),
            lr=lr,
            momentum=0.9,
            weight_decay=5e-4,
        )
    elif dataset in LM_TEMPLATE_MAP.keys():
        assert large_model in SUPPORTED_LLM
        hf_model_name = SUPPORTED_LLM[large_model]
        model = AutoModelForCausalLM.from_pretrained(hf_model_name, torch_dtype=torch_dtype)
        model.model_name = large_model
        if lora:
            # this step initialize lora parameters, which should be under control of seed
            lora_config = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                target_modules=["q_proj", "v_proj"],
            )
            model = get_peft_model(model, lora_config).to(torch_dtype)

        if dataset in ["sst2", "cb", "wsc", "wic", "multirc", "rte", "boolq"]:
            optimizer = torch.optim.SGD(
                model_helpers.get_trainable_model_parameters(model),
                lr=lr,
                momentum=0,
                weight_decay=5e-4,
            )
        elif dataset in ["squad", "drop", "xsum"]:
            optimizer = torch.optim.SGD(
                model_helpers.get_trainable_model_parameters(model),
                lr=lr,
                momentum=0,
                weight_decay=0,
            )
        else:
            raise ValueError(f"Dataset {dataset} is not supported")
    else:
        raise Exception(f"Dataset {dataset} is not supported")

    return model, optimizer


@dataclass
class ModelInferences:
    train_inference: Callable
    test_inference: Callable


@dataclass
class MetricPacks:
    train_loss: Callable
    train_acc: Callable
    test_loss: Callable
    test_acc: Callable


def get_model_inferences_and_metrics(
    dataset: str, hf_model_name: str | None = None
) -> tuple[ModelInferences, MetricPacks]:
    if dataset not in LM_TEMPLATE_MAP.keys():
        return ModelInferences(
            model_helpers.model_forward, model_helpers.model_forward
        ), MetricPacks(nn.CrossEntropyLoss(), accuracy, nn.CrossEntropyLoss(), accuracy)

    assert hf_model_name
    tokenizer = get_hf_tokenizer(hf_model_name)
    if dataset in ["squad", "drop", "xsum"]:
        generation_kwargs = {
            "do_sample": True,
            "temperature": 1.0,
            "num_beams": 2,
            "top_k": None,
            "num_return_sequences": 1,
            "max_new_tokens": 5,  # will be adjusted dynamically later, 500 for xsum, not sure why we need it tho
            "max_length": 2048,
            "early_stopping": True,
            "eos_token_id": [
                tokenizer.encode("\n", add_special_tokens=False)[-1],
                tokenizer.eos_token_id,
            ],
            "top_p": 0.95 if dataset == "xsum" else 0.3,
            "length_penalty": 1 if dataset == "xsum" else 2,
        }
        # write in separate lines to differentiate from above cases, here acc=criterion
        train_criterion = get_lm_loss("full_sentence", verbalizer_id_map={})
        train_accuracy_func = train_criterion
        test_criterion = get_lm_loss("f1", tokenizer=tokenizer)
        test_accuracy_func = test_criterion
        return (
            ModelInferences(
                train_inference=model_helpers.model_forward,
                test_inference=partial(
                    model_helpers.model_generate, generation_kwargs=generation_kwargs
                ),
            ),
            MetricPacks(train_criterion, train_accuracy_func, test_criterion, test_accuracy_func),
        )
    else:
        template = LM_TEMPLATE_MAP[dataset]()
        verbalizer_id_map = template.get_verbalizer_id(tokenizer)  # type: ignore[attr-defined]
        train_criterion = test_criterion = get_lm_loss(
            "last_token", verbalizer_id_map=verbalizer_id_map
        )
        train_accuracy_func = test_accuracy_func = get_lm_loss(
            "accuracy", verbalizer_id_map=verbalizer_id_map
        )
        return ModelInferences(
            model_helpers.model_forward, model_helpers.model_forward
        ), MetricPacks(train_criterion, train_accuracy_func, test_criterion, test_accuracy_func)


def prepare_settings_underseed(
    args, device: torch.device
) -> tuple[nn.Module, torch.optim.SGD, RGE]:
    torch.manual_seed(args.seed)

    model, optimizer = get_model_and_optimizer(
        dataset=args.dataset,
        model_dtype=args.model_dtype,
        lr=args.lr,
        momentum=args.momentum,
        large_model=args.large_model,
        lora=args.lora,
        lora_alpha=args.lora_alpha,
        lora_r=args.lora_r,
    )
    grad_estimator = RGE(
        parameters=model_helpers.get_trainable_model_parameters(model),
        mu=args.mu,
        num_pert=args.num_pert,
        grad_estimate_method=args.grad_estimate_method,
        device=device,
        torch_dtype=get_torch_dtype(args.model_dtype),
        # To save memory consumption, we have to use parameter-wise perturb + no_optim together.
        sgd_only_no_optim=args.no_optim,
        paramwise_perturb=args.no_optim,
    )
    return model, optimizer, grad_estimator
