from typing import Literal, Callable
import torch
import torch.nn as nn

from functools import partial
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer


from cezo_fl.util import model_helpers
from cezo_fl.models.cnn_fashion import CNN_FMNIST
from cezo_fl.models.cnn_mnist import CNN_MNIST
from cezo_fl.models.lenet import LeNet
from cezo_fl.models.lstm import CharLSTM
from cezo_fl.random_gradient_estimator import RandomGradientEstimator as RGE
from cezo_fl.util.language_utils import LM_TEMPLATE_MAP, SUPPORTED_LLM, get_lm_loss
from cezo_fl.util.metrics import accuracy

from dataclasses import dataclass


@dataclass
class MetricPacks:
    train_loss: Callable
    train_acc: Callable
    test_loss: Callable
    test_acc: Callable


@dataclass
class ModelInferences:
    train_inference: Callable
    test_inference: Callable


BASE_GENERATION_KWARGS = {
    "do_sample": True,
    "temperature": 1.0,
    "num_beams": 2,
    "top_k": None,
    "num_return_sequences": 1,
    "max_new_tokens": 5,  # will be adjusted dynamically later
    "max_length": 2048,
    "early_stopping": True,
}


def get_wrapped_criterion(model, criterion):
    def ret_fn(batch_inputs, batch_labels):
        pred = model_helpers.model_forward(model, batch_inputs)
        return criterion(pred, batch_labels)

    return ret_fn


def get_generation_wrapped_criterion(model, criterion, generation_kwargs):
    def ret_fn(batch_inputs, batch_labels):
        pred = model_helpers.model_generate(model, batch_inputs, generation_kwargs)
        return criterion(pred, batch_labels)

    return ret_fn


def get_torch_dtype(args_model_dtype: Literal["float32", "float16", "bfloat16"]) -> torch.dtype:
    return {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[args_model_dtype]


def get_random_gradient_estimator(args, model, device):
    if args.grad_estimate_method in ["rge-central", "rge-forward"]:
        return RGE(
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
    else:
        raise Exception(f"Grad estimate method {args.grad_estimate_method} not supported")


def get_model_and_optimizer(args, device):
    torch_dtype = get_torch_dtype(args.model_dtype)
    if args.dataset == "mnist":
        model = CNN_MNIST().to(torch_dtype).to(device)
        train_model_inference = test_model_inference = partial(
            model_helpers.model_forward, model=model
        )
        model_helpers.model_forward()
        optimizer = torch.optim.SGD(
            model_helpers.get_trainable_model_parameters(model),
            lr=args.lr,
            weight_decay=1e-5,
            momentum=args.momentum,
        )
        train_criterion = test_criterion = nn.CrossEntropyLoss()
        train_accuracy_func = test_accuracy_func = accuracy

        # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.8)
    elif args.dataset == "cifar10":
        model = LeNet().to(torch_dtype).to(device)
        train_model_inference = test_model_inference = partial(
            model_helpers.model_forward, model=model
        )
        model_helpers.model_forward()
        optimizer = torch.optim.SGD(
            model_helpers.get_trainable_model_parameters(model),
            lr=args.lr,
            weight_decay=5e-4,
            momentum=args.momentum,
        )
        train_criterion = test_criterion = nn.CrossEntropyLoss()
        train_accuracy_func = test_accuracy_func = accuracy
        # scheduler = torch.optim.lr_scheduler.MultiStepLR(
        #     optimizer, milestones=[200], gamma=0.1
        # )
    elif args.dataset == "fashion":
        model = CNN_FMNIST().to(torch_dtype).to(device)
        train_model_inference = test_model_inference = partial(
            model_helpers.model_forward, model=model
        )
        model_helpers.model_forward()
        optimizer = torch.optim.SGD(
            model_helpers.get_trainable_model_parameters(model),
            lr=args.lr,
            weight_decay=1e-5,
            momentum=args.momentum,
        )
        train_criterion = test_criterion = nn.CrossEntropyLoss()
        train_accuracy_func = test_accuracy_func = accuracy
        # scheduler = torch.optim.lr_scheduler.MultiStepLR(
        #     optimizer, milestones=[200], gamma=0.1
        # )
    elif args.dataset == "shakespeare":
        model = CharLSTM().to(torch_dtype).to(device)
        train_model_inference = test_model_inference = partial(
            model_helpers.model_forward, model=model
        )
        model_helpers.model_forward()
        optimizer = torch.optim.SGD(
            model_helpers.get_trainable_model_parameters(model),
            lr=args.lr,
            momentum=0.9,
            weight_decay=5e-4,
        )
        train_criterion = test_criterion = nn.CrossEntropyLoss()
        train_accuracy_func = test_accuracy_func = accuracy
        # scheduler = torch.optim.lr_scheduler.MultiStepLR(
        #     optimizer, milestones=[200], gamma=0.1
        # )
    elif args.dataset in LM_TEMPLATE_MAP.keys():
        large_model = args.large_model
        model_name = SUPPORTED_LLM[large_model]
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch_dtype).to(device)
        model.model_name = large_model
        tokenizer = AutoTokenizer.from_pretrained(
            model_name, padding_side="left", truncate_side="left"
        )
        template = LM_TEMPLATE_MAP[args.dataset]()
        if args.lora:
            # this step initialize lora parameters, which should be under control of seed
            lora_config = LoraConfig(
                r=args.lora_r,
                lora_alpha=args.lora_alpha,
                target_modules=["q_proj", "v_proj"],
            )
            model = get_peft_model(model, lora_config).to(torch_dtype)

        if args.dataset in ["sst2", "cb", "wsc", "wic", "multirc", "rte", "boolq"]:
            train_model_inference = test_model_inference = partial(
                model_helpers.model_forward, model=model
            )
            optimizer = torch.optim.SGD(
                model_helpers.get_trainable_model_parameters(model),
                lr=args.lr,
                momentum=0,
                weight_decay=5e-4,
            )
            verbalizer_id_map = template.get_verbalizer_id(tokenizer)
            train_criterion = test_criterion = get_lm_loss(
                "last_token", verbalizer_id_map=verbalizer_id_map
            )
            train_accuracy_func = test_accuracy_func = get_lm_loss(
                "accuracy", verbalizer_id_map=verbalizer_id_map
            )
        elif args.dataset in ["squad", "drop", "xsum"]:
            if args.dataset in ["squad", "drop"]:
                generation_kwargs = BASE_GENERATION_KWARGS | {
                    "top_p": 0.3,
                    "length_penalty": 2,
                    "max_new_tokens": 5,  # will be adjusted dynamically later
                    "eos_token_id": [
                        tokenizer.encode("\n", add_special_tokens=False)[-1],
                        tokenizer.eos_token_id,
                    ],
                }
            elif args.dataset in ["xsum"]:
                generation_kwargs = BASE_GENERATION_KWARGS | {
                    "top_p": 0.95,
                    "num_return_sequences": 1,
                    "max_new_tokens": 500,  # will be adjusted dynamically later
                    "eos_token_id": [
                        tokenizer.encode("\n", add_special_tokens=False)[-1],
                        tokenizer.eos_token_id,
                    ],
                }
            else:
                generation_kwargs = {}

            train_model_inference = partial(model_helpers.model_forward, model=model)
            test_model_inference = partial(
                model_helpers.model_generate, model=model, generation_kwargs=generation_kwargs
            )
            optimizer = torch.optim.SGD(
                model_helpers.get_trainable_model_parameters(model),
                lr=args.lr,
                momentum=0,
                weight_decay=0,
            )
            # write in separate lines to differentiate from above cases, here acc=criterion
            train_criterion = get_lm_loss("full_sentence", verbalizer_id_map={})
            train_accuracy_func = train_criterion
            test_criterion = get_lm_loss("f1", tokenizer=tokenizer)
            test_accuracy_func = test_criterion
        else:
            raise ValueError(f"Dataset {args.dataset} is not supported")
    else:
        raise Exception(f"Dataset {args.dataset} is not supported")

    return (
        model,
        ModelInferences(train_model_inference, test_model_inference),
        optimizer,
        MetricPacks(train_criterion, train_accuracy_func, test_criterion, test_accuracy_func),
    )


def prepare_settings_underseed(
    args, device
) -> tuple[nn.Module, ModelInferences, torch.optim.SGD, MetricPacks, RGE]:
    torch.manual_seed(args.seed)
    model, model_inferences, optimizer, metric_packs = get_model_and_optimizer(args, device)
    grad_estimator = get_random_gradient_estimator(args, model, device)
    return model, model_inferences, optimizer, metric_packs, grad_estimator
