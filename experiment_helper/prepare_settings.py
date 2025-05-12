from typing import Callable, TypeAlias
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
from cezo_fl.gradient_estimators.random_gradient_estimator import RandomGradientEstimator
from cezo_fl.gradient_estimators.adam_forward import AdamForwardGradientEstimator
from cezo_fl.gradient_estimators.hessian_random_gradient_estimator import (
    HessianRandomGradientEstimator,
)
from cezo_fl.util.language_utils import (
    LM_TEMPLATE_MAP,
    SUPPORTED_LLM,
    get_lm_loss,
    get_hf_tokenizer,
)
from cezo_fl.util.metrics import accuracy

from experiment_helper.cli_parser import ModelSetting, OptimizerSetting, RGESetting, EstimatorType
from experiment_helper.data import ImageClassificationTask, LmClassificationTask, LmGenerationTask
from dataclasses import dataclass


SupportedDataset: TypeAlias = ImageClassificationTask | LmClassificationTask | LmGenerationTask


def get_model(
    dataset: SupportedDataset,
    model_setting: ModelSetting,
    seed: int | None = None,
) -> AllModel:
    torch_dtype = model_setting.get_torch_dtype()
    model: AllModel
    if seed:
        torch.manual_seed(seed)

    if dataset == ImageClassificationTask.mnist:
        return CNN_MNIST().to(torch_dtype)
    elif dataset == ImageClassificationTask.cifar10:
        return LeNet().to(torch_dtype)
    elif dataset == ImageClassificationTask.fashion:
        return CNN_FMNIST().to(torch_dtype)
    elif isinstance(dataset, (LmClassificationTask, LmGenerationTask)):
        assert model_setting.large_model.value in SUPPORTED_LLM
        hf_model_name = model_setting.get_hf_model_name()
        model = AutoModelForCausalLM.from_pretrained(hf_model_name, torch_dtype=torch_dtype)
        model.model_name = model_setting.large_model.value
        if model_setting and model_setting.lora:
            # this step initialize lora parameters, which should be under control of seed
            lora_config = LoraConfig(
                r=model_setting.lora_r,
                lora_alpha=model_setting.lora_alpha,
                target_modules=["q_proj", "v_proj"],
            )
            model = get_peft_model(model, lora_config).to(torch_dtype)
        return model
    else:
        raise Exception(f"Dataset {dataset} is not supported")


def get_optimizer(
    model: AllModel, dataset: SupportedDataset, optimizer_setting: OptimizerSetting
) -> torch.optim.SGD | torch.optim.Adam:
    trainable_model_parameters = model_helpers.get_trainable_model_parameters(model)
    if optimizer_setting.optimizer == "sgd":
        if dataset == ImageClassificationTask.mnist:
            return torch.optim.SGD(
                trainable_model_parameters,
                lr=optimizer_setting.lr,
                weight_decay=1e-5,
                momentum=optimizer_setting.momentum,
            )
        elif dataset == ImageClassificationTask.cifar10:
            return torch.optim.SGD(
                trainable_model_parameters,
                lr=optimizer_setting.lr,
                weight_decay=5e-4,
                momentum=optimizer_setting.momentum,
            )
        elif dataset == ImageClassificationTask.fashion:
            return torch.optim.SGD(
                trainable_model_parameters,
                lr=optimizer_setting.lr,
                weight_decay=1e-5,
                momentum=optimizer_setting.momentum,
            )
        elif isinstance(dataset, LmClassificationTask):
            return torch.optim.SGD(
                trainable_model_parameters,
                lr=optimizer_setting.lr,
                momentum=0,
                weight_decay=5e-4,
            )
        elif isinstance(dataset, LmGenerationTask):
            return torch.optim.SGD(
                trainable_model_parameters,
                lr=optimizer_setting.lr,
                momentum=0,
                weight_decay=0,
            )
        else:
            raise Exception(f"dataset {dataset.value} not supported")
    elif optimizer_setting.optimizer == "adam":
        return torch.optim.Adam(
            trainable_model_parameters,
            lr=optimizer_setting.lr,
            betas=(optimizer_setting.beta1, optimizer_setting.beta2),
            weight_decay=5e-4,
        )
    else:
        raise Exception(f"optimizer {optimizer_setting.optimizer} not supported")


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
    dataset: SupportedDataset, model_setting: ModelSetting
) -> tuple[ModelInferences, MetricPacks]:
    if not isinstance(dataset, (LmClassificationTask, LmGenerationTask)):
        return ModelInferences(
            model_helpers.model_forward, model_helpers.model_forward
        ), MetricPacks(
            train_loss=nn.CrossEntropyLoss(),
            train_acc=accuracy,
            test_loss=nn.CrossEntropyLoss(),
            test_acc=accuracy,
        )

    hf_model_name = model_setting.get_hf_model_name()
    tokenizer = get_hf_tokenizer(hf_model_name)
    if isinstance(dataset, LmGenerationTask):
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
        test_accuracy_func = get_lm_loss("f1", tokenizer=tokenizer)
        return (
            ModelInferences(
                train_inference=model_helpers.model_forward,
                test_inference=partial(
                    model_helpers.model_generate, generation_kwargs=generation_kwargs
                ),
            ),
            MetricPacks(
                train_loss=train_criterion,
                train_acc=lambda pred, true: torch.tensor(0.0),  # noop training acc step here
                test_loss=lambda pred, true: torch.tensor(0.0),  # noop test loss step here
                test_acc=test_accuracy_func,
            ),
        )
    else:
        template = LM_TEMPLATE_MAP[dataset.value]()
        verbalizer_id_map = template.get_verbalizer_id(tokenizer)  # type: ignore[attr-defined]
        train_criterion = test_criterion = get_lm_loss(
            "last_token", verbalizer_id_map=verbalizer_id_map
        )
        train_accuracy_func = test_accuracy_func = get_lm_loss(
            "accuracy", verbalizer_id_map=verbalizer_id_map
        )
        return ModelInferences(
            model_helpers.model_forward, model_helpers.model_forward
        ), MetricPacks(
            train_loss=train_criterion,
            train_acc=train_accuracy_func,
            test_loss=test_criterion,
            test_acc=test_accuracy_func,
        )


def get_gradient_estimator(
    model: AllModel, device: torch.device, rge_setting: RGESetting, model_setting: ModelSetting
) -> RandomGradientEstimator | AdamForwardGradientEstimator:
    no_optim = not rge_setting.optim
    if rge_setting.estimator_type == EstimatorType.vanilla:
        return RandomGradientEstimator(
            parameters=model_helpers.get_trainable_model_parameters(model),
            mu=rge_setting.mu,
            num_pert=rge_setting.num_pert,
            grad_estimate_method=rge_setting.grad_estimate_method,
            device=device,
            torch_dtype=model_setting.get_torch_dtype(),
            # To save memory consumption, we have to use parameter-wise perturb + no_optim together.
            sgd_only_no_optim=no_optim,
            paramwise_perturb=no_optim,
        )
    elif rge_setting.estimator_type == EstimatorType.adam_forward:
        return AdamForwardGradientEstimator(
            parameters=model_helpers.get_trainable_model_parameters(model),
            mu=rge_setting.mu,
            num_pert=rge_setting.num_pert,
            device=device,
            torch_dtype=model_setting.get_torch_dtype(),
            k_update_strategy=rge_setting.k_update_strategy,
            hessian_smooth=rge_setting.hessian_smooth,
        )
    else:
        raise ValueError(f"Invalid estimator type: {rge_setting.estimator_type}")


def _get_hessian_random_gradient_estimator(
    model: AllModel, device: torch.device, rge_setting: RGESetting, model_setting: ModelSetting
) -> HessianRandomGradientEstimator:
    return HessianRandomGradientEstimator(
        parameters=model_helpers.get_trainable_model_parameters(model),
        mu=rge_setting.mu,
        num_pert=rge_setting.num_pert,
        device=device,
        torch_dtype=model_setting.get_torch_dtype(),
    )
