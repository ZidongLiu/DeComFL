from typing import Literal
from functools import cached_property

import torch
from pydantic import Field, AliasChoices
from pydantic_settings import BaseSettings, CliImplicitFlag, SettingsConfigDict
from experiment_helper.experiment_typing import (
    LargeModel,
    ModelDtype,
)
from enum import Enum
from cezo_fl.gradient_estimators.random_gradient_estimator import RandomGradEstimateMethod
from cezo_fl.gradient_estimators.adam_forward import KUpdateStrategy
from cezo_fl.util.language_utils import SUPPORTED_LLM
from experiment_helper.data import DataSetting  # noqa: F401, for a central export place for all settings
from fed_avg.server import FOFLStrategy


class FrozenSetting(BaseSettings, cli_parse_args=True, cli_ignore_unknown_args=True):
    # pydantic's config, not neural network model
    model_config = SettingsConfigDict(frozen=True)


class GeneralSetting(FrozenSetting):
    # general
    seed: int = Field(
        default=365,
        description="Random seed used to initialize model, get dataloaders and to sample the RGE's seeds each round",
    )
    log_to_tensorboard: str | None = Field(
        default=None,
        validation_alias=AliasChoices("log-to-tensorboard"),
        description="Provide a valid path, will log training process to the tensorboard in that path",
    )

    @cached_property
    def general_setting(self) -> "GeneralSetting":
        return GeneralSetting()


class DeviceSetting(FrozenSetting):
    # device
    cuda: CliImplicitFlag[bool] = Field(
        default=True, description="--no-cuda will disable cuda training"
    )
    mps: CliImplicitFlag[bool] = Field(
        default=True,
        description="--no-mps will disable macOS GPU training, this command line argument is ignored when cuda is available and choose to use cuda",
    )

    @cached_property
    def device_setting(self) -> "DeviceSetting":
        return DeviceSetting()


class ModelSetting(FrozenSetting):
    # model
    large_model: LargeModel = Field(
        default=LargeModel.opt_125m,
        validation_alias=AliasChoices("large-model"),
        description="Model name for Hugging Face Lanuguage Model. current only support facebook/opt families",
    )
    model_dtype: ModelDtype = Field(
        default=ModelDtype.float32, validation_alias=AliasChoices("model-dtype")
    )

    # LoRA
    lora: CliImplicitFlag[bool] = Field(default=False)
    lora_r: int = Field(default=8, validation_alias=AliasChoices("lora-r"))
    lora_alpha: int = Field(default=16, validation_alias=AliasChoices("lora-alpha"))

    @cached_property
    def model_setting(self) -> "ModelSetting":
        return ModelSetting()

    def get_hf_model_name(self) -> str:
        return SUPPORTED_LLM[self.large_model.value]

    def get_torch_dtype(self):
        return {
            ModelDtype.float16: torch.float16,
            ModelDtype.float32: torch.float32,
            ModelDtype.bfloat16: torch.bfloat16,
        }[self.model_dtype]


class OptimizerSetting(FrozenSetting):
    # optimizer
    optimizer: Literal["sgd", "adam"] = Field(default="sgd")
    lr: float = Field(default=1e-4)
    momentum: float = Field(default=0)
    beta1: float = Field(default=0.9)
    beta2: float = Field(default=0.999)

    @cached_property
    def optimizer_setting(self) -> "OptimizerSetting":
        return OptimizerSetting()


class EstimatorType(Enum):
    vanilla = "vanilla"
    adam_forward = "adam_forward"


class RGESetting(FrozenSetting):
    # zo_grad_estimator
    estimator_type: EstimatorType = Field(
        default=EstimatorType.vanilla,
        validation_alias=AliasChoices("estimator-type"),
        description="Type of gradient estimator, options: vanilla, adam_forward",
    )
    mu: float = Field(default=1e-3, description="Perturbation step to measure local gradients")
    num_pert: int = Field(
        default=1,
        validation_alias=AliasChoices("num-pert"),
        description="Number of perturbations needed to perform when estimating gradient",
    )
    adjust_perturb: CliImplicitFlag[bool] = Field(
        default=False,
        validation_alias=AliasChoices("adjust-perturb"),
        description="Whether to adjust number of perturbation in the training process",
    )
    grad_estimate_method: RandomGradEstimateMethod = Field(
        default=RandomGradEstimateMethod.rge_central,
        validation_alias=AliasChoices("grad-estimate-method"),
        description="Forward or Central",
    )
    optim: CliImplicitFlag[bool] = Field(
        default=True,
        description="Use optimizer or not, when no-optim, update model without torch.optim (SGD only). This can significantly save memory.",
    )
    k_update_strategy: KUpdateStrategy = Field(
        default=KUpdateStrategy.LAST_LOCAL_UPDATE,
        validation_alias=AliasChoices("k-update-strategy"),
        description="Update strategy for K, options: last_local_update, all_local_updates. Only used when estimator-type is adam_forward",
    )
    hessian_smooth: float = Field(
        default=0.95,
        validation_alias=AliasChoices("hessian-smooth"),
        description="Smoothing factor for Hessian. Only used when estimator-type is adam_forward",
    )

    @cached_property
    def rge_setting(self) -> "RGESetting":
        return RGESetting()


class NormalTrainingLoopSetting(FrozenSetting):
    # non-fl training loop
    epoch: int = Field(default=500)
    warmup_epochs: int = Field(default=5, validation_alias=AliasChoices("warmup-epochs"))

    @cached_property
    def normal_training_loop_setting(self) -> "NormalTrainingLoopSetting":
        return NormalTrainingLoopSetting()


class FederatedLearningSetting(FrozenSetting):
    # Federated Learning
    iterations: int = Field(default=100)
    eval_iterations: int = Field(default=20, validation_alias=AliasChoices("eval-iterations"))
    num_clients: int = Field(default=8, validation_alias=AliasChoices("num-clients"))
    num_sample_clients: int = Field(default=2, validation_alias=AliasChoices("num-sample-clients"))
    local_update_steps: int = Field(default=1, validation_alias=AliasChoices("local-update-steps"))

    @cached_property
    def federated_learning_setting(self) -> "FederatedLearningSetting":
        return FederatedLearningSetting()


class ByzantineSetting(FrozenSetting):
    # Byzantinem TODO improve options
    aggregation: Literal["mean", "median", "trim", "krum"] = Field(default="mean")
    byz_type: str = Field(default="no_byz", validation_alias=AliasChoices("byz-type"))
    num_byz: int = Field(
        default=1,
        validation_alias=AliasChoices("num-byz"),
        description="Number of byzantine attackers",
    )

    @cached_property
    def byzantine_setting(self) -> "ByzantineSetting":
        return ByzantineSetting()


class FOFLSetting(FrozenSetting):
    # FO-FL
    fo_fl_strategy: FOFLStrategy = Field(
        default=FOFLStrategy.fedavg,
        validation_alias=AliasChoices("fo-fl-strategy"),
        description="FO-FL strategy, options: fedavg, fedadam, fedadagrad, fedyogi",
    )

    fo_fl_beta1: float = Field(default=0.9, validation_alias=AliasChoices("fo-fl-beta1"))
    fo_fl_beta2: float = Field(default=0.999, validation_alias=AliasChoices("fo-fl-beta2"))

    @cached_property
    def fo_fl_setting(self) -> "FOFLSetting":
        return FOFLSetting()
