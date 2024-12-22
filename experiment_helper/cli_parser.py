from pydantic import Field, AliasChoices
from pydantic_settings import BaseSettings, CliImplicitFlag
from experiment_helper.experiment_typing import (
    LargeModel,
    ModelDtype,
)
from cezo_fl.random_gradient_estimator import RandomGradEstimateMethod
from experiment_helper.data import DataSetting  # noqa: F401


class GeneralSetting(BaseSettings, cli_parse_args=True):
    # general
    seed: int = Field(default=365)
    log_to_tensorboard: str | None = Field(
        default=None, validation_alias=AliasChoices("log-to-tensorboard")
    )


class LoraSetting(BaseSettings, cli_parse_args=True):
    # LoRA
    lora: CliImplicitFlag[bool] = Field(default=False)
    lora_r: int = Field(default=8, validation_alias=AliasChoices("lora-r"))
    lora_alpha: int = Field(default=16, validation_alias=AliasChoices("lora-alpha"))


class ModelSetting(BaseSettings, cli_parse_args=True):
    """
    This warning will go away once we upgraded pydantic-setting to 2.6.2 hopefully.
    See https://github.com/pydantic/pydantic-settings/issues/482
    Warning```
    UserWarning: Field "model_dtype" in Settings has conflict with protected namespace "model_".
    You may be able to resolve this warning by setting `model_config['protected_namespaces'] = ('settings_',)`.
    ```
    """

    # model
    large_model: LargeModel = Field(
        default=LargeModel.opt_125m, validation_alias=AliasChoices("large-model")
    )
    model_dtype: ModelDtype = Field(
        default=ModelDtype.float32, validation_alias=AliasChoices("model-dtype")
    )


class OptimizerSetting(BaseSettings, cli_parse_args=True):
    # optimizer
    lr: float = Field(default=1e-4)
    momentum: float = Field(default=0)


class DeviceSetting(BaseSettings, cli_parse_args=True):
    # device
    no_cuda: CliImplicitFlag[bool] = Field(default=False, validation_alias=AliasChoices("no-cuda"))
    no_mps: CliImplicitFlag[bool] = Field(default=False, validation_alias=AliasChoices("no-mps"))


class RGESetting(BaseSettings, cli_parse_args=True):
    # zo_grad_estimator
    mu: float = Field(default=1e-3)
    num_pert: int = Field(default=1, validation_alias=AliasChoices("num-pert"))
    adjust_perturb: CliImplicitFlag[bool] = Field(
        default=False, validation_alias=AliasChoices("adjust-perturb")
    )
    grad_estimate_method: RandomGradEstimateMethod = Field(
        default=RandomGradEstimateMethod.rge_central,
        validation_alias=AliasChoices("grad-estimate-method"),
    )
    no_optim: CliImplicitFlag[bool] = Field(
        default=False, validation_alias=AliasChoices("no-optim")
    )


class NormalTrainingLoopSetting(BaseSettings, cli_parse_args=True):
    # non-fl training loop
    epoch: int = Field(default=500)
    warmup_epochs: int = Field(default=5, validation_alias=AliasChoices("warmup-epochs"))


class FederatedLearningSetting(BaseSettings, cli_parse_args=True):
    # Federated Learning
    iterations: int = Field(default=100)
    eval_iterations: int = Field(default=20, validation_alias=AliasChoices("eval-iterations"))
    num_clients: int = Field(default=8, validation_alias=AliasChoices("num-clients"))
    num_sample_clients: int = Field(default=2, validation_alias=AliasChoices("num-sample-clients"))
    local_update_steps: int = Field(default=1, validation_alias=AliasChoices("local-update-steps"))


class ByzantineSetting(BaseSettings, cli_parse_args=True):
    # Byzantinem TODO improve options
    aggregation: str = Field(default="mean")  # "mean, median, trim, krum"
    byz_type: str = Field(default="no_byz", validation_alias=AliasChoices("byz-type"))
    num_byz: int = Field(default=1, validation_alias=AliasChoices("num-byz"))
