from pydantic import Field, AliasChoices

from pydantic_settings import BaseSettings, SettingsConfigDict, CliImplicitFlag
from experiment_helper.types import LargeModel, ModelDtype, Dataset, RandomGradEstimateMethod

# class Lora(BaseSettings):
#     '''LoRA related fields'''

# 	lora: bool = Field(default=False)
# 	lora_r: int = Field(default=8)
# 	lora_alpha: int = Field(default=16)


class Settings(BaseSettings, cli_parse_args=True):
    # seed
    seed: int = Field(default=365)
    # LoRA
    lora: CliImplicitFlag[bool] = Field(default=False)
    lora_r: int = Field(default=8, validation_alias=AliasChoices("lora-r"))
    lora_alpha: int = Field(default=16, validation_alias=AliasChoices("lora-alpha"))
    # data
    dataset: Dataset = Field(default=Dataset.mnist.value)
    train_batch_size: int = Field(default=8, validation_alias=AliasChoices("train-batch-size"))
    test_batch_size: int = Field(default=8, validation_alias=AliasChoices("test-batch-size"))
    iid: CliImplicitFlag[bool] = Field(default=True)
    dirichlet_alpha: float = Field(default=1.0, validation_alias=AliasChoices("dirichlet-alpha"))
    num_workers: int = Field(default=2, validation_alias=AliasChoices("num-workers"))
    # model
    large_model: LargeModel = Field(
        default=LargeModel.opt_125m.value, validation_alias=AliasChoices("large-model")
    )
    model_dtype: ModelDtype = Field(
        default=ModelDtype.float32.value, validation_alias=AliasChoices("model-dtype")
    )
    # optimizer
    lr: float = Field(default=1e-4)
    momentum: float = Field(default=0)
    # device
    no_cuda: CliImplicitFlag[bool] = Field(default=False, validation_alias=AliasChoices("no-cuda"))
    no_mps: CliImplicitFlag[bool] = Field(default=False, validation_alias=AliasChoices("no-mps"))
    # non-fl training loop
    epoch: int = Field(default=500)
    warmup_epochs: int = Field(default=5, validation_alias=AliasChoices("warmup-epochs"))
    log_to_tensorboard: str | None = Field(
        default=None, validation_alias=AliasChoices("log-to-tensorboard")
    )
    # zo_grad_estimator
    mu: float = Field(default=1e-3)
    num_pert: int = Field(default=1, validation_alias=AliasChoices("num-pert"))
    adjust_perturb: CliImplicitFlag[bool] = Field(
        default=False, validation_alias=AliasChoices("adjust-perturb")
    )
    grad_estimate_method: RandomGradEstimateMethod = Field(
        default=RandomGradEstimateMethod.rge_central.value,
        validation_alias=AliasChoices("grad-estimate-method"),
    )
    no_optim: CliImplicitFlag[bool] = Field(
        default=False, validation_alias=AliasChoices("no-optim")
    )
    # Federated Learning
    iterations: int = Field(default=100)
    eval_iterations: int = Field(default=20, validation_alias=AliasChoices("eval-iterations"))
    num_clients: int = Field(default=8, validation_alias=AliasChoices("num-clients"))
    num_sample_clients: int = Field(default=2, validation_alias=AliasChoices("num-sample-clients"))
    local_update_steps: int = Field(default=1, validation_alias=AliasChoices("local-update-steps"))
    # Byzantinem TODO improve options
    aggregation: str = Field(default="mean")  # "mean, median, trim, krum"
    byz_type: str = Field(default="no_byz", validation_alias=AliasChoices("byz-type"))
    num_byz: int = Field(default=1, validation_alias=AliasChoices("num-byz"))
