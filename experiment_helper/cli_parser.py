from pydantic import Field

from pydantic_settings import BaseSettings, SettingsConfigDict
from experiment_helper.typing import LargeModel, ModelDtype, Dataset, RandomGradEstimateMethod

# class Lora(BaseSettings):
#     '''LoRA related fields'''

# 	lora: bool = Field(default=False)
# 	lora_r: int = Field(default=8)
# 	lora_alpha: int = Field(default=16)


class Settings(BaseSettings):
    # seed
    seed: int = Field(default=365)
    # LoRA
    lora: bool = Field(default=False)
    lora_r: int = Field(default=8)
    lora_alpha: int = Field(default=16)
    # data
    dataset: Dataset = Field(default=Dataset.mnist.value)
    train_batch_size: int = Field(default=8)
    test_batch_size: int = Field(default=8)
    iid: bool = Field(default=True)
    dirichlet_alpha: float = Field(default=1.0)
    num_workers: int = Field(default=2)
    # model
    large_model: LargeModel = Field(default=LargeModel.opt_125m.value)
    model_dtype: ModelDtype = Field(default=ModelDtype.float32.value)
    # optimizer
    lr: float = Field(default=1e-4)
    momentum: float = Field(default=0)
    # device
    no_cuda: bool = Field(default=False)
    no_mps: bool = Field(default=False)
    # non-fl training loop
    epoch: int = Field(default=500)
    warmup_epochs: int = Field(default=5)
    log_to_tensorboard: str | None = Field(default=None)
    # zo_grad_estimator
    mu: float = Field(default=1e-3)
    num_pert: int = Field(default=1)
    adjust_perturb: bool = Field(default=False)
    grad_estimate_method: RandomGradEstimateMethod = Field(
        default=RandomGradEstimateMethod.rge_central.value
    )
    no_optim: bool = Field(default=False)
    # Federated Learning
    iterations: int = Field(default=100)
    eval_iterations: int = Field(default=20)
    num_clients: int = Field(default=8)
    num_sample_clients: int = Field(default=2)
    local_update_steps: int = Field(default=1)
    # Byzantinem TODO improve options
    aggregation: str = Field(default="mean")  # "mean, median, trim, krum"
    byz_type: str = Field(default="no_byz")
    num_byz: int = Field(default=1)
