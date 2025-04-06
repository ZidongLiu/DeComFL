import sys

from experiment_helper.cli_parser import (
    GeneralSetting,
    DeviceSetting,
    ModelSetting,
    OptimizerSetting,
    RGESetting,
    NormalTrainingLoopSetting,
    FederatedLearningSetting,
    ByzantineSetting,
)
from experiment_helper.experiment_typing import (
    LargeModel,
    ModelDtype,
)
from cezo_fl.gradient_estimators.random_gradient_estimator import RandomGradEstimateMethod


def test_general_setting():
    # default
    sys.argv = ["simplified_test.py"]
    general_setting = GeneralSetting()
    assert general_setting.seed == 365
    assert general_setting.log_to_tensorboard is None

    # some change
    sys.argv = ["simplified_test.py", "--seed=123", "--log-to-tensorboard=./asd/dsa"]
    general_setting = GeneralSetting()
    assert general_setting.seed == 123
    assert general_setting.log_to_tensorboard == "./asd/dsa"


def test_device_setting():
    # default
    sys.argv = ["simplified_test.py"]
    device_setting = DeviceSetting()
    assert device_setting.cuda is True
    assert device_setting.mps is True

    # some change
    sys.argv = ["simplified_test.py", "--no-mps", "--no-cuda"]
    device_setting = DeviceSetting()
    assert device_setting.cuda is False
    assert device_setting.mps is False


def test_model_setting():
    # default
    sys.argv = ["simplified_test.py"]
    model_setting = ModelSetting()
    assert model_setting.large_model == LargeModel.opt_125m
    assert model_setting.model_dtype == ModelDtype.float32
    assert model_setting.lora is False
    assert model_setting.lora_r == 8
    assert model_setting.lora_alpha == 16

    # some change
    sys.argv = [
        "simplified_test.py",
        "--large-model=opt-1.3b",
        "--model-dtype=bfloat16",
        "--lora",
        "--lora-r=12",
        "--lora-alpha=123",
    ]
    model_setting = ModelSetting()
    assert model_setting.large_model == LargeModel.opt_1p3b
    assert model_setting.model_dtype == ModelDtype.bfloat16
    assert model_setting.lora is True
    assert model_setting.lora_r == 12
    assert model_setting.lora_alpha == 123


def test_optimizer_setting():
    # default
    sys.argv = ["simplified_test.py"]
    optimizer_setting = OptimizerSetting()
    assert optimizer_setting.lr == 1e-4
    assert optimizer_setting.momentum == 0

    # some change
    sys.argv = ["simplified_test.py", "--lr=1e-3", "--momentum=0.9"]
    optimizer_setting = OptimizerSetting()
    assert optimizer_setting.lr == 1e-3
    assert optimizer_setting.momentum == 0.9


def test_rge_setting():
    # default
    sys.argv = ["simplified_test.py"]
    rge_setting = RGESetting()
    assert rge_setting.mu == 1e-3
    assert rge_setting.num_pert == 1
    assert rge_setting.adjust_perturb is False
    assert rge_setting.grad_estimate_method == RandomGradEstimateMethod.rge_central
    assert rge_setting.optim is True

    # some change
    sys.argv = [
        "simplified_test.py",
        "--mu=1e-5",
        "--num-pert=5",
        "--adjust-perturb",
        "--grad-estimate-method=rge-forward",
        "--no-optim",
    ]
    rge_setting = RGESetting()
    assert rge_setting.mu == 1e-5
    assert rge_setting.num_pert == 5
    assert rge_setting.adjust_perturb is True
    assert rge_setting.grad_estimate_method == RandomGradEstimateMethod.rge_forward
    assert rge_setting.optim is False


def test_normal_training_loop_setting():
    # default
    sys.argv = ["simplified_test.py"]
    normal_training_loop_setting = NormalTrainingLoopSetting()
    assert normal_training_loop_setting.epoch == 500
    assert normal_training_loop_setting.warmup_epochs == 5

    # some change
    sys.argv = ["simplified_test.py", "--epoch=200", "--warmup-epochs=10"]
    normal_training_loop_setting = NormalTrainingLoopSetting()
    assert normal_training_loop_setting.epoch == 200
    assert normal_training_loop_setting.warmup_epochs == 10


def test_federated_learning_setting():
    # default
    sys.argv = ["simplified_test.py"]
    federate_learning_setting = FederatedLearningSetting()
    assert federate_learning_setting.iterations == 100
    assert federate_learning_setting.eval_iterations == 20
    assert federate_learning_setting.num_clients == 8
    assert federate_learning_setting.num_sample_clients == 2
    assert federate_learning_setting.local_update_steps == 1

    # some change
    sys.argv = [
        "simplified_test.py",
        "--iterations=200",
        "--eval-iterations=10",
        "--num-clients=4",
        "--num-sample-clients=3",
        "--local-update-steps=2",
    ]
    federate_learning_setting = FederatedLearningSetting()
    assert federate_learning_setting.iterations == 200
    assert federate_learning_setting.eval_iterations == 10
    assert federate_learning_setting.num_clients == 4
    assert federate_learning_setting.num_sample_clients == 3
    assert federate_learning_setting.local_update_steps == 2


def test_byzantine_setting():
    # default
    sys.argv = ["simplified_test.py"]
    byzantine_setting = ByzantineSetting()
    assert byzantine_setting.aggregation == "mean"
    assert byzantine_setting.byz_type == "no_byz"
    assert byzantine_setting.num_byz == 1

    # some change
    sys.argv = ["simplified_test.py", "--aggregation=trim", "--byz-type=krum", "--num-byz=3"]
    byzantine_setting = ByzantineSetting()
    assert byzantine_setting.aggregation == "trim"
    assert byzantine_setting.byz_type == "krum"
    assert byzantine_setting.num_byz == 3
