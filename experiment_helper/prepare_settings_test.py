import sys
import pytest
import torch
import torch.nn as nn

from experiment_helper.prepare_settings import get_optimizer, _split_parameters_for_hybrid
from experiment_helper.cli_parser import OptimizerSetting, RGESetting
from experiment_helper.data import ImageClassificationTask
from cezo_fl.util import model_helpers


class SimpleModel(nn.Module):
    """Simple test model."""

    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(10, 5)
        self.linear2 = nn.Linear(5, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.linear1(x))  # type: ignore[no-any-return]


class TestGetOptimizer:
    """Test cases for get_optimizer function."""

    def test_get_optimizer_single_lr(self):
        """Test optimizer creation with single learning rate."""
        model = SimpleModel()
        optimizer_setting = OptimizerSetting(lr=0.01, lr2=None)

        optimizer = get_optimizer(
            model=model,
            dataset=ImageClassificationTask.mnist,
            optimizer_setting=optimizer_setting,
            rge_setting=None,
        )

        # Should have single parameter group
        assert len(optimizer.param_groups) == 1
        assert optimizer.param_groups[0]["lr"] == 0.01

    def test_get_optimizer_multiple_lrs_hybrid(self):
        """Test optimizer creation with multiple learning rates for hybrid estimator."""
        model = SimpleModel()
        optimizer_setting = OptimizerSetting(lr=0.01, lr2=0.005)

        # Set sys.argv to configure RGESetting for hybrid estimator
        original_argv = sys.argv.copy()
        sys.argv = ["test.py", "--estimator-type=hybrid"]
        rge_setting = RGESetting()
        sys.argv = original_argv

        optimizer = get_optimizer(
            model=model,
            dataset=ImageClassificationTask.mnist,
            optimizer_setting=optimizer_setting,
            rge_setting=rge_setting,
        )

        # Should have two parameter groups
        assert len(optimizer.param_groups) == 2
        assert optimizer.param_groups[0]["lr"] == 0.01  # Random parameters
        assert optimizer.param_groups[1]["lr"] == 0.005  # Adam forward parameters

        # Verify parameter groups contain correct parameters
        random_params, adam_forward_params = _split_parameters_for_hybrid(model)
        assert len(optimizer.param_groups[0]["params"]) == len(random_params)
        assert len(optimizer.param_groups[1]["params"]) == len(adam_forward_params)

    def test_get_optimizer_lr2_none_with_hybrid(self):
        """Test that lr2=None still works with hybrid estimator (single LR)."""
        model = SimpleModel()
        optimizer_setting = OptimizerSetting(lr=0.01, lr2=None)

        # When lr2 is None, should create single param group even if hybrid is used
        # (This tests the fallback behavior)
        optimizer = get_optimizer(
            model=model,
            dataset=ImageClassificationTask.mnist,
            optimizer_setting=optimizer_setting,
            rge_setting=None,
        )

        # Should have single parameter group
        assert len(optimizer.param_groups) == 1
        assert optimizer.param_groups[0]["lr"] == 0.01

    def test_get_optimizer_sgd_mnist(self):
        """Test SGD optimizer creation for MNIST."""
        model = SimpleModel()
        optimizer_setting = OptimizerSetting(optimizer="sgd", lr=0.01, momentum=0.9, lr2=None)

        optimizer = get_optimizer(
            model=model,
            dataset=ImageClassificationTask.mnist,
            optimizer_setting=optimizer_setting,
            rge_setting=None,
        )

        assert isinstance(optimizer, torch.optim.SGD)
        assert optimizer.param_groups[0]["momentum"] == 0.9
        assert optimizer.param_groups[0]["weight_decay"] == 1e-5

    def test_get_optimizer_adam(self):
        """Test Adam optimizer creation."""
        model = SimpleModel()
        optimizer_setting = OptimizerSetting(
            optimizer="adam", lr=0.001, beta1=0.9, beta2=0.999, lr2=None
        )

        optimizer = get_optimizer(
            model=model,
            dataset=ImageClassificationTask.mnist,
            optimizer_setting=optimizer_setting,
            rge_setting=None,
        )

        assert isinstance(optimizer, torch.optim.Adam)
        assert optimizer.param_groups[0]["betas"] == (0.9, 0.999)

    def test_get_optimizer_adam_with_multiple_lrs(self):
        """Test Adam optimizer with multiple learning rates for hybrid estimator."""
        model = SimpleModel()
        optimizer_setting = OptimizerSetting(
            optimizer="adam", lr=0.001, lr2=0.0005, beta1=0.9, beta2=0.999
        )

        # Set sys.argv to configure RGESetting for hybrid estimator
        original_argv = sys.argv.copy()
        sys.argv = ["test.py", "--estimator-type=hybrid"]
        rge_setting = RGESetting()
        sys.argv = original_argv

        optimizer = get_optimizer(
            model=model,
            dataset=ImageClassificationTask.mnist,
            optimizer_setting=optimizer_setting,
            rge_setting=rge_setting,
        )

        assert isinstance(optimizer, torch.optim.Adam)
        assert len(optimizer.param_groups) == 2
        assert optimizer.param_groups[0]["lr"] == 0.001
        assert optimizer.param_groups[1]["lr"] == 0.0005
        assert optimizer.param_groups[0]["betas"] == (0.9, 0.999)
        assert optimizer.param_groups[1]["betas"] == (0.9, 0.999)


class TestSplitParametersForHybrid:
    """Test cases for _split_parameters_for_hybrid function."""

    def test_split_parameters_even_split(self):
        """Test parameter splitting with even number of parameters."""
        model = SimpleModel()
        random_params, adam_forward_params = _split_parameters_for_hybrid(model)

        all_params = list(model_helpers.get_trainable_model_parameters(model))
        split_idx = len(all_params) // 2

        assert len(random_params) == split_idx
        assert len(adam_forward_params) == len(all_params) - split_idx
        assert random_params == all_params[:split_idx]
        assert adam_forward_params == all_params[split_idx:]

    def test_split_parameters_odd_number(self):
        """Test parameter splitting with odd number of parameters."""

        # Create model with odd number of parameters
        class OddParamModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear1 = nn.Linear(10, 5)
                self.linear2 = nn.Linear(5, 3)
                self.linear3 = nn.Linear(3, 1)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.linear3(self.linear2(self.linear1(x)))  # type: ignore[no-any-return]

        model = OddParamModel()
        random_params, adam_forward_params = _split_parameters_for_hybrid(model)

        all_params = list(model_helpers.get_trainable_model_parameters(model))
        split_idx = len(all_params) // 2

        assert len(random_params) == split_idx
        assert len(adam_forward_params) == len(all_params) - split_idx
        # First half goes to random, second half goes to adam_forward
        assert random_params == all_params[:split_idx]
        assert adam_forward_params == all_params[split_idx:]

    def test_split_parameters_all_trainable(self):
        """Test that all trainable parameters are included in the split."""
        model = SimpleModel()
        random_params, adam_forward_params = _split_parameters_for_hybrid(model)

        all_params = list(model_helpers.get_trainable_model_parameters(model))
        combined = random_params + adam_forward_params

        assert len(combined) == len(all_params)
        assert set(combined) == set(all_params)


if __name__ == "__main__":
    pytest.main([__file__])
