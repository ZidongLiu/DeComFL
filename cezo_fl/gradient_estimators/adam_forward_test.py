import pytest
import torch
import torch.nn as nn

from cezo_fl.gradient_estimators.adam_forward import (
    AdamForwardGradientEstimatorBatch,
    AdamForwardGradientEstimatorParamwise,
    KUpdateStrategy,
)


class SimpleModel(nn.Module):
    """Simple test model for gradient estimation testing."""

    def __init__(self, input_size: int = 10, hidden_size: int = 5, output_size: int = 1):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.linear1(x))
        return self.linear2(x)  # type: ignore[no-any-return]


def mse_loss_fn(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Simple MSE loss function for testing."""
    return torch.mean((predictions - targets) ** 2)


class TestAdamForwardGradientEstimatorBatch:
    """Test cases for AdamForwardGradientEstimatorBatch."""

    def setup_method(self):
        """Set up test fixtures."""
        self.device = torch.device("cpu")
        self.model = SimpleModel()
        self.parameters = list(self.model.parameters())
        self.batch_size = 4
        self.input_size = 10

        # Create test data
        self.batch_inputs = torch.randn(self.batch_size, self.input_size, device=self.device)
        self.batch_labels = torch.randn(self.batch_size, 1, device=self.device)

        # Initialize estimator
        self.estimator = AdamForwardGradientEstimatorBatch(
            parameters=self.parameters,
            mu=1e-3,
            num_pert=2,
            k_update_strategy=KUpdateStrategy.LAST_LOCAL_UPDATE,
            hessian_smooth=0.95,
            device=self.device,
            torch_dtype=torch.float32,
        )

    def test_initialization(self):
        """Test proper initialization of the estimator."""
        assert self.estimator.mu == 1e-3
        assert self.estimator.num_pert == 2
        assert self.estimator.k_update_strategy == KUpdateStrategy.LAST_LOCAL_UPDATE
        assert self.estimator.hessian_smooth == 0.95
        assert self.estimator.device == self.device
        assert self.estimator.torch_dtype == torch.float32

        # Check K_vec initialization
        expected_dimensions = sum(p.numel() for p in self.parameters)
        assert self.estimator.K_vec.shape == (expected_dimensions,)
        assert torch.allclose(self.estimator.K_vec, torch.ones(expected_dimensions))

        # Check parameters list
        assert len(self.estimator.parameters_list) == len(self.parameters)
        assert all(p.requires_grad for p in self.estimator.parameters_list)

    def test_generate_perturbation_norm(self):
        """Test perturbation generation with K_vec normalization."""
        rng = torch.Generator(device=self.device).manual_seed(42)
        perturbation = self.estimator.generate_perturbation_norm(rng)

        assert perturbation.shape == (self.estimator.total_dimensions,)
        assert perturbation.device == self.device
        assert perturbation.dtype == torch.float32

        # Test that perturbation is normalized by K_vec
        expected_std = 1.0 / torch.sqrt(self.estimator.K_vec)
        assert torch.allclose(torch.std(perturbation), expected_std.mean(), atol=0.1)

    def test_construct_gradient(self):
        """Test gradient construction from directional gradients."""
        seed = 42
        dir_grads = torch.tensor([0.5, -0.3], device=self.device)

        gradient = self.estimator.construct_gradient(dir_grads, seed)

        assert gradient.shape == (self.estimator.total_dimensions,)
        assert gradient.device == self.device
        assert gradient.dtype == torch.float32

    def test_compute_grad(self):
        """Test gradient computation with forward finite differences."""
        seed = 42

        # Store original parameters

        original_params = [p.clone() for p in self.parameters]

        # Compute gradient
        with torch.no_grad():
            dir_grads = self.estimator.compute_grad(
                self.batch_inputs, self.batch_labels, mse_loss_fn, seed
            )

        # Check that parameters are restored
        for orig, current in zip(original_params, self.parameters):
            assert torch.allclose(orig, current, atol=1e-6)

        # Check dir_grads
        assert dir_grads.shape == (self.estimator.num_pert,)
        assert dir_grads.device == self.device

        # Check that gradients are set
        for param in self.parameters:
            assert param.grad is not None
            assert param.grad.shape == param.shape

    def test_update_K_vec(self):
        """Test K_vec update mechanism."""
        seed = 42
        dir_grads = torch.tensor([0.5, -0.3], device=self.device)
        original_K_vec = self.estimator.K_vec.clone()

        self.estimator.update_K_vec(dir_grads, seed)

        # Check that K_vec was updated
        assert not torch.allclose(self.estimator.K_vec, original_K_vec)
        assert self.estimator.K_vec.shape == original_K_vec.shape
        assert torch.all(self.estimator.K_vec > 0)  # Should be positive

    def test_update_gradient_estimator_given_seed_and_grad(self):
        """Test gradient estimator update with different strategies."""
        iteration_seeds = [42, 43, 44]
        iteration_grad_scalar = [
            torch.tensor([0.5, -0.3], device=self.device),
            torch.tensor([0.2, 0.1], device=self.device),
            torch.tensor([-0.4, 0.6], device=self.device),
        ]

        original_K_vec = self.estimator.K_vec.clone()

        # Test LAST_LOCAL_UPDATE strategy
        self.estimator.update_gradient_estimator_given_seed_and_grad(
            iteration_seeds, iteration_grad_scalar
        )

        # K_vec should be updated only with the last gradient
        assert not torch.allclose(self.estimator.K_vec, original_K_vec)

        # Test ALL_LOCAL_UPDATES strategy
        self.estimator.k_update_strategy = KUpdateStrategy.ALL_LOCAL_UPDATES
        original_K_vec = self.estimator.K_vec.clone()

        self.estimator.update_gradient_estimator_given_seed_and_grad(
            iteration_seeds, iteration_grad_scalar
        )

        # K_vec should be updated multiple times
        assert not torch.allclose(self.estimator.K_vec, original_K_vec)

    def test_update_model_given_seed_and_grad(self):
        """Test model update using generated gradients."""
        iteration_seeds = [42, 43]
        iteration_grad_scalar = [
            torch.tensor([0.5, -0.3], device=self.device),
            torch.tensor([0.2, 0.1], device=self.device),
        ]

        # Store original parameters
        original_params = [p.clone() for p in self.parameters]

        # Create optimizer
        optimizer = torch.optim.SGD(self.parameters, lr=0.01)

        # Update model
        with torch.no_grad():
            self.estimator.update_model_given_seed_and_grad(
                optimizer, iteration_seeds, iteration_grad_scalar
            )

        # Check that parameters changed
        for orig, current in zip(original_params, self.parameters):
            assert not torch.allclose(orig, current, atol=1e-6)


class TestAdamForwardGradientEstimatorParamwise:
    """Test cases for AdamForwardGradientEstimatorParamwise."""

    def setup_method(self):
        """Set up test fixtures."""
        self.device = torch.device("cpu")
        self.model = SimpleModel()
        self.parameters = list(self.model.parameters())
        self.batch_size = 4
        self.input_size = 10

        # Create test data
        self.batch_inputs = torch.randn(self.batch_size, self.input_size, device=self.device)
        self.batch_labels = torch.randn(self.batch_size, 1, device=self.device)

        # Initialize estimator
        self.estimator = AdamForwardGradientEstimatorParamwise(
            parameters=self.parameters,
            mu=1e-3,
            num_pert=2,
            k_update_strategy=KUpdateStrategy.LAST_LOCAL_UPDATE,
            hessian_smooth=0.95,
            device=self.device,
            torch_dtype=torch.float32,
        )

    def test_initialization(self):
        """Test proper initialization of the paramwise estimator."""
        assert self.estimator.mu == 1e-3
        assert self.estimator.num_pert == 2
        assert self.estimator.k_update_strategy == KUpdateStrategy.LAST_LOCAL_UPDATE
        assert self.estimator.hessian_smooth == 0.95
        assert self.estimator.device == self.device
        assert self.estimator.torch_dtype == torch.float32

        # Check K_param_list initialization
        assert len(self.estimator.K_param_list) == len(self.parameters)
        for i, (param, k_param) in enumerate(zip(self.parameters, self.estimator.K_param_list)):
            assert k_param.shape == param.shape
            assert torch.allclose(k_param, torch.ones(param.shape))
            assert k_param.device == self.device
            assert k_param.dtype == torch.float32

        # Check parameters list
        assert len(self.estimator.parameters_list) == len(self.parameters)
        assert all(p.requires_grad for p in self.estimator.parameters_list)

    def test_perturb_model_paramwise(self):
        """Test parameterwise model perturbation."""

        original_params = [p.clone() for p in self.parameters]

        # Perturb model
        rng = torch.Generator(device=self.device).manual_seed(42)
        with torch.no_grad():
            self.estimator.perturb_model_paramwise(rng, alpha=0.1)

        # Check that parameters are restored

        # Restore model
        rng = torch.Generator(device=self.device).manual_seed(42)
        with torch.no_grad():
            self.estimator.perturb_model_paramwise(rng, alpha=-0.1)

        # Check that parameters are restored
        for orig, current in zip(original_params, self.parameters):
            assert torch.allclose(orig, current, atol=1e-6)

    def test_generate_then_put_grad_paramwise(self):
        """Test parameterwise gradient generation and setting."""
        seed = 42
        dir_grads = torch.tensor([0.5, -0.3], device=self.device)

        # Clear any existing gradients
        for param in self.parameters:
            param.grad = None

        # Generate and set gradients
        self.estimator.generate_then_put_grad_paramwise(seed, dir_grads)

        # Check that gradients are set for all parameters
        for param in self.parameters:
            assert param.grad is not None
            assert param.grad.shape == param.shape
            assert param.grad.device == self.device
            assert param.grad.dtype == torch.float32

    def test_compute_grad_paramwise(self):
        """Test parameterwise gradient computation."""
        seed = 42

        # Store original parameters
        original_params = [p.clone() for p in self.parameters]

        # Compute gradient
        with torch.no_grad():
            dir_grads = self.estimator.compute_grad(
                self.batch_inputs, self.batch_labels, mse_loss_fn, seed
            )

        # Check that parameters are restored correctly after perturbation
        for orig, current in zip(original_params, self.parameters):
            assert torch.allclose(orig.data, current.data, atol=1e-6)

        # Check dir_grads
        assert dir_grads.shape == (self.estimator.num_pert,)
        assert dir_grads.device == self.device

        # Check that gradients are set
        for param in self.parameters:
            assert param.grad is not None
            assert param.grad.shape == param.shape

    def test_sgd_no_optim_update_model(self):
        """Test SGD update without optimizer."""
        perturbation_dir_grads = torch.tensor([0.5, -0.3], device=self.device)
        seed = 42
        lr = 0.01

        # Store original parameters
        original_params = [p.clone() for p in self.parameters]

        # Update model
        with torch.no_grad():
            self.estimator.sgd_no_optim_update_model(perturbation_dir_grads, seed, lr)

        # Check that parameters changed
        for orig, current in zip(original_params, self.parameters):
            assert not torch.allclose(orig, current, atol=1e-6)

    def test_update_K_param_paramwise(self):
        """Test K_param update mechanism."""
        seed = 42
        dir_grads = torch.tensor([0.5, -0.3], device=self.device)
        original_K_params = [k.clone() for k in self.estimator.K_param_list]

        self.estimator.update_K_param_paramwise(dir_grads, seed)

        # Check that K_params were updated
        for orig, current in zip(original_K_params, self.estimator.K_param_list):
            assert not torch.allclose(orig, current)
            assert current.shape == orig.shape
            assert torch.all(current > 0)  # Should be positive

    def test_update_gradient_estimator_given_seed_and_grad_paramwise(self):
        """Test gradient estimator update with different strategies."""
        iteration_seeds = [42, 43, 44]
        iteration_grad_scalar = [
            torch.tensor([0.5, -0.3], device=self.device),
            torch.tensor([0.2, 0.1], device=self.device),
            torch.tensor([-0.4, 0.6], device=self.device),
        ]

        original_K_params = [k.clone() for k in self.estimator.K_param_list]

        # Test LAST_LOCAL_UPDATE strategy
        self.estimator.update_gradient_estimator_given_seed_and_grad(
            iteration_seeds, iteration_grad_scalar
        )

        # K_params should be updated only with the last gradient
        for orig, current in zip(original_K_params, self.estimator.K_param_list):
            assert not torch.allclose(orig, current)

        # Test ALL_LOCAL_UPDATES strategy
        self.estimator.k_update_strategy = KUpdateStrategy.ALL_LOCAL_UPDATES
        original_K_params = [k.clone() for k in self.estimator.K_param_list]

        self.estimator.update_gradient_estimator_given_seed_and_grad(
            iteration_seeds, iteration_grad_scalar
        )

        # K_params should be updated multiple times
        for orig, current in zip(original_K_params, self.estimator.K_param_list):
            assert not torch.allclose(orig, current)

    def test_update_model_given_seed_and_grad_paramwise(self):
        """Test model update using generated gradients."""
        iteration_seeds = [42, 43]
        iteration_grad_scalar = [
            torch.tensor([0.5, -0.3], device=self.device),
            torch.tensor([0.2, 0.1], device=self.device),
        ]

        # Store original parameters
        original_params = [p.clone() for p in self.parameters]

        # Create optimizer
        optimizer = torch.optim.SGD(self.parameters, lr=0.01)

        # Update model
        with torch.no_grad():
            self.estimator.update_model_given_seed_and_grad(
                optimizer, iteration_seeds, iteration_grad_scalar
            )

        # Check that parameters changed
        for orig, current in zip(original_params, self.parameters):
            assert not torch.allclose(orig, current, atol=1e-6)

    def test_memory_efficiency(self):
        """Test that paramwise approach is more memory efficient."""
        # This is a conceptual test - in practice, you'd measure actual memory usage
        # For now, we just verify that K_param_list has the right structure

        # Check that each K_param has the same shape as its parameter
        for param, k_param in zip(self.parameters, self.estimator.K_param_list):
            assert k_param.shape == param.shape
            assert k_param.numel() == param.numel()

        # Verify that we don't have one large tensor
        total_elements = sum(k.numel() for k in self.estimator.K_param_list)
        assert total_elements == self.estimator.total_dimensions


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_parameters(self):
        """Test behavior with empty parameter list."""
        empty_params = []

        # The estimator should handle empty parameters gracefully
        estimator = AdamForwardGradientEstimatorBatch(empty_params)
        assert estimator.total_dimensions == 0
        assert len(estimator.parameters_list) == 0

    def test_single_parameter(self):
        """Test with single parameter."""
        device = torch.device("cpu")
        param = nn.Parameter(torch.randn(5, 3))
        estimator = AdamForwardGradientEstimatorBatch([param], device=device)

        assert estimator.total_dimensions == 15
        assert estimator.K_vec.shape == (15,)

    def test_different_parameter_shapes(self):
        """Test with parameters of different shapes."""
        device = torch.device("cpu")
        params = [
            nn.Parameter(torch.randn(10, 5)),  # 2D
            nn.Parameter(torch.randn(3)),  # 1D
            nn.Parameter(torch.randn(2, 2, 2)),  # 3D
        ]

        estimator = AdamForwardGradientEstimatorParamwise(params, device=device)

        assert len(estimator.K_param_list) == 3
        assert estimator.K_param_list[0].shape == (10, 5)
        assert estimator.K_param_list[1].shape == (3,)
        assert estimator.K_param_list[2].shape == (2, 2, 2)

    def test_different_devices(self):
        """Test with different devices."""
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
            model = SimpleModel()
            params = list(model.parameters())

            estimator = AdamForwardGradientEstimatorBatch(params, device=device)
            assert str(estimator.device) == str(device)
            assert estimator.K_vec.device == device

            estimator_paramwise = AdamForwardGradientEstimatorParamwise(params, device=device)
            assert str(estimator_paramwise.device) == str(device)
            for k_param in estimator_paramwise.K_param_list:
                assert k_param.device == device


if __name__ == "__main__":
    pytest.main([__file__])
