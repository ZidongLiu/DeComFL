import pytest
import torch
import torch.nn as nn

from cezo_fl.gradient_estimators.hybrid_gradient_estimator import (
    HybridGradientEstimatorBatch,
    HybridGradientEstimatorParamwise,
)
from cezo_fl.gradient_estimators.adam_forward import KUpdateStrategy


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


class TestHybridGradientEstimatorBatch:
    """Test cases for HybridGradientEstimatorBatch."""

    def setup_method(self):
        """Set up test fixtures."""
        self.device = torch.device("cpu")
        self.model = SimpleModel()
        self.all_parameters = list(self.model.parameters())
        # Split parameters: first layer for random, second layer for adam_forward
        self.random_parameters = [self.all_parameters[0], self.all_parameters[1]]
        self.adam_forward_parameters = [self.all_parameters[2], self.all_parameters[3]]
        self.batch_size = 4
        self.input_size = 10

        # Create test data
        self.batch_inputs = torch.randn(self.batch_size, self.input_size, device=self.device)
        self.batch_labels = torch.randn(self.batch_size, 1, device=self.device)

        # Initialize estimator
        self.estimator = HybridGradientEstimatorBatch(
            random_parameters_list=self.random_parameters,
            adam_forward_parameters_list=self.adam_forward_parameters,
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

        # Check dimensions
        expected_random_dimensions = sum(p.numel() for p in self.random_parameters)
        expected_adam_forward_dimensions = sum(p.numel() for p in self.adam_forward_parameters)
        assert self.estimator.random_gradient_dimensions == expected_random_dimensions
        assert self.estimator.adam_forward_dimensions == expected_adam_forward_dimensions
        assert (
            self.estimator.total_dimensions
            == expected_random_dimensions + expected_adam_forward_dimensions
        )

        # Check K_vec initialization (only for adam_forward parameters)
        assert self.estimator.K_vec.shape == (expected_adam_forward_dimensions,)
        assert torch.allclose(
            self.estimator.K_vec, torch.ones(expected_adam_forward_dimensions)
        )

        # Check parameters lists
        assert len(self.estimator.random_parameters_list) == len(self.random_parameters)
        assert len(self.estimator.adam_forward_parameters_list) == len(self.adam_forward_parameters)
        assert all(p.requires_grad for p in self.estimator.random_parameters_list)
        assert all(p.requires_grad for p in self.estimator.adam_forward_parameters_list)

    def test_generate_perturbation_norm(self):
        """Test perturbation generation with hybrid approach."""
        rng = torch.Generator(device=self.device).manual_seed(42)
        perturbation = self.estimator.generate_perturbation_norm(rng)

        assert perturbation.shape == (self.estimator.total_dimensions,)
        assert perturbation.device == self.device
        assert perturbation.dtype == torch.float32

        # Split perturbation into random and adam_forward parts
        random_part = perturbation[: self.estimator.random_gradient_dimensions]
        adam_forward_part = perturbation[self.estimator.random_gradient_dimensions :]

        # Random part should be standard normal
        assert random_part.shape == (self.estimator.random_gradient_dimensions,)

        # Adam forward part should be normalized by K_vec
        assert adam_forward_part.shape == (self.estimator.adam_forward_dimensions,)

    def test_construct_gradient(self):
        """Test gradient construction from directional gradients."""
        seed = 42
        dir_grads = torch.tensor([0.5, -0.3], device=self.device)

        gradient = self.estimator.construct_gradient(dir_grads, seed)

        assert gradient.shape == (self.estimator.total_dimensions,)
        assert gradient.device == self.device
        assert gradient.dtype == torch.float32

    def test_put_grad(self):
        """Test putting gradients into parameters."""
        seed = 42
        dir_grads = torch.tensor([0.5, -0.3], device=self.device)
        gradient = self.estimator.construct_gradient(dir_grads, seed)

        # Clear any existing gradients
        for param in self.random_parameters + self.adam_forward_parameters:
            param.grad = None

        self.estimator.put_grad(gradient)

        # Check that gradients are set for all parameters
        for param in self.random_parameters + self.adam_forward_parameters:
            assert param.grad is not None
            assert param.grad.shape == param.shape

    def test_compute_grad(self):
        """Test gradient computation with forward finite differences."""
        seed = 42

        # Store original parameters
        original_params = [p.clone() for p in self.all_parameters]

        # Compute gradient
        with torch.no_grad():
            dir_grads = self.estimator.compute_grad(
                self.batch_inputs, self.batch_labels, mse_loss_fn, seed
            )

        # Check that parameters are restored
        for orig, current in zip(original_params, self.all_parameters):
            assert torch.allclose(orig, current, atol=1e-6)

        # Check dir_grads
        assert dir_grads.shape == (self.estimator.num_pert,)
        assert dir_grads.device == self.device

        # Check that gradients are set
        for param in self.random_parameters + self.adam_forward_parameters:
            assert param.grad is not None
            assert param.grad.shape == param.shape

    def test_update_K_vec(self):
        """Test K_vec update mechanism (only for adam_forward parameters)."""
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
        original_params = [p.clone() for p in self.all_parameters]

        # Create optimizer
        optimizer = torch.optim.SGD(self.all_parameters, lr=0.01)

        # Update model
        with torch.no_grad():
            self.estimator.update_model_given_seed_and_grad(
                optimizer, iteration_seeds, iteration_grad_scalar
            )

        # Check that parameters changed
        for orig, current in zip(original_params, self.all_parameters):
            assert not torch.allclose(orig, current, atol=1e-6)


class TestHybridGradientEstimatorParamwise:
    """Test cases for HybridGradientEstimatorParamwise."""

    def setup_method(self):
        """Set up test fixtures."""
        self.device = torch.device("cpu")
        self.model = SimpleModel()
        self.all_parameters = list(self.model.parameters())
        # Split parameters: first layer for random, second layer for adam_forward
        self.random_parameters = [self.all_parameters[0], self.all_parameters[1]]
        self.adam_forward_parameters = [self.all_parameters[2], self.all_parameters[3]]
        self.batch_size = 4
        self.input_size = 10

        # Create test data
        self.batch_inputs = torch.randn(self.batch_size, self.input_size, device=self.device)
        self.batch_labels = torch.randn(self.batch_size, 1, device=self.device)

        # Initialize estimator
        self.estimator = HybridGradientEstimatorParamwise(
            random_parameters_list=self.random_parameters,
            adam_forward_parameters_list=self.adam_forward_parameters,
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

        # Check dimensions
        expected_random_dimensions = sum(p.numel() for p in self.random_parameters)
        expected_adam_forward_dimensions = sum(p.numel() for p in self.adam_forward_parameters)
        assert self.estimator.random_gradient_dimensions == expected_random_dimensions
        assert self.estimator.adam_forward_dimensions == expected_adam_forward_dimensions
        assert (
            self.estimator.total_dimensions
            == expected_random_dimensions + expected_adam_forward_dimensions
        )

        # Check K_param_list initialization (only for adam_forward parameters)
        assert len(self.estimator.K_param_list) == len(self.adam_forward_parameters)
        for i, (param, k_param) in enumerate(
            zip(self.adam_forward_parameters, self.estimator.K_param_list)
        ):
            assert k_param.shape == param.shape
            assert torch.allclose(k_param, torch.ones(param.shape))
            assert k_param.device == self.device
            assert k_param.dtype == torch.float32

        # Check parameters list
        assert len(self.estimator.parameters_list) == len(self.all_parameters)
        assert all(p.requires_grad for p in self.estimator.parameters_list)
        assert self.estimator.random_parameters_count == len(self.random_parameters)

    def test_generate_perturbation_norm_paramwise(self):
        """Test parameterwise perturbation generation."""
        rng = torch.Generator(device=self.device).manual_seed(42)

        # Test random parameter (should be standard normal)
        random_idx = 0
        random_perturb = self.estimator.generate_perturbation_norm_paramwise(random_idx, rng)
        assert random_perturb.shape == self.random_parameters[0].shape
        assert random_perturb.device == self.device
        assert random_perturb.dtype == torch.float32

        # Test adam_forward parameter (should be normalized by K)
        adam_forward_idx = len(self.random_parameters)
        rng = torch.Generator(device=self.device).manual_seed(42)
        adam_forward_perturb = self.estimator.generate_perturbation_norm_paramwise(
            adam_forward_idx, rng
        )
        assert adam_forward_perturb.shape == self.adam_forward_parameters[0].shape
        assert adam_forward_perturb.device == self.device
        assert adam_forward_perturb.dtype == torch.float32

    def test_perturb_model_paramwise(self):
        """Test parameterwise model perturbation."""
        original_params = [p.clone() for p in self.all_parameters]

        # Perturb model
        rng = torch.Generator(device=self.device).manual_seed(42)
        with torch.no_grad():
            self.estimator.perturb_model_paramwise(rng, alpha=0.1)

        # Restore model
        rng = torch.Generator(device=self.device).manual_seed(42)
        with torch.no_grad():
            self.estimator.perturb_model_paramwise(rng, alpha=-0.1)

        # Check that parameters are restored
        for orig, current in zip(original_params, self.all_parameters):
            assert torch.allclose(orig, current, atol=1e-6)

    def test_generate_then_put_grad_paramwise(self):
        """Test parameterwise gradient generation and setting."""
        seed = 42
        dir_grads = torch.tensor([0.5, -0.3], device=self.device)

        # Clear any existing gradients
        for param in self.all_parameters:
            param.grad = None

        # Generate and set gradients
        self.estimator.generate_then_put_grad_paramwise(seed, dir_grads)

        # Check that gradients are set for all parameters
        for param in self.all_parameters:
            assert param.grad is not None
            assert param.grad.shape == param.shape
            assert param.grad.device == self.device
            assert param.grad.dtype == torch.float32

    def test_compute_grad_paramwise(self):
        """Test parameterwise gradient computation."""
        seed = 42

        # Store original parameters
        original_params = [p.clone() for p in self.all_parameters]

        # Compute gradient
        with torch.no_grad():
            dir_grads = self.estimator.compute_grad(
                self.batch_inputs, self.batch_labels, mse_loss_fn, seed
            )

        # Check that parameters are restored correctly after perturbation
        for orig, current in zip(original_params, self.all_parameters):
            assert torch.allclose(orig.data, current.data, atol=1e-6)

        # Check dir_grads
        assert dir_grads.shape == (self.estimator.num_pert,)
        assert dir_grads.device == self.device

        # Check that gradients are set
        for param in self.all_parameters:
            assert param.grad is not None
            assert param.grad.shape == param.shape

    def test_update_K_param_paramwise(self):
        """Test K_param update mechanism (only for adam_forward parameters)."""
        seed = 42
        dir_grads = torch.tensor([0.5, -0.3], device=self.device)
        original_K_params = [k.clone() for k in self.estimator.K_param_list]

        self.estimator.update_K_param_paramwise(dir_grads, seed)

        # Check that K_params were updated (only for adam_forward parameters)
        assert len(self.estimator.K_param_list) == len(self.adam_forward_parameters)
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
        original_params = [p.clone() for p in self.all_parameters]

        # Create optimizer
        optimizer = torch.optim.SGD(self.all_parameters, lr=0.01)

        # Update model
        with torch.no_grad():
            self.estimator.update_model_given_seed_and_grad(
                optimizer, iteration_seeds, iteration_grad_scalar
            )

        # Check that parameters changed
        for orig, current in zip(original_params, self.all_parameters):
            assert not torch.allclose(orig, current, atol=1e-6)

    def test_memory_efficiency(self):
        """Test that paramwise approach maintains proper structure."""
        # Check that each K_param has the same shape as its corresponding adam_forward parameter
        for param, k_param in zip(self.adam_forward_parameters, self.estimator.K_param_list):
            assert k_param.shape == param.shape
            assert k_param.numel() == param.numel()

        # Verify that we only have K_params for adam_forward parameters
        total_k_elements = sum(k.numel() for k in self.estimator.K_param_list)
        assert total_k_elements == self.estimator.adam_forward_dimensions


class TestHybridEdgeCases:
    """Test edge cases and error handling for hybrid estimators."""

    def test_empty_random_parameters(self):
        """Test behavior with empty random parameter list."""
        device = torch.device("cpu")
        model = SimpleModel()
        all_params = list(model.parameters())
        # All parameters go to adam_forward
        random_params = []
        adam_forward_params = all_params

        estimator = HybridGradientEstimatorBatch(
            random_parameters_list=random_params,
            adam_forward_parameters_list=adam_forward_params,
            device=device,
        )

        assert estimator.random_gradient_dimensions == 0
        assert estimator.adam_forward_dimensions == sum(p.numel() for p in adam_forward_params)
        assert len(estimator.random_parameters_list) == 0
        assert len(estimator.adam_forward_parameters_list) == len(adam_forward_params)

    def test_empty_adam_forward_parameters(self):
        """Test behavior with empty adam_forward parameter list."""
        device = torch.device("cpu")
        model = SimpleModel()
        all_params = list(model.parameters())
        # All parameters go to random
        random_params = all_params
        adam_forward_params = []

        estimator = HybridGradientEstimatorBatch(
            random_parameters_list=random_params,
            adam_forward_parameters_list=adam_forward_params,
            device=device,
        )

        assert estimator.random_gradient_dimensions == sum(p.numel() for p in random_params)
        assert estimator.adam_forward_dimensions == 0
        assert len(estimator.random_parameters_list) == len(random_params)
        assert len(estimator.adam_forward_parameters_list) == 0
        assert estimator.K_vec.shape == (0,)

    def test_single_parameter_each_type(self):
        """Test with single parameter in each list."""
        device = torch.device("cpu")
        model = SimpleModel()
        all_params = list(model.parameters())
        random_params = [all_params[0]]
        adam_forward_params = [all_params[2]]

        estimator = HybridGradientEstimatorBatch(
            random_parameters_list=random_params,
            adam_forward_parameters_list=adam_forward_params,
            device=device,
        )

        assert estimator.random_gradient_dimensions == random_params[0].numel()
        assert estimator.adam_forward_dimensions == adam_forward_params[0].numel()
        assert estimator.K_vec.shape == (adam_forward_params[0].numel(),)

    def test_different_parameter_shapes_paramwise(self):
        """Test paramwise with parameters of different shapes."""
        device = torch.device("cpu")
        random_params = [
            nn.Parameter(torch.randn(10, 5)),  # 2D
            nn.Parameter(torch.randn(3)),  # 1D
        ]
        adam_forward_params = [
            nn.Parameter(torch.randn(2, 2, 2)),  # 3D
        ]

        estimator = HybridGradientEstimatorParamwise(
            random_parameters_list=random_params,
            adam_forward_parameters_list=adam_forward_params,
            device=device,
        )

        assert len(estimator.K_param_list) == 1
        assert estimator.K_param_list[0].shape == (2, 2, 2)
        assert len(estimator.parameters_list) == 3

    def test_different_devices(self):
        """Test with different devices."""
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
            model = SimpleModel()
            all_params = list(model.parameters())
            random_params = [all_params[0], all_params[1]]
            adam_forward_params = [all_params[2], all_params[3]]

            estimator_batch = HybridGradientEstimatorBatch(
                random_parameters_list=random_params,
                adam_forward_parameters_list=adam_forward_params,
                device=device,
            )
            assert str(estimator_batch.device) == str(device)
            assert estimator_batch.K_vec.device == device

            estimator_paramwise = HybridGradientEstimatorParamwise(
                random_parameters_list=random_params,
                adam_forward_parameters_list=adam_forward_params,
                device=device,
            )
            assert str(estimator_paramwise.device) == str(device)
            for k_param in estimator_paramwise.K_param_list:
                assert k_param.device == device

    def test_hybrid_perturbation_structure(self):
        """Test that hybrid perturbation correctly combines random and adam_forward parts."""
        device = torch.device("cpu")
        model = SimpleModel()
        all_params = list(model.parameters())
        random_params = [all_params[0]]
        adam_forward_params = [all_params[2]]

        estimator = HybridGradientEstimatorBatch(
            random_parameters_list=random_params,
            adam_forward_parameters_list=adam_forward_params,
            device=device,
        )

        rng = torch.Generator(device=device).manual_seed(42)
        perturbation = estimator.generate_perturbation_norm(rng)

        # Verify structure: [random_part, adam_forward_part]
        random_part = perturbation[: estimator.random_gradient_dimensions]
        adam_forward_part = perturbation[estimator.random_gradient_dimensions :]

        assert random_part.shape == (random_params[0].numel(),)
        assert adam_forward_part.shape == (adam_forward_params[0].numel(),)
        assert (
            random_part.shape[0] + adam_forward_part.shape[0] == estimator.total_dimensions
        )


if __name__ == "__main__":
    pytest.main([__file__])

