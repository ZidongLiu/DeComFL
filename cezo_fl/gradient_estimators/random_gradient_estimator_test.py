import pytest
import torch
from torch import nn

from cezo_fl.gradient_estimators import random_gradient_estimator


class LinearModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(3, 2)
        self.linear2 = nn.Linear(2, 1)

    def forward(self, x):
        x = nn.functional.relu(self.linear1(x))
        return self.linear2(x)


@pytest.mark.parametrize(
    "rge_method",
    [
        random_gradient_estimator.RandomGradEstimateMethod.rge_forward,
        random_gradient_estimator.RandomGradEstimateMethod.rge_central,
    ],
)
@pytest.mark.parametrize("num_pert", [2, 4, 5])
def test_parameter_wise_equivalent_all_togther(
    rge_method: random_gradient_estimator.RandomGradEstimateMethod, num_pert: int
) -> None:
    """
    NOTE: Do not extend this test for large model. This test only works when model is small.
    To be specific, works number of parameters <= 10.
    """
    fake_input = torch.randn(5, 3)
    fake_label = torch.randn(5, 1)
    criterion = nn.MSELoss()

    torch.random.manual_seed(123)  # Make sure all models are generated as the same.
    model1 = LinearModel()
    rge1 = random_gradient_estimator.RandomGradientEstimator(
        model1.parameters(),
        num_pert=num_pert,
        grad_estimate_method=rge_method,
        paramwise_perturb=False,
    )
    with torch.no_grad():
        dir_grads1 = rge1.compute_grad(
            fake_input, fake_label, lambda x, y: criterion(model1(x), y), seed=54321
        )

    torch.random.manual_seed(123)  # Make sure all models are generated as the same.
    model2 = LinearModel()
    rge2 = random_gradient_estimator.RandomGradientEstimator(
        model2.parameters(),
        num_pert=num_pert,
        grad_estimate_method=rge_method,
        paramwise_perturb=True,
    )
    with torch.no_grad():
        dir_grads2 = rge2.compute_grad(
            fake_input, fake_label, lambda x, y: criterion(model2(x), y), seed=54321
        )

    torch.testing.assert_close(dir_grads1, dir_grads2)
    for p1, p2 in zip(model1.parameters(), model2.parameters()):
        torch.testing.assert_close(p1.grad, p2.grad)


def test_update_model_given_seed_and_grad():
    # Make the update second times and the output suppose to be the same.
    ouputs = []
    for _ in range(2):
        torch.manual_seed(0)
        fake_model = torch.nn.Sequential(
            torch.nn.Linear(10, 5),
            torch.nn.ReLU(),
            torch.nn.Linear(5, 2),
        )

        optim = torch.optim.SGD(fake_model.parameters(), lr=1e-3)
        rge = random_gradient_estimator.RandomGradientEstimator(
            fake_model.parameters(),
            num_pert=2,
            paramwise_perturb=False,
        )
        rge.update_model_given_seed_and_grad(
            optim,
            iteration_seeds=[1, 2, 3],
            iteration_grad_scalar=[  # two perturbations
                torch.tensor([0.1, 0.2]),
                torch.tensor([0.3, 0.4]),
                torch.tensor([0.5, 0.6]),
            ],
        )
        ouputs.append(
            fake_model(torch.tensor([list(range(i, 10 + i)) for i in range(3)], dtype=torch.float))
        )
    assert ouputs[0].shape == (3, 2)
    assert ouputs[1].shape == (3, 2)
    torch.testing.assert_close(ouputs[0], ouputs[1])


class TestParamwisePerturb:
    """Test suite for paramwise_perturb functionality."""

    def test_paramwise_perturb_compute_grad(self) -> None:
        """Test that paramwise_perturb=True works correctly in compute_grad."""
        torch.manual_seed(42)
        model = LinearModel()
        fake_input = torch.randn(3, 3)
        fake_label = torch.randn(3, 1)
        criterion = nn.MSELoss()

        rge = random_gradient_estimator.RandomGradientEstimator(
            model.parameters(),
            num_pert=3,
            paramwise_perturb=True,
            grad_estimate_method=random_gradient_estimator.RandomGradEstimateMethod.rge_central,
        )

        with torch.no_grad():
            dir_grads = rge.compute_grad(
                fake_input, fake_label, lambda x, y: criterion(model(x), y), seed=12345
            )

        # Check that dir_grads has the correct shape (num_pert)
        assert dir_grads.shape == (3,)
        assert isinstance(dir_grads, torch.Tensor)

        # Check that gradients are set on parameters
        for param in model.parameters():
            assert param.grad is not None
            assert param.grad.shape == param.shape

    def test_paramwise_perturb_generate_then_put_grad_paramwise(self) -> None:
        """Test the generate_then_put_grad_paramwise method."""
        torch.manual_seed(42)
        model = LinearModel()
        rge = random_gradient_estimator.RandomGradientEstimator(
            model.parameters(),
            num_pert=2,
            paramwise_perturb=True,
        )

        with torch.no_grad():
            # Clear any existing gradients
            for param in model.parameters():
                param.grad = None

            # Test generate_then_put_grad_paramwise
            dir_grads = torch.tensor([0.1, 0.2], device=rge.device)
            rge.generate_then_put_grad_paramwise(seed=12345, dir_grads=dir_grads)

            # Check that gradients are set correctly
            for param in model.parameters():
                assert param.grad is not None
                assert param.grad.shape == param.shape

    def test_paramwise_perturb_perturb_model_paramwise(self) -> None:
        """Test the perturb_model_paramwise method."""
        torch.manual_seed(42)
        model = LinearModel()
        rge = random_gradient_estimator.RandomGradientEstimator(
            model.parameters(),
            paramwise_perturb=True,
        )

        with torch.no_grad():
            # Store original parameters
            original_params = [param.clone() for param in model.parameters()]

            # Generate RNG and perturb
            rng = rge.get_rng(seed=12345, perturb_index=0)
            rge.perturb_model_paramwise(rng, alpha=0.1)

            # Check that parameters have changed
            for orig_param, new_param in zip(original_params, model.parameters()):
                assert not torch.allclose(orig_param, new_param)

    def test_paramwise_perturb_zo_grad_estimate_paramwise(self) -> None:
        """Test the _zo_grad_estimate_paramwise method."""
        torch.manual_seed(42)
        model = LinearModel()
        fake_input = torch.randn(2, 3)
        fake_label = torch.randn(2, 1)
        criterion = nn.MSELoss()

        rge = random_gradient_estimator.RandomGradientEstimator(
            model.parameters(),
            num_pert=2,
            paramwise_perturb=True,
            grad_estimate_method=random_gradient_estimator.RandomGradEstimateMethod.rge_forward,
        )

        with torch.no_grad():
            dir_grads = rge._zo_grad_estimate_paramwise(
                fake_input, fake_label, lambda x, y: criterion(model(x), y), seed=12345
            )

        # Check output shape and type
        assert dir_grads.shape == (2,)
        assert isinstance(dir_grads, torch.Tensor)

    def test_paramwise_perturb_validation(self) -> None:
        """Test that paramwise_perturb=True requires normalize_perturbation=False."""
        with pytest.raises(AssertionError):
            random_gradient_estimator.RandomGradientEstimator(
                LinearModel().parameters(),
                paramwise_perturb=True,
                normalize_perturbation=True,
            )


class TestSgdOnlyNoOptim:
    """Test suite for sgd_only_no_optim functionality."""

    def test_sgd_only_no_optim_validation(self) -> None:
        """Test that sgd_only_no_optim=True requires paramwise_perturb=True."""
        with pytest.raises(AssertionError):
            random_gradient_estimator.RandomGradientEstimator(
                LinearModel().parameters(),
                sgd_only_no_optim=True,
                paramwise_perturb=False,
            )

    def test_sgd_only_no_optim_update_model(self) -> None:
        """Test the sgd_no_optim_update_model method."""
        torch.manual_seed(42)
        model = LinearModel()
        rge = random_gradient_estimator.RandomGradientEstimator(
            model.parameters(),
            sgd_only_no_optim=True,
            paramwise_perturb=True,
        )

        with torch.no_grad():
            # Store original parameters
            original_params = [param.clone() for param in model.parameters()]

            # Test sgd_no_optim_update_model
            perturbation_dir_grads = torch.tensor([0.1, 0.2], device=rge.device)
            lr = 0.01
            rge.sgd_no_optim_update_model(perturbation_dir_grads, seed=12345, lr=lr)

            # Check that parameters have changed
            for orig_param, new_param in zip(original_params, model.parameters()):
                assert not torch.allclose(orig_param, new_param)

    def test_sgd_only_no_optim_update_model_given_seed_and_grad(self) -> None:
        """Test update_model_given_seed_and_grad with sgd_only_no_optim=True."""
        torch.manual_seed(42)
        model = LinearModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        rge = random_gradient_estimator.RandomGradientEstimator(
            model.parameters(),
            sgd_only_no_optim=True,
            paramwise_perturb=True,
        )

        with torch.no_grad():
            # Store original parameters
            original_params = [param.clone() for param in model.parameters()]

            # Test update
            iteration_seeds = [12345, 54321]
            iteration_grad_scalar = [
                torch.tensor([0.1, 0.2], device=rge.device),
                torch.tensor([0.3, 0.4], device=rge.device),
            ]

            rge.update_model_given_seed_and_grad(optimizer, iteration_seeds, iteration_grad_scalar)

            # Check that parameters have changed
            for orig_param, new_param in zip(original_params, model.parameters()):
                assert not torch.allclose(orig_param, new_param)

    def test_sgd_only_no_optim_revert_model_given_seed_and_grad(self) -> None:
        """Test that revert_model_given_seed_and_grad raises exception for sgd_only_no_optim."""
        torch.manual_seed(42)
        model = LinearModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        rge = random_gradient_estimator.RandomGradientEstimator(
            model.parameters(),
            sgd_only_no_optim=True,
            paramwise_perturb=True,
        )

        with torch.no_grad():
            iteration_seeds = [12345]
            iteration_grad_scalar = [torch.tensor([0.1, 0.2], device=rge.device)]

            with pytest.raises(AssertionError):
                rge.revert_model_given_seed_and_grad(
                    optimizer, iteration_seeds, iteration_grad_scalar
                )


class TestEdgeCasesAndValidation:
    """Test suite for edge cases and validation logic."""

    def test_empty_parameters_list(self) -> None:
        """Test behavior with empty parameters list."""
        with torch.no_grad():
            # Create a model with no trainable parameters
            model = nn.Sequential(nn.ReLU())  # ReLU has no parameters

            rge = random_gradient_estimator.RandomGradientEstimator(
                model.parameters(),
                paramwise_perturb=True,
            )

            assert len(rge.parameters_list) == 0
            assert rge.total_dimensions == 0

    def test_single_parameter_model(self) -> None:
        """Test with a model that has only one parameter."""
        torch.manual_seed(42)
        model = nn.Linear(3, 1)
        fake_input = torch.randn(2, 3)
        fake_label = torch.randn(2, 1)
        criterion = nn.MSELoss()

        rge = random_gradient_estimator.RandomGradientEstimator(
            model.parameters(),
            num_pert=1,
            paramwise_perturb=True,
        )

        with torch.no_grad():
            dir_grads = rge.compute_grad(
                fake_input, fake_label, lambda x, y: criterion(model(x), y), seed=12345
            )

            assert dir_grads.shape == (1,)
            assert model.weight.grad is not None
            assert model.bias.grad is not None

    def test_large_num_pert(self) -> None:
        """Test with a large number of perturbations."""
        torch.manual_seed(42)
        model = LinearModel()
        fake_input = torch.randn(2, 3)
        fake_label = torch.randn(2, 1)
        criterion = nn.MSELoss()

        rge = random_gradient_estimator.RandomGradientEstimator(
            model.parameters(),
            num_pert=10,
            paramwise_perturb=True,
        )

        with torch.no_grad():
            dir_grads = rge.compute_grad(
                fake_input, fake_label, lambda x, y: criterion(model(x), y), seed=12345
            )

            assert dir_grads.shape == (10,)

    def test_different_mu_values(self) -> None:
        """Test with different mu (perturbation size) values."""
        torch.manual_seed(42)
        model = LinearModel()
        fake_input = torch.randn(2, 3)
        fake_label = torch.randn(2, 1)
        criterion = nn.MSELoss()

        with torch.no_grad():
            for mu in [1e-4, 1e-3, 1e-2]:
                rge = random_gradient_estimator.RandomGradientEstimator(
                    model.parameters(),
                    mu=mu,
                    num_pert=2,
                    paramwise_perturb=True,
                )

                dir_grads = rge.compute_grad(
                    fake_input, fake_label, lambda x, y: criterion(model(x), y), seed=12345
                )

                assert dir_grads.shape == (2,)

    def test_different_torch_dtypes(self) -> None:
        """Test with different torch dtypes."""
        torch.manual_seed(42)
        model = LinearModel()

        with torch.no_grad():
            for dtype in [torch.float32, torch.float64]:
                rge = random_gradient_estimator.RandomGradientEstimator(
                    model.parameters(),
                    torch_dtype=dtype,
                    paramwise_perturb=True,
                )

                assert rge.torch_dtype == dtype

    def test_device_consistency(self) -> None:
        """Test device consistency across operations."""
        if torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"

        torch.manual_seed(42)
        model = LinearModel().to(device)

        with torch.no_grad():
            rge = random_gradient_estimator.RandomGradientEstimator(
                model.parameters(),
                device=device,
                paramwise_perturb=True,
            )

            assert rge.device == device

            # Test that generated tensors are on the correct device
            rng = rge.get_rng(seed=12345, perturb_index=0)
            perturbation = rge.generate_perturbation_norm(rng)
            assert perturbation.device.type == device.split(":")[0]  # Handle cuda:0 vs cuda


class TestReproducibility:
    """Test suite for reproducibility and deterministic behavior."""

    def test_paramwise_perturb_reproducibility(self) -> None:
        """Test that paramwise_perturb results are reproducible with same seed."""
        torch.manual_seed(42)
        model1 = LinearModel()
        fake_input = torch.randn(3, 3)
        fake_label = torch.randn(3, 1)
        criterion = nn.MSELoss()

        rge1 = random_gradient_estimator.RandomGradientEstimator(
            model1.parameters(),
            num_pert=3,
            paramwise_perturb=True,
            grad_estimate_method=random_gradient_estimator.RandomGradEstimateMethod.rge_central,
        )

        with torch.no_grad():
            dir_grads1 = rge1.compute_grad(
                fake_input, fake_label, lambda x, y: criterion(model1(x), y), seed=12345
            )

        # Run again with same seed
        torch.manual_seed(42)
        model2 = LinearModel()
        rge2 = random_gradient_estimator.RandomGradientEstimator(
            model2.parameters(),
            num_pert=3,
            paramwise_perturb=True,
            grad_estimate_method=random_gradient_estimator.RandomGradEstimateMethod.rge_central,
        )

        with torch.no_grad():
            dir_grads2 = rge2.compute_grad(
                fake_input, fake_label, lambda x, y: criterion(model2(x), y), seed=12345
            )

        # Results should be identical
        torch.testing.assert_close(dir_grads1, dir_grads2)
        for p1, p2 in zip(model1.parameters(), model2.parameters()):
            torch.testing.assert_close(p1.grad, p2.grad)

    def test_sgd_only_no_optim_reproducibility(self) -> None:
        """Test that sgd_only_no_optim results are reproducible with same seed."""
        torch.manual_seed(42)
        model1 = LinearModel()
        rge1 = random_gradient_estimator.RandomGradientEstimator(
            model1.parameters(),
            sgd_only_no_optim=True,
            paramwise_perturb=True,
        )

        with torch.no_grad():
            # Test sgd_no_optim_update_model
            perturbation_dir_grads = torch.tensor([0.1, 0.2], device=rge1.device)
            lr = 0.01
            rge1.sgd_no_optim_update_model(perturbation_dir_grads, seed=12345, lr=lr)

        # Run again with same seed
        torch.manual_seed(42)
        model2 = LinearModel()
        rge2 = random_gradient_estimator.RandomGradientEstimator(
            model2.parameters(),
            sgd_only_no_optim=True,
            paramwise_perturb=True,
        )

        with torch.no_grad():
            # Test sgd_no_optim_update_model
            perturbation_dir_grads = torch.tensor([0.1, 0.2], device=rge2.device)
            lr = 0.01
            rge2.sgd_no_optim_update_model(perturbation_dir_grads, seed=12345, lr=lr)

        # Results should be identical
        for p1, p2 in zip(model1.parameters(), model2.parameters()):
            torch.testing.assert_close(p1, p2)

    def test_different_seeds_produce_different_results(self) -> None:
        """Test that different seeds produce different results."""
        torch.manual_seed(42)
        model1 = LinearModel()
        fake_input = torch.randn(2, 3)
        fake_label = torch.randn(2, 1)
        criterion = nn.MSELoss()

        rge1 = random_gradient_estimator.RandomGradientEstimator(
            model1.parameters(),
            num_pert=2,
            paramwise_perturb=True,
        )

        with torch.no_grad():
            dir_grads1 = rge1.compute_grad(
                fake_input, fake_label, lambda x, y: criterion(model1(x), y), seed=12345
            )

        # Run with different seed
        torch.manual_seed(42)
        model2 = LinearModel()
        rge2 = random_gradient_estimator.RandomGradientEstimator(
            model2.parameters(),
            num_pert=2,
            paramwise_perturb=True,
        )

        with torch.no_grad():
            dir_grads2 = rge2.compute_grad(
                fake_input, fake_label, lambda x, y: criterion(model2(x), y), seed=54321
            )

        # Results should be different
        assert not torch.allclose(dir_grads1, dir_grads2, atol=1e-6)

    def test_perturbation_generation_reproducibility(self) -> None:
        """Test that perturbation generation is reproducible with same seed."""
        torch.manual_seed(42)
        model = LinearModel()
        rge = random_gradient_estimator.RandomGradientEstimator(
            model.parameters(),
            paramwise_perturb=True,
        )

        with torch.no_grad():
            # Generate perturbations with same seed
            rng1 = rge.get_rng(seed=12345, perturb_index=0)
            perturbation1 = rge.generate_perturbation_norm(rng1)

            rng2 = rge.get_rng(seed=12345, perturb_index=0)
            perturbation2 = rge.generate_perturbation_norm(rng2)

            # Should be identical
            torch.testing.assert_close(perturbation1, perturbation2)

    def test_perturbation_generation_different_indices(self) -> None:
        """Test that different perturbation indices produce different perturbations."""
        torch.manual_seed(42)
        model = LinearModel()
        rge = random_gradient_estimator.RandomGradientEstimator(
            model.parameters(),
            paramwise_perturb=True,
        )

        with torch.no_grad():
            # Generate perturbations with different indices
            rng1 = rge.get_rng(seed=12345, perturb_index=0)
            perturbation1 = rge.generate_perturbation_norm(rng1)

            rng2 = rge.get_rng(seed=12345, perturb_index=1)
            perturbation2 = rge.generate_perturbation_norm(rng2)

            # Should be different
            assert not torch.allclose(perturbation1, perturbation2, atol=1e-6)

    def test_rge_methods_reproducibility(self) -> None:
        """Test that both RGE methods (forward and central) are reproducible."""
        torch.manual_seed(42)
        model1 = LinearModel()
        fake_input = torch.randn(2, 3)
        fake_label = torch.randn(2, 1)
        criterion = nn.MSELoss()

        # Test forward method
        rge_forward1 = random_gradient_estimator.RandomGradientEstimator(
            model1.parameters(),
            num_pert=2,
            paramwise_perturb=True,
            grad_estimate_method=random_gradient_estimator.RandomGradEstimateMethod.rge_forward,
        )

        with torch.no_grad():
            dir_grads_forward1 = rge_forward1.compute_grad(
                fake_input, fake_label, lambda x, y: criterion(model1(x), y), seed=12345
            )

        # Run forward method again
        torch.manual_seed(42)
        model2 = LinearModel()
        rge_forward2 = random_gradient_estimator.RandomGradientEstimator(
            model2.parameters(),
            num_pert=2,
            paramwise_perturb=True,
            grad_estimate_method=random_gradient_estimator.RandomGradEstimateMethod.rge_forward,
        )

        with torch.no_grad():
            dir_grads_forward2 = rge_forward2.compute_grad(
                fake_input, fake_label, lambda x, y: criterion(model2(x), y), seed=12345
            )

        # Forward method should be reproducible
        torch.testing.assert_close(dir_grads_forward1, dir_grads_forward2)

        # Test central method
        torch.manual_seed(42)
        model3 = LinearModel()
        rge_central1 = random_gradient_estimator.RandomGradientEstimator(
            model3.parameters(),
            num_pert=2,
            paramwise_perturb=True,
            grad_estimate_method=random_gradient_estimator.RandomGradEstimateMethod.rge_central,
        )

        with torch.no_grad():
            dir_grads_central1 = rge_central1.compute_grad(
                fake_input, fake_label, lambda x, y: criterion(model3(x), y), seed=12345
            )

        # Run central method again
        torch.manual_seed(42)
        model4 = LinearModel()
        rge_central2 = random_gradient_estimator.RandomGradientEstimator(
            model4.parameters(),
            num_pert=2,
            paramwise_perturb=True,
            grad_estimate_method=random_gradient_estimator.RandomGradEstimateMethod.rge_central,
        )

        with torch.no_grad():
            dir_grads_central2 = rge_central2.compute_grad(
                fake_input, fake_label, lambda x, y: criterion(model4(x), y), seed=12345
            )

        # Central method should be reproducible
        torch.testing.assert_close(dir_grads_central1, dir_grads_central2)

    def test_multiple_runs_consistency(self) -> None:
        """Test that multiple runs with same parameters produce consistent results."""
        torch.manual_seed(42)
        fake_input = torch.randn(2, 3)
        fake_label = torch.randn(2, 1)
        criterion = nn.MSELoss()

        results = []
        for _ in range(3):
            torch.manual_seed(42)  # Reset seed for each run
            model_copy = LinearModel()
            rge_copy = random_gradient_estimator.RandomGradientEstimator(
                model_copy.parameters(),
                num_pert=3,
                paramwise_perturb=True,
            )

            with torch.no_grad():
                dir_grads = rge_copy.compute_grad(
                    fake_input, fake_label, lambda x, y: criterion(model_copy(x), y), seed=12345
                )
                results.append(dir_grads)

        # All results should be identical
        for i in range(1, len(results)):
            torch.testing.assert_close(results[0], results[i])

    def test_parameter_perturbation_reproducibility(self) -> None:
        """Test that parameter perturbations are reproducible."""
        torch.manual_seed(42)
        model1 = LinearModel()
        rge1 = random_gradient_estimator.RandomGradientEstimator(
            model1.parameters(),
            paramwise_perturb=True,
        )

        with torch.no_grad():
            # Perturb model
            rng1 = rge1.get_rng(seed=12345, perturb_index=0)
            rge1.perturb_model_paramwise(rng1, alpha=0.1)

        # Run again with same seed
        torch.manual_seed(42)
        model2 = LinearModel()
        rge2 = random_gradient_estimator.RandomGradientEstimator(
            model2.parameters(),
            paramwise_perturb=True,
        )

        with torch.no_grad():
            # Perturb model
            rng2 = rge2.get_rng(seed=12345, perturb_index=0)
            rge2.perturb_model_paramwise(rng2, alpha=0.1)

        # Results should be identical
        for p1, p2 in zip(model1.parameters(), model2.parameters()):
            torch.testing.assert_close(p1, p2)
