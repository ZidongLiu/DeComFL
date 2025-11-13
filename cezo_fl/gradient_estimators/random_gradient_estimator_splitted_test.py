import pytest
import torch
from torch import nn

from cezo_fl.gradient_estimators import random_gradient_estimator
from cezo_fl.gradient_estimators.random_gradient_estimator import RandomGradEstimateMethod
from cezo_fl.gradient_estimators.random_gradient_estimator_splitted import (
    RandomGradientEstimatorBatch,
    RandomGradientEstimatorParamwise,
)


class LinearModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(3, 2)
        self.linear2 = nn.Linear(2, 1)

    def forward(self, x):
        x = nn.functional.relu(self.linear1(x))
        return self.linear2(x)


class TestSplitClassesEquivalence:
    """Test suite to ensure split classes produce same results as original."""

    @pytest.mark.parametrize(
        "rge_method",
        [
            RandomGradEstimateMethod.rge_forward,
            RandomGradEstimateMethod.rge_central,
        ],
    )
    @pytest.mark.parametrize("num_pert", [1, 2, 4, 5])
    @pytest.mark.parametrize("mu", [1e-4, 1e-3, 1e-2])
    def test_batch_vs_original_equivalence(
        self,
        rge_method: RandomGradEstimateMethod,
        num_pert: int,
        mu: float,
    ) -> None:
        """Test that RandomGradientEstimatorBatch produces same results as original with paramwise_perturb=False."""
        torch.manual_seed(123)
        fake_input = torch.randn(5, 3)
        fake_label = torch.randn(5, 1)
        criterion = nn.MSELoss()

        # Original class with paramwise_perturb=False
        torch.random.manual_seed(123)
        model_original = LinearModel()
        rge_original = random_gradient_estimator.RandomGradientEstimator(
            model_original.parameters(),
            mu=mu,
            num_pert=num_pert,
            grad_estimate_method=rge_method,
            paramwise_perturb=False,
            normalize_perturbation=False,
        )

        with torch.no_grad():
            dir_grads_original = rge_original.compute_grad(
                fake_input, fake_label, lambda x, y: criterion(model_original(x), y), seed=54321
            )

        # Split batch class
        torch.random.manual_seed(123)
        model_batch = LinearModel()
        rge_batch = RandomGradientEstimatorBatch(
            model_batch.parameters(),
            mu=mu,
            num_pert=num_pert,
            grad_estimate_method=rge_method,
            normalize_perturbation=False,
        )

        with torch.no_grad():
            dir_grads_batch = rge_batch.compute_grad(
                fake_input, fake_label, lambda x, y: criterion(model_batch(x), y), seed=54321
            )

        # Results should be identical
        torch.testing.assert_close(dir_grads_original, dir_grads_batch)
        for p_orig, p_batch in zip(model_original.parameters(), model_batch.parameters()):
            torch.testing.assert_close(p_orig.grad, p_batch.grad)

    @pytest.mark.parametrize(
        "rge_method",
        [
            RandomGradEstimateMethod.rge_forward,
            RandomGradEstimateMethod.rge_central,
        ],
    )
    @pytest.mark.parametrize("num_pert", [1, 2, 4, 5])
    @pytest.mark.parametrize("mu", [1e-4, 1e-3, 1e-2])
    def test_paramwise_vs_original_equivalence(
        self,
        rge_method: RandomGradEstimateMethod,
        num_pert: int,
        mu: float,
    ) -> None:
        """Test that RandomGradientEstimatorParamwise produces same results as original with paramwise_perturb=True."""
        torch.manual_seed(123)
        fake_input = torch.randn(5, 3)
        fake_label = torch.randn(5, 1)
        criterion = nn.MSELoss()

        # Original class with paramwise_perturb=True
        torch.random.manual_seed(123)
        model_original = LinearModel()
        rge_original = random_gradient_estimator.RandomGradientEstimator(
            model_original.parameters(),
            mu=mu,
            num_pert=num_pert,
            grad_estimate_method=rge_method,
            paramwise_perturb=True,
            normalize_perturbation=False,
        )

        with torch.no_grad():
            dir_grads_original = rge_original.compute_grad(
                fake_input, fake_label, lambda x, y: criterion(model_original(x), y), seed=54321
            )

        # Split paramwise class
        torch.random.manual_seed(123)
        model_paramwise = LinearModel()
        rge_paramwise = RandomGradientEstimatorParamwise(
            model_paramwise.parameters(),
            mu=mu,
            num_pert=num_pert,
            grad_estimate_method=rge_method,
            normalize_perturbation=False,
        )

        with torch.no_grad():
            dir_grads_paramwise = rge_paramwise.compute_grad(
                fake_input, fake_label, lambda x, y: criterion(model_paramwise(x), y), seed=54321
            )

        # Results should be identical
        torch.testing.assert_close(dir_grads_original, dir_grads_paramwise)
        for p_orig, p_paramwise in zip(model_original.parameters(), model_paramwise.parameters()):
            torch.testing.assert_close(p_orig.grad, p_paramwise.grad)

    def test_batch_vs_paramwise_equivalence_small_model(self) -> None:
        """Test that batch and paramwise methods produce same results for small models."""
        torch.manual_seed(123)
        fake_input = torch.randn(5, 3)
        fake_label = torch.randn(5, 1)
        criterion = nn.MSELoss()

        # Batch class
        torch.random.manual_seed(123)
        model_batch = LinearModel()
        rge_batch = RandomGradientEstimatorBatch(
            model_batch.parameters(),
            num_pert=2,
            grad_estimate_method=RandomGradEstimateMethod.rge_central,
            normalize_perturbation=False,
        )

        with torch.no_grad():
            dir_grads_batch = rge_batch.compute_grad(
                fake_input, fake_label, lambda x, y: criterion(model_batch(x), y), seed=54321
            )

        # Paramwise class
        torch.random.manual_seed(123)
        model_paramwise = LinearModel()
        rge_paramwise = RandomGradientEstimatorParamwise(
            model_paramwise.parameters(),
            num_pert=2,
            grad_estimate_method=RandomGradEstimateMethod.rge_central,
            normalize_perturbation=False,
        )

        with torch.no_grad():
            dir_grads_paramwise = rge_paramwise.compute_grad(
                fake_input, fake_label, lambda x, y: criterion(model_paramwise(x), y), seed=54321
            )

        # Results should be identical for small models
        torch.testing.assert_close(dir_grads_batch, dir_grads_paramwise)
        for p_batch, p_paramwise in zip(model_batch.parameters(), model_paramwise.parameters()):
            torch.testing.assert_close(p_batch.grad, p_paramwise.grad)

    def test_update_model_given_seed_and_grad_batch_equivalence(self) -> None:
        """Test that update_model_given_seed_and_grad works identically for batch class."""
        torch.manual_seed(42)
        fake_model_original = torch.nn.Sequential(
            torch.nn.Linear(10, 5),
            torch.nn.ReLU(),
            torch.nn.Linear(5, 2),
        )
        optim_original = torch.optim.SGD(fake_model_original.parameters(), lr=1e-3)
        rge_original = random_gradient_estimator.RandomGradientEstimator(
            fake_model_original.parameters(),
            num_pert=2,
            paramwise_perturb=False,
        )

        torch.manual_seed(42)
        fake_model_batch = torch.nn.Sequential(
            torch.nn.Linear(10, 5),
            torch.nn.ReLU(),
            torch.nn.Linear(5, 2),
        )
        optim_batch = torch.optim.SGD(fake_model_batch.parameters(), lr=1e-3)
        rge_batch = RandomGradientEstimatorBatch(
            fake_model_batch.parameters(),
            num_pert=2,
        )

        iteration_seeds = [1, 2, 3]
        iteration_grad_scalar = [
            torch.tensor([0.1, 0.2]),
            torch.tensor([0.3, 0.4]),
            torch.tensor([0.5, 0.6]),
        ]

        with torch.no_grad():
            rge_original.update_model_given_seed_and_grad(
                optim_original, iteration_seeds, iteration_grad_scalar
            )
            rge_batch.update_model_given_seed_and_grad(
                optim_batch, iteration_seeds, iteration_grad_scalar
            )

        # Results should be identical
        for p_orig, p_batch in zip(fake_model_original.parameters(), fake_model_batch.parameters()):
            torch.testing.assert_close(p_orig, p_batch)

    def test_update_model_given_seed_and_grad_paramwise_equivalence(self) -> None:
        """Test that update_model_given_seed_and_grad works identically for paramwise class."""
        torch.manual_seed(42)
        fake_model_original = torch.nn.Sequential(
            torch.nn.Linear(10, 5),
            torch.nn.ReLU(),
            torch.nn.Linear(5, 2),
        )
        optim_original = torch.optim.SGD(fake_model_original.parameters(), lr=1e-3)
        rge_original = random_gradient_estimator.RandomGradientEstimator(
            fake_model_original.parameters(),
            num_pert=2,
            paramwise_perturb=True,
        )

        torch.manual_seed(42)
        fake_model_paramwise = torch.nn.Sequential(
            torch.nn.Linear(10, 5),
            torch.nn.ReLU(),
            torch.nn.Linear(5, 2),
        )
        optim_paramwise = torch.optim.SGD(fake_model_paramwise.parameters(), lr=1e-3)
        rge_paramwise = RandomGradientEstimatorParamwise(
            fake_model_paramwise.parameters(),
            num_pert=2,
        )

        iteration_seeds = [1, 2, 3]
        iteration_grad_scalar = [
            torch.tensor([0.1, 0.2]),
            torch.tensor([0.3, 0.4]),
            torch.tensor([0.5, 0.6]),
        ]

        with torch.no_grad():
            rge_original.update_model_given_seed_and_grad(
                optim_original, iteration_seeds, iteration_grad_scalar
            )
            rge_paramwise.update_model_given_seed_and_grad(
                optim_paramwise, iteration_seeds, iteration_grad_scalar
            )

        # Results should be identical
        for p_orig, p_paramwise in zip(
            fake_model_original.parameters(), fake_model_paramwise.parameters()
        ):
            torch.testing.assert_close(p_orig, p_paramwise)

    def test_revert_model_given_seed_and_grad_batch_equivalence(self) -> None:
        """Test that revert_model_given_seed_and_grad works identically for batch class."""
        torch.manual_seed(42)
        fake_model_original = torch.nn.Sequential(
            torch.nn.Linear(10, 5),
            torch.nn.ReLU(),
            torch.nn.Linear(5, 2),
        )
        optim_original = torch.optim.SGD(fake_model_original.parameters(), lr=1e-3, momentum=0)
        rge_original = random_gradient_estimator.RandomGradientEstimator(
            fake_model_original.parameters(),
            num_pert=2,
            paramwise_perturb=False,
        )

        torch.manual_seed(42)
        fake_model_batch = torch.nn.Sequential(
            torch.nn.Linear(10, 5),
            torch.nn.ReLU(),
            torch.nn.Linear(5, 2),
        )
        optim_batch = torch.optim.SGD(fake_model_batch.parameters(), lr=1e-3, momentum=0)
        rge_batch = RandomGradientEstimatorBatch(
            fake_model_batch.parameters(),
            num_pert=2,
        )

        iteration_seeds = [1, 2]
        iteration_grad_scalar = [
            torch.tensor([0.1, 0.2]),
            torch.tensor([0.3, 0.4]),
        ]

        with torch.no_grad():
            rge_original.revert_model_given_seed_and_grad(
                optim_original, iteration_seeds, iteration_grad_scalar
            )
            rge_batch.revert_model_given_seed_and_grad(
                optim_batch, iteration_seeds, iteration_grad_scalar
            )

        # Results should be identical
        for p_orig, p_batch in zip(fake_model_original.parameters(), fake_model_batch.parameters()):
            torch.testing.assert_close(p_orig, p_batch)

    def test_revert_model_given_seed_and_grad_paramwise_equivalence(self) -> None:
        """Test that revert_model_given_seed_and_grad works identically for paramwise class."""
        torch.manual_seed(42)
        fake_model_original = torch.nn.Sequential(
            torch.nn.Linear(10, 5),
            torch.nn.ReLU(),
            torch.nn.Linear(5, 2),
        )
        optim_original = torch.optim.SGD(fake_model_original.parameters(), lr=1e-3, momentum=0)
        rge_original = random_gradient_estimator.RandomGradientEstimator(
            fake_model_original.parameters(),
            num_pert=2,
            paramwise_perturb=True,
        )

        torch.manual_seed(42)
        fake_model_paramwise = torch.nn.Sequential(
            torch.nn.Linear(10, 5),
            torch.nn.ReLU(),
            torch.nn.Linear(5, 2),
        )
        optim_paramwise = torch.optim.SGD(fake_model_paramwise.parameters(), lr=1e-3, momentum=0)
        rge_paramwise = RandomGradientEstimatorParamwise(
            fake_model_paramwise.parameters(),
            num_pert=2,
        )

        iteration_seeds = [1, 2]
        iteration_grad_scalar = [
            torch.tensor([0.1, 0.2]),
            torch.tensor([0.3, 0.4]),
        ]

        with torch.no_grad():
            rge_original.revert_model_given_seed_and_grad(
                optim_original, iteration_seeds, iteration_grad_scalar
            )
            rge_paramwise.revert_model_given_seed_and_grad(
                optim_paramwise, iteration_seeds, iteration_grad_scalar
            )

        # Results should be identical
        for p_orig, p_paramwise in zip(
            fake_model_original.parameters(), fake_model_paramwise.parameters()
        ):
            torch.testing.assert_close(p_orig, p_paramwise)

    def test_sgd_no_optim_update_model_paramwise_equivalence(self) -> None:
        """Test that sgd_no_optim_update_model works identically for paramwise class."""
        torch.manual_seed(42)
        fake_model_original = LinearModel()
        rge_original = random_gradient_estimator.RandomGradientEstimator(
            fake_model_original.parameters(),
            sgd_only_no_optim=True,
            paramwise_perturb=True,
        )

        torch.manual_seed(42)
        fake_model_paramwise = LinearModel()
        rge_paramwise = RandomGradientEstimatorParamwise(
            fake_model_paramwise.parameters(),
        )

        perturbation_dir_grads = torch.tensor([0.1, 0.2])
        lr = 0.01

        with torch.no_grad():
            rge_original.sgd_no_optim_update_model(perturbation_dir_grads, seed=12345, lr=lr)
            rge_paramwise.sgd_no_optim_update_model(perturbation_dir_grads, seed=12345, lr=lr)

        # Results should be identical
        for p_orig, p_paramwise in zip(
            fake_model_original.parameters(), fake_model_paramwise.parameters()
        ):
            torch.testing.assert_close(p_orig, p_paramwise)

    def test_generate_then_put_grad_paramwise_equivalence(self) -> None:
        """Test that generate_then_put_grad_paramwise works identically for paramwise class."""
        torch.manual_seed(42)
        fake_model_original = LinearModel()
        rge_original = random_gradient_estimator.RandomGradientEstimator(
            fake_model_original.parameters(),
            paramwise_perturb=True,
        )

        torch.manual_seed(42)
        fake_model_paramwise = LinearModel()
        rge_paramwise = RandomGradientEstimatorParamwise(
            fake_model_paramwise.parameters(),
        )

        dir_grads = torch.tensor([0.1, 0.2])

        with torch.no_grad():
            rge_original.generate_then_put_grad_paramwise(seed=12345, dir_grads=dir_grads)
            rge_paramwise.generate_then_put_grad_paramwise(seed=12345, dir_grads=dir_grads)

        # Results should be identical
        for p_orig, p_paramwise in zip(
            fake_model_original.parameters(), fake_model_paramwise.parameters()
        ):
            torch.testing.assert_close(p_orig.grad, p_paramwise.grad)

    def test_perturb_model_paramwise_equivalence(self) -> None:
        """Test that perturb_model_paramwise works identically for paramwise class."""
        torch.manual_seed(42)
        fake_model_original = LinearModel()
        rge_original = random_gradient_estimator.RandomGradientEstimator(
            fake_model_original.parameters(),
            paramwise_perturb=True,
        )

        torch.manual_seed(42)
        fake_model_paramwise = LinearModel()
        rge_paramwise = RandomGradientEstimatorParamwise(
            fake_model_paramwise.parameters(),
        )

        with torch.no_grad():
            rng_original = rge_original.get_rng(seed=12345, perturb_index=0)
            rge_original.perturb_model_paramwise(rng_original, alpha=0.1)

            rng_paramwise = rge_paramwise.get_rng(seed=12345, perturb_index=0)
            rge_paramwise.perturb_model_paramwise(rng_paramwise, alpha=0.1)

        # Results should be identical
        for p_orig, p_paramwise in zip(
            fake_model_original.parameters(), fake_model_paramwise.parameters()
        ):
            torch.testing.assert_close(p_orig, p_paramwise)

    def test_zo_grad_estimate_paramwise_equivalence(self) -> None:
        """Test that _zo_grad_estimate_paramwise works identically for paramwise class."""
        torch.manual_seed(42)
        fake_model_original = LinearModel()
        fake_input = torch.randn(2, 3)
        fake_label = torch.randn(2, 1)
        criterion = nn.MSELoss()

        rge_original = random_gradient_estimator.RandomGradientEstimator(
            fake_model_original.parameters(),
            num_pert=2,
            paramwise_perturb=True,
            grad_estimate_method=RandomGradEstimateMethod.rge_forward,
        )

        torch.manual_seed(42)
        fake_model_paramwise = LinearModel()
        rge_paramwise = RandomGradientEstimatorParamwise(
            fake_model_paramwise.parameters(),
            num_pert=2,
            grad_estimate_method=RandomGradEstimateMethod.rge_forward,
        )

        with torch.no_grad():
            dir_grads_original = rge_original._zo_grad_estimate_paramwise(
                fake_input,
                fake_label,
                lambda x, y: criterion(fake_model_original(x), y),
                seed=12345,
            )
            dir_grads_paramwise = rge_paramwise._zo_grad_estimate_paramwise(
                fake_input,
                fake_label,
                lambda x, y: criterion(fake_model_paramwise(x), y),
                seed=12345,
            )

        # Results should be identical
        torch.testing.assert_close(dir_grads_original, dir_grads_paramwise)

    def test_constructor_equivalence(self) -> None:
        """Test that constructors work identically for all classes."""
        torch.manual_seed(42)
        model = LinearModel()

        # Original class
        rge_original = random_gradient_estimator.RandomGradientEstimator(
            model.parameters(),
            mu=1e-3,
            num_pert=3,
            grad_estimate_method=RandomGradEstimateMethod.rge_central,
            normalize_perturbation=False,
            device="cpu",
            torch_dtype=torch.float32,
            paramwise_perturb=False,
        )

        # Batch class
        rge_batch = RandomGradientEstimatorBatch(
            model.parameters(),
            mu=1e-3,
            num_pert=3,
            grad_estimate_method=RandomGradEstimateMethod.rge_central,
            normalize_perturbation=False,
            device="cpu",
            torch_dtype=torch.float32,
        )

        # Paramwise class
        rge_paramwise = RandomGradientEstimatorParamwise(
            model.parameters(),
            mu=1e-3,
            num_pert=3,
            grad_estimate_method=RandomGradEstimateMethod.rge_central,
            normalize_perturbation=False,
            device="cpu",
            torch_dtype=torch.float32,
        )

        # Check that all have same basic properties
        assert rge_original.mu == rge_batch.mu == rge_paramwise.mu
        assert rge_original.num_pert == rge_batch.num_pert == rge_paramwise.num_pert
        assert (
            rge_original.grad_estimate_method.value
            == rge_batch.grad_estimate_method.value
            == rge_paramwise.grad_estimate_method.value
        )
        assert (
            rge_original.normalize_perturbation
            == rge_batch.normalize_perturbation
            == rge_paramwise.normalize_perturbation
        )
        assert rge_original.device == rge_batch.device == rge_paramwise.device
        assert rge_original.torch_dtype == rge_batch.torch_dtype == rge_paramwise.torch_dtype
        assert (
            rge_original.total_dimensions
            == rge_batch.total_dimensions
            == rge_paramwise.total_dimensions
        )

    def test_reproducibility_across_classes(self) -> None:
        """Test that all classes produce reproducible results with same seeds."""
        torch.manual_seed(42)
        fake_input = torch.randn(3, 3)
        fake_label = torch.randn(3, 1)
        criterion = nn.MSELoss()

        # Test multiple runs with same seed for each class
        for _ in range(3):
            torch.manual_seed(42)
            model_original = LinearModel()
            rge_original = random_gradient_estimator.RandomGradientEstimator(
                model_original.parameters(),
                num_pert=2,
                paramwise_perturb=False,
            )

            torch.manual_seed(42)
            model_batch = LinearModel()
            rge_batch = RandomGradientEstimatorBatch(
                model_batch.parameters(),
                num_pert=2,
            )

            torch.manual_seed(42)
            model_paramwise = LinearModel()
            rge_paramwise = RandomGradientEstimatorParamwise(
                model_paramwise.parameters(),
                num_pert=2,
            )

            with torch.no_grad():
                dir_grads_original = rge_original.compute_grad(
                    fake_input, fake_label, lambda x, y: criterion(model_original(x), y), seed=12345
                )
                dir_grads_batch = rge_batch.compute_grad(
                    fake_input, fake_label, lambda x, y: criterion(model_batch(x), y), seed=12345
                )
                dir_grads_paramwise = rge_paramwise.compute_grad(
                    fake_input,
                    fake_label,
                    lambda x, y: criterion(model_paramwise(x), y),
                    seed=12345,
                )

            # All should produce same results within each class
            torch.testing.assert_close(dir_grads_original, dir_grads_batch)
            torch.testing.assert_close(dir_grads_original, dir_grads_paramwise)

    def test_different_seeds_produce_different_results(self) -> None:
        """Test that different seeds produce different results for all classes."""
        torch.manual_seed(42)
        fake_input = torch.randn(2, 3)
        fake_label = torch.randn(2, 1)
        criterion = nn.MSELoss()

        # Test with different seeds
        model1 = LinearModel()
        rge1 = RandomGradientEstimatorBatch(
            model1.parameters(),
            num_pert=2,
        )

        model2 = LinearModel()
        rge2 = RandomGradientEstimatorBatch(
            model2.parameters(),
            num_pert=2,
        )

        with torch.no_grad():
            dir_grads1 = rge1.compute_grad(
                fake_input, fake_label, lambda x, y: criterion(model1(x), y), seed=12345
            )
            dir_grads2 = rge2.compute_grad(
                fake_input, fake_label, lambda x, y: criterion(model2(x), y), seed=54321
            )

        # Results should be different
        assert not torch.allclose(dir_grads1, dir_grads2, atol=1e-6)

    def test_edge_cases_equivalence(self) -> None:
        """Test edge cases work identically across all classes."""
        # Test with single parameter model
        torch.manual_seed(42)
        model_single = nn.Linear(3, 1)
        fake_input = torch.randn(2, 3)
        fake_label = torch.randn(2, 1)
        criterion = nn.MSELoss()

        rge_original = random_gradient_estimator.RandomGradientEstimator(
            model_single.parameters(),
            num_pert=1,
            paramwise_perturb=True,
        )

        torch.manual_seed(42)
        model_single_paramwise = nn.Linear(3, 1)
        rge_paramwise = RandomGradientEstimatorParamwise(
            model_single_paramwise.parameters(),
            num_pert=1,
        )

        with torch.no_grad():
            dir_grads_original = rge_original.compute_grad(
                fake_input, fake_label, lambda x, y: criterion(model_single(x), y), seed=12345
            )
            dir_grads_paramwise = rge_paramwise.compute_grad(
                fake_input,
                fake_label,
                lambda x, y: criterion(model_single_paramwise(x), y),
                seed=12345,
            )

        # Results should be identical
        torch.testing.assert_close(dir_grads_original, dir_grads_paramwise)
        for p_orig, p_paramwise in zip(
            model_single.parameters(), model_single_paramwise.parameters()
        ):
            torch.testing.assert_close(p_orig.grad, p_paramwise.grad)

    def test_large_num_pert_equivalence(self) -> None:
        """Test with large number of perturbations."""
        fake_input = torch.randn(2, 3)
        fake_label = torch.randn(2, 1)
        criterion = nn.MSELoss()

        torch.manual_seed(42)
        model_original = LinearModel()
        rge_original = random_gradient_estimator.RandomGradientEstimator(
            model_original.parameters(),
            num_pert=10,
            paramwise_perturb=True,
            sgd_only_no_optim=True,
        )

        model_paramwise = LinearModel()
        model_paramwise.load_state_dict(model_original.state_dict())

        for param_original, param_paramwise in zip(
            model_original.parameters(), model_paramwise.parameters()
        ):
            assert param_original.shape == param_paramwise.shape
            torch.testing.assert_close(param_original.data, param_paramwise.data)

        rge_paramwise = RandomGradientEstimatorParamwise(
            model_paramwise.parameters(),
            num_pert=10,
        )

        with torch.no_grad():
            dir_grads_original = rge_original.compute_grad(
                fake_input, fake_label, lambda x, y: criterion(model_original(x), y), seed=12345
            )
            dir_grads_paramwise = rge_paramwise.compute_grad(
                fake_input, fake_label, lambda x, y: criterion(model_paramwise(x), y), seed=12345
            )

        # Results should be identical
        torch.testing.assert_close(dir_grads_original, dir_grads_paramwise)
        assert dir_grads_original.shape == (10,)
        assert dir_grads_paramwise.shape == (10,)
