import abc
import random
import torch
from typing import Any, Iterable, Sequence, Optional
from collections import deque

from cezo_fl.shared import update_model_given_seed_and_grad
from shared.metrics import Metric, accuracy
from gradient_estimators.random_gradient_estimator import RandomGradientEstimator as RGE


class AbstractClient:

    @abc.abstractmethod
    def local_update(self, seeds: Sequence[int]) -> Sequence[torch.Tensor]:
        """Returns a sequence of gradient scalar tensors for each local update.

        The length of the returned sequence should be the same as the length of seeds.
        The inner tensor can be a scalar or a vector. The length of vector is the number
        of perturbations.
        """
        return NotImplemented

    @abc.abstractmethod
    def reset_model(self) -> None:
        """Reset the mode to the state before the local_update."""
        return NotImplemented

    @abc.abstractmethod
    def pull_model(
        self,
        seeds_list: Sequence[Sequence[int]],
        gradient_scalar: Sequence[Sequence[torch.Tensor]],
    ) -> None:
        return NotImplemented


class SeedAndGradientRecords:
    def __init__(self):
        # For seed_records/grad_records, each entry stores info related to 1 iteration
        # seed_records[i]: length = number of local updates K
        # seed_records[i][k]: seed_k
        # grad_records[i]: [vector for local_update_k for k in range(K)]
        # grad_records[i][k]: scalar for 1 perturb or vector for >=1 perturb
        # What should happen on clients pull server using grad_records[i][k]
        # client use seed_records[i][k] to generate perturbation(s)
        # client_grad[i][k]:
        # vector = mean(perturbations[j] * grad_records[i][k][j] for j)

        self.seed_records: deque[list[int]] = deque()
        self.grad_records: deque[list[torch.Tensor]] = deque()
        self.earliest_records = 0
        self.current_iteration = -1

    def add_records(self, seeds: list[int], grad: list[torch.Tensor]) -> int:
        self.current_iteration += 1
        self.seed_records.append(seeds)
        self.grad_records.append(grad)
        return self.current_iteration

    def remove_too_old(self, earliest_record_needs: int):
        if self.earliest_records >= earliest_record_needs:
            return  # No need to do anything
        while self.earliest_records < earliest_record_needs:
            self.seed_records.popleft()
            self.grad_records.popleft()
            self.earliest_records += 1

    def fetch_seed_records(self, earliest_record_needs: int) -> list[list[int]]:
        assert earliest_record_needs >= self.earliest_records
        return [
            self.seed_records[i - self.earliest_records]
            for i in range(earliest_record_needs, self.current_iteration + 1)
        ]

    def fetch_grad_records(self, earliest_record_needs: int) -> list[list[torch.Tensor]]:
        assert earliest_record_needs >= self.earliest_records
        return [
            self.grad_records[i - self.earliest_records]
            for i in range(earliest_record_needs, self.current_iteration + 1)
        ]


# TODO Make sure all client model intialized with same weight.
# TODO Support Gradient Pruning
class CeZO_Server:
    def __init__(
        self,
        clients: Sequence[AbstractClient],
        device: torch.device,
        num_sample_clients: int = 10,
        local_update_steps: int = 10,
    ) -> None:
        self.clients = clients
        self.device = device
        self.num_sample_clients = num_sample_clients
        self.local_update_steps = local_update_steps

        self.seed_grad_records = SeedAndGradientRecords()
        self.client_last_updates = [0 for _ in range(len(self.clients))]

        self.server_model: Optional[torch.nn.Module] = None
        self.server_criterion: Optional[torch.nn.Module] = None
        self.optim: Optional[torch.optim.Optimizer] = None
        self.random_gradient_estimator: Optional[RGE] = None

    def set_server_model_and_criterion(
        self,
        model: torch.nn.Module,
        criterion: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        random_gradient_estimator: RGE,
    ) -> None:
        self.server_model = model
        self.server_criterion = criterion
        self.optim = optimizer
        self.random_gradient_estimator = random_gradient_estimator

    def train(self):
        if self.server_model:
            self.server_model.train()

    def get_sampled_client_index(self):
        return random.sample(range(len(self.clients)), self.num_sample_clients)

    def train_one_step(self, iteration: int) -> None:
        # Step 0: initiate something
        sampled_client_index = self.get_sampled_client_index()
        seeds = [random.randint(0, 1000000) for _ in range(self.local_update_steps)]

        # Step 1 & 2: pull model and local update
        local_grad_scalar_list: list[list[torch.Tensor]] = []  # Clients X Local_update
        for index in sampled_client_index:
            client = self.clients[index]
            last_update_iter = self.client_last_updates[index]
            # The seed and grad in last_update_iter is fetched as well
            # Note at that iteration, we just reset the client model so that iteration
            # information is needed as well.
            seeds_list = self.seed_grad_records.fetch_seed_records(last_update_iter)
            grad_list = self.seed_grad_records.fetch_grad_records(last_update_iter)
            # client will reset model to last pull states before update its model to match server
            client.pull_model(seeds_list, grad_list)

            local_grad_scalar_list.append(client.local_update(seeds=seeds))
            self.client_last_updates[index] = iteration

        # Step 3: server-side aggregation
        avg_grad_scalar: list[torch.Tensor] = []
        for each_client_update in zip(*local_grad_scalar_list):
            avg_grad_scalar.append(sum(each_client_update).div_(self.num_sample_clients))

        self.seed_grad_records.add_records(seeds=seeds, grad=avg_grad_scalar)

        # Optional: optimize the memory. Remove is exclusive, i.e., the min last updates
        # information is still kept.
        self.seed_grad_records.remove_too_old(earliest_record_needs=min(self.client_last_updates))

        if self.server_model:
            self.train()
            update_model_given_seed_and_grad(
                self.optim,
                self.random_gradient_estimator,
                seeds,
                avg_grad_scalar,
            )

    def eval_model(self, test_loader: Iterable[Any]) -> tuple[float, float]:
        if self.server_model is None:
            raise RuntimeError("set_server_model_and_criterion for server first.")
        self.server_model.eval()
        eval_loss = Metric("Eval loss")
        eval_accuracy = Metric("Eval accuracy")
        with torch.no_grad():
            for _, (images, labels) in enumerate(test_loader):
                if self.device != torch.device("cpu"):
                    images, labels = images.to(self.device), labels.to(self.device)
                pred = self.server_model(images)
                eval_loss.update(self.server_criterion(pred, labels))
                eval_accuracy.update(accuracy(pred, labels))
        print(
            f"\nEvaluation(Iteration {self.seed_grad_records.current_iteration}): ",
            f"Eval Loss:{eval_loss.avg:.4f}, " f"Accuracy:{eval_accuracy.avg * 100:.2f}%",
        )
        return eval_loss.avg, eval_accuracy.avg
