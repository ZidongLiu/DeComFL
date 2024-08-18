from __future__ import annotations
import abc
import random
import torch
from typing import Any, Iterable, Sequence
from collections import deque
from config import get_params
from concurrent.futures import ThreadPoolExecutor
from cezo_fl.shared import CriterionType, update_model_given_seed_and_grad
from shared.metrics import Metric
from gradient_estimators.random_gradient_estimator import RandomGradientEstimator as RGE
from dataclasses import dataclass
from byzantine.aggregation import mean, median, trim, krum
from byzantine.attack import no_byz, gaussian_attack, sign_attack, trim_attack, krum_attack

args = get_params().parse_args()


@dataclass
class LocalUpdateResult:
    grad_tensors: list[torch.Tensor]
    step_accuracy: float
    step_loss: float

    # Must add __future__ import to be able to return, see https://stackoverflow.com/a/33533514
    def to(self, device: torch.device) -> LocalUpdateResult:
        self.grad_tensors = [grad_tensor.to(device) for grad_tensor in self.grad_tensors]
        return self


class AbstractClient:

    @abc.abstractmethod
    def local_update(self, seeds: Sequence[int]) -> LocalUpdateResult:
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

    @abc.abstractmethod
    def random_gradient_estimator(self) -> RGE:
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


# TODO fix the multithreading issue (see https://github.com/ZidongLiu/FedDisco/issues/30)
def parallalizable_client_job(
    client: AbstractClient,
    pull_seeds_list: Sequence[Sequence[int]],
    pull_grad_list: Sequence[Sequence[torch.Tensor]],
    local_update_seeds: Sequence[int],
    server_device: torch.device,
) -> LocalUpdateResult:
    """
    Run client pull and local update in parallel.
    This function is added to make better use of multi-gpu set up.
    Each client can be deployed to a separate gpu. Thus we can run all clients in parallel.

    Note:
    This function also make sure the data passed to/from client are converted to correct device.
    We should only do cross device operation here
    """
    # need no_grad because the outer-most no_grad context manager does not affect
    # operation inside sub-thread
    with torch.no_grad():
        # step 1 map pull_grad_list data to client's device

        transfered_grad_list = [
            [tensor.to(client.device) for tensor in tensors] for tensors in pull_grad_list
        ]

        # step 2, client pull to update its model to latest
        client.pull_model(pull_seeds_list, transfered_grad_list)

        # step 3, client local update and get its result
        client_local_update_result = client.local_update(seeds=local_update_seeds)

    # move result to server device and return
    return client_local_update_result.to(server_device)


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

        self.server_model: torch.nn.Module | None = None
        self.server_criterion: CriterionType | None = None
        self.server_accuracy_func = None
        self.optim: torch.optim.Optimizer | None = None
        self.random_gradient_estimator: RGE | None = None

    def set_server_model_and_criterion(
        self,
        model: torch.nn.Module,
        criterion: CriterionType,
        accuracy_func,
        optimizer: torch.optim.Optimizer,
        random_gradient_estimator: RGE,
    ) -> None:
        self.server_model = model
        self.server_criterion = criterion
        self.server_accuracy_func = accuracy_func
        self.optim = optimizer
        self.random_gradient_estimator = random_gradient_estimator

    def train(self) -> None:
        if self.server_model:
            self.server_model.train()

    def get_sampled_client_index(self) -> list[int]:
        return random.sample(range(len(self.clients)), self.num_sample_clients)

    def set_perturbation(self, num_pert: int) -> None:
        for client in self.clients:
            client.random_gradient_estimator().num_pert = num_pert

    def set_learning_rate(self, lr: float) -> None:
        # Client
        for client in self.clients:
            for p in client.optimizer.param_groups:
                p["lr"] = lr
        # Server
        if self.server_model:
            for p in self.optim.param_groups:
                p["lr"] = lr

    def train_one_step(self, iteration: int) -> tuple[float, float]:
        # Step 0: initiate something
        sampled_client_index = self.get_sampled_client_index()
        seeds = [random.randint(0, 1000000) for _ in range(self.local_update_steps)]

        # Step 1 & 2: pull model and local update
        local_grad_scalar_list: list[list[torch.Tensor]] = []  # Clients X Local_update
        step_train_loss = Metric("Step train loss")
        step_train_accuracy = Metric("Step train accuracy")

        client_results = []
        for index in sampled_client_index:
            client = self.clients[index]
            last_update_iter = self.client_last_updates[index]
            # The seed and grad in last_update_iter is fetched as well
            # Note at that iteration, we just reset the client model so that iteration
            # information is needed as well.
            seeds_list = self.seed_grad_records.fetch_seed_records(last_update_iter)
            grad_list = self.seed_grad_records.fetch_grad_records(last_update_iter)

            client_results.append(
                parallalizable_client_job(client, seeds_list, grad_list, seeds, self.device)
            )

        for index, client_local_update_result in zip(sampled_client_index, client_results):
            step_train_loss.update(client_local_update_result.step_loss)
            step_train_accuracy.update(client_local_update_result.step_accuracy)
            local_grad_scalar_list.append(client_local_update_result.grad_tensors)
            self.client_last_updates[index] = iteration

        # Step 3: byzantine attack
        if args.byz_type == "no_byz":
            local_grad_scalar_list = no_byz(local_grad_scalar_list)
        elif args.byz_type == "gaussian":
            local_grad_scalar_list = gaussian_attack(local_grad_scalar_list, args.num_byz)
        elif args.byz_type == "sign":
            local_grad_scalar_list = sign_attack(local_grad_scalar_list, args.num_byz)
        elif args.byz_type == "trim":
            local_grad_scalar_list = trim_attack(local_grad_scalar_list, args.num_byz)
        elif args.byz_type == "krum":
            local_grad_scalar_list = krum_attack(local_grad_scalar_list, args.num_byz)

        # Step 4: server-side aggregation
        if args.aggregation == "mean":
            grad_scalar = mean(args.num_sample_clients, local_grad_scalar_list)
        elif args.aggregation == "median":
            grad_scalar = median(local_grad_scalar_list)
        elif args.aggregation == "trim":
            grad_scalar = trim(args.num_sample_clients, local_grad_scalar_list)
        elif args.aggregation == "krum":
            grad_scalar = krum(local_grad_scalar_list)

        self.seed_grad_records.add_records(seeds=seeds, grad=grad_scalar)

        # Optional: optimize the memory. Remove is exclusive, i.e., the min last updates
        # information is still kept.
        self.seed_grad_records.remove_too_old(earliest_record_needs=min(self.client_last_updates))

        if self.server_model:
            self.train()
            update_model_given_seed_and_grad(
                self.optim,
                self.random_gradient_estimator,
                seeds,
                grad_scalar,
            )

        return step_train_loss.avg, step_train_accuracy.avg

    def eval_model(self, test_loader: Iterable[Any]) -> tuple[float, float]:
        if self.server_model is None:
            raise RuntimeError("set_server_model_and_criterion for server first.")
        self.server_model.eval()
        eval_loss = Metric("Eval loss")
        eval_accuracy = Metric("Eval accuracy")
        with torch.no_grad():
            for _, (batch_inputs, batch_labels) in enumerate(test_loader):
                if (
                    self.device != torch.device("cpu")
                    or self.random_gradient_estimator.torch_dtype != torch.float32
                ):
                    batch_inputs = batch_inputs.to(
                        self.device, self.random_gradient_estimator.torch_dtype
                    )
                    ## NOTE: label does not convert to dtype
                    batch_labels = batch_labels.to(self.device)
                pred = self.random_gradient_estimator.model_forward(batch_inputs)
                eval_loss.update(self.server_criterion(pred, batch_labels))
                eval_accuracy.update(self.server_accuracy_func(pred, batch_labels))
        print(
            f"\nEvaluation(Iteration {self.seed_grad_records.current_iteration}): ",
            f"Eval Loss:{eval_loss.avg:.4f}, " f"Accuracy:{eval_accuracy.avg * 100:.2f}%",
        )
        return eval_loss.avg, eval_accuracy.avg
