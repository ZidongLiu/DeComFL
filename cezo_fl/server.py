import random
import torch
from typing import Any, Iterable, Sequence
from collections import deque
from config import get_params
from cezo_fl.shared import CriterionType, update_model_given_seed_and_grad
from cezo_fl.client import AbstractClient
from cezo_fl.run_client_jobs import execute_sampled_clients
from shared.metrics import Metric
from gradient_estimators.random_gradient_estimator import RandomGradientEstimator as RGE
from dataclasses import dataclass
from byzantine import aggregation as byz_agg
from byzantine import attack as byz_attack

args = get_params().parse_args()


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
        step_train_loss, step_train_accuracy, local_grad_scalar_list = execute_sampled_clients(
            self, sampled_client_index, seeds, parallel=True
        )

        # Step 3: byzantine attack
        if args.byz_type == "no_byz":
            local_grad_scalar_list = byz_attack.no_byz(local_grad_scalar_list)
        elif args.byz_type == "gaussian":
            local_grad_scalar_list = byz_attack.gaussian_attack(
                local_grad_scalar_list, args.num_byz
            )
        elif args.byz_type == "sign":
            local_grad_scalar_list = byz_attack.sign_attack(local_grad_scalar_list, args.num_byz)
        elif args.byz_type == "trim":
            local_grad_scalar_list = byz_attack.trim_attack(local_grad_scalar_list, args.num_byz)
        elif args.byz_type == "krum":
            local_grad_scalar_list = byz_attack.krum_attack(local_grad_scalar_list, args.num_byz)

        # Step 4: server-side aggregation
        if args.aggregation == "mean":
            grad_scalar = byz_agg.mean(args.num_sample_clients, local_grad_scalar_list)
        elif args.aggregation == "median":
            grad_scalar = byz_agg.median(local_grad_scalar_list)
        elif args.aggregation == "trim":
            grad_scalar = byz_agg.trim(args.num_sample_clients, local_grad_scalar_list)
        elif args.aggregation == "krum":
            grad_scalar = byz_agg.krum(local_grad_scalar_list)

        self.seed_grad_records.add_records(seeds=seeds, grad=grad_scalar)

        # Optional: optimize the memory. Remove is exclusive, i.e., the min last updates
        # information is still kept.
        self.seed_grad_records.remove_too_old(earliest_record_needs=min(self.client_last_updates))

        if self.server_model:
            self.train()
            assert self.optim
            assert self.random_gradient_estimator
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
        assert self.random_gradient_estimator
        assert self.server_criterion
        assert self.server_accuracy_func

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
                    # NOTE: label does not convert to dtype
                    batch_labels = batch_labels.to(self.device)
                pred = self.random_gradient_estimator.model_forward(batch_inputs)
                eval_loss.update(self.server_criterion(pred, batch_labels))
                eval_accuracy.update(self.server_accuracy_func(pred, batch_labels))
        print(
            f"\nEvaluation(Iteration {self.seed_grad_records.current_iteration}): ",
            f"Eval Loss:{eval_loss.avg:.4f}, " f"Accuracy:{eval_accuracy.avg * 100:.2f}%",
        )
        return eval_loss.avg, eval_accuracy.avg
