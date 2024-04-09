import torch
from torch.nn import Parameter
from torch import Tensor
from typing import Iterator
from tqdm import tqdm
from tensorboardX import SummaryWriter
from os import path
from shared.model_helpers import get_current_datetime_str


class PDD:

    def __init__(self, params: Iterator[Parameter], lr=1e-3, mu=1e-3):
        self.params_list: list[Parameter] = list(params)
        self.params_shape: list[Tensor] = [p.shape for p in self.params_list]
        self.lr = lr
        self.mu = mu

        self.current_perturbation = None

    def _params_list_add_(self, to_add: list[torch.Tensor]):
        if (not isinstance(to_add, list)) or (not len(to_add) == len(self.params_list)):
            raise Exception("Current to_add does not match controlled parameters")

        for p, added_value in zip(self.params_list, to_add):
            p.add_(added_value)

    def _generate_perturbation(self):
        self.current_perturbation = [torch.randn(shape) for shape in self.params_shape]
        return self.current_perturbation

    def apply_perturbation(self):
        perturbation = self._generate_perturbation()
        self._params_list_add_([self.mu * perturb for perturb in perturbation])

    def cancel_current_perturbation(self):
        if self.current_perturbation is None:
            raise Exception("Current perturbation does not exist yet")

        self._params_list_add_(
            [-self.mu * perturb for perturb in self.current_perturbation]
        )

    def calculate_grad(self, perturbation_loss, original_loss):
        return (perturbation_loss - original_loss) / self.mu

    def step(self, grad, to_cancel_perturbation=True):
        if (not isinstance(self.current_perturbation, list)) or (
            not len(self.current_perturbation) == len(self.params_list)
        ):
            raise Exception("Current perturbation does not match controlled parameters")

        # update_parameters, need to minus perturbation(since model is already changed)
        # then move to new_direction
        # x_t+1 = x_t - learning_rate * grad * perturbation
        # x_t+0.5 = x_t + mu * perturbation
        # x_t+1 = x_t+0.5 - mu * perturbation - learning_rate * grad * perturbation
        # x_t+1 = x_t+0.5 - (mu + learning_rate * grad) * perturbation
        if to_cancel_perturbation:
            perturb_multiplier = -(self.mu + self.lr * grad)
        else:
            perturb_multiplier = -self.lr * grad

        for p, perturb in zip(self.params_list, self.current_perturbation):
            p.add_(perturb_multiplier * perturb)


def _get_accuracy(preds: torch.tensor, labels: torch.tensor):
    return (preds.argmax(dim=1) == labels).float().mean().item()


def PDD_training_loop(
    model1,
    model2,
    pdd1,
    pdd2,
    criterion,
    train_loader,
    test_loader,
    n_epoch,
    train_update_iteration,
    eval_iteration,
    tensorboard_path,
):
    trainset_len = len(train_loader)
    total_steps = n_epoch * trainset_len
    tensorboard_sub_folder = f"{n_epoch}-{get_current_datetime_str()}"
    writer = SummaryWriter(path.join(tensorboard_path, tensorboard_sub_folder))
    with tqdm(total=total_steps, desc="Training:") as t:
        with torch.no_grad():
            running_loss = 0.0
            running_accuracy = 0.0
            for epoch_idx in range(n_epoch):
                for train_batch_idx, data in enumerate(train_loader):
                    (images, labels) = data

                    # model 1
                    original_out_1 = model1(images)
                    pdd1.apply_perturbation()
                    perturbed_out_1 = model1(images)

                    # model 2 and calulate loss and grad
                    original_out_2 = model2(original_out_1)
                    pdd2.apply_perturbation()
                    perturbed_out_2 = model2(perturbed_out_1)

                    original_loss = criterion(original_out_2, labels)
                    perturbed_loss = criterion(perturbed_out_2, labels)
                    grad = pdd2.calculate_grad(perturbed_loss, original_loss)

                    # update model
                    pdd2.step(grad)
                    pdd1.step(grad)

                    running_loss += original_loss.item()
                    running_accuracy += _get_accuracy(original_out_2, labels)

                    eval_round = epoch_idx * trainset_len + train_batch_idx
                    if train_batch_idx % train_update_iteration == (
                        train_update_iteration - 1
                    ):
                        train_loss = running_loss / train_update_iteration
                        train_accuracy = running_accuracy / train_update_iteration
                        t.set_postfix(
                            {
                                "Loss": train_loss,
                                "Accuracy": train_accuracy,
                            }
                        )
                        writer.add_scalar("Loss/train", train_loss, eval_round)
                        writer.add_scalar("Accuracy/train", train_accuracy, eval_round)
                        t.update(train_update_iteration)
                        running_loss = 0.0
                        running_accuracy = 0.0

                    if train_batch_idx % eval_iteration == (eval_iteration - 1):

                        loss = 0.0
                        accuracy = 0.0

                        for test_idx, data in enumerate(train_loader):
                            (test_images, test_labels) = data
                            pred = model2(model1(test_images))

                            accuracy += _get_accuracy(original_out_2, labels)

                            batch_eval_loss = criterion(pred, test_labels)
                            loss += batch_eval_loss

                        eval_loss = loss / (test_idx + 1)
                        eval_accuracy = accuracy / (test_idx + 1)

                        writer.add_scalar("Loss/test", eval_loss, eval_round)
                        writer.add_scalar("Accuracy/test", eval_accuracy, eval_round)
                        print(
                            f"Evaluation(round {eval_round}): {eval_loss=:.3f}"
                            + f"{eval_accuracy=:.3f}"
                        )
                        print(
                            f"Eval Loss:{eval_loss:.4f}, Accuracy: {eval_accuracy: .4f}"
                        )

    writer.close()
