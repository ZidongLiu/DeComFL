import torch
from torch.nn import Parameter
from typing import Iterator
from torch.nn.utils import parameters_to_vector, vector_to_parameters


class ZCD:

    def __init__(
        self,
        params: Iterator[Parameter],
        lr=1e-3,
        mu=1e-3,
        grad_estimate_method="forward",
    ):
        self.parameters_list = list(params)
        self.lr = lr
        self.mu = mu
        self.grad_estimate_method = grad_estimate_method

        params_vec = parameters_to_vector(self.parameters_list)
        self.params_len = params_vec.shape[0]

    def perturb_and_step(self, images, labels, model, criterion):
        # index
        perturb_index = torch.randint(0, self.params_len, (128,))
        #
        current_parameters_vec = parameters_to_vector(self.parameters_list)

        if self.grad_estimate_method == "forward":
            pred1 = model(images)
            perturbed_param_vec = current_parameters_vec.clone()
            perturbed_param_vec[perturb_index] += self.mu
            vector_to_parameters(perturbed_param_vec, self.parameters_list)
            pred2 = model(images)
        elif self.grad_estimate_method == "middle":
            negative_param_vec = current_parameters_vec.clone()
            negative_param_vec[perturb_index] -= self.mu
            vector_to_parameters(negative_param_vec, self.parameters_list)
            pred1 = model(images)

            positive_param_vec = current_parameters_vec.clone()
            positive_param_vec[perturb_index] += self.mu
            vector_to_parameters(positive_param_vec, self.parameters_list)
            pred2 = model(images)
        else:
            raise Exception("something went wrong")

        loss1 = criterion(pred1, labels)
        loss2 = criterion(pred2, labels)

        grad = self.calculate_grad(loss1, loss2)

        current_parameters_vec[perturb_index] += self.lr * grad
        return grad

    @property
    def divider(self):
        if self.grad_estimate_method == "forward":
            return self.mu

        if self.grad_estimate_method == "middle":
            return self.mu * 2

    def calculate_grad(self, perturbation_1_loss, perturbation_2_loss):
        return (perturbation_2_loss - perturbation_1_loss) / self.divider
