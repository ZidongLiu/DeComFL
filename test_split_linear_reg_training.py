import numpy as np
import torch
from models.splitted_linear_regression.splitted_linear_regression import (
    LinearRegSlope,
    LinearRegIntercept,
)
from optimizers.perturbation_direction_descent import PDD
from torch.nn import MSELoss
from tqdm import tqdm


learning_rate = 1e-4
mu = 1e-4

slope_model = LinearRegSlope()
slope_pdd = PDD(slope_model.parameters(), lr=learning_rate, mu=mu)

intercept_model = LinearRegIntercept()
intercept_pdd = PDD(intercept_model.parameters(), lr=learning_rate, mu=mu)

mse_criterion = MSELoss()
# generate training set
true_slope = 1
true_intercept = 1

torch.manual_seed(0)
np.random.seed(1)

xs = torch.arange(-5, 5, 0.1)
ys = true_slope * xs + true_intercept + torch.randn_like(xs) / 3
test_ys = true_slope * xs + true_intercept + torch.randn_like(xs) / 3

print(
    "true slope: {},  start slope: {:.4f}".format(
        true_slope, slope_pdd.params_list[0].data.item()
    )
)
print(
    "true intercept: {},  start intercept: {:.4f}".format(
        true_intercept, intercept_pdd.params_list[0].data.item()
    )
)
print("start MSE loss", mse_criterion(torch.zeros_like(ys), ys))
#
indexes_to_be_sampled = range(len(xs))
batch_size = 2
n_round = 10000
eval_iteration = 500
train_update_iteration = 100


with tqdm(total=n_round, desc="Training:") as t:
    with torch.no_grad():
        running_loss = 0.0
        for cur_round in range(n_round):
            sampled_indices = torch.from_numpy(
                np.random.choice(indexes_to_be_sampled, 2, replace=False)
            )
            train_x = xs[sampled_indices]
            train_y = ys[sampled_indices]
            # model 1
            original_out_1 = slope_model(train_x)
            slope_pdd.apply_perturbation()
            perturbed_out_1 = slope_model(train_x)

            # model 2 and calulate loss and grad
            original_out_2 = intercept_model(original_out_1)
            intercept_pdd.apply_perturbation()
            perturbed_out_2 = intercept_model(perturbed_out_1)

            original_loss = mse_criterion(original_out_2, train_y)
            perturbed_loss = mse_criterion(perturbed_out_2, train_y)
            grad = intercept_pdd.calculate_grad(perturbed_loss, original_loss)

            # update model
            intercept_pdd.step(grad)
            slope_pdd.step(grad)

            running_loss += original_loss.item()
            if cur_round % 500 == 499:
                eval_loss = mse_criterion(
                    intercept_model(slope_model(xs)), test_ys
                ).item()
                print(
                    f"\nEval Round: {cur_round+1}. Eval MSE loss: {eval_loss:.5f}. "
                    + f"Slope: {slope_pdd.params_list[0].data.item():.5f}"
                    + f" Intercept: {intercept_pdd.params_list[0].data.item():.5f}"
                )

            if cur_round % train_update_iteration == (train_update_iteration - 1):
                t.set_postfix({"MSE loss": running_loss / train_update_iteration})
                t.update(train_update_iteration)
                running_loss = 0.0

print(
    "true slope: {},  trained slope: {:.4f}".format(
        true_slope, slope_pdd.params_list[0].data.item()
    )
)
print(
    "true intercept: {},  trained intercept: {:.4f}".format(
        true_intercept, intercept_pdd.params_list[0].data.item()
    )
)
