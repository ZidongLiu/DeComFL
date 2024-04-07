import numpy as np
import torch
from models.splitted_linear_regression.splitted_linear_regression import LinearRegSlope, LinearRegIntercept
from optimizers.perturbation_direction_descent import PDD
from torch.nn import MSELoss

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

xs = torch.arange(-5, 5, 0.1)
ys = true_slope * xs + true_intercept

print('true slope: {},  start slope: {:.4f}'.format(true_slope, slope_pdd.params_list[0].data.item()))
print('start MSE loss', mse_criterion(torch.zeros_like(ys), ys))
#
indexes_to_be_sampled = range(len(xs))
batch_size = 2

with torch.no_grad():
    for i in range(10000):
        sampled_indices = torch.from_numpy(np.random.choice(indexes_to_be_sampled, 2, replace=False))
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
        # print('grad:', grad.item())
        # update model
        intercept_pdd.step(grad)
        slope_pdd.step(grad)
        # print('slope: ', slope_pdd.params_list[0].data.item(), 'intercept: ', intercept_pdd.params_list[0].data.item())
        if i % 500 == 499:
            print(f'Training Step: {i+1}. Overall MSE loss:', mse_criterion(intercept_model(slope_model(xs)), ys).item())

print('true slope: {},  trained slope: {:.4f}'.format(true_slope, slope_pdd.params_list[0].data.item()))
print('true intercept: {},  trained intercept: {:.4f}'.format(true_intercept, intercept_pdd.params_list[0].data.item()))