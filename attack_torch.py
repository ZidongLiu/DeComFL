# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 21:06:55 2024

@author: Zidong
"""
import torch
from torch import nn
import numpy as np

n_channel = 3
height = 32
width = 32

h = 0.001
LOSS_EPS = 1e-30
KAI = 0

BETA_1 = 0.9
BETA_2 = 0.999
ADAM_EPS = 1e-8

learning_rate = 0.001

image_value_lower_bound = -1
image_value_upper_bound = 1

CLASSIFICATION_LOSS_MULTIPLIER = 0.5

N_LABEL = 10
image_shape = (n_channel, height, width)
sample_per_channel = 50
indexes_to_be_sampled = np.arange(height * width)


def sample_channel_height_width(channel_index):
    indices_1d = np.random.choice(indexes_to_be_sampled,
                                  sample_per_channel,
                                  replace=False)
    return np.stack([
        channel_index * np.ones(sample_per_channel, dtype='int32'),
        indices_1d // width, indices_1d % width
    ],
                    axis=1)


def get_image_sampling_pixels():
    channel_samplings = []
    for i in range(n_channel):
        channel_samplings.append(sample_channel_height_width(i))

    sampling_indices_3d = np.concatenate(channel_samplings, axis=0)

    return sampling_indices_3d


def get_loss(original_image: torch.tensor, image_true_label: int,
             change_images: torch.tensor,
             change_images_label_probability: torch.tensor):
    '''
    original_image: (n_channel, height, width)
    change_images: (n_images, n_channel, height, width)
    model_out_probability: (n_images, n_label = 10)
    '''

    l2dist_loss = torch.sum((change_images[0] - original_image)**2)
    non_target_mask = torch.ones(N_LABEL, dtype=int)
    non_target_mask[image_true_label] = 0
    max_log_non_target_prob = torch.log(
        torch.max(change_images_label_probability[:, non_target_mask]) +
        LOSS_EPS)

    classification_loss = torch.log(
        change_images_label_probability[:, image_true_label]
    ) - max_log_non_target_prob
    classification_loss[classification_loss < -KAI] = -KAI

    return l2dist_loss + CLASSIFICATION_LOSS_MULTIPLIER * classification_loss


def get_grad_in_1_pixel(model: nn.Module, cur_image: torch.tensor,
                        original_image: torch.tensor, image_true_label: int,
                        pixel_index_3d: torch.tensor):
    cur_pixel_value = cur_image[*pixel_index_3d]
    pixel_minus = cur_pixel_value + h
    pixel_plus = cur_pixel_value - h
    if pixel_minus < image_value_lower_bound or pixel_plus > image_value_upper_bound:
        return None

    image_offset = torch.zeros(image_shape)
    image_offset[*pixel_index_3d] = h
    grad_images = torch.stack(
        [cur_image - image_offset, cur_image + image_offset])
    model_out = model(grad_images)
    model_out_probability = model_out.softmax(dim=1)

    grad_image_loss = get_loss(original_image, image_true_label, grad_images,
                               model_out_probability)
    grad = (grad_image_loss[1] - grad_image_loss[0]) / (2 * h)
    return grad


def get_grad_in_many_pixels(model, cur_image: torch.tensor,
                            original_image: torch.tensor,
                            image_true_label: int,
                            many_pixel_indices_3d: torch.tensor):
    n_sample = len(many_pixel_indices_3d)
    grad_images = torch.clone(cur_image).unsqueeze(0).repeat(
        n_sample * 2, 1, 1, 1)
    plus_indexes = range(n_sample)

    plus_tensor = grad_images[plus_indexes, many_pixel_indices_3d[:, 0],
                              many_pixel_indices_3d[:, 1],
                              many_pixel_indices_3d[:, 2]] + h

    minus_indexes = range(n_sample, n_sample * 2)
    minus_tensor = grad_images[minus_indexes, many_pixel_indices_3d[:, 0],
                               many_pixel_indices_3d[:, 1],
                               many_pixel_indices_3d[:, 2]] - h

    plus_tensor[plus_tensor >
                image_value_upper_bound] = image_value_upper_bound
    minus_tensor[minus_tensor <
                 image_value_lower_bound] = image_value_lower_bound
    grad_images[plus_indexes, many_pixel_indices_3d[:, 0],
                many_pixel_indices_3d[:, 1],
                many_pixel_indices_3d[:, 2]] = plus_tensor
    grad_images[minus_indexes, many_pixel_indices_3d[:, 0],
                many_pixel_indices_3d[:, 1],
                many_pixel_indices_3d[:, 2]] = minus_tensor

    grad_step_size = plus_tensor - minus_tensor

    ####
    grad_images_probability = model(grad_images).softmax(dim=1)
    grad_image_losses = get_loss(original_image, image_true_label, grad_images,
                                 grad_images_probability)

    sample_grads = (grad_image_losses[plus_indexes] -
                    grad_image_losses[minus_indexes]) / grad_step_size

    return sample_grads


def single_step(T_matrix: torch.tensor, M_matrix: torch.tensor,
                V_matrix: torch.tensor, model: torch.nn.Module,
                last_image: torch.tensor, original_image: torch.tensor,
                true_label: int) -> torch.tensor:
    sample_indices = torch.from_numpy(get_image_sampling_pixels())
    sample_indices_tuple = (sample_indices[:, 0], sample_indices[:, 1],
                            sample_indices[:, 2])

    grads = get_grad_in_many_pixels(model, last_image, original_image,
                                    true_label, sample_indices)

    T_matrix[sample_indices_tuple] += 1
    M_matrix[sample_indices_tuple] = BETA_1 * M_matrix[
        *sample_indices_tuple] + (1 - BETA_1) * grads
    V_matrix[sample_indices_tuple] = BETA_2 * V_matrix[
        *sample_indices_tuple] + (1 - BETA_2) * grads**2

    T_i = T_matrix[sample_indices_tuple]
    M_i = M_matrix[sample_indices_tuple] / (1 - BETA_1 ** T_i )
    V_i = V_matrix[sample_indices_tuple] / (1 - BETA_2 ** T_i )

    updated_pixel_values = last_image[sample_indices_tuple] + (
        -learning_rate * M_i / (V_i**0.5 + ADAM_EPS))
    clamped_updates = torch.clamp(updated_pixel_values,
                                  image_value_lower_bound,
                                  image_value_upper_bound)
    last_image[sample_indices_tuple] = clamped_updates

    return last_image


def attack(model, image, label):
    T_matrix = torch.zeros((n_channel, height, width))
    M_matrix = torch.zeros((n_channel, height, width))
    V_matrix = torch.zeros((n_channel, height, width))

    loss = []
    last_image = image.clone()
    success = False
    for i in range(100):
        next_image = single_step(T_matrix, M_matrix, V_matrix, model,
                                 last_image, image, label)

        pred_prob = model(next_image).softmax(dim=1)
        pred_label = pred_prob.argmax().item()
        if pred_label != label:
            success = True
            break
        cur_loss = get_loss(image, label,
                            next_image.reshape(1, *image_shape),
                            pred_prob.reshape(1, -1))
        loss.append(cur_loss[0])

        last_image = next_image

    return success, next_image, loss


from model_helpers import get_model_and_optimizer

model, optimizer = get_model_and_optimizer(
    r'C:\research\zoo_attack\models\simple-2024-3-24-20-45-56.pt')
import torchvision.transforms as transforms

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

import torchvision

# Load CIFAR-10 dataset
trainset = torchvision.datasets.CIFAR10(root='./data',
                                        train=True,
                                        download=False,
                                        transform=transform)

np.random.seed(1)
result, result_images, losses = [], [], []

from time import time

with torch.no_grad():
    for i in range(100):
        image, label = trainset[i]
        model_pred = model(image).argmax().item()
        if label == model_pred:
            t1 = time()
            is_success, attack_image, loss = attack(model, image, label)
            result += [is_success]
            result_images += [(image, attack_image)]
            losses += [loss]
            t2 = time()
            print(i, (t2 - t1) / (len(loss) + 1))