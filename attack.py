# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 21:06:55 2024

@author: Zidong
"""
from matplotlib import pyplot as plt
from model_helpers import get_model_and_optimizer
import torchvision.transforms as transforms
import torchvision
from time import time
import numpy as np
import torch
from os import path

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
    indices_1d = np.random.choice(indexes_to_be_sampled, sample_per_channel, replace=False)
    return np.stack([channel_index * np.ones(sample_per_channel, dtype='int32'), indices_1d // width, indices_1d % width], axis=1)


def get_image_sampling_pixels():
    channel_samplings = []
    for i in range(n_channel):
        channel_samplings.append(sample_channel_height_width(i))

    sampling_indices_3d = np.concatenate(channel_samplings, axis=0)

    return sampling_indices_3d


def get_loss(original_image: np.ndarray, image_true_label: int, change_images: np.ndarray, change_images_label_probability: np.ndarray):
    '''
    original_image: (n_channel, height, width)
    change_images: (n_images, n_channel, height, width)
    model_out_probability: (n_images, n_label = 10)
    '''

    l2dist_loss = np.sum((change_images[0] - original_image)**2)
    non_target_mask = np.ones(N_LABEL, dtype=int)
    non_target_mask[image_true_label] = 0
    max_log_non_target_prob = np.log(np.max(change_images_label_probability[:, non_target_mask]) + LOSS_EPS)

    classification_loss = np.log(change_images_label_probability[:, image_true_label]) - max_log_non_target_prob
    classification_loss[classification_loss < -KAI] = -KAI

    return l2dist_loss + CLASSIFICATION_LOSS_MULTIPLIER * classification_loss


def get_grad_in_1_pixel(model, cur_image, original_image, image_true_label: int, pixel_index_3d: np.ndarray):
    cur_pixel_value = cur_image[tuple(pixel_index_3d)]
    pixel_minus = cur_pixel_value + h
    pixel_plus = cur_pixel_value - h
    if pixel_minus < image_value_lower_bound or pixel_plus > image_value_upper_bound:
        return None

    image_offset = torch.zeros(image_shape)
    image_offset[tuple(pixel_index_3d)] = h
    grad_images = torch.stack([cur_image - image_offset, cur_image + image_offset])
    model_out = model(grad_images)
    model_out_probability = model_out.softmax(dim=1).detach().numpy()

    grad_image_loss = get_loss(original_image.detach().numpy(), image_true_label, grad_images.detach().numpy(), model_out_probability)
    grad = (grad_image_loss[1] - grad_image_loss[0]) / (2 * h)
    return grad


def single_step(
    T_matrix: np.ndarray, M_matrix: np.ndarray, V_matrix: np.ndarray, model: torch.nn.Module, last_image: torch.tensor,
    original_image: torch.tensor, true_label: int
) -> torch.tensor:
    sample_indices = get_image_sampling_pixels()
    next_image = torch.clone(last_image)

    for sample_index in sample_indices:
        grad_i = get_grad_in_1_pixel(model, last_image, original_image, true_label, sample_index)
        if grad_i is None:
            continue
        T_matrix[tuple(sample_index)] += 1
        M_matrix[tuple(sample_index)] = BETA_1 * M_matrix[tuple(sample_index)] + (1 - BETA_1) * grad_i
        V_matrix[tuple(sample_index)] = BETA_2 * V_matrix[tuple(sample_index)] + (1 - BETA_2) * grad_i**2

        T_i = T_matrix[tuple(sample_index)]
        M_i = M_matrix[tuple(sample_index)] / (1 - BETA_1**T_i)
        V_i = V_matrix[tuple(sample_index)] / (1 - BETA_2**T_i)

        pixel_value = next_image[tuple(sample_index)] + (-learning_rate * M_i / (V_i**0.5 + ADAM_EPS))
        if pixel_value > image_value_upper_bound:
            pixel_value = image_value_upper_bound

        if pixel_value < image_value_lower_bound:
            pixel_value = image_value_lower_bound

        next_image[tuple(sample_index)] = pixel_value

    return next_image


def attack(model, image, label):
    T_matrix = np.zeros((n_channel, height, width))
    M_matrix = np.zeros((n_channel, height, width))
    V_matrix = np.zeros((n_channel, height, width))

    np.random.seed(1)

    loss = []
    last_image = image
    success = False
    for i in range(100):
        next_image = single_step(T_matrix, M_matrix, V_matrix, model, last_image, image, label)

        pred_prob = model(next_image).softmax(dim=1).detach().numpy()
        pred_label = np.argmax(pred_prob)
        if pred_label != label:
            success = True
            break
        cur_loss = get_loss(image.detach().numpy(), label, next_image.detach().numpy().reshape(1, *image_shape), pred_prob.reshape(1, -1))
        loss.append(cur_loss[0])

        last_image = next_image

    return success, next_image, i


def compare_images(image1, image2):
    f, axarr = plt.subplots(1, 2)
    axarr[0].imshow((image1.permute(1, 2, 0) + 1) / 2)
    axarr[1].imshow((image2.permute(1, 2, 0) + 1) / 2)


model, optimizer = get_model_and_optimizer(path.join(path.dirname(__file__), 'models/simple-2024-3-24-20-45-56.pt'))

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# Load CIFAR-10 dataset
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=False, transform=transform)

result, result_images = [], []
# success percentage = 32.5%

with torch.no_grad():
    for i in range(3):
        image, label = trainset[i]
        model_pred = int(model(image).argmax())
        if label == model_pred:
            t1 = time()
            is_success, attack_image, n_ite = attack(model, image, label)
            result += [is_success]
            result_images += [(image, attack_image)]
            t2 = time()
            print(i, (t2 - t1) / (n_ite + 1))
