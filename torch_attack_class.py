# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 16:15:41 2024

@author: Zidong
"""

from typing import TypedDict
from torch import nn
import torch
import numpy as np


class HyperParams(TypedDict):
    sample_per_channel: int
    n_channel: int
    height: int
    width: int
    h: float
    LOSS_EPS: float
    KAI: float
    BETA_1: float
    BETA_2: float
    ADAM_EPS: float
    learning_rate: float
    image_value_lower_bound: float
    image_value_upper_bound: float
    CLASSIFICATION_LOSS_MULTIPLIER: float
    N_LABEL: int


DEFAULT_HYPER_PARAMS: HyperParams = HyperParams(
    {
        'sample_per_channel': 50,
        'n_channel': 3,
        'height': 32,
        'width': 32,
        'h': 0.001,
        'LOSS_EPS': 1e-30,
        'KAI': 0.0,
        'BETA_1': 0.9,
        'BETA_2': 0.999,
        'ADAM_EPS': 1e-8,
        'learning_rate': 0.001,
        'image_value_lower_bound': -1.0,
        'image_value_upper_bound': 1.0,
        'CLASSIFICATION_LOSS_MULTIPLIER': 0.5,
        'N_LABEL': 10,
    }
)


class Attack:

    def __init__(self, model: nn.Module, hyper_params: HyperParams = DEFAULT_HYPER_PARAMS):
        # hyper_params
        self.sample_per_channel: int = hyper_params['sample_per_channel']
        self.n_channel: int = hyper_params['n_channel']
        self.height: int = hyper_params['height']
        self.width: int = hyper_params['width']
        self.h: float = hyper_params['h']
        self.LOSS_EPS: float = hyper_params['LOSS_EPS']
        self.KAI: float = hyper_params['KAI']
        self.BETA_1: float = hyper_params['BETA_1']
        self.BETA_2: float = hyper_params['BETA_2']
        self.ADAM_EPS: float = hyper_params['ADAM_EPS']
        self.learning_rate: float = hyper_params['learning_rate']
        self.image_value_lower_bound: float = hyper_params['image_value_lower_bound']
        self.image_value_upper_bound: float = hyper_params['image_value_upper_bound']
        self.CLASSIFICATION_LOSS_MULTIPLIER: float = hyper_params['CLASSIFICATION_LOSS_MULTIPLIER']
        self.N_LABEL: int = hyper_params['N_LABEL']
        # other
        self.model = model
        self.image_shape = (self.n_channel, self.height, self.width)
        # handy nonchanging stuff
        self.indexes_to_be_sampled: np.ndarray = np.arange(self.height * self.width)
        self._reset_adam_helper_matrices()
        self._set_image_example_data()

    def _generate_adam_helper_matrices(self):
        T_matrix = torch.zeros((self.n_channel, self.height, self.width))
        M_matrix = torch.zeros((self.n_channel, self.height, self.width))
        V_matrix = torch.zeros((self.n_channel, self.height, self.width))
        return T_matrix, M_matrix, V_matrix

    def _reset_adam_helper_matrices(self):
        self.T_matrix, self.M_matrix, self.V_matrix = self._generate_adam_helper_matrices()

    def _set_image_example_data(self, image: torch.tensor = None, label: int = None):
        if image is None and label is None:
            self.original_image = None
            self.image_label = None
        elif image is not None and label is not None:
            self.original_image = image.clone()
            self.image_label = label
        else:
            raise Exception('Not valid')

    def check_image_data_valid(self):
        if (self.original_image is None) or (self.image_label is None):
            raise Exception('Not valid')

    def sample_channel_height_width(self, channel_index: int):
        indices_1d = np.random.choice(self.indexes_to_be_sampled, self.sample_per_channel, replace=False)
        return np.stack(
            [channel_index * np.ones(self.sample_per_channel, dtype='int32'), indices_1d // self.width, indices_1d % self.width], axis=1
        )

    def get_image_sampling_pixels(self):
        channel_samplings = []
        for i in range(self.n_channel):
            channel_samplings.append(self.sample_channel_height_width(i))

        return np.concatenate(channel_samplings, axis=0)

    def get_loss(self, change_images: torch.tensor, change_images_label_probability: torch.tensor) -> torch.tensor:
        '''
        original_image: (n_channel, height, width)
        change_images: (n_images, n_channel, height, width)
        model_out_probability: (n_images, n_label = 10)
        '''
        self.check_image_data_valid()

        l2dist_loss = torch.sum((change_images[0] - self.original_image)**2)
        non_target_mask = torch.ones(self.N_LABEL, dtype=int)
        non_target_mask[self.image_label] = 0
        max_log_non_target_prob = torch.log(torch.max(change_images_label_probability[:, non_target_mask]) + self.LOSS_EPS)

        classification_loss = torch.log(change_images_label_probability[:, self.image_label]) - max_log_non_target_prob
        classification_loss[classification_loss < -self.KAI] = -self.KAI

        return l2dist_loss + self.CLASSIFICATION_LOSS_MULTIPLIER * classification_loss

    def get_grad_in_many_pixels(self, cur_image: torch.tensor, many_pixel_indices_3d: torch.tensor) -> torch.tensor:
        self.check_image_data_valid()

        n_sample = len(many_pixel_indices_3d)
        many_pixel_indices_tuple = (many_pixel_indices_3d[:, 0], many_pixel_indices_3d[:, 1], many_pixel_indices_3d[:, 2])
        grad_images = torch.clone(cur_image).unsqueeze(0).repeat(n_sample * 2, 1, 1, 1)
        plus_indexes: tuple = (range(n_sample), *many_pixel_indices_tuple)
        plus_tensor = grad_images[plus_indexes] + self.h

        minus_indexes: tuple = (range(n_sample, n_sample * 2), *many_pixel_indices_tuple)
        minus_tensor = grad_images[minus_indexes] - self.h

        plus_tensor[plus_tensor > self.image_value_upper_bound] = self.image_value_upper_bound
        minus_tensor[minus_tensor < self.image_value_lower_bound] = self.image_value_lower_bound
        grad_images[plus_indexes] = plus_tensor
        grad_images[minus_indexes] = minus_tensor

        grad_step_size = plus_tensor - minus_tensor

        ####
        grad_images_probability = self.model(grad_images).softmax(dim=1)
        grad_image_losses = self.get_loss(grad_images, grad_images_probability)

        sample_grads = (grad_image_losses[plus_indexes[0]] - grad_image_losses[minus_indexes[0]]) / grad_step_size

        return sample_grads

    def single_step(self, last_image: torch.tensor) -> torch.tensor:
        self.check_image_data_valid()

        T_matrix, M_matrix, V_matrix = self.T_matrix, self.M_matrix, self.V_matrix

        BETA_1, BETA_2, learning_rate, ADAM_EPS = self.BETA_1, self.BETA_2, self.learning_rate, self.ADAM_EPS
        sample_indices = torch.from_numpy(self.get_image_sampling_pixels())
        sample_indices_tuple = (sample_indices[:, 0], sample_indices[:, 1], sample_indices[:, 2])

        grads = self.get_grad_in_many_pixels(last_image, sample_indices)

        T_matrix[sample_indices_tuple] += 1
        M_matrix[sample_indices_tuple] = BETA_1 * M_matrix[sample_indices_tuple] + (1 - BETA_1) * grads
        V_matrix[sample_indices_tuple] = BETA_2 * V_matrix[sample_indices_tuple] + (1 - BETA_2) * grads**2

        T_i = T_matrix[sample_indices_tuple]
        M_i = M_matrix[sample_indices_tuple] / (1 - BETA_1**T_i)
        V_i = V_matrix[sample_indices_tuple] / (1 - BETA_2**T_i)

        updated_pixel_values = last_image[sample_indices_tuple] + (-learning_rate * M_i / (V_i**0.5 + ADAM_EPS))
        clamped_updates = torch.clamp(updated_pixel_values, self.image_value_lower_bound, self.image_value_upper_bound)
        last_image[sample_indices_tuple] = clamped_updates

        return last_image

    def attack(self, image: torch.tensor, label: int, iteration: int = 100):
        with torch.no_grad():
            self._set_image_example_data(image, label)

            loss = []
            success = False
            last_image = image.clone()
            for i in range(iteration):
                next_image = self.single_step(last_image)
                pred_prob = self.model(next_image).softmax(dim=1)
                pred_label = pred_prob.argmax().item()
                if pred_label != label:
                    success = True
                    break
                cur_loss = self.get_loss(next_image.reshape(1, *self.image_shape), pred_prob.reshape(1, -1))
                loss.append(cur_loss[0])

                last_image = next_image

            return success, next_image, loss
