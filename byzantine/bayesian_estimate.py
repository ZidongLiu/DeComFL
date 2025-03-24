import torch
from torch.distributions import Gamma
import numpy as np
from scipy.stats import norm
from byzantine.aggregation import mean

'''
    Now, we assume that the local gradients are 8 * 1 * 1 dimensional list[list[tensor]]. 
    Local update=1
    len(tensor)=1, len(first_list=8), len(second_list=1)
'''

def Bayesian_estimate(local_grad_scalar_list: list[list[torch.Tensor]], 
                      mean: float, 
                      variance: float, 
                      pi: float, 
                      list_flag: list)->list[list[torch.Tensor]]: 
    updated_pi = sample_pi(list_flag=list_flag)
    updated_list_flag = sample_flag(list_scalar=local_grad_scalar_list, 
                                    pi_val=updated_pi, 
                                    mean=mean, 
                                    std=variance**0.5)
    updated_local_grad_scalar_list = sample_scalar(local_grad_scalar_list=local_grad_scalar_list, 
                                           mean=mean, 
                                           variance=variance**0.5, 
                                           list_flag=updated_list_flag)
    updated_mean = sample_and_update_mean(list_scalar=updated_local_grad_scalar_list, 
                                         list_flag=updated_list_flag, 
                                         mean=mean, 
                                         variance=variance, 
                                         tau0_sq=1)
    updated_variance = sample_variance(list_scalar=updated_local_grad_scalar_list,
                                      list_flag=updated_list_flag,
                                      mean=updated_mean,
                                      alpha=1,
                                      beta=1)    
    
    # updated_local_grad_scalar_list is num_clients * 1 * 1 dimensional list[list[tensor]]
    global_grad_scalar = mean(updated_local_grad_scalar_list)
    return updated_pi, updated_list_flag, updated_mean, updated_variance, global_grad_scalar


def sample_pi(alpha_pi: float=1, beta_pi: float=1, list_flag: list=[])->float:
    num_modified_scalars = np.sum(list_flag)
    num_total_scalars = len(list_flag)
    alpha_post = alpha_pi + num_modified_scalars
    beta_post = beta_pi + num_total_scalars - num_modified_scalars
    sampled_pi = np.random.beta(alpha_post, beta_post)
    return sampled_pi


def sample_flag(list_scalar: list[list[torch.tensor]], 
                pi_val: float, 
                mean: float, 
                std: float, 
                outlier_pdf: float=1e-3)->list: 
    
    list_flag = [0] * len(list_scalar)
    
    for i in range(len(list_scalar)): 
        pdf_inlier = norm.pdf(list_scalar[i][0].item(), loc=mean, scale=std)
        pdf_outlier = outlier_pdf
        denom = pi_val * pdf_outlier + (1 - pi_val) * pdf_inlier
        if denom == 0.0: 
            p_z1 = 0.5
        else: 
            p_z1 = (pi_val * pdf_outlier) / denom
        
        list_flag[i] = np.random.binomial(1, p_z1)
    
    return list_flag


def sample_scalar(list_scalar: list[list[torch.tensor]], 
                  mean: float, 
                  variance: float, 
                  list_flag: list)->list[list[torch.tensor]]:
    assert len(list_scalar)==len(list_flag)
    for i in range(len(list_scalar)):
        if list_flag[i] == 1: 
            new_scalar = torch.normal(mean=mean, std=variance)
            list_scalar[i][0] = torch.tensor(new_scalar)
    return list_scalar


def sample_and_update_mean(list_scalar: list[list[torch.tensor]], 
                           list_flag: list,
                           mean: float, 
                           variance: float, 
                           tau0_sq: float)->float: 
    sum = 0
    num_flag = 0
    
    for i in range(len(list_scalar)):
        if list_flag.item() == 0:
            sum += list_scalar[i][0].item()
            num_flag += 1
    
    if num_flag == 0: 
        mean_sample = mean * torch.randn(1).item() * (tau0_sq ** 0.5)
        return mean_sample

    tau_post_sq = 1.0 / (1.0 / tau0_sq + num_flag / variance)
    mean_post = tau_post_sq * (mean / tau0_sq + sum / variance)
    mean_sample = mean_post + torch.randn(1).item() * (tau_post_sq ** 0.5)
    return mean_sample


def sample_variance(list_scalar: list[list[torch.tensor]], list_flag, mean, alpha, beta)->float:
    sum_sq = 0.0
    num_flag = 0
    for i in range(len(list_scalar)):
        if list_flag[i] == 0:
            val = list_scalar[i][0].item()
            diff = val - mean
            sum_sq += diff * diff
            num_flag += 1
            
    alpha_post = alpha + num_flag / 2.0
    beta_post  = beta  + 0.5 * sum_sq
    gamma_dist = Gamma(alpha_post, beta_post)
    gamma_sample = gamma_dist.sample()
    sigma2_sample = 1.0 / gamma_sample.item()
    
    return sigma2_sample
