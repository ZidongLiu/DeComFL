import torch
import numpy as np

def Bayesian_estimate(g, max_iter=10, tol=1e-3):
    flags = torch.zeros_like(g)
    mu = torch.mean(g)
    sigma2 = torch.var(g, unbiased=True)
    pi = 0.5

    for iter in range(max_iter):
        pi_new = sample_pi(flags, alpha_prior=1.0, beta_prior=1.0)
        
        flags_new = sample_flag(g, mu, sigma2, pi_new)
        
        g_recovered = g.clone()
        for idx in range(g.numel()):
            if flags_new[idx].item() == 1:
                g_recovered[idx] = sample_scalar(g[idx], 1, mu, sigma2)
        
        mu_new = sample_mean(g_recovered, flags_new)
        sigma2_new = sample_variance(g_recovered, flags_new, mu_new)
        
        if (torch.abs(mu_new - mu) < tol and 
            torch.abs(sigma2_new - sigma2) < tol and 
            abs(pi_new - pi) < tol):
            mu, sigma2, pi, flags = mu_new, sigma2_new, pi_new, flags_new
            g = g_recovered
            break
        
        mu, sigma2, pi, flags = mu_new, sigma2_new, pi_new, flags_new
        g = g_recovered

    return mu, sigma2, pi, flags, g

def sample_pi(flags, alpha_prior=1.0, beta_prior=1.0):
    n = flags.numel()
    sum_z = torch.sum(flags).item()
    alpha_post = alpha_prior + sum_z
    beta_post = beta_prior + n - sum_z
    beta_dist = torch.distributions.Beta(alpha_post, beta_post)
    return beta_dist.sample().item()

def sample_flag(g, mu, sigma2, pi):
    norm_pdf = (1.0 / torch.sqrt(2 * torch.pi * sigma2)) * torch.exp(-((g - mu) ** 2) / (2 * sigma2))
    p_modified = pi / (pi + (1 - pi) * norm_pdf)
    flags = torch.bernoulli(p_modified)
    return flags

def sample_scalar(g, flag, mu, sigma2):
    if flag == 0:
        return g
    else:
        return torch.normal(mu, torch.sqrt(sigma2))

def sample_mean(g, flags):
    unmodified = g[flags == 0]
    if unmodified.numel() > 0:
        unmodified_np = unmodified.cpu().numpy()
        mu_n_np = np.mean(unmodified_np)
        if unmodified_np.size > 1:
            var_np = np.var(unmodified_np, ddof=1)
        else:
            var_np = 0.0
        sigma_n2 = var_np / unmodified_np.size
        mu_n = torch.tensor(mu_n_np, dtype=g.dtype, device=g.device)
        std = torch.sqrt(torch.tensor(sigma_n2, dtype=g.dtype, device=g.device))
        if std < 1e-6:
            return mu_n
        return torch.normal(mu_n, std)
    else:
        return torch.mean(g)

def sample_variance(g, flags, mu):
    unmodified = g[flags == 0]
    n_x = unmodified.numel()
    if n_x == 0:
        return torch.var(g, unbiased=True)
    
    alpha_prior = 1.0
    beta_prior = 1.0
    alpha_post = alpha_prior + n_x / 2.0
    beta_post = beta_prior + 0.5 * torch.sum((unmodified - mu) ** 2)
    
    inv_gamma_dist = torch.distributions.InverseGamma(alpha_post, beta_post)
    return inv_gamma_dist.sample()
