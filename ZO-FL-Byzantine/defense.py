import numpy as np
from scipy.stats import norm

def run_defense(g, num_iters=100, tol=1e-5, alpha_pi=1, beta_pi=1):
    g = g.copy()
    M = len(g)
    z = np.zeros(M, dtype=int)
    mu = np.mean(g)
    sigma2 = np.var(g) + 1e-6
    
    for it in range(num_iters):
        mu_old = mu
        sigma2_old = sigma2
        pi_val = np.random.beta(alpha_pi + np.sum(z), beta_pi + M - np.sum(z))
        new_z = np.zeros(M, dtype=int)
        for i in range(M):
            pdf_val = norm.pdf(g[i], loc=mu, scale=np.sqrt(sigma2))
            prob_modified = pi_val / (pi_val + (1 - pi_val) * pdf_val)
            new_z[i] = int(np.random.rand() < prob_modified)
        z = new_z
        for i in range(M):
            if z[i] == 1:
                g[i] = np.random.normal(mu, np.sqrt(sigma2))
        unmodified_indices = (z == 0)
        if np.sum(unmodified_indices) > 0:
            mu = np.mean(g[unmodified_indices])
            sigma2 = np.var(g[unmodified_indices]) + 1e-6
        else:
            mu = mu_old
            sigma2 = sigma2_old
        if abs(mu - mu_old) < tol and abs(sigma2 - sigma2_old) < tol:
            break

    recovered_g = g
    return recovered_g, mu, sigma2, pi_val, z
