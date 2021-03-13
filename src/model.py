"""
This file contains functions to compute the model's full and partial log-densities, as well as their gradients.
The variables are named as follows:
- As in the code corresonds to (A_1, ..., A_N) in the paper
- Xs in the code corresonds to (X_1, ..., X_N) in the paper
- ls in the code corresonds to (lambda_1, ..., lambda_N) in the paper

The partial log-likelihood functions, which are used extensively in the MCMC-SAEM, are compiled with Numba.
In some function, including the normalizing constant is optional, as it is the most time-intensive step.

The parameter theta is composed of:
- F and mu designate the vMF parameter and the mean eigenvalues
- sigma and sigma_l in the code correspond to sigma_epsilon and sigma_lambda in the paper
"""

import numpy as np
from numba import njit

import src.stiefel as st
from src import spa


def log_lk(Xs, ls, As, theta, normalized=False):
    """
    Log-density [Xs, ls, As | theta]
    """
    n_samples, n_nodes, _ = As.shape
    F, mu, sigma, sigma_l = theta
    res = 0
    
    # (As | Xs, ls, sigma)
    Xs_comp = st.comp(Xs, ls)
    res += -0.5*((As-Xs_comp)**2).sum()/sigma**2 - n_samples*(n_nodes**2)*np.log(sigma)
    
    # (Xs | F)
    res += (F*Xs).sum()
    if normalized:
        res += -n_samples*spa.log_vmf(F)
    
    # (ls | mu, sigma_l)
    p = F.shape[1]
    res += -0.5*((ls-mu[None,:])**2).sum()/sigma_l**2 - n_samples*p*np.log(sigma_l)
    
    return res


@njit
def log_lk_partial(X, l, A, theta, normalized=False):
    """
    Log-density [Xs[i], ls[i], As[i] | theta]: log-likelihood term for one individual
    """
    n_nodes, _ = A.shape
    F, mu, sigma, sigma_l = theta
    res = 0
    # (As | Xs)
    X_comp = st.comp_numba_single(X, l)
    As_Xs = -0.5*((A-X_comp)**2).sum()/sigma**2 - (n_nodes**2)*np.log(sigma)
    res += As_Xs
    
    # (Xs | F)
    res += (F*X).sum()
    if normalized:
        res += -spa.log_vmf(F)
        
    # (ls | mu, sigma_l)
    p = F.shape[1]
    res += -0.5*((l-mu)**2).sum()/sigma_l**2 - p*np.log(sigma_l)
    
    return res

@njit
def log_lk_partial_grad_lambda(X, l, A, theta):
    """Gradient of log_lk_partial with respect to lambda"""
    F, mu, sigma, sigma_l = theta
    n, p = X.shape
    grad = -(1/sigma**2 + 1/sigma_l**2) * l
    tmp = np.zeros(p)
    for k in range(p):
        for i in range(n):
            for j in range(n):
                tmp[k] += X[i,k] * A[i,j] * X[j,k]
    grad += tmp / sigma**2
    grad += mu / sigma_l**2
    return grad

@njit
def log_lk_partial_grad_X(X, l, A, theta):
    """Riemannian gradient of log_lk_partial with respect to X"""
    F, mu, sigma, sigma_l = theta
    grad_E = A@X@np.diag(l) / sigma**2 + F
    grad_R = (grad_E@X.T - X@grad_E.T)@X # Riemannian gradient for the canonical metric
    return grad_R