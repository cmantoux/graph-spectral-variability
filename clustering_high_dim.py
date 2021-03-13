"""
This script reproduces the experiment of section 5.1.3 on mixture models in high dimension.
The constants are named as follow:
- K is the number of clusters. In this experiment, K=4.
- n_samples is the number of model samples. In this experiment, n_samples=500.
- n is the number of nodes. In this experiment, n=20.
- p is the number of orthonormal columns. In this experiment, p=10.

The parameter theta is composed of:
- F and mu designate the list of vMF parameters and mean eigenvalues for each cluster
- sigma and sigma_l in the code correspond to sigma_epsilon and sigma_lambda in the paper
- pi is the list of cluster probabilities

The variables used in the code are:
- Xs (n_samples,n,p) is the list of vMF samples (X_1, ..., X_N)
- ls (n_samples,p) is the list of patterns amplitudes for each individual (lambda_1, ..., lambda_N)
- zs (n_samples) is the list of the categorical labels of each sample
- Xs_mh, ls_mh and zs_mh are the MCMC estimates of Xs and ls once the MCMC has converged
"""

import numpy as np
from matplotlib.pyplot import *
import os

from src.utils import *
from src.stiefel import *
from src import spa, vmf, mcmc, model, model_cluster, saem

np.random.seed(0)
set_cmap("bwr")
fig_folder = "figures"
try:
    os.mkdir(fig_folder)
except:
    pass

print("Experiment: mixture model estimation in high dimension.")
print("Generating the synthetic data set.")

n = 20
p = 10

# Generate the center of each cluster on the Stiefel manifold.
# X2, X3 and X4 are chosen to be small deviations of X1
X1 = unif(n, p)
X2 = proj_V(X1 + 0.15*np.random.randn(*X1.shape))
X3 = proj_V(X1 + 0.15*np.random.randn(*X1.shape))
X4 = proj_V(X1 + 0.15*np.random.randn(*X1.shape))
# Define the parameter F by scaling the cluster centers with the concentration parameters. In this experiment, all concentration parameters are taken equal for the sake of simplicity.
F0 = np.array([
    80 * X1,
    80 * X2,
    80 * X3,
    80 * X4
])
mu0 = np.array([
    [40,20,10,10,10,10,10,10,5,5],
    [20,20,10,10,10,10,10,10,5,5],
    [30,20,10,10,10,10,10,10,5,5],
    [35,25,10,10,10,10,10,10,5,5]
], dtype=np.float)
pi0 = np.ones(4)/4  # cluster probabilities

sigma0 = np.array([1,1,1,1])       # sigma_epsilon
sigma_l0 = np.array([20,20,20,20]) # sigma_lambda
n_samples = 500 

theta0 = (F0, mu0, sigma0, sigma_l0, pi0)
K, n, p = F0.shape

# Simulate matrices depending on their cluster
zs = np.random.choice(np.arange(K), p=pi0, size=n_samples)
ls = mu0[zs]
ls += sigma_l0[zs][:,None]*np.random.randn(*ls.shape)
Xs = np.zeros((n_samples, n, p))
for k in range(K):
    idx = np.where(zs==k)[0]
    Xs[idx] = vmf.sample_von_mises_fisher(F0[k], n_iter=len(idx), burn=10000, stride=10)
As = comp(Xs, ls)
idx = np.triu_indices(n)
noise = sigma0[zs][:,None,None]*np.random.randn(*As.shape)
noise[:,idx[0],idx[1]] = noise[:,idx[1],idx[0]]
As += noise

print("Initializing the MCMC-SAEM algorithm.")
theta_init, Xs_mh_init, ls_mh_init, zs_mh_init = saem.init_saem_cluster(As, p=10, K=4, n_iter=10, step=0.05)
Xs_mh_init, ls_mh_init, zs_mh_init, theta_init, ms, ss = saem.map_to_ground_truth_cluster(
    Xs_mh_init, ls_mh_init, zs_mh_init,
    theta_init, theta0)

print(f"Proportion of K-Means correct predictions: {np.round((zs_mh_init==zs).mean(), 3)}\n")


# Run the tempered MCMC-SAEM for mixtures for 1000 iterations with 20 MCMC steps per SAEM step and initial temperature 200.
print("Running the tempered MCMC-SAEM algorithm.")
np.random.seed(0)
result = saem.mcmc_saem_cluster(As, Xs_mh_init, ls_mh_init, zs_mh_init, theta_init, n_mcmc=20, n_iter=1000, T=200)


# Open result
theta = result["theta"]
F, mu, sigma, sigma_l, pi = theta
Xs_mh = result["Xs_mh"].copy()
ls_mh = result["ls_mh"].copy()
zs_mh = result["zs_mh"].copy()

# Permute the result to align with the ground truth clusters
Xs_mh, ls_mh, zs_mh, theta, ms, ss = saem.map_to_ground_truth_cluster(Xs_mh, ls_mh, zs_mh, theta, theta0)

print(f"Proportion of correct model predictions: {np.round((zs_mh==zs).mean(), 3)}\n")

print(f"Relative Root Mean Square Error for the estimation of the von Mises-Fisher parameters: {relative_error(F, F0, rnd=3)}")

print(f"Relative Root Mean Square Error for the estimation of the mean eigenvalues: {relative_error(mu, mu0, rnd=3)}")