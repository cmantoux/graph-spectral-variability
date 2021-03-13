"""
This script reproduces the experiment of section 5.1.3 on mixture models in low dimension.
The constants are named as follow:
- K is the number of clusters. In this experiment, K=3.
- n_samples is the number of model samples. In this experiment, n_samples=100.
- n is the number of nodes. In this experiment, n=3.
- p is the number of orthonormal columns. In this experiment, p=2.

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


print("Experiment: mixture model estimation in low dimension.")
print("Generating the synthetic data set.")

# Generate the center of each cluster on the Stiefel manifold.
# X1 and X2 are chosen to be close, and their distributions overlap
# X3 is defined far from X1 and X2
X1 = proj_V(np.array([
    [0.8,0.5],
    [0.2,-0.7],
    [0.5,-0.5]
]))
X2 = proj_V(X1 + 0.15*np.random.randn(*X1.shape))
X3 = proj_V(np.array([
    [0,-1],
    [-0.3,-0.3],
    [-1,0]
]))
# Define the parameter F by scaling X1, X2, X3 with the concentration parameters
F0 = np.array([
    [100,50]*X1,
    [20,20]*X2,
    [30,30]*X3
])
mu0 = np.array([
    [20,10],
    [20,10],
    [30,10]
], dtype=np.float)
pi0 = np.array([0.35, 0.35, 0.3]) # cluster probabilities

sigma0 = np.array([0.1, 0.2, 1]) # sigma_epsilon
sigma_l0 = np.array([2, 2, 3])   # sigma_lambda
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
theta, Xs_mh, ls_mh, zs_mh, lks = saem.init_saem_grad_cluster(As, p=2, K=3, n_iter=10, step=0.05)
Xs_mh, ls_mh, zs_mh, theta, ms, ss = saem.map_to_ground_truth_cluster(Xs_mh, ls_mh, zs_mh, theta, theta0)

print(f"Proportion of K-Means correct predictions: {np.round((zs_mh==zs).mean(), 3)}\n")


# Run the tempered MCMC-SAEM for mixtures for 1000 iterations with 20 MCMC steps per SAEM step and initial temperature 50.
print("Running the tempered MCMC-SAEM algorithm.")
np.random.seed(0)
result = saem.mcmc_saem_cluster(As, Xs_mh, ls_mh, zs_mh, theta, n_mcmc=20, n_iter=1000, T=50)


# Open result
theta = result["theta"]
F, mu, sigma, sigma_l, pi = theta
Xs_mh = result["Xs_mh"].copy()
ls_mh = result["ls_mh"].copy()
zs_mh = result["zs_mh"].copy()

# Permute the result to align with the ground truth clusters
Xs_mh, ls_mh, zs_mh, theta, ms, ss = saem.map_to_ground_truth_cluster(Xs_mh, ls_mh, zs_mh, theta, theta0)

print(f"Proportion of correct model predictions: {np.round((zs_mh==zs).mean(), 3)}\n")



# Figure representing the true and estimated latent Stiefel distributions

def set_3D_plot():
    ax = gca()
    ticks = [-1,-0.5,0,0.5,1]
    font = 17
    ax.set_xlim(-1,1);ax.set_ylim(-1,1);ax.set_zlim(-1,1)
    ax.set_xticks(ticks);ax.set_yticks(ticks);ax.set_zticks(ticks)

font = 14
fig = figure(figsize=(15,10))
tt = [f"({chr(k)})" for k in range(97, 103)]
for k in range(K):
    ax = fig.add_subplot(2, 3, k+1, projection="3d")
    set_3D_plot()
    ax.quiver(*np.zeros((3,2)), *proj_V(F0[k]), color="red", linewidth=3)
    idx = np.where(zs==k)[0]
    ax.scatter(*Xs[idx].transpose(1,0,2).reshape(3,-1), marker="o", alpha=0.5, color="green")
    title(tt[k], fontsize=font)
for k in range(K):
    ax = fig.add_subplot(2, 3, k+4, projection="3d")
    set_3D_plot()
    ax.quiver(*np.zeros((3,2)), *proj_V(F[k]), color="red", linewidth=3, zorder=100)
    idx = np.where(zs_mh==k)[0]
    ax.scatter(*Xs_mh[idx].transpose(1,0,2).reshape(3,-1), marker="o", alpha=0.5)
    title(tt[3+k], fontsize=font)

fig_path = fig_folder+"/clustering_small_dim.pdf"
savefig(fig_path)
print(f"Saved figure at {fig_path}.")


print(f"Relative Root Mean Square Error for the estimation of the mean eigenvalues: {relative_error(mu, mu0, rnd=3)}")