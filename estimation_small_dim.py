"""
This script reproduces the experiment of section 5.1.1 on parameter estimation in low dimension.
The constants are named as follow:
- n_samples is the number of model samples. In this experiment, n_samples=100.
- n is the number of nodes. In this experiment, n=3.
- p is the number of orthonormal columns. In this experiment, p=2.

The parameter theta is composed of:
- F and mu designate the vMF parameter and the mean eigenvalues
- sigma and sigma_l in the code correspond to sigma_epsilon and sigma_lambda in the paper

The variables used in the code are:
- Xs (n_samples,n,p) is the list of vMF samples (X_1, ..., X_N)
- ls (n_samples,p) is the list of patterns amplitudes for each individual (lambda_1, ..., lambda_N)
- Xs_mh and ls_mh are the MCMC estimate of Xs and ls once the MCMC has converged
"""

import numpy as np
from matplotlib.pyplot import *
import os

from src.utils import *
from src.stiefel import *
from src import spa, vmf, mcmc, model, saem

np.random.seed(0)
set_cmap("bwr")
fig_folder = "figures"
try:
    os.mkdir(fig_folder)
except:
    pass


#==============================
# EXPERIMENT 1 : SMALL NOISE
#==============================

print("Experiment: parameter estimation with small noise.")
print("Generating the synthetic data set.")

F0 = [25,10]*unif(3,2)
mu0 = np.array([20.,10.])
mu0 = mu0[np.abs(mu0).argsort()][::-1]
sigma0 = 0.1 # sigma_epsilon
sigma_l0 = 2 # sigma_lambda

theta0 = (F0, mu0, sigma0, sigma_l0)

n_samples = 100
n, p = F0.shape

ls = mu0[None,:].repeat(n_samples, axis=0)
ls += sigma_l0*np.random.randn(*ls.shape)
Xs = vmf.sample_von_mises_fisher(F0, n_iter=n_samples, burn=10000, stride=100)
As = comp(Xs, ls)
idx = np.triu_indices(F0.shape[0])
noise = sigma0*np.random.randn(*As.shape)
noise[:,idx[0],idx[1]] = noise[:,idx[1],idx[0]] # Symmetrize the noise
As += noise

# Initialize the parameters at randomly chosen values
print("Initializing the MCMC-SAEM algorithm.")
F = 5*np.random.randn(*F0.shape)
mu = np.random.randn(F.shape[1])


# Perform initial MCMC steps on X and lambda
prop_X = 0.02
prop_l = 0.1
Xs_mh, ls_mh, _, _, _ = mcmc.mh(As, (F, mu, 1, 1), n_iter=200, init=None, prop_X=prop_X, prop_l=prop_l)

# Run the MCMC-SAEM for 100 iterations with 20 MCMC steps per SAEM step
print("Running the MCMC-SAEM algorithm.")
result = saem.mcmc_saem(As, Xs_mh, ls_mh, (F, mu, 1, 1), n_iter=100, n_mcmc=20)


#==============================
# EXPERIMENT 2 : STRONG NOISE
#==============================

print("Experiment: parameter estimation with strong noise")
print("Generating the synthetic data set.")

F0_strong = F0.copy()
mu0_strong = mu0.copy()

sigma0_strong = 4
sigma_l0_strong = 2
n_samples = 100

ls_strong = ls.copy()
Xs_strong = Xs.copy()
As_strong = comp(Xs_strong, ls_strong)
idx = np.triu_indices(F0.shape[0])
noise = sigma0_strong*np.random.randn(*As.shape)
noise[:,idx[0],idx[1]] = noise[:,idx[1],idx[0]] # Symmetrize the noise
As_strong += noise


# Initialize the parameters at randomly chosen values
print("Initializing the MCMC-SAEM algorithm.")
F = 5*np.random.randn(*F0.shape)
mu = np.random.randn(F.shape[1])

# Perform initial MCMC steps on X and lambda
prop_X = 0.02
prop_l = 0.1
Xs_mh, ls_mh, _, _, _ = mcmc.mh(As_strong, (F, mu, 1, 1), n_iter=200, init=None, prop_X=prop_X, prop_l=prop_l)

# Run the MCMC-SAEM for 100 iterations with 20 MCMC steps per SAEM step
print("Running the MCMC-SAEM algorithm.")
result_strong = saem.mcmc_saem(As_strong, Xs_mh, ls_mh, (F, mu, 1, 1), n_iter=100, n_mcmc=20)

#==============================
# DISPLAY THE RESULTS
#==============================

# Open results with small noise
theta = result["theta"]
Xs_mh = result["Xs_mh"]
ls_mh = result["ls_mh"]

# Align the signs of the columns of F to the ground truth to ease visualization
Xs_mh, _, (F, _, _, _), m, s = saem.map_to_ground_truth(Xs_mh, ls_mh, theta, theta0)
Xs_mhs = result["history"]["Xs_mh"]
Xs_mh_bar = s*np.array([proj_V(x) for x in np.mean(Xs_mhs[50:], axis=0)])[:,:,m]

# Open results with strong noise
theta_strong = result_strong["theta"]
Xs_mh_strong = result_strong["Xs_mh"]
ls_mh_strong = result_strong["ls_mh"]

# Align the signs of the columns of F to the ground truth to ease visualization
Xs_mh_strong, _, (F_strong, _, _, _), m, s = saem.map_to_ground_truth(Xs_mh_strong, ls_mh_strong, theta_strong, theta0)
Xs_mhs_strong = result_strong["history"]["Xs_mh"]
Xs_mh_bar_strong = s*np.array([proj_V(x) for x in np.mean(Xs_mhs_strong[50:], axis=0)])[:,:,m]


def set_3D_plot():
    ax = gca()
    ticks = [-1,-0.5,0,0.5,1]
    ax.set_xlim(-1,1);ax.set_ylim(-1,1);ax.set_zlim(-1,1)
    ax.set_xticks(ticks);ax.set_yticks(ticks);ax.set_zticks(ticks)

font = 17

fig = figure(figsize=(15,5))
subplots_adjust(left=.01, right=.99, wspace=0.0)
ax = fig.add_subplot(131, projection="3d")
title("(a)", fontsize=font)
ax.quiver(*np.zeros((3,2)), *proj_V(F0), color="red", linewidth=3)
ax.scatter(*Xs.transpose(1,0,2).reshape(3,-1), marker="o", alpha=0.5, color="green")
set_3D_plot()

ax = fig.add_subplot(132, projection="3d")
title("(b)", fontsize=font)
ax.quiver(*np.zeros((3,2)), *proj_V(F), color="red", linewidth=3)
ax.scatter(*Xs_mh_bar.transpose(1,0,2).reshape(3,-1), marker="o", alpha=0.5)
set_3D_plot()

ax = fig.add_subplot(133, projection="3d")
title("(c)", fontsize=font)
ax.quiver(*np.zeros((3,2)), *proj_V(F_strong), color="red", linewidth=3)
ax.scatter(*Xs_mh_bar_strong.transpose(1,0,2).reshape(3,-1), marker="o", alpha=0.5)
set_3D_plot()

fig_path = fig_folder+"/small_dim.pdf"
savefig(fig_path)
print(f"Saved figure at {fig_path}.")

print("True concentration parameters:")
print(norm(F0, axis=0))

print("Estimated concentration parameters with small noise:")
print(norm(F, axis=0))

print("Estimated concentration parameters with strong noise:")
print(norm(F_strong, axis=0))