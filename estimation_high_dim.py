"""
This script reproduces the experiment of section 5.1.1 on parameter estimation in high dimension.
The constants are named as follow:
- n_samples is the number of model samples. In this experiment, n_samples=200.
- n is the number of nodes. In this experiment, n=40.
- p is the number of orthonormal columns. In this experiment, p=20.

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


print("Experiment: parameter estimation in high dimension.")
print("Generating the synthetic data set.")

n = 40
p = 20
svf = 2*[100]+4*[60]+6*[40]+8*[20]
F0 = svf*unif(n, p)
mu0 = np.array([60,-60,50,-50,40,-40,30,-30,30,-30,20,-20,20,-20,10,-10,10,-10,10,-10], dtype=np.float)

sigma0 = 1   # sigma_epsilon
sigma_l0 = 2 # sigma_lambda
n_samples = 200

theta0 = (F0, mu0, sigma0, sigma_l0)
n_nodes = F0.shape[0]
ls = mu0[None,:].repeat(n_samples, axis=0)
ls += sigma_l0*np.random.randn(*ls.shape)

Xs = vmf.sample_von_mises_fisher(F0, n_iter=n_samples, burn=10000, stride=100)
As = comp(Xs, ls)
noise = sigma0*np.random.randn(*As.shape)
noise = (noise+noise.transpose(0,2,1))/np.sqrt(2)
As += noise

# Initialize the parameters
print("Initializing the MCMC-SAEM algorithm.")
theta_init, Xs_mh, ls_mh = saem.init_saem(As, p=p)

# Run the MCMC-SAEM for 100 iterations with 100 MCMC steps per SAEM step
print("Running the MCMC-SAEM algorithm.")
np.random.seed(0)
result = saem.mcmc_saem(As, Xs_mh, ls_mh, theta_init, n_mcmc=100, n_iter=100)


# Open results
theta = result["theta"]
F, mu, sigma, sigma_l = theta

# Align the signs of the columns of F to the ground truth to ease visualization
m, s = greedy_permutation(proj_V(F0), proj_V(F)) 
F = s*F[:,m]
mu = mu[m]
Xs_mh = s*result["Xs_mh"][:,:,m]
ls_mh = result["ls_mh"][:,m]
mode = proj_V(F)

Fs = result["history"]["F"]
mus = result["history"]["mu"]
sigmas = result["history"]["sigma"]
sigma_ls = result["history"]["sigma_l"]
Xs_mhs = result["history"]["Xs_mh"]
ls_mhs = result["history"]["ls_mh"]
for i in range(len(Fs)):
    Fs[i] = s*Fs[i][:,m]
    mus[i] = mus[i][m]
    Xs_mhs[i] = s*Xs_mhs[i][:,:,m]
    ls_mhs[i] = ls_mhs[i][:,m]


# Convergence figure
font1 = 14
font2 = 13
figure(figsize=(13,4))
subplot(1,2,1)
plot(np.minimum(sv(result["history"]["F"][:100]), 120), color="red", alpha=0.5)
for x, y in zip(sv(F0), sv(vmf.mle(Xs.mean(axis=0)))):
    axhline(x, c="black", linestyle="--", zorder=20, alpha=0.5)
ylim(bottom=0)
title(r"Convergence of $(|\hat f_1|, ..., |\hat f_p|)_t$", fontsize=font1)
xlabel("Iterations", fontsize=font2)
ylabel("Concentration parameters", fontsize=font2)

subplot(1,2,2)
plot(np.minimum(result["history"]["mu"][:100], 100), color="red", alpha=0.8)
for m in mu0:
    axhline(m, linestyle="--", c="black", alpha=1)
title(r"Convergence of $\hat \mu_t$", fontsize=font1)
xlabel("Iterations", fontsize=font2)
ylabel("Eivengalues", fontsize=font2)

fig_path = fig_folder+"/high_dim_convergence.pdf"
savefig(fig_path)
print(f"Saved figure at {fig_path}.")


# Figure for parameter F
M = np.abs(F0).max()
figure(figsize=(8,4))
subplot(1,3,1)
imshow(F0, vmin=-M, vmax=M)
no_axis(); colorbar()
title("True F")
subplot(1,3,2)
imshow(F, vmin=-M, vmax=M)
no_axis(); colorbar()
title("Estimated $\hat F$")
subplot(1,3,3)
imshow(F0-F, vmin=-M, vmax=M)
no_axis(); colorbar()
title("Difference $F-\hat F$")

fig_path = fig_folder+"/high_dim_F.pdf"
savefig(fig_path)
print(f"Saved figure at {fig_path}.")

# Figure for the columns of F (which correspond to the projection of F onto the Stiefel manifold)
M = 0.5
figure(figsize=(8,4))
subplot(1,3,1)
imshow(proj_V(F0), vmin=-M, vmax=M)
no_axis(); colorbar()
title("True $\pi_V(F)$")
subplot(1,3,2)
imshow(proj_V(F), vmin=-M, vmax=M)
no_axis(); colorbar()
title("Estimated $\pi_V(\hat F)$")
subplot(1,3,3)
imshow(proj_V(F0)-proj_V(F), vmin=-M, vmax=M)
no_axis(); colorbar()
title("Difference $\pi_V(F)-\pi_V(\hat F)$")

fig_path = fig_folder+"/high_dim_F_proj.pdf"
savefig(fig_path)
print(f"Saved figure at {fig_path}.")

print("Relative Root Mean Square Error for the template patterns:")
print(relative_error(F, F0))
