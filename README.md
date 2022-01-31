# Understanding the Spectral Variability of Graph Data Sets through Statistical Modeling on the Stiefel Manifold

This repository hosts the code for the [paper of the same name](https://www.mdpi.com/1099-4300/23/4/490).

## Requirements

The code was tested on Python 3.8. In order to run the code, the Python packages listed in `requirements.txt` are needed. They can be installed for instance with `conda`:

```
conda create -n gsv -c conda-forge --file requirements.txt
conda activate gsv
```

On a 2,9 GHz Intel Core i5 double core processor, the cumulated running time of all experiments does not exceed half an hour.

## Reproducing the experiments

The experiments presented on simulated data can be reproduced by running the following scripts with `python script_name.py`.

- `estimation_small_dim.py` and `estimation_high_dim.py` reproduce the results of section 5.1.1.

- `imputation.py` reproduces the missing link imputation experiment of section 5.1.2.

- `clustering_small_dim.py` and `clustering_high_dim.py` reproduce the experiments on mixture models in section 5.1.3.

## How to run the algorithm

The implementation we provide can be used on other data sets of weighted or binary networks. It takes as input a list of adjacency matrices with shape `(n_samples, n, n)``.  In order to run the MCMC-SAEM algorithm and retrieve the optimization results, use the following code.

#### Parameter estimation

```python
from src import saem

As = ... # Array of adjacency matrices with shape (n_samples, n, n)
p = ... # Number of eigenvectors used in the model

# Initialize the parameter theta and the MCMC on X (Xs_mh) and lambda (ls_mh)
theta_init, Xs_mh, ls_mh = saem.init_saem(As, p=p) # to a perform gradient ascent on X, use saem.init_saem_grad instead

# Run the MCMC-SAEM algorithm for 100 iterations with 20 MCMC steps per SAEM iteration
# By default, the algorithm does not store the MCMC values of X and lambda along the trajectory.
# This behavior can be changed by using the argument history=True.
result = saem.mcmc_saem(As, Xs_mh, ls_mh, theta_init, n_iter=100, n_mcmc=20, history=True)

# Retrieve the optimal parameters
theta = result["theta"]
F, mu, sigma, sigma_l = theta

# Retrieve the last MCMC samples...
Xs_mh = result["Xs_mh"]
ls_mh = result["ls_mh"]
# ...or the entire chain (requires history=True)
Xs_mh_history = result["history"]["Xs_mh"]
ls_mh_history = result["history"]["ls_mh"]
```

#### Missing link imputation

Imputing the value of missing link requires an estimation `theta`of the model parameters.

```python
from src import mcmc

A = ... # Adjacency matrix with missing coefficients
theta = ... # Estimated model parameters

mask_x = [...] # List of the row indices of masked coefficients
mask_y = [...] # List of the column indices of masked coefficients
mask = [mask_x, mask_y]

# Compute the MAP of X, lambda and the missing coefficients given the others.
# log_lk is the history of log-likelihood values during the algorithm
A_map, X_map, l_map, log_lk = mcmc.map_mask(A, mask, theta, n_iter=5000)

# Compute MCMC posterior samples of X, lambda and the missing coefficients given the others.
A_mcmc, X_mcmc, l_mcmc, log_lk = mcmc.mh_mask(A, mask, theta, n_iter=10000)
```

#### Mixture model estimation

The mixture model usage is very similar to the base model:

```python
from src import saem

As = ... # Array of adjacency matrices with shape (n_samples, n, n)
p = ... # Number of eigenvectors used in the model

# Initialize the parameter theta and the MCMC on X (Xs_mh), lambda (ls_mh) and the labels z (zs_mh)
theta, Xs_mh, ls_mh, zs_mh = saem.init_saem_cluster(As, p=p) # to a perform gradient ascent on X, use saem.init_saem_grad_cluster instead

# Run the MCMC-SAEM algorithm for 1000 iterations with 20 MCMC steps per SAEM iteration and initial temperature 50
# By default, the algorithm does not store the MCMC values of X and lambda along the trajectory.
# This behavior can be changed by using the argument history=True.
result = saem.mcmc_saem_cluster(As, Xs_mh, ls_mh, zs_mh, theta, n_iter=1000, n_mcmc=20, T=50, history=True)

# Retrieve the optimal parameters
theta = result["theta"]
F, mu, sigma, sigma_l, pi = theta

# Retrieve the last MCMC samples...
Xs_mh = result["Xs_mh"]
ls_mh = result["ls_mh"]
zs_mh = result["ls_mh"]
# ...or the entire chain (requires history=True)
Xs_mh_history = result["history"]["Xs_mh"]
ls_mh_history = result["history"]["ls_mh"]
zs_mh_history = result["history"]["zs_mh"]
```

#### Working on binary data

All the functions used above assume by default that the networks have weighted edges. Binary matrices can be considered by adding a `setting="binary"` in each function call. The functions `mcmc.mh_mask` and `mcmc.map_mask` do not support binary matrices for now.
