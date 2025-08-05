# Nonparametric-Bayesian-Multi-Facet-Clustering-for-Longitudinal-Data
Code for UAI paper: [Nonparametric Bayesian Multi-Facet Clustering for Longitudinal Data]([https://openreview.net/forum?id=o7xZLAheJ5](https://proceedings.mlr.press/v286/wang25c.html))

- Author: [Luwei Wang](https://demi-wlw.github.io/)

We implemented two example time series models for multi-facet clustering:
1. Nonlinear growth model (single-dimensional, B-spline-based, `MFNLG`, allows missing values)
   - Facets: intercept at time T; shape; noise
3. Vector autoregressive model (multi-dimensional, one lag, `MFVAR`, allows varying time series lengths)
   - Facets: intercept vector; transition matrix (interaction); noise vector

## Function descriptions ##
1. `.MFNLG()` and `.MFVAR()`: create an initial model with data information and specific time series model parameters.
2. `.initialize()`: initialize model parameters and change hyperparameters of priors through a dictionary.
3. `.fit()`: start fitting by mean-field Variational Inference.
   - We recommend running the model fitting through multiple parallel runs to avoid getting stuck in the bad local optima.
   - `prune_threshold` should not be greater than `1/trunc_level`. `trunc_level` controls the maximum possible number of clusters and also affects the training speed. `prune_threshold` affects the learned number of clusters, and can be specified for each facet separately.
4. The fitted model only returns important hyperparameters, ELBOs and training time. If you want to get the estimated parameters, call `.getEstimates()`. If you want to get the estimated posterior probability matrix for each individual, call `.getPiMat()`.
5. There are built-in print and plot functions to print fitting results (with ground truth if provided) `.printRes()`, visualise facets `.plotFacet()` and show facet cluster assignments `.showClust()`. You can also define your own print and plot functions.
6. `.save()`: save the model

## Example code for running the method
```python
import numpy as np
from NPBayesMFVI import MFNLG, MFVAR
from joblib import Parallel, delayed

def runParal(s): # for parallel runs
  model = MFVAR(data=data, trunc_level = 10, seed=s)
  # model = MFNLG(data=data, trunc_level = 10, inter_knots=int_knots, seed=s)
  model.initialize()
  model.fit(maxIter = 300, prune_threshold=0.01)
  return model

data = (numpy array in N*T or N*D*T) # for MFNLG or MFVAR model
seeds = [] # generate random seeds to use for parallel runs
num_cores = 50
paral_res = Parallel(n_jobs=num_cores)(delayed(runParal)(s) for s in seeds)
# choose the highest ELBO
ELBOs = [(i,np.round(paral_res[i].ELBO_iters[-1],2)) for i in range(len(paral_res))]
print('Optimal ELBOs of diff models:\n', ELBOs)
model_idx = np.nanargmax(ELBOs, axis=0)[1]
model = paral_res[model_idx]
print('Best model at index:', model_idx, "with ELBO:", np.round(model.ELBO_iters[-1],2),"; Model seed:", model.seed)
model.printRes() # print estimated results and sd; can also print ground truth if provided (a,pi_a,beta/Beta,pi_beta/pi_Beta,sigma,pi_sigma)
model.plotFacet()
model.showClust() # plot contingency table of cluster assignments for intercepts and coefficients
model.getEstimates() # get all estimates for customized analysis and plots
model.save(filePath)
```
