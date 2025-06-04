# Nonparametric-Bayesian-Multi-Facet-Clustering-for-Longitudinal-Data
Code for UAI paper: Nonparametric Bayesian Multi-Facet Clustering for Longitudinal Data

We implemented two example time series models for multi-facet clustering:
1. Nonlinear growth model (single-dimensional, **MFNLG**)
2. Vector autoregressive model (multi-dimensional, **MFVAR**)

## Example for running the method
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
