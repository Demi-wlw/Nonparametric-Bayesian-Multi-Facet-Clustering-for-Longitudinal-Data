from abc import ABC, abstractmethod
from typing import Union, List
import numpy as np
from numpy import linalg as LA
from tqdm import tqdm
import time
from scipy.special import softmax, psi, gammaln, betaln
import matplotlib.pyplot as plt

### Utility functions ###
# relative l2 error
def rel_l2(x, y):
    return LA.norm(x-y)/LA.norm(x) # (0, inf)

def l2_mat(a, b):
    a = np.array(a); b = np.array(b)
    l2Mat = np.zeros((a.shape[0], b.shape[0]))
    for i in range(a.shape[0]):
        for j in range(b.shape[0]):
            l2Mat[i,j] = rel_l2(a[i].flatten(), b[j].flatten())
    return l2Mat

def radar_chart(data, var, pi_sigma, fgsize,dpi,fontsize, colors: list=None, save_folder: str=None, min_max_scales: Union[List[tuple], None]=None):
    """
    Plot a radar chart for given data.
    
    Parameters:
    - data: 2D array-like, where each row corresponds to a series in the radar chart.
    - var: List of labels for each axis (dimensions).
    - pi_sigma: Array of probabilities for each cluster in the sigma.
    - fig attributes: fgsize, dpi, fontsize
    - colors: List of colors for each series in the radar chart.
    - save_folder: String, folder to save the plot.
    - min_max_scales: List of tuples, each tuple contains (min, max) for scaling each variable.
    """
    def normalize_row(row, scales):
        if scales is None:
            return row.tolist()
        else:
            return [(v - mn) / (mx - mn) if mx != mn else 0 for v, (mn, mx) in zip(row, scales)]
    num_vars = len(var)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]  # Close the radar chart
    fig, ax = plt.subplots(figsize=fgsize, dpi=dpi, subplot_kw=dict(polar=True))
    # If you want the first axis to be on top:
    ax.set_theta_offset(np.pi/2)
    ax.set_theta_direction(-1)
    # Draw one line per row in data
    for idx, row in enumerate(data):
        values = normalize_row(row, min_max_scales)
        values += values[:1]  # Close the radar chart
        color = colors[idx] if colors else "blue"
        labels = f"Clust{idx+1} Prob {pi_sigma[idx]:.2f}"
        ax.fill(angles, values, color=color, alpha=0.25)
        ax.plot(angles, values, color=color, linewidth=1.5, label=labels)
    # Add labels for each axis
    plt.xticks(angles[:-1], var, fontsize=fontsize)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3,1.1)) # "upper left" -0.05, 1.1; "upper right" 1.3,1.1
    plt.tight_layout()
    if min_max_scales:
        ax.set_yticklabels([])  # Hide radial ticks
        tick_radii = np.linspace(0, 1, 6)[[1,3,5]]  # positions on normalized scale
        for i, angle in enumerate(angles[:-1]):  # skip the repeated closing angle
            min_val, max_val = min_max_scales[i]
            tick_vals = np.linspace(min_val, max_val, 6)[[1,3,5]]
            for r, label in zip(tick_radii, tick_vals):
                if i == 0:
                    ax.text(x=angle, y=r+0.08,s=f"{label:.2f}", ha="left",va="top",fontsize=fontsize - 8,color="black")
                else:
                    ax.text(x=angle*(1.03+(i-2)*0.03), y=r+0.08*(len(min_max_scales)-i-1),s=f"{label:.2f}", ha="right",va="top",fontsize=fontsize - 8,color="black")
    # Save plot if requested
    if save_folder:
        plt.savefig(f"{save_folder}EstSIGMA.png", bbox_inches="tight")
        plt.show()
    else:
        plt.show()

def stacked_Pis_barplot(pis, fgsize, dpi, fontsize, xlabels=["Intercept","Coef","Noise"]):
    import pandas as pd
    import seaborn as sns
    # Validate input
    if len(pis) != len(xlabels):
        raise ValueError("The number of Pi arrays must match the number of xlabels!")
    # Prepare data for the stacked bar plot
    plot_data = []
    for i, prob_array in enumerate(pis):
    #    if not np.isclose(prob_array.sum(), 1.0):
    #        raise ValueError(f"Probabilities in variable '{xlabels[i]}' do not sum to 1.")    
        for cluster_idx, cluster_percentage in enumerate(prob_array):
            plot_data.append({"Facet": xlabels[i], "Cluster": f"{cluster_idx+1}", "Percentage": cluster_percentage * 100})
    # Create a DataFrame for plotting
    df = pd.DataFrame(plot_data)
    # Define colors using tab10 colormap
    num_clusters_total = max(len(arr) for arr in pis)  # Max number of clusters across all facets
    tab10_colors = plt.cm.tab10.colors  # Tab10 contains 10 distinct colors
    colors = [tab10_colors[i % len(tab10_colors)] for i in range(num_clusters_total)]
    palette = {f"{i + 1}": colors[i] for i in range(num_clusters_total)}
    # Create the stacked bar plot
    plt.figure(figsize=(fgsize[0],fgsize[1]+0.5), dpi=dpi)
    # Plot the stacked bar chart
    sns.barplot(data=df, x="Facet", y="Percentage", hue="Cluster", palette=palette)
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.ylabel("Percentage", fontsize=fontsize+2)
    plt.xlabel("Facets", fontsize=fontsize+2)
    plt.legend(title="Cluster", loc="upper right") # bbox_to_anchor(x,y) to move legend outside
    plt.tight_layout()

def merge_clust(est_dict: dict, merge_idx_dict: dict):
    """Post-analysis (Not Used): merge the similar clusters based on the given indices for each facet
    """
    est_a = est_dict["intercept"]
    est_a_sd = est_dict["intercept_sd"]
    pi_a = est_dict["pi_a"]
    a_clust = est_dict["a_clust"]
    est_beta = est_dict["B_coef"]
    est_beta_sd = est_dict["B_coef_sd"]
    pi_beta = est_dict.get("pi_beta", est_dict.get("pi_Beta"))
    beta_clust = est_dict.get("beta_clust", est_dict.get("Beta_clust"))
    est_sigma = est_dict["sigma"]
    est_sigma_sd = est_dict["sigma_sd"]
    pi_sigma = est_dict["pi_sigma"]
    sigma_clust = est_dict["sigma_clust"]
    merged_est_dict = est_dict.copy()
    # merge the resulting clusters for a, beta, sigma
    for key, value in merge_idx_dict.items():
        if key == "intercept":
            old_idx = set(range(est_a.shape[0]))
            new_est_a = []
            new_est_a_sd = []
            new_pi_a = []
            new_idx = []
            for idx in value:
                idx = set(idx)
                new_est_a.append(np.mean(est_a[list(idx)], axis=0))
                new_est_a_sd.append(np.mean(est_a_sd[list(idx)], axis=0))
                new_pi_a.append(np.sum(pi_a[list(idx)]))
                new_idx.append(np.array(list(idx)))
                old_idx -= idx
            new_est_a = np.array(new_est_a)
            new_est_a_sd = np.array(new_est_a_sd)
            new_pi_a = np.array(new_pi_a)
            merged_est_dict["intercept"] = np.concatenate((est_a[list(old_idx)],new_est_a), axis=0)
            merged_est_dict["intercept_sd"] = np.concatenate((est_a_sd[list(old_idx)],new_est_a_sd), axis=0)
            merged_est_dict["pi_a"] = np.concatenate((pi_a[list(old_idx)],new_pi_a))
            new_idx.insert(0, np.array(list(old_idx)))
            old_idx = np.concatenate(new_idx)
            new_idx[0] = np.arange(len(new_idx[0]))
            for i, idx in enumerate(new_idx[1:]):
                new_idx[i+1] = np.array([len(new_idx[0])+i]*len(idx))
            new_idx = np.concatenate(new_idx)
            idx_mapping = {old:new for old, new in zip(old_idx, new_idx)}
            merged_est_dict["a_clust"] = np.array([idx_mapping[old] for old in a_clust])
        elif key == "B_coef":
            old_idx = set(range(est_beta.shape[0]))
            new_est_beta = []
            new_est_beta_sd = []
            new_pi_beta = []
            new_idx = []
            for idx in value:
                idx = set(idx)
                new_est_beta.append(np.mean(est_beta[list(idx)], axis=0))
                new_est_beta_sd.append(np.mean(est_beta_sd[list(idx)], axis=0))
                new_pi_beta.append(np.sum(pi_beta[list(idx)]))
                new_idx.append(np.array(list(idx)))
                old_idx -= idx
            new_est_beta = np.array(new_est_beta)
            new_est_beta_sd = np.array(new_est_beta_sd)
            new_pi_beta = np.array(new_pi_beta)
            merged_est_dict["B_coef"] = np.concatenate((est_beta[list(old_idx)],new_est_beta), axis=0)
            merged_est_dict["B_coef_sd"] = np.concatenate((est_beta_sd[list(old_idx)],new_est_beta_sd), axis=0)
            if "pi_beta" in est_dict.keys():
                merged_est_dict["pi_beta"] = np.concatenate((pi_beta[list(old_idx)],new_pi_beta))
            else:
                merged_est_dict["pi_Beta"] = np.concatenate((pi_beta[list(old_idx)],new_pi_beta))
            new_idx.insert(0, np.array(list(old_idx)))
            old_idx = np.concatenate(new_idx)
            new_idx[0] = np.arange(len(new_idx[0]))
            for i, idx in enumerate(new_idx[1:]):
                new_idx[i+1] = np.array([len(new_idx[0])+i]*len(idx))
            new_idx = np.concatenate(new_idx)
            idx_mapping = {old:new for old, new in zip(old_idx, new_idx)}
            if "beta_clust" in est_dict.keys():
                merged_est_dict["beta_clust"] = np.array([idx_mapping[old] for old in beta_clust])
            else:
                merged_est_dict["Beta_clust"] = np.array([idx_mapping[old] for old in beta_clust])
        elif key == "sigma":
            old_idx = set(range(est_sigma.shape[0]))
            new_est_sigma = []
            new_est_sigma_sd = []
            new_pi_sigma = []
            new_idx = []
            for idx in value:
                idx = set(idx)
                new_est_sigma.append(np.mean(est_sigma[list(idx)], axis=0))
                new_est_sigma_sd.append(np.mean(est_sigma_sd[list(idx)], axis=0))
                new_pi_sigma.append(np.sum(pi_sigma[list(idx)]))
                new_idx.append(np.array(list(idx)))
                old_idx -= idx
            new_est_sigma = np.array(new_est_sigma)
            new_est_sigma_sd = np.array(new_est_sigma_sd)
            new_pi_sigma = np.array(new_pi_sigma)
            merged_est_dict["sigma"] = np.concatenate((est_sigma[list(old_idx)],new_est_sigma), axis=0)
            merged_est_dict["sigma_sd"] = np.concatenate((est_sigma_sd[list(old_idx)],new_est_sigma_sd), axis=0)
            merged_est_dict["pi_sigma"] = np.concatenate((pi_sigma[list(old_idx)],new_pi_sigma))
            new_idx.insert(0, np.array(list(old_idx)))
            old_idx = np.concatenate(new_idx)
            new_idx[0] = np.arange(len(new_idx[0]))
            for i, idx in enumerate(new_idx[1:]):
                new_idx[i+1] = np.array([len(new_idx[0])+i]*len(idx))
            new_idx = np.concatenate(new_idx)
            idx_mapping = {old:new for old, new in zip(old_idx, new_idx)}
            merged_est_dict["sigma_clust"] = np.array([idx_mapping[old] for old in sigma_clust])
        else:
            print(f'Wrong key "{key}"! Key should be one of ["intercept", "B_coef", "sigma"]!')
    return merged_est_dict
##########################

class MFVI(ABC):
    def __init__(self, data: np.ndarray, trunc_level: int, seed=None):
        self.Y = data.copy()
        # check dimension of data
        if len(data.shape) == 2:
            self.N, self.T = self.Y.shape
        elif len(data.shape) == 3:
            self.N, self.D, self.T = self.Y.shape
        else:
            raise ValueError('The dimension of the input data should be 2 or 3!')
        self.K = trunc_level
        self.ELBO_iters = [1]
        self.seed = seed
        self.exc_time = 0
        #todo# Specific hyperparameters for the model ###
        #todo# Hyperparameters of priors or hyperpriors ###
        #todo# Initialize variational parameters ###
        #todo# Estimated means of model parameters ###

    @abstractmethod
    def fit(self):
        pass

    def _has_converged(self, ELBO_rel_tol: float, ELBO_abs_tol: float):
        ELBO_abs = np.abs(self.ELBO_iters[-1] - self.ELBO_iters[-2])
        return np.abs(ELBO_abs/self.ELBO_iters[-2]) < ELBO_rel_tol or ELBO_abs < ELBO_abs_tol

    def save(self, filepath: str):
        """Save the model object to a file

        Parameters:
            filepath (str): file path to save the model object
        
        Raises:
            ValueError: if the model has not been fitted
        """
        if len(self.ELBO_iters) == 1:
            raise ValueError('Please fit the model first!')
        import pickle as pkl
        with open(filepath, 'wb') as f:
            pkl.dump(self, f)


class MFNLG(MFVI):
    def __init__(self, inter_knots, times=None, intercept_shift_to: int=0, degree: int=1, Bscale: bool=True, **kwargs):
        """Nonparametric Bayesian Multi-Facet Nonlinear growth model with Mean-Field Variational Inference

        Parameters:
            ---Super-class-Args:
            data (np.ndarray): Data matrix of size N x T
            trunc_level (int): truncation level for the model
            seed (int, optional): Random seed for reproducibility. Defaults to None.
            ---NPBMF-NLG-Args:
            inter_knots (list/array): interior knots for B-spline basis
            times (list/array): predictive timepoints for B-spline basis. Defaults to None, then it's initalised with np.arange(T).
            intercept_shift_to (int, optional): timepoint to shift the intercept. Defaults to 0.
            degree (int, optional): degree of B-splines. Defaults to 1.
            Bscale (bool, optional): whether to scale the B-spline matrix. Defaults to True.
        """
        print("######################################################")
        print("### Mean-Field Variational Inference for NPBMF-NLG ###")
        print("######################################################")
        super().__init__(**kwargs)
        #--- data distribution ---#
        # y_n ~ N(a_k1 + beta_k2 @ B_n, tau_k3^-1)
        ##############################################
        ### Specific hyperparameters for the model ###
        from patsy import dmatrix
        # B-spline basis matrix
        self.inter_knots = inter_knots
        if times is None:
            self.times = np.arange(self.T)
        else:
            self.times = np.array(times)
        self.degree = degree
        if len(self.times) != self.T:
            raise ValueError('The length of times should be equal to the number of timepoints in the data matrix!')
        if self.times[0] != 0:
            raise ValueError('The first element of times should be 0!')
        B = dmatrix("bs(x, knots=knots, degree=degree) - 1", {"x": self.times, "knots": self.inter_knots, "degree": self.degree})
        if Bscale:
            B = B * np.max(self.times) # for real data, equal scale up to the given time range so that beta won't be too large
        self.B = np.asarray(B)
        if intercept_shift_to > 0 and intercept_shift_to in self.times:
            self.B = self.B - self.B[self.times==intercept_shift_to]
        elif intercept_shift_to != 0 and intercept_shift_to not in self.times:
            print(f'[Warning] The intercept shift timepoint {intercept_shift_to} is not in the given timepoints! Applying the default no shift.')
        self.B = self.B.T # I x T matrix
        self.__I = self.B.shape[0]
        self.intercept_shift_to = intercept_shift_to
        # separate dynamic truncation level for facets
        self.Ka = self.K; self.Kbeta = self.K; self.Ktau = self.K
        ################################################
        ### Hyperparameters of priors or hyperpriors ###
        # Priors
        # a_k ~ N(0, 1) can be nonzero means
        self.__lambda_a1 = 0 #? a_prior_mean can be changed
        self.__lambda_a2 = 1 #? a_sd can be changed
        self.__lambda_a2 = 1/self.__lambda_a2**2 # to precision
        # beta_k ~ N(0, 1)
        self.__lambda_beta1 = 0; self.__lambda_beta2 = 1
        self.__lambda_beta2 = 1/self.__lambda_beta2**2
        # tau_k ~ Gamma(1, 1) or (1, 0.1) for smaller variance
        self.__lambda_tau1 = 1; self.__lambda_tau2 = 1
        # v ~ Beta(1, alpha)
        self.__alphfix = 1 # not changed
        # alpha ~ Gamma(1, 1)
        self.__s1 = 1; self.__s2 = 1 #? s can be changed
        #########################################
        ### Initialize variational parameters ###
        # Random initialization
        print('[INFO] Random initialization...')
        if self.seed is not None:
            np.random.seed(self.seed)
        # variational parameters for a
        lamStar_a1 = np.random.normal(self.__lambda_a1, np.sqrt(1/self.__lambda_a2), self.K) # kth element is for a_k
        lamStar_a2 = np.random.gamma(1, 1, self.K) # precisions
        self.__lamStar_a = np.column_stack((lamStar_a1, lamStar_a2))
        # variational parameters for tau
        self.__lamStar_tau = np.random.gamma(1, 1, [self.K, 2]) # kth params for tau_k
        # variational parameters for alpha
        self.__sStar_a = np.random.gamma(1, 1, 2) # length 2, shape and rate for Gamma
        self.__sStar_beta = np.random.gamma(1, 1, 2) # length 2
        self.__sStar_tau = np.random.gamma(1, 1, 2) # length 2
        # variational parameters for z
        self.__piStar_a = np.random.dirichlet([1]*self.K, self.N) # matrix of size N x K
        self.__piStar_beta = np.random.dirichlet([1]*self.K, self.N) # matrix of size N x K
        self.__piStar_tau = np.random.dirichlet([1]*self.K, self.N) # matrix of size N x K
        #-----------------------------------------#
        # Create storage for the rest var params
        self.__lamStar_beta1 = np.zeros((self.K, self.__I))
        self.__lamStar_beta2 = np.zeros((self.K, self.__I, self.__I)) # precision matrices
        self.__alphaStar_a = np.zeros((self.K-1, 2)) # only K-1 alphas
        self.__alphaStar_beta = np.zeros((self.K-1, 2))
        self.__alphaStar_tau = np.zeros((self.K-1, 2))
        ###########################################
        ### Estimated means of model parameters ###
        self.__est_a = lamStar_a1
        self.__est_beta = self.__lamStar_beta1
        self.__est_tau = self.__lamStar_tau[:, 0]/self.__lamStar_tau[:, 1]
        self.__est_Za = np.argmax(self.__piStar_a, axis=1)
        self.__est_Zbeta = np.argmax(self.__piStar_beta, axis=1)
        self.__est_Ztau = np.argmax(self.__piStar_tau, axis=1)
        self.__est_pi_a = np.zeros(self.K)
        self.__est_pi_beta = np.zeros(self.K)
        self.__est_pi_tau = np.zeros(self.K)
        #----------------------------------------#
        self.__est_alpha_a = self.__sStar_a[0]/self.__sStar_a[1]
        self.__est_alpha_beta = self.__sStar_beta[0]/self.__sStar_beta[1]
        self.__est_alpha_tau = self.__sStar_tau[0]/self.__sStar_tau[1]

    def initialize(self, param_dict: dict):
        """Initialize the variational parameters with specified values

        Parameters:
            param_dict (dict): dictionary of specified values for initialization. 
                This includes: 'a_mean', 'a_prior_mean', 'a_prior_sd', 'alpha_prior_s', 'tau_gamma', 'a_pi', 'beta_pi', 'tau_pi'.
            Examples:
            'a_prior_mean' (list/array): specified prior means for a_k
            'a_prior_sd' (float): specified prior standard deviation for a_k
            'alpha_prior_s' (tuple): specified prior shape and rate for alpha ~ Gamma(shape, rate)
            'a_mean' (list/array): specified means for a_k
            'tau_gamma' (2D list/array): specified shape and rate for tau_k ~ Gamma(shape, rate)
            'a_pi' (2D list/array): specified pi_n for a_k
            'beta_pi' (2D list/array): specified pi_n for beta_k
            'tau_pi' (2D list/array): specified pi_n for tau_k
        """
        orig_lamStar_a = self.__lamStar_a.copy(); change_lamStar_a = False
        orig_lambda_a1 = self.__lambda_a1; change_lambda_a1 = False
        orig_lambda_a2 = self.__lambda_a2; change_lambda_a2 = False
        orig_s1 = self.__s1
        orig_s2 = self.__s2; change_s = False
        orig_lamStar_tau = self.__lamStar_tau.copy(); change_lamStar_tau = False
        orig_piStar_a = self.__piStar_a.copy(); change_piStar_a = False
        orig_piStar_beta = self.__piStar_beta.copy(); change_piStar_beta = False
        orig_piStar_tau = self.__piStar_tau.copy(); change_piStar_tau = False
        # update the specified values
        lamStar_a1_new = np.array(param_dict.get('a_mean', self.__lamStar_a[:,0]))
        # check length
        if len(lamStar_a1_new) < self.K:
            lamStar_a1_new = np.concatenate((lamStar_a1_new, np.array([0.]*(self.K-len(lamStar_a1_new)))))
        self.__lamStar_a[:,0] = lamStar_a1_new
        self.__lambda_a1 = param_dict.get('a_prior_mean', self.__lambda_a1)
        self.__lambda_a2 = param_dict.get('a_prior_sd', np.sqrt(1/self.__lambda_a2))
        self.__lambda_a2 = 1/self.__lambda_a2**2
        changed_s = param_dict.get('alpha_prior_s', (self.__s1,self.__s2))
        self.__s1 = changed_s[0]
        self.__s2 = changed_s[1]
        self.__lamStar_tau = np.array(param_dict.get('tau_gamma', self.__lamStar_tau))
        self.__lamStar_tau = self.__lamStar_tau.astype('float64')
        self.__piStar_a = np.array(param_dict.get('a_pi', self.__piStar_a))
        self.__piStar_beta = np.array(param_dict.get('beta_pi', self.__piStar_beta))
        self.__piStar_tau = np.array(param_dict.get('tau_pi', self.__piStar_tau))
        self.__piStar_a = self.__piStar_a.astype('float64')
        self.__piStar_beta = self.__piStar_beta.astype('float64')
        self.__piStar_tau = self.__piStar_tau.astype('float64')
        # check if the values are changed
        if (self.__lamStar_a - orig_lamStar_a).any() != 0:
            change_lamStar_a = True
        if self.__lambda_a1 != orig_lambda_a1:
            change_lambda_a1 = True
        if self.__lambda_a2 != orig_lambda_a2:
            change_lambda_a2 = True
        if (self.__s1, self.__s2) != (orig_s1, orig_s2):
            change_s = True
        if (self.__lamStar_tau - orig_lamStar_tau).any() != 0:
            change_lamStar_tau = True
        if (self.__piStar_a - orig_piStar_a).any() != 0:
            change_piStar_a = True
        if (self.__piStar_beta - orig_piStar_beta).any() != 0:
            change_piStar_beta = True
        if (self.__piStar_tau - orig_piStar_tau).any() != 0:
            change_piStar_tau = True
        if change_lamStar_a or change_lambda_a1 or change_lambda_a2 or change_s or change_lamStar_tau or change_piStar_a or change_piStar_beta or change_piStar_tau:
            print("[INFO] Set specified initialization for {}{}{}{}{}{}{}{}".format("'a_mean'; " if change_lamStar_a else '', "'a_prior_mean'; " if change_lambda_a1 else '', "'a_prior_sd'; " if change_lambda_a2 else '', "'alpha_prior_s'; " if change_s else '', "'tau_gamma'; " if change_lamStar_tau else '', "'a_pi'; " if change_piStar_a else '', "'beta_pi'; " if change_piStar_beta else '', "'tau_pi'" if change_piStar_tau else ''))
        if not change_lamStar_a and (change_lambda_a1 or change_lambda_a2):
            # re-initalize variational parameters for a
            self.__lamStar_a[:,0] = np.random.normal(self.__lambda_a1, np.sqrt(1/self.__lambda_a2), self.K) # kth element is for a_k
        # recompute the estimated means
        if change_lamStar_a or change_lambda_a1 or change_lambda_a2:
            self.__est_a = self.__lamStar_a[:,0]
        if change_lamStar_tau:
            self.__est_tau = self.__lamStar_tau[:,0]/self.__lamStar_tau[:,1] 

    def __get_obs_data(self, n):
        obs_idx = np.where(np.isfinite(self.Y[n,:]))[0]
        Yobs = self.Y[n, obs_idx]
        Bobs = self.B[:, obs_idx]
        Tobs = len(obs_idx)
        return Yobs, Bobs, Tobs

    def _impute_missing_data(self):
        miss_idx_ls = []
        for n in range(self.N):
            miss_idx = np.where(~np.isfinite(self.Y[n,:]))[0]
            miss_idx_ls.append(miss_idx)
            if len(miss_idx) != 0:
                E_a_n = sum([self.__piStar_a[n,j]* self.__est_a[j] for j in range(self.K)])
                E_beta_n = sum([self.__piStar_beta[n,j]* self.__est_beta[j, :] for j in range(self.K)]) # dim I
                est_Y_n = E_a_n + E_beta_n @ self.B[:, miss_idx]
                self.Y[n, miss_idx] = est_Y_n
        return miss_idx_ls

    def __compute_ELBO(self, est_logV_a, est_log1V_a, est_logV_beta, est_log1V_beta, est_logV_tau, est_log1V_tau):
        """Compute the Evidence Lower Bound (ELBO) for the model
        """
        ELBO = 0
        # prior -q on a
        ELBO += sum([1/2*(np.log(self.__lambda_a2)-np.log(self.__lamStar_a[k,1])) +1/2 - 1/2*self.__lambda_a2* (1/self.__lamStar_a[k,1]+self.__est_a[k]**2-2*self.__lambda_a1*self.__est_a[k]+self.__lambda_a1**2) for k in range(self.Ka)])
        # prior -q on beta
        ELBO += sum([self.__I/2*(np.log(self.__lambda_beta2))-1/2*np.array(LA.slogdet(self.__lamStar_beta2[k])).prod() +self.__I/2 - 1/2*self.__lambda_beta2* (np.sum(self.__est_beta[k,:]**2) + np.trace(LA.pinv(self.__lamStar_beta2[k])) -2*self.__lambda_beta1*np.sum(self.__est_beta[k,:]) +self.__I*self.__lambda_beta1**2) for k in range(self.Kbeta)])
        # prior -q on tau
        ELBO += sum([self.__lambda_tau1*np.log(self.__lambda_tau2) - self.__lamStar_tau[k,0]*np.log(self.__lamStar_tau[k,1]) -gammaln(self.__lambda_tau1)+gammaln(self.__lamStar_tau[k,0]) +(self.__lambda_tau1-self.__lamStar_tau[k,0])*(psi(self.__lamStar_tau[k,0])-np.log(self.__lamStar_tau[k,1])) +self.__lamStar_tau[k,0] -self.__lambda_tau2*self.__est_tau[k] for k in range(self.Ktau)])
        # prior -q on v
        ELBO += sum([(self.__alphfix-self.__alphaStar_a[k,0])*est_logV_a[k] + (self.__est_alpha_a-self.__alphaStar_a[k,1])*est_log1V_a[k] - psi(self.__sStar_a[0]) +np.log(self.__sStar_a[1]) +betaln(self.__alphaStar_a[k,0],self.__alphaStar_a[k,1]) for k in range(self.Ka-1)])
        ELBO += sum([(self.__alphfix-self.__alphaStar_beta[k,0])*est_logV_beta[k] + (self.__est_alpha_beta-self.__alphaStar_beta[k,1])*est_log1V_beta[k] - psi(self.__sStar_beta[0]) +np.log(self.__sStar_beta[1]) +betaln(self.__alphaStar_beta[k,0],self.__alphaStar_beta[k,1]) for k in range(self.Kbeta-1)])
        ELBO += sum([(self.__alphfix-self.__alphaStar_tau[k,0])*est_logV_tau[k] + (self.__est_alpha_tau-self.__alphaStar_tau[k,1])*est_log1V_tau[k] - psi(self.__sStar_tau[0]) +np.log(self.__sStar_tau[1]) +betaln(self.__alphaStar_tau[k,0],self.__alphaStar_tau[k,1]) for k in range(self.Ktau-1)])
        for n in range(self.N):
            # for Z_a_n - max piStar_a[n, :]
            ELBO += sum([np.sum(self.__piStar_a[n, (k+1):self.Ka])*est_log1V_a[k] +self.__piStar_a[n,k]*est_logV_a[k] for k in range(self.Ka-1)]) - (np.log(self.__piStar_a[n, self.__est_Za[n]]) if self.__piStar_a[n, self.__est_Za[n]] >0 else 0)
            # for Z_beta_n
            ELBO += sum([np.sum(self.__piStar_beta[n, (k+1):self.Kbeta])*est_log1V_beta[k] +self.__piStar_beta[n,k]*est_logV_beta[k] for k in range(self.Kbeta-1)]) - (np.log(self.__piStar_beta[n, self.__est_Zbeta[n]]) if self.__piStar_beta[n, self.__est_Zbeta[n]] >0 else 0)
            # for Z_tau_n
            ELBO += sum([np.sum(self.__piStar_tau[n, (k+1):self.Ktau])*est_log1V_tau[k] +self.__piStar_tau[n,k]*est_logV_tau[k] for k in range(self.Ktau-1)]) - (np.log(self.__piStar_tau[n, self.__est_Ztau[n]]) if self.__piStar_tau[n, self.__est_Ztau[n]] >0 else 0)
            # likelihood for y_n
            Yobs, Bobs, Tobs = self.__get_obs_data(n)
            E_a_n = sum([self.__piStar_a[n,j]* self.__est_a[j] for j in range(self.Ka)])
            E_beta_n = sum([self.__piStar_beta[n,j]* self.__est_beta[j, :] for j in range(self.Kbeta)])
            E_tau_n = sum([self.__piStar_tau[n, j]*self.__est_tau[j] for j in range(self.Ktau)])
            E_log_tau_n = sum([self.__piStar_tau[n, j]*(psi(self.__lamStar_tau[j,0]) - np.log(self.__lamStar_tau[j,1])) for j in range(self.Ktau)])
            E_a2_n = sum([self.__piStar_a[n,j]* (1/self.__lamStar_a[j,1]+self.__est_a[j]**2) for j in range(self.Ka)])
            ELBO += -Tobs/2*np.log(2*np.pi) +Tobs/2*E_log_tau_n - 1/2*E_tau_n * (np.sum(Yobs**2) - 2*np.sum(E_a_n*Yobs) -2*E_beta_n @Bobs @Yobs +Tobs*E_a2_n +2*np.sum(E_a_n*E_beta_n @Bobs) +E_beta_n @Bobs @Bobs.T @E_beta_n)
        # prior -q on alpha
        ELBO += self.__s1*np.log(self.__s2) - self.__sStar_a[0]*np.log(self.__sStar_a[1]) -gammaln(self.__s1)+gammaln(self.__sStar_a[0]) +(self.__s1-self.__sStar_a[0])*(psi(self.__sStar_a[0])-np.log(self.__sStar_a[1])) -self.__s2*self.__est_alpha_a +self.__sStar_a[0]
        ELBO += self.__s1*np.log(self.__s2) - self.__sStar_beta[0]*np.log(self.__sStar_beta[1]) -gammaln(self.__s1)+gammaln(self.__sStar_beta[0]) +(self.__s1-self.__sStar_beta[0])*(psi(self.__sStar_beta[0])-np.log(self.__sStar_beta[1])) -self.__s2*self.__est_alpha_beta +self.__sStar_beta[0]
        ELBO += self.__s1*np.log(self.__s2) - self.__sStar_tau[0]*np.log(self.__sStar_tau[1]) -gammaln(self.__s1)+gammaln(self.__sStar_tau[0]) +(self.__s1-self.__sStar_tau[0])*(psi(self.__sStar_tau[0])-np.log(self.__sStar_tau[1])) -self.__s2*self.__est_alpha_tau +self.__sStar_tau[0]
        return ELBO

    def fit(self, maxIter: int=200, prune_threshold: Union[float, List[float]]=0.01, ELBO_rel_tol: float=1e-8, ELBO_abs_tol: float=1e-4, verbose: bool=False):
        """Coordinate Ascent algorithm for MFVI

        Parameters:
            maxIter (int, optional): Number of iterations. Defaults to 200.
            prune_threshold (float, optional): Probability threshold for pruning the clusters. Defaults to 0.01.
            ELBO_rel_tol (float, optional): Relative tolerance of changes in ELBO for conergence check. Defaults to 1e-8.
            ELBO_abs_tol (float, optional): Absolute tolerance of changes in ELBO for conergence check. Defaults to 1e-4.
            verbose (bool, optional): Whether to print out detailed training information. Defaults to False.
        """
        if isinstance(prune_threshold, list):
            if len(prune_threshold) != 3:
                raise ValueError('The length of prune_threshold should be 3!')
            if max(prune_threshold) > 1/self.K: #* Check pruning threshold
                raise ValueError('Pruning threshold may be too large considering current truncation level!')
            prune_threshold_a = prune_threshold[0]
            prune_threshold_beta = prune_threshold[1]
            prune_threshold_tau = prune_threshold[2]
        else:
            if prune_threshold > 1/self.K: #* Check pruning threshold
                raise ValueError('Pruning threshold may be too large considering current truncation level!')
            prune_threshold_a = prune_threshold
            prune_threshold_beta = prune_threshold
            prune_threshold_tau = prune_threshold
        start_time = time.perf_counter()
        for it in tqdm(range(maxIter), desc="[INFO] Iterate the Coordinate Ascent algorithm"): # loop over iterations for MFVI
            if verbose: print(f'#---------------- Iteration {it+1} ----------------#')
            est_Va = np.zeros(self.Ka-1)
            est_Vbeta = np.zeros(self.Kbeta-1)
            est_Vtau = np.zeros(self.Ktau-1)
            est_logV_a = np.zeros(self.Ka-1)
            est_log1V_a = np.zeros(self.Ka-1)
            est_logV_beta = np.zeros(self.Kbeta-1)
            est_log1V_beta = np.zeros(self.Kbeta-1)
            est_logV_tau = np.zeros(self.Ktau-1)
            est_log1V_tau = np.zeros(self.Ktau-1)
            # _ = self._impute_missing_data() # impute missing data
            # update variational parameters for kth cluster
            for k in (tqdm(range(self.K), desc='[INFO] Update variational parameters for all Ks',leave=False) if verbose else range(self.K)): 
                if k < self.Ka-1:
                    #- update variational parameters for v: alphaStar -#
                    self.__alphaStar_a[k, 0] = self.__alphfix + np.sum(self.__piStar_a[:, k])
                    self.__alphaStar_a[k, 1] = self.__est_alpha_a + np.sum(self.__piStar_a[:, (k+1):self.Ka])
                    #--- get means for V ---#
                    est_Va[k] = self.__alphaStar_a[k, 0]/np.sum(self.__alphaStar_a[k,:])
                    #--- get estimates for pi ---#
                    self.__est_pi_a[k] = np.exp(np.log(est_Va[k])+ np.log(1-est_Va)[:k].sum())
                if k < self.Kbeta-1:
                    self.__alphaStar_beta[k, 0] = self.__alphfix + np.sum(self.__piStar_beta[:, k])
                    self.__alphaStar_beta[k, 1] = self.__est_alpha_beta + np.sum(self.__piStar_beta[:, (k+1):self.Kbeta])
                    #--- get means for V ---#
                    est_Vbeta[k] = self.__alphaStar_beta[k, 0]/np.sum(self.__alphaStar_beta[k,:])
                    #--- get estimates for pi ---#
                    self.__est_pi_beta[k] = np.exp(np.log(est_Vbeta[k])+ np.log(1-est_Vbeta)[:k].sum())
                if k < self.Ktau-1:
                    self.__alphaStar_tau[k, 0] = self.__alphfix + np.sum(self.__piStar_tau[:, k])
                    self.__alphaStar_tau[k, 1] = self.__est_alpha_tau + np.sum(self.__piStar_tau[:, (k+1):self.Ktau])
                    #--- get means for V ---#
                    est_Vtau[k] = self.__alphaStar_tau[k, 0]/np.sum(self.__alphaStar_tau[k,:])
                    #--- get estimates for pi ---#
                    self.__est_pi_tau[k] = np.exp(np.log(est_Vtau[k])+ np.log(1-est_Vtau)[:k].sum())
                #- update variational parameters for beta: lamStar_beta -#
                if k < self.Kbeta:
                    beta2_sum = np.zeros((self.__I, self.__I))
                    beta1_sum = np.zeros((self.__I))
                    for n in range(self.N):
                        E_tau_n = sum([self.__piStar_tau[n, j]*self.__est_tau[j] for j in range(self.Ktau)])
                        E_a_n = sum([self.__piStar_a[n,j]* self.__est_a[j] for j in range(self.Ka)])
                        Yobs, Bobs, _ = self.__get_obs_data(n)
                        beta2_sum_n = self.__piStar_beta[n, k]* E_tau_n
                        beta2_sum += beta2_sum_n * Bobs @ Bobs.T
                        beta1_sum_n = beta2_sum_n * (Yobs - E_a_n) @ Bobs.T
                        beta1_sum += beta1_sum_n
                    self.__lamStar_beta2[k] = self.__lambda_beta2* np.eye(self.__I) + beta2_sum
                    self.__lamStar_beta1[k, :] = LA.solve(self.__lamStar_beta2[k], (self.__lambda_beta1*self.__lambda_beta2 + beta1_sum))
                    #--- get means for beta ---#
                    self.__est_beta[k, :] = self.__lamStar_beta1[k, :]
                #- update variational parameters for a: lamStar_a -#
                if k < self.Ka:
                    a2_sum = 0
                    a1_sum = 0
                    for n in range(self.N):
                        E_tau_n = sum([self.__piStar_tau[n, j]*self.__est_tau[j] for j in range(self.Ktau)])
                        E_beta_n = sum([self.__piStar_beta[n,j]* self.__est_beta[j, :] for j in range(self.Kbeta)]) # dim I
                        Yobs, Bobs, Tobs = self.__get_obs_data(n)
                        a2_sum_n = self.__piStar_a[n, k] * E_tau_n
                        a2_sum += Tobs * a2_sum_n
                        a1_sum_n = a2_sum_n * (np.sum(Yobs - E_beta_n @ Bobs))
                        a1_sum += a1_sum_n
                    self.__lamStar_a[k, 1] = self.__lambda_a2 + a2_sum
                    self.__lamStar_a[k, 0] = (self.__lambda_a1*self.__lambda_a2 + a1_sum)/self.__lamStar_a[k, 1]
                    #--- get means for a ---#
                    self.__est_a[k] = self.__lamStar_a[k, 0]
                #- update variational parameters for tau: lamStar_tau -#
                if k < self.Ktau:
                    tau2_sum = 0
                    Tobs_ls = []
                    for n in range(self.N):
                        E_a_n = sum([self.__piStar_a[n,j]* self.__est_a[j] for j in range(self.Ka)])
                        E_beta_n = sum([self.__piStar_beta[n,j]* self.__est_beta[j, :] for j in range(self.Kbeta)])
                        Yobs, Bobs, Tobs = self.__get_obs_data(n)
                        Tobs_ls.append(Tobs)
                        tau2_sum_n = self.__piStar_tau[n, k] * np.sum((Yobs - E_a_n - E_beta_n @ Bobs)**2)
                        tau2_sum += tau2_sum_n
                    self.__lamStar_tau[k, 0] = self.__lambda_tau1 + 1/2 * np.sum(np.array(Tobs_ls)* self.__piStar_tau[:, k])
                    self.__lamStar_tau[k, 1] = self.__lambda_tau2 + 1/2 * tau2_sum
                    #--- get means for tau ---#
                    self.__est_tau[k] = self.__lamStar_tau[k, 0]/self.__lamStar_tau[k, 1]
            # get pi for the last cluster
            self.__est_pi_a[self.Ka-1] = 1 - self.__est_pi_a[:(self.Ka-1)].sum()
            self.__est_pi_beta[self.Kbeta-1] = 1 - self.__est_pi_beta[:(self.Kbeta-1)].sum()
            self.__est_pi_tau[self.Ktau-1] = 1 - self.__est_pi_tau[:(self.Ktau-1)].sum()

            ## Cluster reordering ##
            order_idx_a = np.flip(np.argsort(self.__est_pi_a[:self.Ka])) # sort by est. pi in descending order
            order_idx_a = np.concatenate((order_idx_a, np.arange(self.Ka, len(self.__est_pi_a)))) # add zeros to the end
            order_idx_beta = np.flip(np.argsort(self.__est_pi_beta[:self.Kbeta]))
            order_idx_beta = np.concatenate((order_idx_beta, np.arange(self.Kbeta, len(self.__est_pi_beta))))
            order_idx_tau = np.flip(np.argsort(self.__est_pi_tau[:self.Ktau]))
            order_idx_tau = np.concatenate((order_idx_tau, np.arange(self.Ktau, len(self.__est_pi_tau))))
            self.__est_pi_a = self.__est_pi_a[order_idx_a]
            self.__est_pi_beta = self.__est_pi_beta[order_idx_beta]
            self.__est_pi_tau = self.__est_pi_tau[order_idx_tau]
            self.__est_a = self.__est_a[order_idx_a]
            self.__est_beta = self.__est_beta[order_idx_beta]
            self.__est_tau = self.__est_tau[order_idx_tau]
            self.__piStar_a = self.__piStar_a[:, order_idx_a]
            self.__piStar_beta = self.__piStar_beta[:, order_idx_beta]
            self.__piStar_tau = self.__piStar_tau[:, order_idx_tau]
            self.__lamStar_a = self.__lamStar_a[order_idx_a]
            self.__lamStar_beta2 = self.__lamStar_beta2[order_idx_beta]
            self.__lamStar_tau = self.__lamStar_tau[order_idx_tau]
            for k in range(self.K-1):
                if k < self.Ka-1:
                    # recalculate the variational parameters for v
                    self.__alphaStar_a[k, 0] = self.__alphfix + np.sum(self.__piStar_a[:, k])
                    self.__alphaStar_a[k, 1] = self.__est_alpha_a + np.sum(self.__piStar_a[:, (k+1):self.Ka])
                    #--- get means for log V and 1-V ---#
                    est_logV_a[k] = psi(self.__alphaStar_a[k, 0]) - psi(np.sum(self.__alphaStar_a[k,:]))
                    est_log1V_a[k] = psi(self.__alphaStar_a[k, 1]) - psi(np.sum(self.__alphaStar_a[k,:]))
                if k < self.Kbeta-1:
                    self.__alphaStar_beta[k, 0] = self.__alphfix + np.sum(self.__piStar_beta[:, k])
                    self.__alphaStar_beta[k, 1] = self.__est_alpha_beta + np.sum(self.__piStar_beta[:, (k+1):self.Kbeta])
                    #--- get means for log V and 1-V ---#
                    est_logV_beta[k] = psi(self.__alphaStar_beta[k, 0]) - psi(np.sum(self.__alphaStar_beta[k,:]))
                    est_log1V_beta[k] = psi(self.__alphaStar_beta[k, 1]) - psi(np.sum(self.__alphaStar_beta[k,:]))
                if k < self.Ktau-1:
                    self.__alphaStar_tau[k, 0] = self.__alphfix + np.sum(self.__piStar_tau[:, k])
                    self.__alphaStar_tau[k, 1] = self.__est_alpha_tau + np.sum(self.__piStar_tau[:, (k+1):self.Ktau])
                    #--- get means for log V and 1-V ---#
                    est_logV_tau[k] = psi(self.__alphaStar_tau[k, 0]) - psi(np.sum(self.__alphaStar_tau[k,:]))
                    est_log1V_tau[k] = psi(self.__alphaStar_tau[k, 1]) - psi(np.sum(self.__alphaStar_tau[k,:]))

            #- update variational parameters for z_n: piStar (can be paralleled for each n to speed up) -#
            for n in (tqdm(range(self.N), desc='[INFO] Update variational parameters for Zn', leave=False) if verbose else range(self.N)): 
                E_a_n = sum([self.__piStar_a[n,j]* self.__est_a[j] for j in range(self.Ka)]) # can be rewritten as a matrix-vector multiplication?
                E_beta_n = sum([self.__piStar_beta[n,j]* self.__est_beta[j, :] for j in range(self.Kbeta)])
                E_tau_n = sum([self.__piStar_tau[n, j]* self.__est_tau[j] for j in range(self.Ktau)])
                Yobs, Bobs, Tobs = self.__get_obs_data(n)
                for k in range(self.K):
                    if k < self.Ka:
                        # update variational parameters for a: PiStar_a
                        Sa_kn = -1/2 * E_tau_n * np.sum((Yobs-self.__est_a[k]-E_beta_n @ Bobs)**2)
                        self.__piStar_a[n, k] = est_logV_a[k:(k+1)].sum() + est_log1V_a[:k].sum() + Sa_kn # enable indexing out of bounds -> 0
                    if k < self.Kbeta:
                        # update variational parameters for beta: PiStar_beta
                        Sb_kn = -1/2 * E_tau_n * np.sum((Yobs-E_a_n- self.__est_beta[k,:] @ Bobs)**2)
                        self.__piStar_beta[n, k] = est_logV_beta[k:(k+1)].sum() + est_log1V_beta[:k].sum() + Sb_kn
                    if k < self.Ktau:
                        # update variational parameters for tau: PiStar_tau
                        Stau_kn = Tobs/2*np.log(self.__est_tau[k]) -1/2* self.__est_tau[k] * np.sum((Yobs-E_a_n- E_beta_n @ Bobs)**2)
                        self.__piStar_tau[n, k] = est_logV_tau[k:(k+1)].sum() + est_log1V_tau[:k].sum() + Stau_kn

            #- update variational parameters for alpha: sStar -#
            if verbose: print('[INFO] Update variational parameters for alpha...')
            self.__sStar_a[0] = self.__s1 + self.Ka -1
            self.__sStar_a[1] = self.__s2 - est_log1V_a.sum()
            self.__sStar_beta[0] = self.__s1 + self.Kbeta -1
            self.__sStar_beta[1] = self.__s2 - est_log1V_beta.sum()
            self.__sStar_tau[0] = self.__s1 + self.Ktau -1
            self.__sStar_tau[1] = self.__s2 - est_log1V_tau.sum()
            #--- get means for alpha ---#
            self.__est_alpha_a = self.__sStar_a[0]/self.__sStar_a[1]
            self.__est_alpha_beta = self.__sStar_beta[0]/self.__sStar_beta[1]
            self.__est_alpha_tau = self.__sStar_tau[0]/self.__sStar_tau[1]
            if verbose: print('Updated alpha for a, beta, tau:', np.round([self.__est_alpha_a, self.__est_alpha_beta, self.__est_alpha_tau],2))

            ## Cluster pruning - remove the cluster with pi_k < prune_thres ##
            pruned_idx_a = np.where(self.__est_pi_a[:self.Ka] < prune_threshold_a)[0] 
            pruned_idx_beta = np.where(self.__est_pi_beta[:self.Kbeta] < prune_threshold_beta)[0]
            pruned_idx_tau = np.where(self.__est_pi_tau[:self.Ktau] < prune_threshold_tau)[0]
            self.K -= np.min([len(pruned_idx_a), len(pruned_idx_beta), len(pruned_idx_tau)]) # also consider noise cluster
            #self.K -= np.min([len(pruned_idx_a), len(pruned_idx_beta)]) # new K
            self.Ka -= len(pruned_idx_a); self.Kbeta -= len(pruned_idx_beta); self.Ktau -= len(pruned_idx_tau)
            #? may also consider empirical number of clusters as new Ka, Kbeta, Ktau
            self.__piStar_a[:,:self.Ka] = softmax(self.__piStar_a[:,:self.Ka], axis=1) # renormalize to sum to 1 (this function is stable)
            self.__piStar_beta[:,:self.Kbeta] = softmax(self.__piStar_beta[:,:self.Kbeta], axis=1) # (no floating point error)
            self.__piStar_tau[:,:self.Ktau] = softmax(self.__piStar_tau[:,:self.Ktau], axis=1)
            self.__est_Za = np.argmax(self.__piStar_a[:,:self.Ka], axis=1)
            self.__est_Zbeta = np.argmax(self.__piStar_beta[:,:self.Kbeta], axis=1)
            self.__est_Ztau = np.argmax(self.__piStar_tau[:,:self.Ktau], axis=1)

            #- Measure convergence using the relative change (& abs change) in the log marginal probability bound,
            # stopping the algorithm when it was less than ELBO_rel_tol (ELBO_abs_tol).
            # Alternatively can stop the iterative scheme when the change of the L2-norm of the vector of variational parameters is smaller than some e = 1e−5.
            if verbose: print('[INFO] Compute ELBO...')
            ELBO = self.__compute_ELBO(est_logV_a, est_log1V_a, est_logV_beta, est_log1V_beta, est_logV_tau, est_log1V_tau)
            if verbose: print('-- ELBO:', round(ELBO, 4), '[√]' if ELBO > self.ELBO_iters[-1] or self.ELBO_iters[-1]==1 else '[X]')
            self.ELBO_iters.append(ELBO) # store ELBO
            if verbose: print('[INFO] Check convergence...')
            if self._has_converged(ELBO_rel_tol, ELBO_abs_tol):
                if verbose: 
                    print('Est. Pi.a:', np.round(self.__est_pi_a, 4))
                    print('Est. Pi.beta:', np.round(self.__est_pi_beta, 4))
                    print('Est. Pi.tau:', np.round(self.__est_pi_tau, 4))
                print('[INFO] Converged! Optimal ELBO is reached:', round(ELBO, 4))
                break
            elif ELBO is np.nan:
                print('[INFO] ELBO is NaN! Break the loop.')
                break
            elif it == maxIter-1:
                if verbose: 
                    print('Est. Pi.a:', np.round(self.__est_pi_a, 4))
                    print('Est. Pi.beta:', np.round(self.__est_pi_beta, 4))
                    print('Est. Pi.tau:', np.round(self.__est_pi_tau, 4))
                print('[INFO] Maximum number of iterations reached! Current ELBO:', round(ELBO, 4))
        end_time = time.perf_counter()
        total_time = end_time - start_time
        hours, remainder = divmod(total_time, 3600)
        minutes, seconds = divmod(remainder, 60)
        self.exc_time = f'{int(hours)}h:{int(minutes):02d}m:{seconds:.2f}s'

    def getEstimates(self, verbose: bool=False):
        """Get the estimates of the model parameters

        Returns:
            estimates (dict): dictionary of estimated values for a, beta, sigma (with sd), pi and Z.
        """
        if len(self.ELBO_iters) == 1:
            raise ValueError('Please fit the model first!')
        # get unique clusters
        est_Za_unique = np.unique(self.__est_Za)
        est_Zbeta_unique = np.unique(self.__est_Zbeta)
        est_Ztau_unique = np.unique(self.__est_Ztau)
        if verbose:
            print('Unique clusters for a ({}): {}'.format(len(est_Za_unique), est_Za_unique))
            print('Unique clusters for beta ({}): {}'.format(len(est_Zbeta_unique), est_Zbeta_unique))
            print('Unique clusters for sigma/tau ({}): {}'.format(len(est_Ztau_unique), est_Ztau_unique))
        # get the estimates for a, beta, tau
        est_a_ = self.__est_a[est_Za_unique]
        est_beta_ = self.__est_beta[est_Zbeta_unique]
        est_tau_ = self.__est_tau[est_Ztau_unique]
        est_sigma = np.sqrt(1/est_tau_)
        # get the sd for estimation means
        est_a_sd = np.sqrt(1/self.__lamStar_a[:,1][est_Za_unique])
        est_beta_sd = np.array([np.max(np.sqrt(1/np.diag(self.__lamStar_beta2[i]))) for i in est_Zbeta_unique])
        est_tau_sd = np.sqrt(self.__lamStar_tau[:,0]/self.__lamStar_tau[:,1]**2)[est_Ztau_unique]
        est_sigma_min = np.sqrt(1/(est_tau_+est_tau_sd))
        est_sigma_max = np.sqrt(1/(est_tau_-est_tau_sd))
        est_sigma_sd = np.array(list(map(max,zip(est_sigma - est_sigma_min, est_sigma_max - est_sigma))))
        # get the estimates for pi
        est_pi_a_ = self.__est_pi_a[est_Za_unique]
        est_pi_beta_ = self.__est_pi_beta[est_Zbeta_unique]
        est_pi_tau_ = self.__est_pi_tau[est_Ztau_unique]
        # map the cluster index to the index from 0 to K
        est_Za_map = {old_idx: i for i, old_idx in enumerate(est_Za_unique)}
        est_Zbeta_map = {old_idx: i for i, old_idx in enumerate(est_Zbeta_unique)}
        est_Ztau_map = {old_idx: i for i, old_idx in enumerate(est_Ztau_unique)}
        est_Za = np.array([est_Za_map[i] for i in self.__est_Za])
        est_Zbeta = np.array([est_Zbeta_map[i] for i in self.__est_Zbeta])
        est_Ztau = np.array([est_Ztau_map[i] for i in self.__est_Ztau])
        return {"intercept":est_a_, "intercept_sd": est_a_sd, "B_coef":est_beta_, "B_coef_sd": est_beta_sd, "sigma":est_sigma, "sigma_sd":est_sigma_sd,\
            "a_clust":est_Za, "beta_clust":est_Zbeta, "sigma_clust":est_Ztau, "pi_a":est_pi_a_, "pi_beta":est_pi_beta_, "pi_sigma":est_pi_tau_}

    def getPiMat(self):
        """Get the original pi est. and matrices for a, beta, tau

        Returns:
            piMat (dict): dictionary of pi matrices and estimated pi for a, beta, tau.
        """
        if len(self.ELBO_iters) == 1:
            raise ValueError('Please fit the model first!')
        piMat = {"PiMat_a": self.__piStar_a[:,:self.Ka], "PiMat_beta": self.__piStar_beta[:,:self.Kbeta], "PiMat_tau": self.__piStar_tau[:,:self.Ktau], "pi_a": self.__est_pi_a[:self.Ka], "pi_Beta": self.__est_pi_beta[:self.Kbeta], "pi_tau": self.__est_pi_tau[:self.Ktau]}
        return piMat

    def checkEstError(self, estDict: dict=None, Gtruth: dict=None, plot: bool=False):
        """Check the relative L2 error of the estimates (with ground truth)

        Parameters:
            estDict (dict, optional): specified dictionary of estimated values for a, beta, sigma. Defaults to None, then use the estimates from the model.
            Gtruth (dict, optional): Ground truth values for a, beta, sigma. Defaults to None.
            plot (bool, optional): To plot the heatmaps or not. Defaults to False.

        Returns:
            a_l2mat, beta_l2mat, sigma_l2mat: numpy matrices of relative L2 errors for a, beta, sigma.
        """
        if len(self.ELBO_iters) == 1:
            raise ValueError('Please fit the model first!')
        if estDict:
            est_dict = estDict
        else:
            est_dict = self.getEstimates()
        est_a_ = est_dict['intercept'] # Ka
        est_beta_ = est_dict['B_coef'] # Kbeta x I
        est_sigma = est_dict['sigma'] # Ksigma
        if Gtruth:
            # check l2 error of estimates with ground truth
            true_a = Gtruth['a']
            true_beta = Gtruth['beta']
            true_sigma = Gtruth['sigma']
            a_l2mat = l2_mat(true_a, est_a_)
            beta_l2mat = l2_mat(true_beta, est_beta_)
            sigma_l2mat = l2_mat(true_sigma, est_sigma)
        else:
            # check l2 error of estimates with itself 
            a_l2mat = l2_mat(est_a_, est_a_)
            a_l2mat[range(a_l2mat.shape[0]),range(a_l2mat.shape[0])] += 10
            beta_l2mat = l2_mat(est_beta_, est_beta_)
            beta_l2mat[range(beta_l2mat.shape[0]),range(beta_l2mat.shape[0])] += 10
            sigma_l2mat = l2_mat(est_sigma, est_sigma)
            sigma_l2mat[range(sigma_l2mat.shape[0]),range(sigma_l2mat.shape[0])] += 10
        if plot:
            import seaborn as sns
            ax = sns.heatmap(a_l2mat, annot=True, fmt=".2f", cmap="Reds", vmin=0, cbar=False)
            for i, row in enumerate(a_l2mat):
                min_col_idx = np.argmin(row)
                ax.add_patch(plt.Rectangle((min_col_idx, i), 1, 1, fill=False, edgecolor='red', lw=2))
            ax.set(xlabel="Est", ylabel="Gtruth" if Gtruth else "Est")
            plt.title('Relative L2 error of a')
            plt.show()
            ax = sns.heatmap(beta_l2mat, annot=True, fmt=".2f", cmap="Reds", vmin=0, cbar=False)
            for i, row in enumerate(beta_l2mat):
                min_col_idx = np.argmin(row)
                ax.add_patch(plt.Rectangle((min_col_idx, i), 1, 1, fill=False, edgecolor='red', lw=2))
            ax.set(xlabel="Est", ylabel="Gtruth" if Gtruth else "Est")
            plt.title('Relative L2 error of beta')
            plt.show()
            ax = sns.heatmap(sigma_l2mat, annot=True, fmt=".2f", cmap="Reds", vmin=0, cbar=False)
            for i, row in enumerate(sigma_l2mat):
                min_col_idx = np.argmin(row)
                ax.add_patch(plt.Rectangle((min_col_idx, i), 1, 1, fill=False, edgecolor='red', lw=2))
            ax.set(xlabel="Est", ylabel="Gtruth" if Gtruth else "Est")
            plt.title('Relative L2 error of sigma')
            plt.show()
        return a_l2mat, beta_l2mat, sigma_l2mat

    def printRes(self, estDict: dict=None, Gtruth: dict=None):
        """Print the results of the model

        Parameters:
            estDict (dict, optional): specified dictionary of estimated values for a, beta, sigma. Defaults to None, then use the estimates from the model.
            Gtruth (dict, optional): Ground truth values for a, beta, sigma. Defaults to None.
        """
        if len(self.ELBO_iters) == 1:
            raise ValueError('Please fit the model first!')
        from prettytable import PrettyTable, ALL
        import sys
        if sys.__stdin__.isatty():
            import plotext as plt
        else:
            import matplotlib.pyplot as plt
        # plot ELBO
        plt.plot(self.ELBO_iters[1:])
        step = np.ceil(len(self.ELBO_iters[1:])/15).astype('int32')
        xtcks = np.arange(len(self.ELBO_iters[1:]),step=step)[1:]
        if len(xtcks) == 0: xtcks = np.array([1])
        if len(self.ELBO_iters[1:])-xtcks[-1]<step:
            xtcks = xtcks[:-1]
        xtcks = np.append(xtcks,len(self.ELBO_iters[1:]))
        plt.xticks(ticks=xtcks-1, labels=xtcks)
        plt.xlabel('Iteration')
        plt.ylabel('ELBO')
        plt.show()
        if estDict:
            est_dict = estDict
        else:
            est_dict = self.getEstimates(verbose=True)
        # get the estimates for a, beta, tau
        est_a_ = est_dict['intercept']
        est_beta_ = est_dict['B_coef']
        est_sigma = est_dict['sigma']
        # get the sd for estimation means
        est_a_sd = est_dict['intercept_sd']
        est_beta_sd = est_dict['B_coef_sd']
        est_sigma_sd = est_dict['sigma_sd']
        # get the estimates for pi
        est_pi_a_ = est_dict['pi_a']
        est_pi_beta_ = est_dict['pi_beta']
        est_pi_tau_ = est_dict['pi_sigma']
        # get the estimates for Z
        est_Za = est_dict['a_clust']
        est_Zbeta = est_dict['beta_clust']
        est_Ztau = est_dict['sigma_clust']
        print('#=============== Estimations ===============#')
        print('Estimated a:')
        table_a = PrettyTable(header=False)
        table_a.add_row(['a']+ list(map(' ±'.join, zip(map(str, np.round(est_a_,2).tolist()), map(str, np.round(est_a_sd,3).tolist())))))
        table_a.add_row(['π_a']+np.round(est_pi_a_,3).tolist())
        print(table_a)
        print('Estimated beta:')
        table_beta = PrettyTable(header=False, hrules=ALL)
        table_beta.add_column(None,['K x I matrix']+ list(map(' ±'.join, zip(map(str, np.round(est_beta_,2).tolist()), map(str, np.round(est_beta_sd,3).tolist())))))
        table_beta.add_column(None,['π_beta']+np.round(est_pi_beta_,3).tolist())
        print(table_beta)
        print('Estimated sigma:')
        table_tau = PrettyTable(header=False)
        table_tau.add_row(['sigma']+ list(map(' ±'.join, zip(map(str, np.round(est_sigma, 2).tolist()), map(str, np.round(est_sigma_sd,3).tolist())))))
        table_tau.add_row(['π_sigma']+np.round(est_pi_tau_, 3).tolist())
        print(table_tau)
        if Gtruth:
            from sklearn.metrics import adjusted_rand_score
            print('#=============== Ground truth ===============#')
            # ground truth
            true_a = Gtruth.get('a', [None])
            true_pi_a = Gtruth.get('pi_a', [None])
            true_beta = Gtruth.get('beta', [None])
            true_pi_beta = Gtruth.get('pi_beta', [None])
            true_sigma = Gtruth.get('sigma', [None])
            true_pi_sigma = Gtruth.get('pi_sigma', [None])
            print('True a:')
            table_a = PrettyTable(header=False)
            table_a.add_row(['a']+list(true_a))
            table_a.add_row(['π_a']+list(true_pi_a))
            print(table_a)
            print('True beta:')
            table_beta = PrettyTable(header=False, hrules=ALL)
            table_beta.add_column(None,['K x I matrix']+list(true_beta))
            table_beta.add_column(None,['π_beta']+list(true_pi_beta))
            print(table_beta)
            print('True sigma:')
            table_tau = PrettyTable(header=False)
            table_tau.add_row(['sigma']+list(true_sigma))
            table_tau.add_row(['π_sigma']+list(true_pi_sigma))
            print(table_tau)
            print('#============ Relative L2 error ============#')
            a_l2mat, beta_l2mat, sigma_l2mat = self.checkEstError(Gtruth=Gtruth)
            print('L2_err_a:', np.round(np.min(a_l2mat, axis=1), 3), 'Mean:', np.round(np.min(a_l2mat, axis=1).mean(), 3))
            print('L2_err_beta:', np.round(np.min(beta_l2mat, axis=1), 3), 'Mean:', np.round(np.min(beta_l2mat, axis=1).mean(), 3))
            print('L2_err_sigma:', np.round(np.min(sigma_l2mat, axis=1), 3), 'Mean:', np.round(np.min(sigma_l2mat, axis=1).mean(), 3))
            print('#=================== ARI ===================#')
            # ARI
            print('ARI_a:', round(adjusted_rand_score(Gtruth['true_a_clust'], est_Za),3))
            print('ARI_beta:', round(adjusted_rand_score(Gtruth['true_beta_clust'], est_Zbeta),3))
            print('ARI_sigma/tau:', round(adjusted_rand_score(Gtruth['true_sigma_clust'], est_Ztau),3))

    def plotFacet(self, estDict: dict=None, facet=['a', 'beta', 'sigma'], aTshift: tuple=(0,0), save_folder: str=None, figDict: dict=None):
        """Plot the facets of the model

        Parameters:
            estDict (dict, optional): specified dictionary of estimated values for a, beta, sigma. Defaults to None, then use the estimates from the model.
            facet (str/list, optional): string names of facets. Defaults to ['a', 'beta', 'sigma'].
            aTshift (tuple, optional): shift of intercepts and Time axis. Defaults to (0,0).
            save_folder (str, optional): folder to save the plot. Defaults to None, then only show the plot.
            figDict (dict, optional): dictionary of figure parameters. Defaults to {"figsize":(6,4),"dpi":200,"fontsize":14}.
        """
        if len(self.ELBO_iters) == 1:
            raise ValueError('Please fit the model first!')
        facets = ['a', 'beta', 'sigma']
        if type(facet) != list:
            facet = [facet]
        facet_cp = facet.copy()
        if estDict:
            est_dict = estDict
        else:
            est_dict = self.getEstimates()
        if figDict:
            figDict_update = figDict.copy()
            figDict = {"figsize":(6,4),"dpi":200,"fontsize":14}
            figDict.update(figDict_update)
        else:
            figDict = {"figsize":(6,4),"dpi":200,"fontsize":14}
        fgsize = figDict["figsize"]; dpi = figDict["dpi"]; fontsize = figDict["fontsize"]
        Pis = []
        if facets[0] in facet:
            facet_cp.remove(facets[0])
            est_a = est_dict['intercept']
            pi_a = est_dict["pi_a"]
            # Plot the intercepts
            plt.figure(figsize=fgsize, dpi=dpi)
            for i in range(len(est_a)):
                plt.plot(self.times+aTshift[1], [est_a[i]+aTshift[0]]*len(self.times), "--o", alpha=min(pi_a[i]*100,1), markevery=[self.intercept_shift_to], label="a{}".format(i+1))
            plt.xticks(fontsize=fontsize)
            plt.yticks(fontsize=fontsize)
            plt.xlabel('Time', fontsize=fontsize+2)
            plt.ylabel('Intercept', fontsize=fontsize+2)
            plt.legend(loc='upper right')
            if save_folder:
                plt.savefig(f"{save_folder}pointEstA.png", bbox_inches="tight")
                plt.close()
            else:
                plt.show()
            # Pi_a
            Pis.append(pi_a)
        if facets[1] in facet:
            facet_cp.remove(facets[1])
            est_beta = est_dict['B_coef']
            pi_beta = est_dict["pi_beta"]
            # Plot the coefficients
            plt.figure(figsize=fgsize, dpi=dpi)
            for i in range(est_beta.shape[0]):
                y_beta = est_beta[i]@ self.B
                plt.plot(self.times+aTshift[1], y_beta, alpha=min(pi_beta[i]*100,1), label="Beta{}".format(i+1))
            plt.xticks(fontsize=fontsize)
            plt.yticks(fontsize=fontsize)
            plt.xlabel('Time', fontsize=fontsize+2)
            plt.ylabel('Shape', fontsize=fontsize+2)
            plt.legend(loc='upper right')
            if save_folder:
                plt.savefig(f"{save_folder}pointEstbeta.png", bbox_inches="tight")
                plt.close()
            else:
                plt.show()
            # Pi_beta
            Pis.append(pi_beta)
        if facets[2] in facet:
            from scipy.stats import norm
            facet_cp.remove(facets[2])
            est_sigma = est_dict['sigma']
            pi_sigma = est_dict["pi_sigma"]
            max_sigma = np.max(est_sigma)
            x = np.linspace(-2*max_sigma, 2*max_sigma, 1000)
            # Plot the sigmas
            plt.figure(figsize=fgsize, dpi=dpi)
            for i in range(len(est_sigma)):
                plt.plot(x, norm.pdf(x, 0, est_sigma[i]), alpha=min(pi_sigma[i]*100,1), label="Sigma{}".format(i+1))
            plt.xticks(fontsize=fontsize)
            plt.yticks(fontsize=fontsize)
            plt.xlabel('Sigma', fontsize=fontsize+2)
            plt.legend(loc='upper right')
            if save_folder:
                plt.savefig(f"{save_folder}pointEstSIGMA.png", bbox_inches="tight")
                plt.close()
            else:
                plt.show()
            # Pi_sigma
            Pis.append(pi_sigma)
        if len(Pis) != 0:
            stacked_Pis_barplot(Pis, fgsize, dpi, fontsize)
            if save_folder:
                plt.savefig(f"{save_folder}Pis.png", bbox_inches="tight")
                plt.close()
            else:
                plt.show()
        if len(facet_cp) != 0:
            print('Facet {} not found!'.format(facet_cp))
        if save_folder:
            print(f"[INFO] Plots saved in '{save_folder}'")

    def showClust(self, estDict: dict=None, save_folder: str=None, figDict: dict=None):
        """Show the cluster assignments of the model

        Parameters:
            estDict (dict, optional): specified dictionary of estimated values for a, beta, sigma. Defaults to None, then use the estimates from the model.
            save_folder (str, optional): folder to save the plot. Defaults to None, then only show the plot.
            figDict (dict, optional): dictionary of figure parameters. Defaults to {"figsize":(6,4),"dpi":200,"fontsize":14}.
        """
        if len(self.ELBO_iters) == 1:
            raise ValueError('Please fit the model first!')
        if estDict:
            est_dict = estDict
        else:
            est_dict = self.getEstimates()
        if figDict:
            figDict_update = figDict.copy()
            figDict = {"figsize":(6,4),"dpi":200,"fontsize":14}
            figDict.update(figDict_update)
        else:
            figDict = {"figsize":(6,4),"dpi":200,"fontsize":14}
        fgsize = figDict["figsize"]; dpi = figDict["dpi"]; fontsize = figDict["fontsize"]
        est_Za = est_dict['a_clust']
        est_Zbeta = est_dict['beta_clust']
        # Initialize the co-occurrence matrix
        probability_matrix = np.zeros((np.max(est_Za)+1,np.max(est_Zbeta)+1))
        # Count co-occurrences of cluster pairs
        for c1, c2 in zip(est_Za, est_Zbeta):
            probability_matrix[c1, c2] += 1
        probability_matrix = probability_matrix / probability_matrix.sum()  # Normalize to probabilities
        probability_matrix = probability_matrix * 100 # Scale to percentage
        import seaborn as sns
        plt.figure(figsize=fgsize, dpi=dpi)
        sns.heatmap(probability_matrix, annot=True, fmt=".1f", cmap="Blues",cbar_kws={'format': '%.0f%%'},vmin=0,xticklabels=[f"{i+1}" for i in range(np.max(est_Zbeta)+1)],yticklabels=[f"{i+1}" for i in range(np.max(est_Za)+1)])
        plt.xlabel("Coefficient Cluster", fontsize=fontsize)
        plt.ylabel("Intercept Cluster", fontsize=fontsize)
        if save_folder:
            plt.savefig(f"{save_folder}ClusterAssign.png", bbox_inches="tight")
            plt.close()
        else:
            plt.show()


class MFVAR(MFVI):
    def __init__(self, **kwargs):
        """Nonparametric Bayesian Multi-Facet Vector autoregressive model with Mean-Field Variational Inference

        Parameters:
            ---Super-class-Args:
            data (np.ndarray): Data matrix of size N x D x T
            trunc_level (int): truncation level for the model
            seed (int, optional): Random seed for reproducibility. Defaults to None.
            ---NPBMF-VAR-Args:
        """
        print("######################################################")
        print("### Mean-Field Variational Inference for NPBMF-VAR ###")
        print("######################################################")
        super().__init__(**kwargs)
        #--- data distribution ---#
        # y_nt ~ N_D(a_k1 + Beta_k2(y_n(t-1) - a_k1), diag(tau_k3^-1))
        ##############################################
        ### Specific hyperparameters for the model ###
        ################################################
        self.Ka = self.K; self.KBeta = self.K; self.Ktau = self.K
        ### Hyperparameters of priors or hyperpriors ###
        # Priors
        # a_k ~ N(0, 1) can be nonzero means
        self.__prior_mua = 0 #? a_prior_mean can be changed
        self.__prior_epsa = 1 #? a_prior_sd can be changed
        self.__prior_epsa = 1/self.__prior_epsa**2 # to precision
        # Beta_k ~ N(0, 1)
        self.__prior_muBeta = 0; self.__prior_epsBeta = 1
        self.__prior_epsBeta = 1/self.__prior_epsBeta**2
        # tau_k ~ Gamma(1, 1)
        self.__lambda_tau1 = 1; self.__lambda_tau2 = 1
        # v ~ Beta(1, alpha)
        self.__alphfix = 1 # not changed
        # alpha ~ Gamma(1, 1)
        self.__s1 = 1; self.__s2 = 1 #? s can be changed
        #########################################
        ### Initialize variational parameters ###
        # Random initialization
        print('[INFO] Random initialization...')
        if self.seed is not None:
            np.random.seed(self.seed)
        # variational parameters for a (vector of D)
        self.__MuStar_a = np.random.normal(self.__prior_mua, np.sqrt(1/self.__prior_epsa), (self.K, self.D)) # dim K x D
        self.__LamStar_a = np.random.gamma(1, 1, (self.K, self.D,self.D)) # precisions
        # variational parameters for tau (vector of D)
        self.__lamStar_tau = np.random.gamma(1, 1, (self.K, self.D, 2)) # kth params for tau_k
        # variational parameters for alpha
        self.__sStar_a = np.random.gamma(1, 1, 2) # length 2, shape and rate for Gamma
        self.__sStar_Beta = np.random.gamma(1, 1, 2) # length 2
        self.__sStar_tau = np.random.gamma(1, 1, 2) # length 2
        # variational parameters for z
        self.__piStar_a = np.random.dirichlet([1]*self.K, self.N) # matrix of size N x K
        self.__piStar_Beta = np.random.dirichlet([1]*self.K, self.N) # matrix of size N x K
        self.__piStar_tau = np.random.dirichlet([1]*self.K, self.N) # matrix of size N x K
        #-----------------------------------------#
        # Create storage for the rest var params
        self.__MuStar_Beta = np.zeros((self.K, self.D, self.D)) # dim K x D x D
        self.__RowprecStar_Beta = np.zeros((self.K, self.D, self.D)) # row precision matrices (U)
        self.__PrecStar_Beta = np.zeros((self.K, self.D, self.D)) # column precision matrices (V)
        self.__LamStar_Beta = np.zeros((self.K, self.D**2, self.D**2)) # precision matrices for vec(Beta) = np.kron(V, U)
        self.__alphaStar_a = np.zeros((self.K-1, 2)) # only K-1 alphas
        self.__alphaStar_Beta = np.zeros((self.K-1, 2))
        self.__alphaStar_tau = np.zeros((self.K-1, 2))
        ###########################################
        ### Estimated means of model parameters ###
        self.__est_a = self.__MuStar_a
        self.__est_Beta = self.__MuStar_Beta
        self.__est_tau = self.__lamStar_tau[:,:, 0]/self.__lamStar_tau[:,:, 1] # dim K x D
        self.__est_Za = np.argmax(self.__piStar_a, axis=1)
        self.__est_ZBeta = np.argmax(self.__piStar_Beta, axis=1)
        self.__est_Ztau = np.argmax(self.__piStar_tau, axis=1)
        self.__est_pi_a = np.zeros(self.K)
        self.__est_pi_Beta = np.zeros(self.K)
        self.__est_pi_tau = np.zeros(self.K)
        #----------------------------------------#
        self.__est_alpha_a = self.__sStar_a[0]/self.__sStar_a[1]
        self.__est_alpha_Beta = self.__sStar_Beta[0]/self.__sStar_Beta[1]
        self.__est_alpha_tau = self.__sStar_tau[0]/self.__sStar_tau[1]

    def initialize(self, param_dict: dict):
        """Initialize the variational parameters with specified values

        Parameters:
            param_dict (dict): dictionary of specified values for initialization. 
                This includes: 'a_mean', 'a_prior_mean', 'a_prior_sd', 'alpha_prior_s', 'tau_gamma', 'a_pi', 'Beta_pi', 'tau_pi'.
            Examples:
            'a_prior_mean' (float): specified prior means for a_k
            'a_prior_sd' (float): specified prior standard deviation for a_k
            'alpha_prior_s' (tuple): specified shape and rate for alpha ~ Gamma(shape, rate)
            'a_mean' (2D list/array): specified means for a_k
            'tau_gamma' (3D list/array): specified shape and rate for tau_k ~ Gamma(shape, rate)
            'a_pi' (2D list/array): specified pi_n for a_k
            'Beta_pi' (2D list/array): specified pi_n for Beta_k
            'tau_pi' (2D list/array): specified pi_n for tau_k
        """
        orig_prior_mua = self.__prior_mua; change_prior_mua = False
        orig_prior_epsa = self.__prior_epsa; change_prior_epsa = False
        orig_s1 = self.__s1
        orig_s2 = self.__s2; change_s = False
        orig_MuStar_a = self.__MuStar_a.copy(); change_MuStar_a = False
        orig_lamStar_tau = self.__lamStar_tau.copy(); change_lamStar_tau = False
        orig_piStar_a = self.__piStar_a.copy(); change_piStar_a = False
        orig_piStar_Beta = self.__piStar_Beta.copy(); change_piStar_Beta = False
        orig_piStar_tau = self.__piStar_tau.copy(); change_piStar_tau = False
        # update the specified values
        self.__prior_mua = param_dict.get('a_prior_mean', orig_prior_mua)
        self.__prior_epsa = param_dict.get('a_prior_sd', np.sqrt(1/orig_prior_epsa))
        self.__prior_epsa = 1/self.__prior_epsa**2
        changed_s = param_dict.get('alpha_prior_s', (orig_s1,orig_s2))
        self.__s1 = changed_s[0]
        self.__s2 = changed_s[1]
        MuStar_a_new = np.array(param_dict.get('a_mean', self.__MuStar_a))
        # check length
        if MuStar_a_new.shape[0] < self.K:
            MuStar_a_new = np.vstack((MuStar_a_new, np.zeros((self.K-MuStar_a_new.shape[0], self.D))))
        self.__MuStar_a = MuStar_a_new.astype('float64')
        self.__lamStar_tau = np.array(param_dict.get('tau_gamma', self.__lamStar_tau))
        self.__lamStar_tau = self.__lamStar_tau.astype('float64')
        self.__piStar_a = np.array(param_dict.get('a_pi', orig_piStar_a))
        self.__piStar_Beta = np.array(param_dict.get('Beta_pi', orig_piStar_Beta))
        self.__piStar_tau = np.array(param_dict.get('tau_pi', orig_piStar_tau))
        self.__piStar_a = self.__piStar_a.astype('float64')
        self.__piStar_Beta = self.__piStar_Beta.astype('float64')
        self.__piStar_tau = self.__piStar_tau.astype('float64')
        # check if the values are changed
        if self.__prior_mua != orig_prior_mua:
            change_prior_mua = True
        if self.__prior_epsa != orig_prior_epsa:
            change_prior_epsa = True
        if (self.__s1, self.__s2) != (orig_s1, orig_s2):
            change_s = True
        if (self.__MuStar_a - orig_MuStar_a).any() != 0:
            change_MuStar_a = True
        if (self.__lamStar_tau - orig_lamStar_tau).any() != 0:
            change_lamStar_tau = True
        if (self.__piStar_a - orig_piStar_a).any() != 0:
            change_piStar_a = True
        if (self.__piStar_Beta - orig_piStar_Beta).any() != 0:
            change_piStar_Beta = True
        if (self.__piStar_tau - orig_piStar_tau).any() != 0:
            change_piStar_tau = True
        if change_prior_mua or change_prior_epsa or change_s or change_MuStar_a or change_lamStar_tau or change_piStar_a or change_piStar_Beta or change_piStar_tau:
            print("[INFO] Set specified initialization for {}{}{}{}{}{}{}{}".format("'a_mean'; " if change_MuStar_a else '', "'a_prior_mean'; " if change_prior_mua else '', "'a_prior_sd'; " if change_prior_epsa else '', "'alpha_prior_s'; " if change_s else '', "'tau_gamma'; " if change_lamStar_tau else '', "'a_pi'; " if change_piStar_a else '', "'Beta_pi'; " if change_piStar_Beta else '', "'tau_pi'" if change_piStar_tau else ''))
        if not change_MuStar_a and (change_prior_mua or change_prior_epsa):
            # variational parameters for a (vector of D)
            self.__MuStar_a = np.random.normal(self.__prior_mua, np.sqrt(1/self.__prior_epsa), (self.K, self.D)) # dim K x D
        # recompute the estimated means
        if change_MuStar_a or change_prior_mua or change_prior_epsa:
            self.__est_a = self.__MuStar_a
        if change_lamStar_tau:
            self.__est_tau = self.__lamStar_tau[:,:, 0]/self.__lamStar_tau[:,:, 1] # dim K x D

    def __get_obs_data(self, n):
        obs_idx = np.isfinite(self.Y[n])
        Yn = self.Y[n, obs_idx].reshape(self.D, -1)
        Tn = Yn.shape[1]
        return Yn, Tn
    
    def __compute_ELBO(self, est_logV_a, est_log1V_a, est_logV_Beta, est_log1V_Beta, est_logV_tau, est_log1V_tau):
        """Compute the Evidence Lower Bound (ELBO) for the model
        """
        ELBO = 0
        # prior -q on a
        ELBO += sum([self.D/2*np.log(self.__prior_epsa) -1/2*np.array(LA.slogdet(self.__LamStar_a[k,:,:])).prod() - self.__prior_epsa/2* (LA.norm(self.__est_a[k,:]-self.__prior_mua)**2) for k in range(self.Ka)]) #  + np.trace(LA.pinv(self.__LamStar_a[k,:,:]))
        # prior -q on Beta
        ELBO += sum([self.D**2/2*np.log(self.__prior_epsBeta) -1/2*np.array(LA.slogdet(self.__LamStar_Beta[k,:,:])).prod() - self.__prior_epsBeta/2* (LA.norm(self.__est_Beta[k]-self.__prior_muBeta)**2 + np.trace(LA.pinv(self.__LamStar_Beta[k]))) for k in range(self.KBeta)])
        # prior -q on tau
        for k in range(self.Ktau):
            ELBO += sum([self.__lambda_tau1*np.log(self.__lambda_tau2) - self.__lamStar_tau[k,d,0]*np.log(self.__lamStar_tau[k,d,1]) -gammaln(self.__lambda_tau1)+gammaln(self.__lamStar_tau[k,d,0]) +(self.__lambda_tau1-self.__lamStar_tau[k,d,0])*(psi(self.__lamStar_tau[k,d,0])-np.log(self.__lamStar_tau[k,d,1])) +self.__lamStar_tau[k,d,0] -self.__lambda_tau2*self.__est_tau[k,d] for d in range(self.D)])
        # prior -q on v
        ELBO += sum([(self.__alphfix-self.__alphaStar_a[k,0])*est_logV_a[k] + (self.__est_alpha_a-self.__alphaStar_a[k,1])*est_log1V_a[k] - psi(self.__sStar_a[0]) +np.log(self.__sStar_a[1]) +betaln(self.__alphaStar_a[k,0],self.__alphaStar_a[k,1]) for k in range(self.Ka-1)])
        ELBO += sum([(self.__alphfix-self.__alphaStar_Beta[k,0])*est_logV_Beta[k] + (self.__est_alpha_Beta-self.__alphaStar_Beta[k,1])*est_log1V_Beta[k] - psi(self.__sStar_Beta[0]) +np.log(self.__sStar_Beta[1]) +betaln(self.__alphaStar_Beta[k,0],self.__alphaStar_Beta[k,1]) for k in range(self.KBeta-1)])
        ELBO += sum([(self.__alphfix-self.__alphaStar_tau[k,0])*est_logV_tau[k] + (self.__est_alpha_tau-self.__alphaStar_tau[k,1])*est_log1V_tau[k] - psi(self.__sStar_tau[0]) +np.log(self.__sStar_tau[1]) +betaln(self.__alphaStar_tau[k,0],self.__alphaStar_tau[k,1]) for k in range(self.Ktau-1)])
        for n in range(self.N):
            # for Z_a_n - max piStar_a[n, :]
            ELBO += sum([np.sum(self.__piStar_a[n, (k+1):self.Ka])*est_log1V_a[k] +self.__piStar_a[n,k]*est_logV_a[k] for k in range(self.Ka-1)]) - (np.log(self.__piStar_a[n, self.__est_Za[n]]) if self.__piStar_a[n, self.__est_Za[n]] >0 else 0)
            # for Z_beta_n
            ELBO += sum([np.sum(self.__piStar_Beta[n, (k+1):self.KBeta])*est_log1V_Beta[k] +self.__piStar_Beta[n,k]*est_logV_Beta[k] for k in range(self.KBeta-1)]) - (np.log(self.__piStar_Beta[n, self.__est_ZBeta[n]]) if self.__piStar_Beta[n, self.__est_ZBeta[n]] >0 else 0)
            # for Z_tau_n
            ELBO += sum([np.sum(self.__piStar_tau[n, (k+1):self.Ktau])*est_log1V_tau[k] +self.__piStar_tau[n,k]*est_logV_tau[k] for k in range(self.Ktau-1)]) - (np.log(self.__piStar_tau[n, self.__est_Ztau[n]]) if self.__piStar_tau[n, self.__est_Ztau[n]] >0 else 0)
            # likelihood for y_n
            E_a_n = sum([self.__piStar_a[n,j]* self.__est_a[j,:] for j in range(self.Ka)])
            E_Beta_n = sum([self.__piStar_Beta[n,j]* self.__est_Beta[j, :,:] for j in range(self.KBeta)])
            E_tau_n = sum([self.__piStar_tau[n, j]*self.__est_tau[j,:] for j in range(self.Ktau)])
            sum_E_log_tau_n = sum([sum([self.__piStar_tau[n, j]*(psi(self.__lamStar_tau[j,d,0]) - np.log(self.__lamStar_tau[j,d,1])) for j in range(self.Ktau)]) for d in range(self.D)])
            Yn, Tn = self.__get_obs_data(n)
            ELBO += 1/2*sum_E_log_tau_n - 1/2*(Yn[:,0]-E_a_n).T@ np.diag(E_tau_n)@(Yn[:,0]-E_a_n) + sum([1/2*sum_E_log_tau_n - 1/2*(Yn[:,t]-E_a_n-E_Beta_n@Yn[:,t-1]+E_Beta_n@E_a_n).T@np.diag(E_tau_n)@(Yn[:,t]-E_a_n-E_Beta_n@Yn[:,t-1]+E_Beta_n@E_a_n) for t in range(1,Tn)])
        # prior -q on alpha
        ELBO += self.__s1*np.log(self.__s2) - self.__sStar_a[0]*np.log(self.__sStar_a[1]) -gammaln(self.__s1)+gammaln(self.__sStar_a[0]) +(self.__s1-self.__sStar_a[0])*(psi(self.__sStar_a[0])-np.log(self.__sStar_a[1])) -self.__s2*self.__est_alpha_a +self.__sStar_a[0]
        ELBO += self.__s1*np.log(self.__s2) - self.__sStar_Beta[0]*np.log(self.__sStar_Beta[1]) -gammaln(self.__s1)+gammaln(self.__sStar_Beta[0]) +(self.__s1-self.__sStar_Beta[0])*(psi(self.__sStar_Beta[0])-np.log(self.__sStar_Beta[1])) -self.__s2*self.__est_alpha_Beta +self.__sStar_Beta[0]
        ELBO += self.__s1*np.log(self.__s2) - self.__sStar_tau[0]*np.log(self.__sStar_tau[1]) -gammaln(self.__s1)+gammaln(self.__sStar_tau[0]) +(self.__s1-self.__sStar_tau[0])*(psi(self.__sStar_tau[0])-np.log(self.__sStar_tau[1])) -self.__s2*self.__est_alpha_tau +self.__sStar_tau[0]
        return ELBO

    def fit(self, maxIter: int=200, prune_threshold: Union[float, List[float]]=0.01, ELBO_rel_tol: float=1e-8, ELBO_abs_tol: float=1e-4, verbose: bool=False):
        """Coordinate Ascent algorithm for MFVI

        Parameters:
            maxIter (int, optional): Number of iterations. Defaults to 200.
            prune_threshold (float/List of floats, optional): Probability threshold for pruning the clusters. Defaults to 0.01.
            ELBO_rel_tol (float, optional): Relative tolerance of changes in ELBO for conergence check. Defaults to 1e-8.
            ELBO_abs_tol (float, optional): Absolute tolerance of changes in ELBO for conergence check. Defaults to 1e-4.
            verbose (bool, optional): Whether to print out detailed training information. Defaults to False.
        """
        if isinstance(prune_threshold, list):
            if len(prune_threshold) != 3:
                raise ValueError('The length of prune_threshold should be 3!')
            if max(prune_threshold) > 1/self.K: #* Check pruning threshold
                raise ValueError('Pruning threshold may be too large considering current truncation level!')
            prune_threshold_a = prune_threshold[0]
            prune_threshold_Beta = prune_threshold[1]
            prune_threshold_tau = prune_threshold[2]
        else:
            if prune_threshold > 1/self.K: #* Check pruning threshold
                raise ValueError('Pruning threshold may be too large considering current truncation level!')
            prune_threshold_a = prune_threshold
            prune_threshold_Beta = prune_threshold
            prune_threshold_tau = prune_threshold
        start_time = time.perf_counter()
        for it in tqdm(range(maxIter), desc="[INFO] Iterate the Coordinate Ascent algorithm"): # loop over iterations for MFVI
            if verbose: print(f'#---------------- Iteration {it+1} ----------------#')
            est_Va = np.zeros(self.Ka-1)
            est_VBeta = np.zeros(self.KBeta-1)
            est_Vtau = np.zeros(self.Ktau-1)
            est_logV_a = np.zeros(self.Ka-1)
            est_log1V_a = np.zeros(self.Ka-1)
            est_logV_Beta = np.zeros(self.KBeta-1)
            est_log1V_Beta = np.zeros(self.KBeta-1)
            est_logV_tau = np.zeros(self.Ktau-1)
            est_log1V_tau = np.zeros(self.Ktau-1)
            # update variational parameters for kth cluster
            for k in (tqdm(range(self.K), desc='[INFO] Update variational parameters for all Ks',leave=False) if verbose else range(self.K)): 
                if k < self.Ka-1:
                    #- update variational parameters for v: alphaStar -#
                    self.__alphaStar_a[k, 0] = self.__alphfix + np.sum(self.__piStar_a[:, k])
                    self.__alphaStar_a[k, 1] = self.__est_alpha_a + np.sum(self.__piStar_a[:, (k+1):self.Ka])
                    #--- get means for V ---#
                    est_Va[k] = self.__alphaStar_a[k, 0]/np.sum(self.__alphaStar_a[k,:])
                    #--- get estimates for pi ---#
                    self.__est_pi_a[k] = np.exp(np.log(est_Va[k])+ np.log(1-est_Va)[:k].sum())
                if k < self.KBeta-1:
                    self.__alphaStar_Beta[k, 0] = self.__alphfix + np.sum(self.__piStar_Beta[:, k])
                    self.__alphaStar_Beta[k, 1] = self.__est_alpha_Beta + np.sum(self.__piStar_Beta[:, (k+1):self.KBeta])
                    #--- get means for V ---#
                    est_VBeta[k] = self.__alphaStar_Beta[k, 0]/np.sum(self.__alphaStar_Beta[k,:])
                    #--- get estimates for pi ---#
                    self.__est_pi_Beta[k] = np.exp(np.log(est_VBeta[k])+ np.log(1-est_VBeta)[:k].sum())
                if k < self.Ktau-1:
                    self.__alphaStar_tau[k, 0] = self.__alphfix + np.sum(self.__piStar_tau[:, k])
                    self.__alphaStar_tau[k, 1] = self.__est_alpha_tau + np.sum(self.__piStar_tau[:, (k+1):self.Ktau])
                    #--- get means for V ---#
                    est_Vtau[k] = self.__alphaStar_tau[k, 0]/np.sum(self.__alphaStar_tau[k,:])
                    #--- get estimates for pi ---#
                    self.__est_pi_tau[k] = np.exp(np.log(est_Vtau[k])+ np.log(1-est_Vtau)[:k].sum())
                #- update variational parameters for Beta: lamStar_Beta -#
                if k < self.KBeta:
                    Beta2_sum = np.zeros((self.D**2,self.D**2))
                    Beta1_sum = np.zeros((self.D,self.D))
                    for n in range(self.N):
                        E_tau_n = sum([self.__piStar_tau[n, j]*self.__est_tau[j,:] for j in range(self.Ktau)])
                        E_a_n = sum([self.__piStar_a[n,j]* self.__est_a[j,:] for j in range(self.Ka)])
                        Yn, Tn = self.__get_obs_data(n)
                        E_a_mat_n = (E_a_n+np.zeros((self.D,Tn-1)).T).T
                        Betap1_sum_n = self.__piStar_Beta[n, k] * np.diag(E_tau_n)
                        BetaXX_n = Yn[:,:-1]-E_a_mat_n
                        Beta2_sum += self.__piStar_Beta[n, k] * np.kron(BetaXX_n @ BetaXX_n.T, np.diag(E_tau_n))
                        Beta1_sum += Betap1_sum_n @ (Yn[:,1:]-E_a_mat_n) @ BetaXX_n.T
                    self.__LamStar_Beta[k] = self.__prior_epsBeta * np.eye(self.D**2) + Beta2_sum
                    self.__RowprecStar_Beta[k,:,:] = np.diag(np.diag(self.__LamStar_Beta[k]).reshape(self.D, self.D, order='F').mean(axis=1))
                    self.__PrecStar_Beta[k,:,:] = np.diag(self.__LamStar_Beta[k]).reshape(self.D, self.D, order='F')
                    self.__MuStar_Beta[k,:,:] = (LA.solve(self.__LamStar_Beta[k], (self.__prior_epsBeta *(np.zeros((self.D,self.D))+self.__prior_muBeta) + Beta1_sum).flatten('F'))).reshape(self.D, self.D, order='F')
                    #--- get means for Beta ---#
                    self.__est_Beta[k,:,:] = self.__MuStar_Beta[k,:,:]
                #- update variational parameters for a: lamStar_a -#
                if k < self.Ka:
                    a1_sum = np.zeros((self.D,self.D)) # for precison update
                    a2_sum = np.zeros(self.D) # for Mua update
                    for n in range(self.N):
                        E_tau_n = sum([self.__piStar_tau[n, j]*self.__est_tau[j,:] for j in range(self.Ktau)])
                        E_Beta_n = sum([self.__piStar_Beta[n,j]* self.__est_Beta[j,:,:] for j in range(self.KBeta)]) # dim I
                        Yn, Tn = self.__get_obs_data(n)
                        a1_sum_n = self.__piStar_a[n, k] * np.diag(E_tau_n)
                        a2_sum_n = self.__piStar_a[n, k] * E_Beta_n.T @ np.diag(E_tau_n) @ E_Beta_n
                        a3_sum_n = self.__piStar_a[n, k] * E_Beta_n.T @ np.diag(E_tau_n)
                        a4_sum_n = self.__piStar_a[n, k] * np.diag(E_tau_n) @ E_Beta_n
                        a1_sum += Tn * a1_sum_n + (Tn-1)*a2_sum_n - (Tn-1)*a3_sum_n - (Tn-1)*a4_sum_n
                        a2_sum += a1_sum_n @ Yn[:,0] + a1_sum_n @ np.sum(Yn[:,1:], axis=1) - a4_sum_n @ np.sum(Yn[:,:-1], axis=1) + a2_sum_n@np.sum(Yn[:,:-1], axis=1) - a3_sum_n@np.sum(Yn[:,1:], axis=1)
                    self.__LamStar_a[k,:,:] = self.__prior_mua*np.eye(self.D) + a1_sum
                    self.__MuStar_a[k,:] = LA.solve(self.__LamStar_a[k,:,:], (self.__prior_epsa*(np.zeros(self.D)+self.__prior_mua) + a2_sum))
                    #--- get means for a ---#
                    self.__est_a[k,:] = self.__MuStar_a[k,:]
                #- update variational parameters for tau: lamStar_tau -#
                if k < self.Ktau:
                    tau2_sum = 0
                    Tn_ls = []
                    for n in range(self.N):
                        E_a_n = sum([self.__piStar_a[n,j]* self.__est_a[j,:] for j in range(self.Ka)])
                        E_Beta_n = sum([self.__piStar_Beta[n,j]* self.__est_Beta[j,:,:] for j in range(self.KBeta)])
                        Yn, Tn = self.__get_obs_data(n)
                        Tn_ls.append(Tn)
                        tau2_sum_n = self.__piStar_tau[n, k] * ((Yn[:,0]-E_a_n)**2 + sum([(Yn[:,t]-E_a_n- E_Beta_n@Yn[:,t-1]+E_Beta_n@E_a_n)**2 for t in range(1,Tn)]))
                        tau2_sum += tau2_sum_n
                    self.__lamStar_tau[k,:,0] = np.zeros(self.D) + self.__lambda_tau1 + 1/2 * np.sum(np.array(Tn_ls)* self.__piStar_tau[:, k])
                    self.__lamStar_tau[k,:,1] = self.__lambda_tau2 + 1/2 * tau2_sum
                    #--- get means for tau ---#
                    self.__est_tau[k,:] = self.__lamStar_tau[k,:, 0]/self.__lamStar_tau[k,:, 1]
            # get pi for the last cluster
            self.__est_pi_a[self.Ka-1] = 1 - self.__est_pi_a[:(self.Ka-1)].sum()
            self.__est_pi_Beta[self.KBeta-1] = 1 - self.__est_pi_Beta[:(self.KBeta-1)].sum()
            self.__est_pi_tau[self.Ktau-1] = 1 - self.__est_pi_tau[:(self.Ktau-1)].sum()

            ## Cluster reordering ##
            order_idx_a = np.flip(np.argsort(self.__est_pi_a[:self.Ka])) # sort by est. pi in descending order
            order_idx_a = np.concatenate((order_idx_a, np.arange(self.Ka, len(self.__est_pi_a))))
            order_idx_Beta = np.flip(np.argsort(self.__est_pi_Beta[:self.KBeta]))
            order_idx_Beta = np.concatenate((order_idx_Beta, np.arange(self.KBeta, len(self.__est_pi_Beta))))
            order_idx_tau = np.flip(np.argsort(self.__est_pi_tau[:self.Ktau]))
            order_idx_tau = np.concatenate((order_idx_tau, np.arange(self.Ktau, len(self.__est_pi_tau))))
            self.__est_pi_a = self.__est_pi_a[order_idx_a]
            self.__est_pi_Beta = self.__est_pi_Beta[order_idx_Beta]
            self.__est_pi_tau = self.__est_pi_tau[order_idx_tau]
            self.__est_a = self.__est_a[order_idx_a]
            self.__est_Beta = self.__est_Beta[order_idx_Beta]
            self.__est_tau = self.__est_tau[order_idx_tau]
            self.__piStar_a = self.__piStar_a[:, order_idx_a]
            self.__piStar_Beta = self.__piStar_Beta[:, order_idx_Beta]
            self.__piStar_tau = self.__piStar_tau[:, order_idx_tau]
            self.__LamStar_a = self.__LamStar_a[order_idx_a]
            self.__LamStar_Beta = self.__LamStar_Beta[order_idx_Beta]
            self.__RowprecStar_Beta = self.__RowprecStar_Beta[order_idx_Beta]
            self.__PrecStar_Beta = self.__PrecStar_Beta[order_idx_Beta]
            self.__lamStar_tau = self.__lamStar_tau[order_idx_tau]
            for k in range(self.K-1):
                if k < self.Ka-1:
                    # recalculate the variational parameters for v
                    self.__alphaStar_a[k, 0] = self.__alphfix + np.sum(self.__piStar_a[:, k])
                    self.__alphaStar_a[k, 1] = self.__est_alpha_a + np.sum(self.__piStar_a[:, (k+1):self.Ka])
                    #--- get means for log V and 1-V ---#
                    est_logV_a[k] = psi(self.__alphaStar_a[k, 0]) - psi(np.sum(self.__alphaStar_a[k,:]))
                    est_log1V_a[k] = psi(self.__alphaStar_a[k, 1]) - psi(np.sum(self.__alphaStar_a[k,:]))
                if k < self.KBeta-1:
                    self.__alphaStar_Beta[k, 0] = self.__alphfix + np.sum(self.__piStar_Beta[:, k])
                    self.__alphaStar_Beta[k, 1] = self.__est_alpha_Beta + np.sum(self.__piStar_Beta[:, (k+1):self.KBeta])
                    #--- get means for log V and 1-V ---#
                    est_logV_Beta[k] = psi(self.__alphaStar_Beta[k, 0]) - psi(np.sum(self.__alphaStar_Beta[k,:]))
                    est_log1V_Beta[k] = psi(self.__alphaStar_Beta[k, 1]) - psi(np.sum(self.__alphaStar_Beta[k,:]))
                if k < self.Ktau-1:
                    self.__alphaStar_tau[k, 0] = self.__alphfix + np.sum(self.__piStar_tau[:, k])
                    self.__alphaStar_tau[k, 1] = self.__est_alpha_tau + np.sum(self.__piStar_tau[:, (k+1):self.Ktau])
                    #--- get means for log V and 1-V ---#
                    est_logV_tau[k] = psi(self.__alphaStar_tau[k, 0]) - psi(np.sum(self.__alphaStar_tau[k,:]))
                    est_log1V_tau[k] = psi(self.__alphaStar_tau[k, 1]) - psi(np.sum(self.__alphaStar_tau[k,:]))

            #- update variational parameters for z_n: piStar (can be paralleled for each n to speed up) -#
            for n in (tqdm(range(self.N), desc='[INFO] Update variational parameters for Zn', leave=False) if verbose else range(self.N)): 
                E_a_n = sum([self.__piStar_a[n,j]* self.__est_a[j,:] for j in range(self.Ka)]) # can be rewritten as a matrix-vector multiplication?
                E_Beta_n = sum([self.__piStar_Beta[n,j]* self.__est_Beta[j,:,:] for j in range(self.KBeta)])
                E_tau_n = sum([self.__piStar_tau[n, j]* self.__est_tau[j,:] for j in range(self.Ktau)])
                Yn, Tn = self.__get_obs_data(n)
                for k in range(self.K):
                    if k < self.Ka:
                        # update variational parameters for a: PiStar_a
                        mat_a = (Yn[:,1:].T-self.__est_a[k,:]).T - E_Beta_n@(Yn[:,:-1].T-self.__est_a[k,:]).T
                        Sa_kn = -1/2 * ( (Yn[:,0]-self.__est_a[k,:]).T @ np.diag(E_tau_n) @ (Yn[:,0]-self.__est_a[k,:]) + np.trace(np.diag(E_tau_n)@mat_a@mat_a.T) ) 
                        self.__piStar_a[n, k] = est_logV_a[k:(k+1)].sum() + est_log1V_a[:k].sum() + Sa_kn # enable indexing out of bounds -> 0
                    if k < self.KBeta:
                        # update variational parameters for beta: PiStar_Beta
                        mat_Beta = (Yn[:,1:].T-E_a_n).T - self.__est_Beta[k,:,:]@(Yn[:,:-1].T-E_a_n).T
                        Sb_kn = -1/2 * ( (Yn[:,0]-E_a_n).T @ np.diag(E_tau_n) @ (Yn[:,0]-E_a_n) + np.trace(np.diag(E_tau_n)@mat_Beta@mat_Beta.T) ) 
                        self.__piStar_Beta[n, k] = est_logV_Beta[k:(k+1)].sum() + est_log1V_Beta[:k].sum() + Sb_kn
                    if k < self.Ktau:
                        # update variational parameters for tau: PiStar_tau
                        mat_tau = (Yn[:,1:].T-E_a_n).T - E_Beta_n@(Yn[:,:-1].T-E_a_n).T
                        Stau_kn = Tn/2* np.log(self.__est_tau[k,:]).sum() -1/2 * ( (Yn[:,0]-E_a_n).T @ np.diag(self.__est_tau[k,:]) @ (Yn[:,0]-E_a_n) + np.trace(np.diag(self.__est_tau[k,:])@mat_tau@mat_tau.T) ) 
                        self.__piStar_tau[n, k] = est_logV_tau[k:(k+1)].sum() + est_log1V_tau[:k].sum() + Stau_kn

            #- update variational parameters for alpha: sStar -#
            if verbose: print('[INFO] Update variational parameters for alpha...')
            self.__sStar_a[0] = self.__s1 + self.Ka -1
            self.__sStar_a[1] = self.__s2 - est_log1V_a.sum()
            self.__sStar_Beta[0] = self.__s1 + self.KBeta -1
            self.__sStar_Beta[1] = self.__s2 - est_log1V_Beta.sum()
            self.__sStar_tau[0] = self.__s1 + self.Ktau -1
            self.__sStar_tau[1] = self.__s2 - est_log1V_tau.sum()
            #--- get means for alpha ---#
            self.__est_alpha_a = self.__sStar_a[0]/self.__sStar_a[1]
            self.__est_alpha_Beta = self.__sStar_Beta[0]/self.__sStar_Beta[1]
            self.__est_alpha_tau = self.__sStar_tau[0]/self.__sStar_tau[1]
            if verbose: print('Updated alpha for a, Beta, tau:', np.round([self.__est_alpha_a, self.__est_alpha_Beta, self.__est_alpha_tau],2))

            ## Cluster pruning - remove the cluster with pi_k < prune_thres ##
            pruned_idx_a = np.where(self.__est_pi_a[:self.Ka] < prune_threshold_a)[0] 
            pruned_idx_Beta = np.where(self.__est_pi_Beta[:self.KBeta] < prune_threshold_Beta)[0]
            pruned_idx_tau = np.where(self.__est_pi_tau[:self.Ktau] < prune_threshold_tau)[0]
            self.K -= np.min([len(pruned_idx_a), len(pruned_idx_Beta), len(pruned_idx_tau)]) # also consider noise cluster
            #self.K -= np.min([len(pruned_idx_a), len(pruned_idx_Beta)]) # new K
            self.Ka -= len(pruned_idx_a); self.KBeta -= len(pruned_idx_Beta); self.Ktau -= len(pruned_idx_tau)
            #? may also consider empirical number of clusters as new Ka, KBeta, Ktau
            self.__piStar_a[:,:self.Ka] = softmax(self.__piStar_a[:,:self.Ka], axis=1) # renormalize to sum to 1 (this function is stable)
            self.__piStar_Beta[:,:self.KBeta] = softmax(self.__piStar_Beta[:,:self.KBeta], axis=1) # (no floating point error)
            self.__piStar_tau[:,:self.Ktau] = softmax(self.__piStar_tau[:,:self.Ktau], axis=1)
            self.__est_Za = np.argmax(self.__piStar_a[:,:self.Ka], axis=1)
            self.__est_ZBeta = np.argmax(self.__piStar_Beta[:,:self.KBeta], axis=1)
            self.__est_Ztau = np.argmax(self.__piStar_tau[:,:self.Ktau], axis=1)

            #- Measure convergence using the relative change (& abs change) in the log marginal probability bound,
            # stopping the algorithm when it was less than ELBO_rel_tol (ELBO_abs_tol).
            # Alternatively can stop the iterative scheme when the change of the L2-norm of the vector of variational parameters is smaller than some e = 1e−5.
            if verbose: print('[INFO] Compute ELBO...')
            ELBO = self.__compute_ELBO(est_logV_a, est_log1V_a, est_logV_Beta, est_log1V_Beta, est_logV_tau, est_log1V_tau)
            if verbose: print('-- ELBO:', round(ELBO, 4), '[√]' if ELBO > self.ELBO_iters[-1] or self.ELBO_iters[-1]==1 else '[X]')
            self.ELBO_iters.append(ELBO) # store ELBO
            if verbose: print('[INFO] Check convergence...')
            if self._has_converged(ELBO_rel_tol, ELBO_abs_tol):
                if verbose: 
                    print('Est. Pi.a:', np.round(self.__est_pi_a, 4))
                    print('Est. Pi.Beta:', np.round(self.__est_pi_Beta, 4))
                    print('Est. Pi.tau:', np.round(self.__est_pi_tau, 4))
                print('[INFO] Converged! Optimal ELBO is reached:', round(ELBO, 4))
                break
            elif ELBO is np.nan:
                print('[INFO] ELBO is NaN! Break the loop.')
                break
            elif it == maxIter-1:
                if verbose: 
                    print('Est. Pi.a:', np.round(self.__est_pi_a, 4))
                    print('Est. Pi.Beta:', np.round(self.__est_pi_Beta, 4))
                    print('Est. Pi.tau:', np.round(self.__est_pi_tau, 4))
                print('[INFO] Maximum number of iterations reached! Current ELBO:', round(ELBO, 4))
        end_time = time.perf_counter()
        total_time = end_time - start_time
        hours, remainder = divmod(total_time, 3600)
        minutes, seconds = divmod(remainder, 60)
        self.exc_time = f'{int(hours)}h:{int(minutes):02d}m:{seconds:.2f}s'

    def getEstimates(self, verbose: bool=False):
        """Get the estimates of the model parameters

        Returns:
            estimates (dict): dictionary of estimated values for a, Beta, sigma (with sd), pi and Z.
        """
        if len(self.ELBO_iters) == 1:
            raise ValueError('Please fit the model first!')
        # get unique clusters
        est_Za_unique = np.unique(self.__est_Za)
        est_ZBeta_unique = np.unique(self.__est_ZBeta)
        est_Ztau_unique = np.unique(self.__est_Ztau)
        if verbose:
            print('Unique clusters for a vec ({}): {}'.format(len(est_Za_unique), est_Za_unique))
            print('Unique clusters for Beta mat ({}): {}'.format(len(est_ZBeta_unique), est_ZBeta_unique))
            print('Unique clusters for sigma vec ({}): {}'.format(len(est_Ztau_unique), est_Ztau_unique))
        # get the estimates for a, Beta, tau
        est_a_ = self.__est_a[est_Za_unique] # Ka x D
        est_Beta_ = self.__est_Beta[est_ZBeta_unique,:,:] # KBeta x D x D
        est_tau_ = self.__est_tau[est_Ztau_unique]
        est_sigma = np.sqrt(1/est_tau_) # Ktau x D
        # get the sd for estimation means
        est_a_sd = np.array([np.max(np.sqrt(1/np.diag(self.__LamStar_a[i,:,:]))) for i in est_Za_unique]) # array of Ka scalars
        est_Beta_sd = np.array([np.sqrt(1/np.diag(self.__RowprecStar_Beta[i,:,:])) for i in est_ZBeta_unique]) # array of KBeta vectors for row sd of Beta
        est_Beta_all_sd = np.array([np.sqrt(1/self.__PrecStar_Beta[i,:,:]) for i in est_ZBeta_unique]) # array of KBeta x D vectors for all sd of Beta
        est_tau_sd = np.sqrt(self.__lamStar_tau[:,:,0]/self.__lamStar_tau[:,:,1]**2)[est_Ztau_unique] # Ktau x D
        est_sigma_min = np.sqrt(1/(est_tau_+est_tau_sd))
        est_sigma_max = np.sqrt(1/(est_tau_-est_tau_sd))
        est_sigma_sd = list(map(max,zip((est_sigma - est_sigma_min).flatten(), (est_sigma_max - est_sigma).flatten())))
        est_sigma_sd = np.array(est_sigma_sd).reshape(est_sigma.shape)
        # get the estimates for pi
        est_pi_a_ = self.__est_pi_a[est_Za_unique]
        est_pi_Beta_ = self.__est_pi_Beta[est_ZBeta_unique]
        est_pi_tau_ = self.__est_pi_tau[est_Ztau_unique]
        # map the cluster index to the index from 0 to K
        est_Za_map = {old_idx: i for i, old_idx in enumerate(est_Za_unique)}
        est_ZBeta_map = {old_idx: i for i, old_idx in enumerate(est_ZBeta_unique)}
        est_Ztau_map = {old_idx: i for i, old_idx in enumerate(est_Ztau_unique)}
        est_Za = np.array([est_Za_map[i] for i in self.__est_Za])
        est_ZBeta = np.array([est_ZBeta_map[i] for i in self.__est_ZBeta])
        est_Ztau = np.array([est_Ztau_map[i] for i in self.__est_Ztau])
        return {"intercept":est_a_, "intercept_sd": est_a_sd, "B_coef":est_Beta_, "B_coef_sd": est_Beta_sd, "B_coef_all_sd": est_Beta_all_sd, "sigma":est_sigma, "sigma_sd":est_sigma_sd,\
            "a_clust":est_Za, "Beta_clust":est_ZBeta, "sigma_clust":est_Ztau, "pi_a":est_pi_a_, "pi_Beta":est_pi_Beta_, "pi_sigma":est_pi_tau_}

    def getPiMat(self):
        """Get the original pi est. and matrix for a, Beta, tau

        Returns:
            piMat (dict): dictionary of pi matrices and estimated pi for a, Beta, tau.
        """
        if len(self.ELBO_iters) == 1:
            raise ValueError('Please fit the model first!')
        piMat = {"PiMat_a": self.__piStar_a[:,:self.Ka], "PiMat_Beta": self.__piStar_Beta[:,:self.KBeta], "PiMat_tau": self.__piStar_tau[:,:self.Ktau], "pi_a": self.__est_pi_a[:self.Ka], "pi_Beta": self.__est_pi_Beta[:self.KBeta], "pi_tau": self.__est_pi_tau[:self.Ktau]}
        return piMat

    def checkEstError(self, estDict: dict=None, Gtruth: dict=None, plot: bool=False):
        """Check the relative L2 error of the estimates (with ground truth)

        Parameters:
            estDict (dict, optional): specified dictionary of estimated values for a, beta, sigma. Defaults to None, then use the estimates from the model.
            Gtruth (dict, optional): Ground truth values for a, Beta, sigma. Defaults to None.
            plot (bool, optional): To plot the heatmaps or not. Defaults to False.

        Returns:
            a_l2mat, Beta_l2mat, sigma_l2mat: numpy matrices of relative L2 errors for a, Beta, sigma.
        """
        if len(self.ELBO_iters) == 1:
            raise ValueError('Please fit the model first!')
        if estDict:
            est_dict = estDict
        else:
            est_dict = self.getEstimates()
        est_a_ = est_dict['intercept'] # Ka x D
        est_Beta_ = est_dict['B_coef'] # Kbeta x D x D
        est_sigma = est_dict['sigma'] # Ksigma x D
        if Gtruth:
            # check l2 error of estimates with ground truth
            true_a = Gtruth['a']
            true_Beta = Gtruth['Beta']
            true_sigma = Gtruth['sigma']
            a_l2mat = l2_mat(true_a, est_a_)
            Beta_l2mat = l2_mat(true_Beta, est_Beta_)
            sigma_l2mat = l2_mat(true_sigma, est_sigma)
        else:
            # check l2 error of estimates with itself 
            a_l2mat = l2_mat(est_a_, est_a_)
            a_l2mat[range(a_l2mat.shape[0]),range(a_l2mat.shape[0])] += 10
            Beta_l2mat = l2_mat(est_Beta_, est_Beta_)
            Beta_l2mat[range(Beta_l2mat.shape[0]),range(Beta_l2mat.shape[0])] += 10
            sigma_l2mat = l2_mat(est_sigma, est_sigma)
            sigma_l2mat[range(sigma_l2mat.shape[0]),range(sigma_l2mat.shape[0])] += 10
        if plot:
            import seaborn as sns
            ax = sns.heatmap(a_l2mat, annot=True, fmt=".2f", cmap="Reds", vmin=0, cbar=False)
            # Highlight the minimum value in each row
            for i, row in enumerate(a_l2mat):
                min_col_idx = np.argmin(row)
                ax.add_patch(plt.Rectangle((min_col_idx, i), 1, 1, fill=False, edgecolor='red', lw=2))
            ax.set(xlabel="Est", ylabel="Gtruth" if Gtruth else "Est")
            plt.title('Relative L2 error of a vec')
            plt.show()
            ax = sns.heatmap(Beta_l2mat, annot=True, fmt=".2f", cmap="Reds", vmin=0, cbar=False)
            for i, row in enumerate(Beta_l2mat):
                min_col_idx = np.argmin(row)
                ax.add_patch(plt.Rectangle((min_col_idx, i), 1, 1, fill=False, edgecolor='red', lw=2))
            ax.set(xlabel="Est", ylabel="Gtruth" if Gtruth else "Est")
            plt.title('Relative L2 error of Beta mat')
            plt.show()
            ax = sns.heatmap(sigma_l2mat, annot=True, fmt=".2f", cmap="Reds", vmin=0, cbar=False)
            for i, row in enumerate(sigma_l2mat):
                min_col_idx = np.argmin(row)
                ax.add_patch(plt.Rectangle((min_col_idx, i), 1, 1, fill=False, edgecolor='red', lw=2))
            ax.set(xlabel="Est", ylabel="Gtruth" if Gtruth else "Est")
            plt.title('Relative L2 error of sigma vec')
            plt.show()
        return a_l2mat, Beta_l2mat, sigma_l2mat

    def printRes(self, estDict: dict=None, Gtruth: dict=None):
        """Print the results of the model

        Parameters:
            estDict (dict, optional): specified dictionary of estimated values for a, Beta, sigma. Defaults to None, then use the estimates from the model.
            Gtruth (dict, optional): Ground truth values for a, Beta, sigma. Defaults to None.
        """
        if len(self.ELBO_iters) == 1:
            raise ValueError('Please fit the model first!')
        from prettytable import PrettyTable, ALL
        import sys
        if sys.__stdin__.isatty():
            import plotext as plt
        else:
            import matplotlib.pyplot as plt
        # plot ELBO
        plt.plot(self.ELBO_iters[1:])
        step = np.ceil(len(self.ELBO_iters[1:])/15).astype('int32')
        xtcks = np.arange(len(self.ELBO_iters[1:]),step=step)[1:]
        if len(xtcks) == 0: xtcks = np.array([1])
        if len(self.ELBO_iters[1:])-xtcks[-1]<step:
            xtcks = xtcks[:-1]
        xtcks = np.append(xtcks,len(self.ELBO_iters[1:]))
        plt.xticks(ticks=xtcks-1, labels=xtcks)
        plt.xlabel('Iteration')
        plt.ylabel('ELBO')
        plt.show()
        if estDict:
            est_dict = estDict
        else:
            est_dict = self.getEstimates(verbose=True)
        # get the estimates for a, Beta, tau
        est_a_ = est_dict['intercept'] # Ka x D
        est_Beta_ = est_dict['B_coef'] # KBeta x D x D
        est_sigma = est_dict['sigma'] # Ktau x D
        # get the sd for estimation means
        est_a_sd = est_dict['intercept_sd'] # Ka scalars
        est_Beta_sd = est_dict['B_coef_sd'] # KBeta vectors (dim D) for row sd of Beta
        est_sigma_sd = np.max(est_dict['sigma_sd'], axis=1) # Ktau scalars
        # get the estimates for pi
        est_pi_a_ = est_dict['pi_a']
        est_pi_Beta_ = est_dict['pi_Beta']
        est_pi_tau_ = est_dict['pi_sigma']
        # get the estimates for Z
        est_Za = est_dict['a_clust']
        est_ZBeta = est_dict['Beta_clust']
        est_Ztau = est_dict['sigma_clust']
        print('#=============== Estimations ===============#')
        print('Estimated a vec:')
        table_a = PrettyTable(header=False, hrules=ALL)
        table_a.add_column(None,['K x D matrix']+ list(map(' ±'.join, zip(map(str, np.round(est_a_,2).tolist()), map(str, np.round(est_a_sd,3).tolist())))))
        table_a.add_column(None,['π_a']+np.round(est_pi_a_,3).tolist())
        print(table_a)
        print('Estimated Beta mat:')
        for i in range(est_Beta_.shape[0]):
            print('-K{} π={}-'.format(i+1, np.round(est_pi_Beta_[i],3)))
            table_beta = PrettyTable(header=False, hrules=ALL)
            table_beta.add_column(None,['D x D matrix']+ list(map(' ±'.join, zip(map(str, np.round(est_Beta_[i],2).tolist()), map(str, np.round(est_Beta_sd[i],3).tolist())))))
            print(table_beta)
        print('Estimated sigma vec:')
        table_tau = PrettyTable(header=False, hrules=ALL)
        table_tau.add_column(None,['K x D matrix']+ list(map(' ±'.join, zip(map(str, np.round(est_sigma,2).tolist()), map(str, np.round(est_sigma_sd,3).tolist())))))
        table_tau.add_column(None,['π_sigma']+np.round(est_pi_tau_,3).tolist())
        print(table_tau)
        if Gtruth:
            from sklearn.metrics import adjusted_rand_score
            print('#=============== Ground truth ===============#')
            # ground truth
            true_a = Gtruth.get('a', [None])
            true_pi_a = Gtruth.get('pi_a', [None])
            true_Beta = Gtruth.get('Beta', [None])
            true_pi_Beta = Gtruth.get('pi_Beta', [None])
            true_sigma = Gtruth.get('sigma', [None])
            true_pi_sigma = Gtruth.get('pi_sigma', [None])
            print('True a vec:')
            table_a = PrettyTable(header=False, hrules=ALL)
            table_a.add_column(None,['K x D matrix']+list(true_a))
            table_a.add_column(None,['π_a']+list(true_pi_a))
            print(table_a)
            print('True Beta mat:')
            for i in range(np.array(true_Beta).shape[0]):
                print('-K{} π={}-'.format(i+1, np.round(true_pi_Beta[i],3)))
                table_beta = PrettyTable(header=False, hrules=ALL)
                table_beta.add_column(None,['D x D matrix']+ list(true_Beta[i]))
                print(table_beta)
            print('True sigma vec:')
            table_tau = PrettyTable(header=False, hrules=ALL)
            table_tau.add_column(None,['K x D matrix']+list(true_sigma))
            table_tau.add_column(None,['π_sigma']+list(true_pi_sigma))
            print(table_tau)
            print('#============ Relative L2 error ============#')
            a_l2mat, Beta_l2mat, sigma_l2mat = self.checkEstError(Gtruth=Gtruth)
            print('L2_err_a:', np.round(np.min(a_l2mat, axis=1), 3), 'Mean:', np.round(np.min(a_l2mat, axis=1).mean(), 3))
            print('L2_err_Beta:', np.round(np.min(Beta_l2mat, axis=1), 3), 'Mean:', np.round(np.min(Beta_l2mat, axis=1).mean(), 3))
            print('L2_err_sigma:', np.round(np.min(sigma_l2mat, axis=1), 3), 'Mean:', np.round(np.min(sigma_l2mat, axis=1).mean(), 3))
            print('#=================== ARI ===================#')
            # ARI
            print('ARI_a:', round(adjusted_rand_score(Gtruth['true_a_clust'], est_Za),3))
            print('ARI_Beta:', round(adjusted_rand_score(Gtruth['true_Beta_clust'], est_ZBeta),3))
            print('ARI_sigma/tau:', round(adjusted_rand_score(Gtruth['true_sigma_clust'], est_Ztau),3))

    def plotFacet(self, estDict: dict=None, facet=['a', 'Beta', 'sigma'], ashift: float=0, save_folder: str=None, figDict: dict=None, varnames: list=None):
        """Plot the facets of the model

        Parameters:
            estDict (dict, optional): specified dictionary of estimated values for a, Beta, sigma. Defaults to None, then use the estimates from the model.
            facet (str/list, optional): string names of facets. Defaults to ['a', 'Beta', 'sigma'].
            ashift (float, optional): shift of intercepts. Defaults to 0.
            save_folder (str, optional): folder to save the plot. Defaults to None, then only show the plot.
            figDict (dict, optional): dictionary of figure parameters. Defaults to {"figsize":(6,4),"dpi":200,"fontsize":14}.
            varnames (list, optional): list of variable names for different dimensions.
        """
        if len(self.ELBO_iters) == 1:
            raise ValueError('Please fit the model first!')
        from matplotlib import patches as mpatches
        from matplotlib.colors import ListedColormap
        import networkx as nx
        facets = ['a', 'Beta', 'sigma']
        if type(facet) != list:
            facet = [facet]
        facet_cp = facet.copy()
        if estDict:
            est_dict = estDict
        else:
            est_dict = self.getEstimates()
        if figDict:
            figDict_update = figDict.copy()
            figDict = {"figsize":(6,4),"dpi":200,"fontsize":14}
            figDict.update(figDict_update)
        else:
            figDict = {"figsize":(6,4),"dpi":200,"fontsize":14}
        fgsize = figDict["figsize"]; dpi = figDict["dpi"]; fontsize = figDict["fontsize"]
        Pis = []
        if varnames is None:
            varnames = [f"V{i}" for i in range(1, self.D+1)]
        if facets[0] in facet:
            facet_cp.remove(facets[0])
            est_a = est_dict['intercept']
            pi_a = est_dict["pi_a"]
            # plot intercepts
            plt.figure(figsize=fgsize, dpi=dpi)
            for i in range(est_a.shape[0]):
                sc = plt.scatter([i+1] *est_a.shape[1], est_a[i] +ashift, alpha=min(pi_a[i]*100,1), c=range(est_a.shape[1]), cmap=ListedColormap(plt.cm.tab10.colors[:est_a.shape[1]]), s=40)
            plt.xticks(ticks=np.arange(1,est_a.shape[0]+1), labels=np.arange(1,est_a.shape[0]+1), fontsize=fontsize)
            plt.yticks(fontsize=fontsize)
            plt.xlabel("Cluster", fontsize=fontsize+2)
            plt.ylabel("Intercept", fontsize=fontsize+2)
            handles = [plt.plot([],color=sc.get_cmap()(sc.norm(c)),ls="", marker="o")[0] for c in range(est_a.shape[1])]
            plt.legend(handles, varnames, loc='upper right')
            if save_folder:
                plt.savefig(f"{save_folder}pointEstA.png", bbox_inches="tight")
                plt.close()
            else:
                plt.show()
            # Pi_a
            Pis.append(pi_a)
        if facets[1] in facet:
            def coeff_matrix_plot(coeff_matrix, var_name, fontsize, pos=None):
                edges = np.where(coeff_matrix != 0)
                edge_weights = coeff_matrix[edges]
                edge_list = [(var_name[edges[1][i]], var_name[edges[0][i]], edge_weights[i]) for i in range(len(edge_weights))]
                g = nx.DiGraph()
                g.add_weighted_edges_from(edge_list)
                edge_colors = ["green" if w > 0 else "red" for _, _, w in g.edges(data="weight")]
                edge_widths = [abs(w) * 10 for _, _, w in g.edges(data="weight")]
                if pos is None:
                    pos = nx.circular_layout(g)
                nx.draw(g,pos, connectionstyle='arc3, rad = 0.1', with_labels=True,node_color="lightblue",node_size=500*est_Beta.shape[1]**2,edge_color=edge_colors,width=edge_widths,font_size=fontsize,font_color="black")
                return pos
            facet_cp.remove(facets[1])
            est_Beta = est_dict['B_coef']
            pi_Beta = est_dict["pi_Beta"]
            # plot coefficients matrix
            nrow = int(np.ceil(est_Beta.shape[0]/5))
            ncol = 5 if est_Beta.shape[0] > 5 else est_Beta.shape[0]
            fig, axes = plt.subplots(nrow, ncol, figsize=(ncol*fgsize[0], nrow*fgsize[1]*2.5), dpi=dpi)
            axes = np.array(axes).flatten()
            for i in range(len(axes)):
                ax = axes[i]
                if i < est_Beta.shape[0]:
                    plt.sca(ax)
                    if i == 0:
                        pos = coeff_matrix_plot(est_Beta[i], varnames, fontsize*4) 
                    else:
                        pos = coeff_matrix_plot(est_Beta[i], varnames, fontsize*4, pos)
                    ax.set_title(f"Cluster {i + 1}", fontsize=fontsize*4, pad=10*fgsize[1])
                else:
                    ax.axis("off")
            handles = [mpatches.Patch(color=colour, label=label) for colour,label in [('green', 'positive'), ('red', 'negative')]]
            fig.legend(handles=handles, loc='lower right', bbox_to_anchor=(0.92, 0.1), fontsize=fontsize*max(ncol/2,1.5)) #2.5
            if save_folder:
                plt.savefig(f"{save_folder}pointEstBeta.png", bbox_inches="tight")
                plt.close()
            else:
                plt.show()
            # Pi_Beta
            Pis.append(pi_Beta)
        if facets[2] in facet:
            facet_cp.remove(facets[2])
            est_sigma = est_dict['sigma']
            pi_sigma = est_dict["pi_sigma"]
            colors = plt.cm.tab10.colors[:est_sigma.shape[0]]
            #sigma_vec = np.vstack([np.max(est_sigma, axis=0),np.zeros(est_sigma.shape[1]),est_sigma])
            radar_chart(est_sigma, varnames, pi_sigma, fgsize,dpi,fontsize, colors, save_folder)
            # Pi_sigma
            Pis.append(pi_sigma)
        if len(Pis) != 0:
            stacked_Pis_barplot(Pis, fgsize, dpi, fontsize)
            if save_folder:
                plt.savefig(f"{save_folder}Pis.png", bbox_inches="tight")
                plt.close()
            else:
                plt.show()
        if len(facet_cp) != 0:
            print('Facet {} not found!'.format(facet_cp))
        if save_folder:
            print(f"[INFO] Plots saved in '{save_folder}'")

    def showClust(self, estDict: dict=None, save_folder: str=None, figDict: dict=None):
        """Show the cluster assignments of the model

        Parameters:
            estDict (dict, optional): specified dictionary of estimated values for a, beta, sigma. Defaults to None, then use the estimates from the model.
            save_folder (str, optional): folder to save the plot. Defaults to None, then only show the plot.
            figDict (dict, optional): dictionary of figure parameters. Defaults to {"figsize":(6,4),"dpi":200,"fontsize":14}.
        """
        if len(self.ELBO_iters) == 1:
            raise ValueError('Please fit the model first!')
        import sys
        if estDict:
            est_dict = estDict
        else:
            est_dict = self.getEstimates()
        if figDict:
            figDict_update = figDict.copy()
            figDict = {"figsize":(6,4),"dpi":200,"fontsize":14}
            figDict.update(figDict_update)
        else:
            figDict = {"figsize":(6,4),"dpi":200,"fontsize":14}
        fgsize = figDict["figsize"]; dpi = figDict["dpi"]; fontsize = figDict["fontsize"]
        est_Za = est_dict['a_clust']
        est_ZBeta = est_dict['Beta_clust']
        # Initialize the co-occurrence matrix
        probability_matrix = np.zeros((np.max(est_Za)+1,np.max(est_ZBeta)+1))
        # Count co-occurrences of cluster pairs
        for c1, c2 in zip(est_Za, est_ZBeta):
            probability_matrix[c1, c2] += 1
        probability_matrix = probability_matrix / probability_matrix.sum()  # Normalize to probabilities
        probability_matrix = probability_matrix * 100 # Scale to percentage
        import seaborn as sns
        plt.figure(figsize=fgsize, dpi=dpi)
        sns.heatmap(probability_matrix, annot=True, fmt=".1f",cmap="Blues",cbar_kws={'format': '%.0f%%'},vmin=0,xticklabels=[f"{i+1}" for i in range(np.max(est_ZBeta)+1)],yticklabels=[f"{i+1}" for i in range(np.max(est_Za)+1)])
        plt.xlabel("Coefficient Cluster", fontsize=fontsize)
        plt.ylabel("Intercept Cluster", fontsize=fontsize)
        if save_folder:
            plt.savefig(f"{save_folder}ClusterAssign.png", bbox_inches="tight")
            plt.close()
        else:
            plt.show()

