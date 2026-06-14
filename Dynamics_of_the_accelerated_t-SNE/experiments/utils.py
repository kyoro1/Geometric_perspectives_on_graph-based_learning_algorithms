import os
import numpy as np
from numpy.random import *
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.utils import check_random_state
from sklearn.decomposition import PCA
from sklearn.manifold import SpectralEmbedding, MDS
from scipy.special import jv, iv
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
from collections import Counter

import matplotlib.pyplot as plt

class experiment():
    def __init__(self,
                labels: list,
                sample_data: str,
                sample_number: int):
        ## Define the original data
        self.X, self.y = None, None
        ## random choiced data
        self.X_sampled, self.y_sampled = None, None
        self.N = 0
        self.sample_data = sample_data
        ## filename for kddcup
        self.filename = 'data/kddcup.data_10_percent.gz'
        self.selected_labels = [18, 9, 11, 0, 17]
        self.df_preprocessed = pd.DataFrame()
        self.df_sample = pd.DataFrame()
        self.labels = labels
        ## sample size for each label
        self.sample_number = sample_number

        ## number of unique labels
        self.num_unique_labels = None

    def load_data(self):
        if self.sample_data == 'mnist':
            X, y = fetch_openml("mnist_784", version=1, return_X_y=True, as_frame=False)

            random_state = check_random_state(0)
            permutation = random_state.permutation(X.shape[0])
            X = X[permutation]
            X = X.reshape((X.shape[0], -1))
            y = y[permutation]
            y = y.astype(str)

            # Check the distribution of labels
            label_counts = Counter(y)
            print("Label distribution in the dataset:", label_counts)

            # Filter the dataset for digits
            filtered_indices = np.isin(y, self.labels)
            print(f"Number of samples for selected digits: {np.sum(filtered_indices)}")

            X_filtered = X[filtered_indices]
            y_filtered = y[filtered_indices]

            # Check the size of the filtered dataset
            print(f"Number of samples for selected digits after filtering: {X_filtered.shape[0]}")

            # Initialize lists to hold the sampled data
            sampled_X, sampled_y = [], []

            # Sample 10 instances for each digit
            for digit in self.labels:
                digit_indices = np.where(y_filtered == digit)[0]
                print(f"Number of samples for digit {digit}: {len(digit_indices)}")
                if len(digit_indices) >= 10:
                    digit_sample_indices = random_state.choice(digit_indices, self.sample_number, replace=True)
                    sampled_X.extend(X_filtered[digit_sample_indices])
                    sampled_y.extend(y_filtered[digit_sample_indices])
                else:
                    print(f"Digit {digit} has only {len(digit_indices)} samples, skipping...")

            # Convert lists to numpy arrays
            self.X_sampled = np.array(sampled_X)
            self.y_sampled = np.array(sampled_y)

            # Scale the data to avoid numerical issues
            scaler = StandardScaler()
            self.X_sampled = scaler.fit_transform(self.X_sampled)

        ## Get the number of data points
        self.N = self.X_sampled.shape[0]
        print('Filtered and randomly choiced data: {}, {}'.format(self.X_sampled.shape, self.y_sampled.shape))
        ## Get unique labels
        unique_labels = np.unique(self.y_sampled)
        self.num_unique_labels = unique_labels.shape[0]
        print('Got unique labels: {}'.format(self.num_unique_labels))

    def encode_features(self, df):
        '''
        Encode categorical features
        '''
        for column in df.columns:
            if df[column].dtype == type(object):  ## in case of categorical data
                le = LabelEncoder()
                df[column] = le.fit_transform(df[column])
        return df


class tSNE_experiment(experiment):
    def __init__(self, 
                labels: list, 
                sample_data: str,
                perplexity: float,
                alpha: float,
                h: int,
                total_iteration:int,
                momentum: float,
                sample_number: int,
                R: float):
        super().__init__(labels, sample_data, sample_number)
        ## inheritance of class experiment
        self.labels = labels
        ## bandwidth of Gaussian kernel
        self.distances = None
        self.tau = None
        self.perplexity = perplexity
        self.alpha = alpha
        self.h = h
        self.ee_iteration = total_iteration
        ## number of connected components
        self.R = R
        ## Define the location of labels
        self.P = None
        self.Y = dict()  ## Embedded coordinates in 2-dim.
        self.Q = dict()
        self.S = dict()
        self.cost = dict()
        self.beta = None
        ## For Nestrov method
        self.Y_nes = dict()
        ## momentum 
        self.momentum = momentum
        ## Supplemental matrix
        self.P_star = None
        self.squared_distances = dict()
        self.student_kernel = dict()

    ## Calculate tau_i
    def Hbeta(self, D, beta):
        '''
        Calculate entropy
        '''
        ## Calculation of Gauss distribution
        P = np.exp(-D * beta)
        ## Sum of P
        sumP = np.sum(P)
        if sumP == 0:
            sumP = 1e-12
        P = P / sumP
        ## Calculation of entropy
        H = np.log(sumP) + beta * np.sum(D * P)
        return H, P, sumP

    def compute_joint_probabilities(self,
                                    perplexity, 
                                    tol,
                                    initial_beta_coefficient=1.0,
                                    verbose=False):
        '''
        Calculate matrix P and beta
        '''
        P = np.zeros((self.N, self.N))
        beta = np.ones(self.N) * initial_beta_coefficient
        logU = np.log(perplexity)

        for i in range(self.N):
            if verbose:
                if i % 10 == 0:
                    print(i)
            betamin = -np.inf
            betamax = np.inf
            Di = self.distances[i, np.concatenate((np.r_[0:i], np.r_[i+1:self.N]))]
            
            (H, thisP, sumP) = self.Hbeta(Di, beta[i])
            Hdiff = H - logU
            tries = 0
            
            while np.abs(Hdiff) > tol and tries < 100:
                if Hdiff > 0:
                    betamin = beta[i]
                    if betamax == np.inf or betamax == -np.inf:
                        beta[i] = beta[i] * 1.2
                    else:
                        beta[i] = (beta[i] + betamax) / 2.0
                else:
                    betamax = beta[i]
                    if betamin == np.inf or betamin == -np.inf:
                        beta[i] = beta[i] / 1.2
                    else:
                        beta[i] = (beta[i] + betamin) / 2.0
                if verbose:
                    print(tries)

                (H, thisP, sumP) = self.Hbeta(Di, beta[i])
                if verbose:
                    print(f' SumP: {sumP}')
                Hdiff = H - logU
                tries += 1
                # Early stopping if sumP is too small to prevent divergence
                if sumP < 1e-12 or Hdiff > 1e3:
                    if verbose:
                        print(f"Early stopping at try {tries}, sumP {np.sum(thisP)}, Hdiff {Hdiff}")
                    break

            P[i, np.concatenate((np.r_[0:i], np.r_[i+1:self.N]))] = thisP

        P = (P + P.T) / (2 * self.N)
        self.P, self.beta = P, beta

    def tsne_probabilities(self,
                           X, 
                           perplexity,
                           tol,
                           initial_beta_coefficient=1.0):
        '''
        Compute t-SNE joint probabilities
        '''
        # Calculate pairwise distances
        self.distances = pairwise_distances(X, metric='euclidean', squared=True)
        # Compute joint probabilities
        self.compute_joint_probabilities(perplexity=perplexity, 
                                        tol=tol,
                                        initial_beta_coefficient=initial_beta_coefficient)


    def show_P_matrix(self):
        '''
        Show the matrix P
        '''
        plt.imshow(self.P, cmap='hot', interpolation='nearest')
        plt.colorbar()
        plt.show()

    def initialize_Y(self,
                     seed: int = 42,
                     method: str = 'random',
                     **kwargs):
        '''
        Initialize the vector Y
        '''
        np.random.seed(seed)
        ## Random initialization
        if method == 'random':
            ## Initialize Y for embedded coordinates
            threshold = kwargs.get('threshold', 0.1)
            self.Y[0] = np.random.uniform(low=-threshold, 
                                        high=threshold, 
                                        size=(self.N,2))
        ## with PCA
        elif method == 'pca':
            n_components = kwargs.get('n_components', 2)
            print(f'Initialization: PCA with seed:{seed}, n_components:{n_components}')
            pca = PCA(n_components=n_components,
                      random_state=seed)
            self.Y[0] = pca.fit_transform(self.X_sampled)
        ## with Spectral embedding
        elif method =='se':
            n_neighbors = kwargs.get('n_neighbors', 10)
            n_components = kwargs.get('n_components', 2)
            print(f'Initialization: Spectral Embedding with seed:{seed}, n_components:{n_components}, n_neighbors:{n_neighbors}')
            spectral_embedding = SpectralEmbedding(n_components=n_components,
                                                   n_neighbors=n_neighbors, 
                                                   random_state=seed)    
            self.Y[0] = spectral_embedding.fit_transform(self.X_sampled)
        ## with MDS
        elif method == 'mds':
            n_components = kwargs.get('n_components', 2)
            n_init = kwargs.get('n_init', 4)
            print(f'Initialization: MDS with seed:{seed}, n_components:{n_components}')
            mds = MDS(n_components=n_components, 
                      random_state=seed,
                      n_init=n_init)
            self.Y[0] = mds.fit_transform(self.X_sampled)

        self.Y[-1] = self.Y[0].copy()
        ## For Nestrov method
        self.Y_nes[0] = self.Y[0].copy()

    def calc_cost_with_KL(self,
                  k: int,) -> float:
        '''
        Calculate cost with KL divergence
        '''
        mask = (self.P != 0) & (self.Q[k] > 0)
        return np.sum(self.P[mask] * np.log(self.P[mask] / self.Q[k][mask]))

    def _get_pairwise_squared_distances(self,
                                        Y: np.ndarray) -> np.ndarray:
        squared_norm = np.sum(np.square(Y), axis=1)
        squared_distances = squared_norm[:, np.newaxis] + squared_norm[np.newaxis, :] - 2 * (Y @ Y.T)
        return np.maximum(squared_distances, 0.0)

    def from_Y_to_Q(self,
                    k: int,
                    Y_override: np.ndarray = None):
        '''
        Calculate Q from Y
        '''
        target_Y = self.Y[k] if Y_override is None else Y_override
        squared_distances = self._get_pairwise_squared_distances(target_Y)
        student_kernel = 1.0 / (1.0 + squared_distances)
        np.fill_diagonal(student_kernel, 0.0)

        denominator = np.sum(student_kernel)
        if denominator <= 0:
            denominator = 1e-12

        self.squared_distances[k] = squared_distances
        self.student_kernel[k] = student_kernel
        self.Q[k] = student_kernel / denominator

        ## Calculate cost between P and Q
        self.cost[k] = self.calc_cost_with_KL(k=k)
        log_interval = getattr(self, '_log_interval', 10)
        if log_interval is not None and log_interval > 0 and k % log_interval == 0:
            log_prefix = getattr(self, '_log_prefix', None)
            log_stage = getattr(self, '_log_stage', None)
            if log_prefix is not None or log_stage is not None:
                prefix_parts = []
                if log_prefix is not None:
                    prefix_parts.append(str(log_prefix))
                if log_stage is not None:
                    prefix_parts.append(f'stage={log_stage}')
                prefix = ' | '.join(prefix_parts)
                print(f'[{prefix}] iteration={k}, cost={self.cost[k]}')
            else:
                print(f'iteration: {k}, cost: {self.cost[k]}')

    def from_Q_to_S(self,
                    k: int,
                    Y_override: np.ndarray = None):
        '''
        Calculate S from Q
        '''
        if k not in self.student_kernel or Y_override is not None:
            target_Y = self.Y[k] if Y_override is None else Y_override
            squared_distances = self._get_pairwise_squared_distances(target_Y)
            self.squared_distances[k] = squared_distances
            student_kernel = 1.0 / (1.0 + squared_distances)
            np.fill_diagonal(student_kernel, 0.0)
            self.student_kernel[k] = student_kernel

        self.S[k] = (self.alpha * self.P - self.Q[k]) * self.student_kernel[k]

    def _get_nag_evaluation_point(self,
                                  k: int) -> np.ndarray:
        return self.Y_nes[k]

    def from_S_to_Y(self,
                    k:int,
                    optimization_method: str) -> None:
        '''
        Calculate Y from S
        '''
        ## Initialize Y for next embedded coordinates
        self.Y[k+1] = np.zeros((self.N, 2))
        if optimization_method == 'NAG':
            self.Y_nes[k+1] = np.zeros((self.N, 2))

        if optimization_method == 'NAG':
            current_Y = self.Y_nes[k]
        else:
            current_Y = self.Y[k]

        row_sums = np.sum(self.S[k], axis=1, keepdims=True)
        drift = self.S[k] @ current_Y - row_sums * current_Y

        if optimization_method == 'GD':
            self.Y[k+1] = self.Y[k] + self.h * drift
        elif optimization_method == 'MM':
            self.Y[k+1] = self.Y[k] + self.h * drift + self.momentum * (self.Y[k] - self.Y[k-1])
        elif optimization_method == 'NAG':
            self.Y[k+1] = current_Y + self.h * drift
            self.Y_nes[k+1] = self.Y[k+1] + (k) / (k+3) * (self.Y[k+1] - self.Y[k])

    def calculate_YQS(self,
                      optimization_method: str,
                      start_iteration: int = 0,
                      end_iteration: int = None,
                      momentum_schedule=None,
                      stage_name: str = None,
                      track_kl_history: bool = False) -> None:
        '''
        Calculate Y, Q, S
        '''
        if end_iteration is None:
            end_iteration = self.ee_iteration

        original_momentum = self.momentum
        ## Calculate Q and S
        for k in range(start_iteration, end_iteration):
            if momentum_schedule is not None:
                self.momentum = momentum_schedule(k)

            evaluation_Y = self._get_nag_evaluation_point(k) if optimization_method == 'NAG' else self.Y[k]
            self.from_Y_to_Q(k, Y_override=evaluation_Y)
            self.from_Q_to_S(k, Y_override=evaluation_Y)
            self.from_S_to_Y(k=k,
                             optimization_method=optimization_method)
            if track_kl_history and hasattr(self, 'kl_history'):
                self.kl_history.append(self.cost[k])
            if stage_name is not None and hasattr(self, 'stage_history'):
                self.stage_history[k] = stage_name

        self.momentum = original_momentum
        print('Done')
    
    def plot_Y(self,
                limit: float):
        '''
        plot Y with given limit
        '''
        for k in range(self.ee_iteration):
            plt.clf()
            if k % 20 == 0:
                print(str(k).zfill(3))
            plt.scatter(self.Y[k][:, 0], self.Y[k][:, 1], c=np.array(self.y_sampled, dtype=int), cmap='jet', alpha=0.5, s=20)
            plt.title(f'Iteration {str(k).zfill(3)}')
        plt.xlim(-limit, limit)
        plt.ylim(-limit, limit)
        plt.savefig(f'./images/iteration_{str(k).zfill(3)}.png')
        plt.close()


class original_tSNE(tSNE_experiment):
    def __init__(self,
                 labels: list,
                 sample_data: str = 'mnist',
                 perplexity: float = 30,
                 sample_number: int = 400,
                 total_iteration: int = 1000,
                 ee_iteration: int = 50,
                 alpha: float = 4.0,
                 learning_rate: float = 500.0,
                 initial_momentum: float = 0.5,
                 final_momentum: float = 0.8,
                 momentum_switch_iteration: int = 250,
                 pca_components: int = 30,
                 initial_beta_coefficient: float = 1e-4,
                 tol: float = 1e-5,
                 random_state: int = 0,
                 min_gain: float = 0.01,
                 initialization_std: float = 1e-4):
        super().__init__(labels=labels,
                         sample_data=sample_data,
                         perplexity=perplexity,
                         alpha=alpha,
                         h=learning_rate,
                         total_iteration=total_iteration,
                         momentum=initial_momentum,
                         sample_number=sample_number,
                         R=len(labels))
        self.total_iteration = total_iteration
        self.ee_iteration = ee_iteration
        self.learning_rate = learning_rate
        self.initial_momentum = initial_momentum
        self.final_momentum = final_momentum
        self.momentum_switch_iteration = momentum_switch_iteration
        self.pca_components = pca_components
        self.initial_beta_coefficient = initial_beta_coefficient
        self.tol = tol
        self.random_state = random_state
        self.min_gain = min_gain
        self.initialization_std = initialization_std
        self.kl_history = []
        self.embedding_ = None
        self.X_original_sampled = None
        self.pca_model = None
        self.gains = None
        self.stage_history = None

    def load_original_mnist(self):
        '''
        Load MNIST with the preprocessing used in the original t-SNE paper.
        '''
        if self.sample_data != 'mnist':
            raise ValueError('load_original_mnist only supports sample_data="mnist".')

        X, y = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False)
        y = y.astype(str)
        rng = check_random_state(self.random_state)

        sampled_X = []
        sampled_y = []
        for label in self.labels:
            label_indices = np.where(y == label)[0]
            if label_indices.shape[0] < self.sample_number:
                raise ValueError(f'Not enough samples for label {label}: {label_indices.shape[0]}')
            chosen = rng.choice(label_indices, size=self.sample_number, replace=False)
            sampled_X.append(X[chosen])
            sampled_y.extend([label] * self.sample_number)

        self.X_original_sampled = np.vstack(sampled_X)
        self.y_sampled = np.array(sampled_y)

        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(self.X_original_sampled)

        max_components = min(X_scaled.shape[0], X_scaled.shape[1])
        n_components = min(self.pca_components, max_components) if self.pca_components is not None else None

        if n_components is not None and n_components >= 1 and n_components < X_scaled.shape[1]:
            self.pca_model = PCA(n_components=n_components, random_state=self.random_state)
            self.X_sampled = self.pca_model.fit_transform(X_scaled)
        else:
            self.X_sampled = X_scaled

        self.N = self.X_sampled.shape[0]
        self.num_unique_labels = np.unique(self.y_sampled).shape[0]
        print('Loaded original MNIST setting:', self.X_sampled.shape, self.y_sampled.shape)
        return self.X_sampled, self.y_sampled

    def _stage_momentum(self,
                        iteration: int,
                        optimization_method: str) -> float:
        if optimization_method == 'GD':
            return 0.0
        if iteration <= self.momentum_switch_iteration:
            return self.initial_momentum
        return self.final_momentum

    def _initialize_optimizer_state(self,
                                    seed: int,
                                    store_history: bool,
                                    initial_Y: np.ndarray = None):
        if initial_Y is None:
            rng = np.random.RandomState(seed)
            Y = rng.normal(loc=0.0,
                           scale=self.initialization_std,
                           size=(self.N, 2))
        else:
            Y = np.array(initial_Y, dtype=float, copy=True)
            if Y.shape != (self.N, 2):
                raise ValueError(f'initial_Y must have shape {(self.N, 2)}, got {Y.shape}')

        update = np.zeros_like(Y)
        gains = np.ones_like(Y)

        self.Y = {0: Y.copy(), -1: Y.copy()} if store_history else dict()
        self.Y_nes = {0: Y.copy()}
        self.Q = dict()
        self.cost = dict()
        self.kl_history = []
        self.stage_history = dict()
        return Y, update, gains

    def fit_transform(self,
                      X=None,
                      store_history: bool = True,
                      verbose: bool = True,
                      seed: int = None,
                      optimization_method: str = 'MM',
                      initial_Y: np.ndarray = None):
        '''
        Run the original t-SNE optimization schedule.
        '''
        if X is None:
            X = self.X_sampled
        if X is None:
            raise ValueError('Input data is not set. Load data first or pass X explicitly.')

        if optimization_method not in {'GD', 'MM', 'NAG'}:
            raise ValueError('optimization_method must be one of {"GD", "MM", "NAG"}.')

        if seed is None:
            seed = self.random_state

        self.X_sampled = X
        self.N = X.shape[0]
        self.tsne_probabilities(X=X,
                                perplexity=self.perplexity,
                                tol=self.tol,
                                initial_beta_coefficient=self.initial_beta_coefficient)

        Y, update, gains = self._initialize_optimizer_state(seed=seed,
                                                            store_history=store_history,
                                                            initial_Y=initial_Y)

        self.verbose = verbose
        self.Q = dict()
        self.S = dict()
        self.cost = dict()
        self.kl_history = []
        self.stage_history = dict()

        ee_end = min(self.ee_iteration, self.total_iteration)
        alpha_original = self.alpha
        if ee_end > 0:
            self.alpha = alpha_original
            self.calculate_YQS(optimization_method=optimization_method,
                               start_iteration=0,
                               end_iteration=ee_end,
                               momentum_schedule=lambda k: self._stage_momentum(k + 1, optimization_method),
                               stage_name='early_exaggeration',
                               track_kl_history=True)

        if ee_end < self.total_iteration:
            self.alpha = 1.0
            self.calculate_YQS(optimization_method=optimization_method,
                               start_iteration=ee_end,
                               end_iteration=self.total_iteration,
                               momentum_schedule=lambda k: self._stage_momentum(k + 1, optimization_method),
                               stage_name='embedding',
                               track_kl_history=True)

        self.alpha = alpha_original
        self.embedding_ = self.Y[self.total_iteration]
        self.gains = gains
        return self.embedding_

class SolutionPath(tSNE_experiment):
    def __init__(self
                ,labels: np.ndarray
                ,sample_data: str
                ,perplexity: float
                ,alpha: float
                ,h: float
                ,total_iteration: int
                ,momentum: float
                ,sample_number: int
                ,R: float
                ,m: float
                ):
        tSNE_experiment.__init__(self, labels, sample_data, perplexity, alpha, h, total_iteration, momentum, sample_number, R)
        ## inheritance of class experiment
        self.labels = labels
        self.N = None  ## record number of data points
        self.sample_data = sample_data
        self.perplexity = perplexity
        ## bandwidth of Gaussian kernel
        self.alpha = alpha
        self.h = h
        self.ee_iteration = total_iteration
        ## Define the location of labels
        self.P = dict()  ## original similarity matrix
        ## Store diameters for each iteration
        self.momentum = momentum
        ## Eigen values and vectors
        self.w = None
        self.v = None
        self.cost = dict()
        ## For visualization
        self.cost_list = []
        ## For data frame for grid search 
        self.df = pd.DataFrame()
        ## for Adjusted Rand Index value
        self.ari = None

        ## optimization methods
        self.optimization_methods = ['GD', 'MM', 'NAG']
        
        ## For termination of Early exaggeration strage
        self.k_GD = None
        self.k_MM = None
        self.k_NAG = None

        self.Y_GD = None
        self.Y_MM = None
        self.Y_NAG = None

        self.m = m
        self.initial_projection = None
        self.solution_cache = dict()

    def getSolution(self):
        '''
        L(alpha * P-H_n)
        '''
        ## eigen decomposition
        D = np.diag(self.P.sum(axis=1))
        L = D - self.P
        self.w, self.v = np.linalg.eigh(L)

        ## Sort by eigenvalues
        sorted_indices = np.argsort(self.w)
        self.w = self.w[sorted_indices]
        self.v = self.v[:, sorted_indices]

        self.w = (self.alpha * self.w) - 1 / (self.N - 1)
        self.w[0] = self.alpha * self.w[0] + 1 / (self.N - 1)
        self.initial_projection = self.v.T @ self.Y[0]
        self.solution_cache = dict()

    def calc_solution_with_Bessel(self,
                                  t: float,
                                  value:float):
        '''
        Calculate Bessel function and modified Bessel function
        '''
        if t * value == 0:
            return 1
        elif value > 0:
            return (2/(t * np.sqrt(value))) \
                        * jv(1, t * np.sqrt(value))    ## normal Bessel function
        else:
            return (2/(t * np.sqrt(-value))) \
                        * iv(1, t * np.sqrt(-value))  ## modified Bessel function

    def calc_solution_with_Bessel_vectorized(self,
                                             t: float,
                                             values: np.ndarray) -> np.ndarray:
        '''
        Vectorized version of the Bessel-based coefficient computation.
        '''
        values = np.asarray(values)
        result = np.ones_like(values, dtype=float)

        nonzero_mask = (t * values) != 0
        positive_mask = nonzero_mask & (values > 0)
        negative_mask = nonzero_mask & (values < 0)

        if np.any(positive_mask):
            positive_values = values[positive_mask]
            result[positive_mask] = (2 / (t * np.sqrt(positive_values))) * jv(1, t * np.sqrt(positive_values))

        if np.any(negative_mask):
            negative_values = values[negative_mask]
            result[negative_mask] = (2 / (t * np.sqrt(-negative_values))) * iv(1, t * np.sqrt(-negative_values))

        return result

    def getSolutionWithoptimizationMethod(self,
                                        k: int,
                                        optimization_method: str) -> np.array:
        '''
        get solution with optimization method
        '''
        cache_key = (optimization_method, k)
        if cache_key in self.solution_cache:
            return self.solution_cache[cache_key].copy()

        Y = np.zeros(self.Y[0].shape)
        if optimization_method == 'GD':
            t = k * self.h
            scales = np.exp(-t * self.w)
            Y = self.v @ (scales[:, np.newaxis] * self.initial_projection)
        elif optimization_method == 'MM':
            t = k * self.h
            scales = np.exp(-t * self.w / (1 - self.momentum))
            Y = self.v @ (scales[:, np.newaxis] * self.initial_projection)
        elif optimization_method == 'NAG':
            t = k * np.sqrt(self.h)
            scales = self.calc_solution_with_Bessel_vectorized(t=t, values=self.w)
            Y = self.v @ (scales[:, np.newaxis] * self.initial_projection)

        self.solution_cache[cache_key] = Y.copy()
        return Y

    def getSolutionPath(self,
                        optimization_method: str) -> None:
        '''
        get solution path with iteration number `k`
        '''
        ## Initialize cost
        self.cost = dict()
        ## trace
        for k in range(self.ee_iteration):
            self.Y[k+1] = np.zeros(self.Y[0].shape)
            Y = self.getSolutionWithoptimizationMethod(k=k, optimization_method=optimization_method)
            self.Y[k+1][:, 0], self.Y[k+1][:, 1] = Y[:, 0], Y[:, 1]

            ## Calculate Q matrix to evaluate KL divergence between P and Q
            self.from_Y_to_Q(k=k)
            ## Calculate cost between P and Q
            self.cost[k] = self.calc_cost_with_KL(k=k)
