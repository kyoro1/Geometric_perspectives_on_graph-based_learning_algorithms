import os
import time
import numpy as np
from numpy.random import *
import pandas as pd
import numpy.linalg as LA
from sklearn.datasets import fetch_openml, fetch_olivetti_faces
from sklearn.utils import check_random_state
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import SpectralEmbedding, MDS, trustworthiness
from scipy.special import jv, iv
from sklearn.metrics import pairwise_distances, adjusted_rand_score
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
from collections import Counter

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.animation as animation

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
        if self.sample_data == 'gmm':
            mu1 = [0, 0, 0]
            mu2 = [150, -110, 170]
            mu3 = [-130, 150, -150]
            sigma1 = [[30, 20, 25], 
                     [20, 50, 10], 
                     [25, 10, 30]]

            # Generate random samples from multivariate normal distribution
            values1 = multivariate_normal(mu1, sigma1, self.sample_number)
            values2 = multivariate_normal(mu2, sigma1, self.sample_number)
            values3 = multivariate_normal(mu3, sigma1, self.sample_number)

            self.X = np.vstack((values1, values2, values3))
            self.y = np.hstack((np.array([0] * self.sample_number)
                                , np.array([1] * self.sample_number)
                                , np.array([2] * self.sample_number)))
            ## Convert to pandas dataframe
            df = pd.DataFrame(self.X)
            ## Add labels to control the order of data
            df['label'] = self.y
            ## Sampling with labels
            self.X_sampled = df.sample(n=self.sample_number, random_state=42)
            ## sort with labels
            self.X_sampled = self.X_sampled.sort_values('label')

            ## Define the labels
            self.y_sampled = np.array(self.X_sampled['label'])
            ## Drop the label column
            self.X_sampled = np.array(self.X_sampled.drop('label', axis=1))

        elif self.sample_data == 'kddcup':
            # Column names with the KDD Cup 1999 dataset
            columns = [
                "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes", "land",
                "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in", "num_compromised",
                "root_shell", "su_attempted", "num_root", "num_file_creations", "num_shells",
                "num_access_files", "num_outbound_cmds", "is_host_login", "is_guest_login", "count",
                "srv_count", "serror_rate", "srv_serror_rate", "rerror_rate", "srv_rerror_rate",
                "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate", "dst_host_count",
                "dst_host_srv_count", "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
                "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate", "dst_host_serror_rate",
                "dst_host_srv_serror_rate", "dst_host_rerror_rate", "dst_host_srv_rerror_rate", "label"
            ]
            # Load data
            df = pd.read_csv(self.filename, compression='gzip', header=None, names=columns)
            # encoding
            self.df_preprocessed = self.encode_features(df)

            #  Use LabelEncoder to encode labels as numerical values
            le = LabelEncoder()
            self.df_preprocessed['label'] = le.fit_transform(self.df_preprocessed['label'])

            df_sample = pd.concat([self.df_preprocessed[self.df_preprocessed['label'] == label].sample(n=self.sample_number, random_state=42)
                                   for label in self.selected_labels
                                   ])

            ## Get the number of data points
            features = df_sample.drop(columns=['label'])

            # Scale the data to a 0-1 range using MinMaxScaler
            scaler = MinMaxScaler()
            scaled_features = scaler.fit_transform(features)

            # Convert the scaled data into a DataFrame
            self.df_scaled = pd.DataFrame(scaled_features, columns=features.columns)

            # Add the original 'label' column back
            self.df_scaled['label'] = df_sample['label'].values

            self.X_sampled = np.array(self.df_scaled.drop(columns=['label']))
            self.y_sampled = self.df_scaled['label']
        elif self.sample_data == 'mnist':
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

        elif self.sample_data == 'olivetti':
            data = fetch_olivetti_faces()
            self.X_sampled = data.data  # flattened face images
            self.y_sampled = data.target  # labels for face images

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
        return sum([self.P[i][j] * np.log(self.P[i][j] / self.Q[k][i,j]) 
                    for i in range(self.N) 
                    for j in range(self.N) 
                        if self.Q[k][i,j] >0 and self.P[i][j] != 0])

    def from_Y_to_Q(self,
                    k: int):
        '''
        Calculate Q from Y
        '''
        ## Initialize Q for embedded coordinates
        self.Q[k] = np.zeros((self.N, self.N))
        denominator = sum(
                        [(1 + np.linalg.norm(self.Y[k][l, :] - self.Y[k][s, :]) ** 2) ** (-1) \
                            for l in range(self.N) for s in range(self.N) if l != s]
                        )
        ## Populate Q[k] with Y[k]
        for i in range(self.N):
            for j in range(self.N):
                if i == j:
                    self.Q[k][i, j] = 0
                elif i > j:
                    self.Q[k][i, j] = (1 + np.linalg.norm(self.Y[k][i, :] - self.Y[k][j, :]) ** 2) ** (-1) / denominator
                    self.Q[k][j, i] = self.Q[k][i, j]
        ## Calculate cost between P and Q
        self.cost[k] = self.calc_cost_with_KL(k=k)
        if k % 10 == 0:
            print(f'iteration: {k}, cost: {self.cost[k]}')

    def from_Q_to_S(self,
                    k: int):
        '''
        Calculate S from Q
        '''
        ## Initialize S
        self.S[k] = np.zeros((self.N, self.N))
        ## Populate Q[k] with Y[k]
        for i in range(self.N):
            for j in range(self.N):
                self.S[k][i, j] = (self.alpha * self.P[i, j] - self.Q[k][i, j]) / (1 + np.linalg.norm(self.Y[k][i, :] - self.Y[k][j, :]) ** 2)

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

        for i in range(self.N):
            if optimization_method == 'GD':
                self.Y[k+1][i, :] = self.Y[k][i, :] + self.h * sum([ (self.Y[k][j, :] - self.Y[k][i, :]) * self.S[k][i, j] for j in range(self.N) if j != i ])
            elif optimization_method == 'MM':
                self.Y[k+1][i, :] = self.Y[k][i, :] + self.h * sum([ (self.Y[k][j, :] - self.Y[k][i, :]) * self.S[k][i, j] for j in range(self.N) if j != i ]) \
                                    + self.momentum * (self.Y[k][i, :] - self.Y[k-1][i, :])
            elif optimization_method == 'NAG':
                ## Update y[k+1]
                self.Y[k+1][i, :] = self.Y_nes[k][i, :] + self.h * sum([ (self.Y_nes[k][j, :] - self.Y_nes[k][i, :]) * self.S[k][i, j] for j in range(self.N) if j != i ])
                ## Update Y_nes[k+1]
                self.Y_nes[k+1][i, :] = self.Y[k+1][i, :] + (k) / (k+3) * (self.Y[k+1][i, :] - self.Y[k][i, :])

    def calculate_YQS(self,
                      optimization_method: str) -> None:
        '''
        Calculate Y, Q, S
        '''
        ## Calculate Q and S
        for k in range(self.ee_iteration):
            self.from_Y_to_Q(k)
            self.from_Q_to_S(k)
            self.from_S_to_Y(k=k,
                             optimization_method=optimization_method)
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

        for i in range(len(self.w)):
            if i == 0:
                self.w[i] = self.alpha * self.w[i]
            else:
                self.w[i] = (self.alpha * self.w[i]) - 1 / (self.N -1)

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

    def getSolutionWithoptimizationMethod(self,
                                        k: int,
                                        optimization_method: str) -> np.array:
        '''
        get solution with optimization method
        '''
        Y = np.zeros(self.Y[0].shape)
        if optimization_method == 'GD':
            t = k * self.h
            ## x coordinate
            Y[:, 0] = sum([np.exp(-t * self.w[l]) * (self.v[:, l] @ self.Y[0][:,0]) * self.v[:, l] for l in range(len(self.w))])
            ## y coordinate
            Y[:, 1] = sum([np.exp(-t * self.w[l]) * (self.v[:, l] @ self.Y[0][:,1]) * self.v[:, l] for l in range(len(self.w))])
        elif optimization_method == 'MM':
            t = k * self.h
            ## x coordinate
            Y[:, 0] = sum([np.exp(-t * self.w[l] / (1-self.momentum)) * (self.v[:, l] @ self.Y[0][:,0]) * self.v[:, l] for l in range(len(self.w))])
            ## y coordinate
            Y[:, 1] = sum([np.exp(-t * self.w[l] / (1-self.momentum)) * (self.v[:, l] @ self.Y[0][:,1]) * self.v[:, l] for l in range(len(self.w))])
        elif optimization_method == 'NAG':
            t = k * np.sqrt(self.h)               
            ## x coordinate
            Y[:, 0] = sum([self.Y[0][l][0] * self.calc_solution_with_Bessel(t=t, value=self.w[l]) * self.v[:, l] for l in range(len(self.w))])
            ## y coordinate
            Y[:, 1] = sum([self.Y[0][l][1] * self.calc_solution_with_Bessel(t=t, value=self.w[l]) * self.v[:, l] for l in range(len(self.w))])
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

    def generate_statistic(self,
                            df: pd.DataFrame, 
                            optimization_method: str,
                            proc: str,
                            x_max: int) -> pd.DataFrame:
        '''
        Calculate mean and std of cost
        '''
        tmp = df[(df.optimization_method == optimization_method) & (df.proc == proc)]['cost']
        df_tmp = pd.DataFrame(tmp.values.tolist())
        df_tmp_mean = np.mean(df_tmp, axis=0)
        df_tmp_std = np.std(df_tmp, axis=0)
        return df_tmp_mean.iloc[:x_max], df_tmp_std.iloc[:x_max]

    def proc_compare(self,
                     optimization_method: str, 
                     proc: str) -> np.array:
        '''
        Judge the process type between `continuous` and `iterative`
        '''
        if proc == 'continuous':
            ## continuous process
            self.getSolutionPath(optimization_method=optimization_method)
            return self.cost.copy()
        else:
            ## iterative process
            self.calculate_YQS(optimization_method=optimization_method)
            return self.cost.copy()

    def calc_cost_transition(self,
                             variations: int):
        '''
        calculate cost transition
        '''
        self.cost_list = []
        for i in range(variations):
            start = time.time()
            print(f'{i}: start')
            ## initialize Y
            self.initialize_Y(threshold=0.1,
                              seed=i,
                              method='random')
            for optimization_method in self.optimization_methods:
                for proc in ['continuous', 'iterative']:
                    processed_cost = self.proc_compare(optimization_method=optimization_method, 
                                                       proc=proc)
                    ## Save proc
                    tmp_list = []
                    tmp_list.append(optimization_method)
                    tmp_list.append(proc)
                    tmp_list.append(processed_cost)
                    self.cost_list.append(tmp_list)
            process_time = time.time() - start
            print(f'{process_time} seconds')

    def visualize_cost_transition(self,
                                  x_max: int):
        '''
        visualize cost transition
        '''
        x_range = range(self.ee_iteration)[:x_max]

        df = pd.DataFrame(self.cost_list)
        df.columns = ['optimization_method', 'proc', 'cost']

        plt.figure()

        ## GD ODE
        df_tmp_mean, df_tmp_std = self.generate_statistic(df=df, optimization_method='GD', proc='continuous', x_max=x_max)
        plt.plot(x_range, df_tmp_mean, label='GD_ODE', linestyle='--', color='pink')
        plt.fill_between(x_range, df_tmp_mean - df_tmp_std, df_tmp_mean + df_tmp_std, color='pink', alpha=0.2)

        ## GD iterative
        df_tmp_mean, df_tmp_std = self.generate_statistic(df=df, optimization_method='GD', proc='iterative', x_max=x_max)
        plt.plot(x_range, df_tmp_mean, label='GD_iterative', linestyle='-', color='r', marker='x')
        plt.fill_between(x_range, df_tmp_mean - df_tmp_std, df_tmp_mean + df_tmp_std, color='r', alpha=0.2)

        ## MM ODE
        df_tmp_mean, df_tmp_std = self.generate_statistic(df=df, optimization_method='MM', proc='continuous', x_max=x_max)
        plt.plot(x_range, df_tmp_mean, label='MM_ODE', linestyle='--', color='purple')
        plt.fill_between(x_range, df_tmp_mean - df_tmp_std, df_tmp_mean + df_tmp_std, color='purple', alpha=0.2)

        ## MM iterative
        df_tmp_mean, df_tmp_std = self.generate_statistic(df=df, optimization_method='MM', proc='iterative', x_max=x_max)
        plt.plot(x_range, df_tmp_mean, label='MM_iterative', linestyle='-', color='b', marker='x')
        plt.fill_between(x_range, df_tmp_mean - df_tmp_std, df_tmp_mean + df_tmp_std, color='b', alpha=0.2)

        ## NAG ODE
        df_tmp_mean, df_tmp_std = self.generate_statistic(df=df, optimization_method='NAG', proc='continuous', x_max=x_max)
        plt.plot(x_range, df_tmp_mean, label='NAG_ODE', linestyle='--', color='g')
        plt.fill_between(x_range, df_tmp_mean - df_tmp_std, df_tmp_mean + df_tmp_std, color='g', alpha=0.2)

        ## NAG iterative
        df_tmp_mean, df_tmp_std = self.generate_statistic(df=df, optimization_method='NAG', proc='iterative', x_max=x_max)
        plt.plot(x_range, df_tmp_mean, label='NAG_iterative', linestyle='-', color='y', marker='x')
        plt.fill_between(x_range, df_tmp_mean - df_tmp_std, df_tmp_mean + df_tmp_std, color='y', alpha=0.2)

        plt.legend(fontsize=15)
        plt.ylabel('KL-divergence', fontsize=15)

        plt.show()

    def visualize_vectors(self):
        '''
        Visualize eigen functions of the Laplacian matrix
        '''
        fig, axes = plt.subplots(2, 3, figsize=(8, 8), sharex=True, sharey=True)

        axes[0,0].set_title('values of $u_1$')
        axes[0,0].plot(self.v[:,0])
        axes[0,0].axhline(y=0, color='red', linestyle='--', alpha=0.3)

        axes[0,1].set_title('values of $u_2$')
        axes[0,1].plot(self.v[:,1])
        axes[0,1].axhline(y=0, color='red', linestyle='--', alpha=0.3)

        axes[0,2].set_title('values of $u_3$')
        axes[0,2].plot(self.v[:,2])
        axes[0,2].axhline(y=0, color='red', linestyle='--', alpha=0.3)

        axes[1,0].set_title('values of $u_4$')
        axes[1,0].plot(self.v[:,3])
        axes[1,0].axhline(y=0, color='red', linestyle='--', alpha=0.3)

        axes[1,1].set_title('values of $u_5$')
        axes[1,1].plot(self.v[:,4])
        axes[1,1].axhline(y=0, color='red', linestyle='--', alpha=0.3)

        axes[1,2].axis('off')

    def visualize_eigenvalues(self):
        '''
        Visalize distribution of eigenvalues of the Laplacian matrix
        '''
        sns.histplot(self.w, bins=100)
        plt.title('Distribution of Eigenvalues')

    def visualize_SolutionPath(self,
                               k0: int,
                               k1: int,
                                k2: int,
                                k3: int) -> None:
        '''
        Visualize solution path with 4 different iterations
        '''
        Y0 = self.getSolutionWithoptimizationMethod(k=k0, optimization_method='GD')
        Y1 = self.getSolutionWithoptimizationMethod(k=k1, optimization_method='GD')
        Y2 = self.getSolutionWithoptimizationMethod(k=k2, optimization_method='GD')
        Y3 = self.getSolutionWithoptimizationMethod(k=k3, optimization_method='GD')

        Y4 = self.getSolutionWithoptimizationMethod(k=k0, optimization_method='MM')
        Y5 = self.getSolutionWithoptimizationMethod(k=k1, optimization_method='MM')
        Y6 = self.getSolutionWithoptimizationMethod(k=k2, optimization_method='MM')
        Y7 = self.getSolutionWithoptimizationMethod(k=k3, optimization_method='MM')

        Y8 = self.getSolutionWithoptimizationMethod(k=k0, optimization_method='NAG')
        Y9 = self.getSolutionWithoptimizationMethod(k=k1, optimization_method='NAG')
        Y10 = self.getSolutionWithoptimizationMethod(k=k2, optimization_method='NAG')
        Y11 = self.getSolutionWithoptimizationMethod(k=k3, optimization_method='NAG')

        fig, axes = plt.subplots(3, 4, figsize=(15, 15))  # Create 3 subplots arranged horizontally

        axes[0,0].scatter(Y0[:, 0], Y0[:, 1], c=np.array(self.y_sampled, dtype=int), cmap='jet', alpha=0.3, s=20)
        axes[0,0].set_title(f'GD (k={k0},h={self.h},t={k0*self.h})')
        axes[0,1].scatter(Y1[:, 0], Y1[:, 1], c=np.array(self.y_sampled, dtype=int), cmap='jet', alpha=0.3, s=20)
        axes[0,1].set_title(f'GD (k={k1},h={self.h},t={k1*self.h})')
        axes[0,1].set_yticks([])  # Hide y-axis ticks
        axes[0,2].scatter(Y2[:, 0], Y2[:, 1], c=np.array(self.y_sampled, dtype=int), cmap='jet', alpha=0.3, s=20)
        axes[0,2].set_title(f'GD (k={k2},h={self.h},t={k2*self.h})')
        axes[0,2].set_yticks([])  # Hide y-axis ticks
        axes[0,3].scatter(Y3[:, 0], Y3[:, 1], c=np.array(self.y_sampled, dtype=int), cmap='jet', alpha=0.3, s=20)
        axes[0,3].set_title(f'GD (k={k3},h={self.h},t={k3*self.h})')
        axes[0,3].set_yticks([])  # Hide y-axis ticks

        axes[1,0].scatter(Y4[:, 0], Y4[:, 1], c=np.array(self.y_sampled, dtype=int), cmap='jet', alpha=0.3, s=20)
        axes[1,0].set_title(f'MM (k={k0},h={self.h},t={k0*self.h})')
        axes[1,1].scatter(Y5[:, 0], Y5[:, 1], c=np.array(self.y_sampled, dtype=int), cmap='jet', alpha=0.3, s=20)
        axes[1,1].set_title(f'MM (k={k1},h={self.h},t={k1*self.h})')
        axes[1,1].set_yticks([])  # Hide y-axis ticks
        axes[1,2].scatter(Y6[:, 0], Y6[:, 1], c=np.array(self.y_sampled, dtype=int), cmap='jet', alpha=0.3, s=20)
        axes[1,2].set_title(f'MM (k={k2},h={self.h},t={k2*self.h})')
        axes[1,2].set_yticks([])  # Hide y-axis ticks
        axes[1,3].scatter(Y7[:, 0], Y7[:, 1], c=np.array(self.y_sampled, dtype=int), cmap='jet', alpha=0.3, s=20)
        axes[1,3].set_title(f'MM (k={k3},h={self.h},t={k3*self.h})')
        axes[1,3].set_yticks([])  # Hide y-axis ticks

        axes[2,0].scatter(Y8[:, 0], Y8[:, 1], c=np.array(self.y_sampled, dtype=int), cmap='jet', alpha=0.3, s=20)
        axes[2,0].set_title(f'NAG (k={k0},h={self.h},t={k0*np.sqrt(self.h):.2f})')
        axes[2,1].scatter(Y9[:, 0], Y9[:, 1], c=np.array(self.y_sampled, dtype=int), cmap='jet', alpha=0.3, s=20)
        axes[2,1].set_title(f'NAG (k={k1},h={self.h},t={k1*np.sqrt(self.h):.2f})')
        axes[2,1].set_yticks([])  # Hide y-axis ticks
        axes[2,2].scatter(Y10[:, 0], Y10[:, 1], c=np.array(self.y_sampled, dtype=int), cmap='jet', alpha=0.3, s=20)
        axes[2,2].set_title(f'NAG (k={k2},h={self.h},t={k2*np.sqrt(self.h):.2f})')
        axes[2,2].set_yticks([])  # Hide y-axis ticks
        axes[2,3].scatter(Y11[:, 0], Y11[:, 1], c=np.array(self.y_sampled, dtype=int), cmap='jet', alpha=0.3, s=20)
        axes[2,3].set_title(f'NAG (k={k3},h={self.h},t={k3*np.sqrt(self.h):.2f})')
        axes[2,3].set_yticks([])  # Hide y-axis ticks

        # Set colormap and normalization
        unique_labels = np.unique(self.y_sampled)
        cmap = plt.get_cmap('jet')
        norm = plt.Normalize(vmin=unique_labels.min(), vmax=unique_labels.max())

        # Create handles for the legend
        handles = [plt.Line2D([0], [0], marker='o', color=cmap(norm(label)), linestyle='', markersize=10) for label in unique_labels]

        if self.sample_data == 'kddcup':
            # Create the legend
            axes[2,3].legend(handles, self.labels, title="Values")


    def visualize_SolutionPath_For_GMM(self,
                               k0: int,
                               k1: int,
                                k2: int) -> None:
        '''
        Visualize solution path with 3 different iterations
        '''
        Y0 = self.getSolutionWithoptimizationMethod(k=k0, optimization_method='GD')
        Y1 = self.getSolutionWithoptimizationMethod(k=k1, optimization_method='GD')
        Y2 = self.getSolutionWithoptimizationMethod(k=k2, optimization_method='GD')

        Y4 = self.getSolutionWithoptimizationMethod(k=k0, optimization_method='MM')
        Y5 = self.getSolutionWithoptimizationMethod(k=k1, optimization_method='MM')
        Y6 = self.getSolutionWithoptimizationMethod(k=k2, optimization_method='MM')

        Y8 = self.getSolutionWithoptimizationMethod(k=k0, optimization_method='NAG')
        Y9 = self.getSolutionWithoptimizationMethod(k=k1, optimization_method='NAG')
        Y10 = self.getSolutionWithoptimizationMethod(k=k2, optimization_method='NAG')

        fig, axes = plt.subplots(3, 3, figsize=(15, 15))  # Create 3 subplots arranged horizontally

        # Set the default font size for all elements
        plt.rcParams.update({'font.size': 15})

        axes[0,0].scatter(Y0[:, 0], Y0[:, 1], c=np.array(self.y_sampled, dtype=int), cmap='jet', alpha=0.3, s=20)
        axes[0,0].set_title(f'GD (k={k0},h={self.h},t={k0*self.h})')
        axes[0,1].scatter(Y1[:, 0], Y1[:, 1], c=np.array(self.y_sampled, dtype=int), cmap='jet', alpha=0.3, s=20)
        axes[0,1].set_title(f'GD (k={k1},h={self.h},t={k1*self.h})')
        axes[0,1].set_yticks([])  # Hide y-axis ticks
        axes[0,2].scatter(Y2[:, 0], Y2[:, 1], c=np.array(self.y_sampled, dtype=int), cmap='jet', alpha=0.3, s=20)
        axes[0,2].set_title(f'GD (k={k2},h={self.h},t={k2*self.h})')
        axes[0,2].set_yticks([])  # Hide y-axis ticks

        axes[1,0].scatter(Y4[:, 0], Y4[:, 1], c=np.array(self.y_sampled, dtype=int), cmap='jet', alpha=0.3, s=20)
        axes[1,0].set_title(f'MM (k={k0},h={self.h},t={k0*self.h})')
        axes[1,1].scatter(Y5[:, 0], Y5[:, 1], c=np.array(self.y_sampled, dtype=int), cmap='jet', alpha=0.3, s=20)
        axes[1,1].set_title(f'MM (k={k1},h={self.h},t={k1*self.h})')
        axes[1,1].set_yticks([])  # Hide y-axis ticks
        axes[1,2].scatter(Y6[:, 0], Y6[:, 1], c=np.array(self.y_sampled, dtype=int), cmap='jet', alpha=0.3, s=20)
        axes[1,2].set_title(f'MM (k={k2},h={self.h},t={k2*self.h})')
        axes[1,2].set_yticks([])  # Hide y-axis ticks

        axes[2,0].scatter(Y8[:, 0], Y8[:, 1], c=np.array(self.y_sampled, dtype=int), cmap='jet', alpha=0.3, s=20)
        axes[2,0].set_title(f'NAG (k={k0},h={self.h},t={k0*np.sqrt(self.h):.2f})')
        axes[2,1].scatter(Y9[:, 0], Y9[:, 1], c=np.array(self.y_sampled, dtype=int), cmap='jet', alpha=0.3, s=20)
        axes[2,1].set_title(f'NAG (k={k1},h={self.h},t={k1*np.sqrt(self.h):.2f})')
        axes[2,1].set_yticks([])  # Hide y-axis ticks
        axes[2,2].scatter(Y10[:, 0], Y10[:, 1], c=np.array(self.y_sampled, dtype=int), cmap='jet', alpha=0.3, s=20)
        axes[2,2].set_title(f'NAG (k={k2},h={self.h},t={k2*np.sqrt(self.h):.2f})')
        axes[2,2].set_yticks([])  # Hide y-axis ticks

        fig.savefig('gmm_plot.pdf')

    def calc_ARI(self,
                k: int,
                optimization_method: str) -> float:
        '''
        Calculate adjusted Rand Index for each k, optimization method        
        '''
        Y = self.getSolutionWithoptimizationMethod(k=k, 
                                                optimization_method=optimization_method)
        kmeans = KMeans(n_clusters=self.num_unique_labels, random_state=42)
        clusters = kmeans.fit_predict(Y)

        # get the most frequent class label in each cluster
        cluster_to_class = np.zeros(self.num_unique_labels, dtype=int)
        for i in range(self.num_unique_labels):
            mask_cluster = (clusters == i)
            if np.sum(mask_cluster) == 0:
                continue
            cluster_labels = self.y_sampled[mask_cluster]
            cluster_mode = pd.Series(cluster_labels).mode()
            cluster_to_class[i] = cluster_mode[0]

        predicted_classes = cluster_to_class[clusters]

        return adjusted_rand_score(predicted_classes, self.y_sampled)


    def getKmeans(self):
        '''
        Calculate ARI with k-means clustering
        '''
        X_filtered = self.X_sampled
        y_filtered = self.y_sampled

        # k-means clustering
        kmeans = KMeans(n_clusters=self.num_unique_labels, 
                        random_state=42)
        clusters = kmeans.fit_predict(X_filtered)

        # Calculate Adjusted Rand Index score
        self.ari = adjusted_rand_score(y_filtered, 
                                  clusters)
        print(f"Adjusted Rand Index (k-means): {self.ari}")

    def getMEAN_STD(self,
                    df: pd.DataFrame,
                    optimization_method: str) -> tuple:
        '''
        Calculate mean and std of ARI
        '''
        ## filter ARI by optimization_method
        df_tmp = df[df['optimization_method'] == optimization_method][['t','ARI']]
        df_tmp_mean = df_tmp.groupby('t').mean()
        df_tmp_std = df_tmp.groupby('t').std()
        return df_tmp_mean['ARI'], df_tmp_std['ARI']

    def calc_ARI_with_initial_values(self,
                            ks: list,
                            trials: int,
                            threshold: float) -> pd.DataFrame:
        '''
        calculate ARI with various initial values
        '''
        list_ari = []
        self.tsne_probabilities(X=self.X_sampled, 
                            perplexity=self.perplexity,
                            tol=1e-10,
                            initial_beta_coefficient=1e-4)
        ## Solve eigenvalue problem
        self.getSolution()
        for iter_num in range(trials):
            self.initialize_Y(threshold=threshold,
                              seed=iter_num,
                              method='random')
            for optimization_method in self.optimization_methods:
                for k in ks:
                    ari = self.calc_ARI(k=k, 
                                     optimization_method=optimization_method)
                    if optimization_method == 'GD' or optimization_method == 'MM':
                        t = k * self.h
                    elif optimization_method == 'NAG':
                        t = k * np.sqrt(self.h)
                    list_ari.append([iter_num, self.perplexity, k, t, ari, optimization_method])

        return pd.DataFrame(list_ari, columns=['iter_num', 'perplexity', 'k', 't', 'ARI', 'optimization_method'])

    def calc_ARI_with_various_initial_values(self,
                            initialization_methods: list,
                            ks: list,
                            trials: int,
                            **kwargs) -> pd.DataFrame:
        '''
        calculate ARI with various initial values
        '''
        list_ari = []
        self.tsne_probabilities(X=self.X_sampled, 
                            perplexity=self.perplexity,
                            tol=1e-10,
                            initial_beta_coefficient=1e-4)
        ## Solve eigenvalue problem
        self.getSolution()
        for iter_num in range(trials):
            ## Initialize
            for initialization_method in initialization_methods:
                if initialization_method == 'random':
                    threshold = kwargs.get('threshold', 0.1)
                    self.initialize_Y(seed=iter_num,
                                    method=initialization_method,
                                    threshold=threshold)
                elif initialization_method == 'pca':
                    n_components = kwargs.get('n_components', 2)
                    self.initialize_Y(seed=iter_num,
                                    method=initialization_method,
                                    n_components=n_components)
                elif initialization_method == 'se':
                    n_neighbors = kwargs.get('n_neighbors', 10)
                    n_components = kwargs.get('n_components', 2)
                    self.initialize_Y(seed=iter_num,
                                    method=initialization_method,
                                    n_components=n_components,
                                    n_neighbors=n_neighbors)
                elif initialization_method == 'mds':
                    n_components = kwargs.get('n_components', 2)
                    self.initialize_Y(seed=iter_num,
                                    method=initialization_method,
                                    n_components=n_components)

                for optimization_method in self.optimization_methods:
                    for k in ks:
                        ari = self.calc_ARI(k=k, 
                                        optimization_method=optimization_method)
                        if optimization_method == 'GD' or optimization_method == 'MM':
                            t = k * self.h
                        elif optimization_method == 'NAG':
                            t = k * np.sqrt(self.h)
                        list_ari.append([iter_num, self.perplexity, k, t, ari, optimization_method, initialization_method])

        return pd.DataFrame(list_ari, columns=['iter_num', 'perplexity', 'k', 't', 'ARI', 'optimization_method', 'initialization_method'])


    def visualize_ARI(self,
                      df: pd.DataFrame) -> None:
        '''
        Visualize ARI
        '''
        print(f"Adjusted Rand Index (k-means): {self.ari}")

        plt.figure()

        ## GD ODE
        df_tmp_mean, df_tmp_std = self.getMEAN_STD(df=df, optimization_method='GD')
        plt.plot(df_tmp_mean.index, df_tmp_mean, label='GD', linestyle='--', color='pink')
        plt.fill_between(df_tmp_mean.index, df_tmp_mean - df_tmp_std, df_tmp_mean + df_tmp_std, color='pink', alpha=0.2)

        df_tmp_mean, df_tmp_std = self.getMEAN_STD(df=df, optimization_method='MM')
        plt.plot(df_tmp_mean.index, df_tmp_mean, label='MM', linestyle='--', color='purple')
        plt.fill_between(df_tmp_mean.index, df_tmp_mean - df_tmp_std, df_tmp_mean + df_tmp_std, color='purple', alpha=0.2)

        df_tmp_mean, df_tmp_std = self.getMEAN_STD(df=df, optimization_method='NAG')
        plt.plot(df_tmp_mean.index, df_tmp_mean, label='NAG', linestyle='--', color='green')
        plt.fill_between(df_tmp_mean.index, df_tmp_mean - df_tmp_std, df_tmp_mean + df_tmp_std, color='green', alpha=0.2)

        plt.xlabel('t')
        plt.ylabel('Adjusted Rand Index', fontsize=12)
        plt.legend()

        plt.savefig(f'ari_plot_{self.sample_data}.pdf')

    def calcTerms(self,
                  start_t: int,
                   end_t: int,
                   t: float,
                   optimization_method: str,
                   initial_value: bool) -> float:
        '''
        calculate terms for optimization method
        '''
        val_x, val_y = None, None
        if initial_value:
            if optimization_method == 'GD':
                val_x = np.sum([np.exp(-t * self.w[i]) * np.abs(self.v[:,i] @ self.Y[0][:,0]) for i in range(start_t,end_t)])
                val_y = np.sum([np.exp(-t * self.w[i]) * np.abs(self.v[:,i] @ self.Y[0][:,1]) for i in range(start_t,end_t)])
            elif optimization_method == 'MM':
                val_x = np.sum([np.exp(-t * self.w[i]/(1-self.momentum)) * np.abs(self.v[:,i] @ self.Y[0][:,0]) for i in range(start_t,end_t)])
                val_y = np.sum([np.exp(-t * self.w[i]/(1-self.momentum)) * np.abs(self.v[:,i] @ self.Y[0][:,1]) for i in range(start_t,end_t)])
            elif optimization_method == 'NAG':
                val_x = np.sum([np.abs(self.Y[0][i][0] * self.calc_solution_with_Bessel(t=t, value=self.w[i])) for i in range(start_t,end_t)])
                val_y = np.sum([np.abs(self.Y[0][i][1] * self.calc_solution_with_Bessel(t=t, value=self.w[i])) for i in range(start_t,end_t)])
            value = val_x + val_y
        else:
            if optimization_method == 'GD':
                value = np.sum([np.exp(-t * self.w[i]) for i in range(start_t,end_t)])
            elif optimization_method == 'MM':
                value = np.sum([np.exp(-t * self.w[i]/(1-self.momentum)) for i in range(start_t,end_t)])
            elif optimization_method == 'NAG':
                value = np.sum([np.abs(self.calc_solution_with_Bessel(t=t, value=self.w[i])) for i in range(start_t,end_t)])
        return value

    def calcTermsByOptimization(self,
                                k_max: int,
                                initial_value: bool,
                                n_neighbors: int) -> pd.DataFrame:
        '''
        calculate terms by optimization method
        '''
        total_values = []
        for k in range(0, k_max):
            for optimization_method in self.optimization_methods:
                if optimization_method == 'GD' or optimization_method == 'MM':
                    t = k * self.h
                elif optimization_method == 'NAG':
                    t = k * np.sqrt(self.h)
                values = []
                major_val = self.calcTerms(start_t=0,
                                            end_t=self.R,
                                            t=t, 
                                            optimization_method=optimization_method,
                                            initial_value=initial_value)
                minor_val = self.calcTerms(start_t=self.R,
                                             end_t=self.N,
                                             t=t,
                                             optimization_method=optimization_method,
                                             initial_value=initial_value)
                ## Calculate trustworthiness
                Y_k = self.getSolutionWithoptimizationMethod(k=k, optimization_method=optimization_method)
                tw_val = trustworthiness(self.X_sampled, Y_k, n_neighbors=n_neighbors)


                values.append(k)
                values.append(t)
                values.append(optimization_method)
                values.append((1/(self.N-self.R)) * minor_val/((1/(self.N-self.R)) * minor_val + (1/self.R) * major_val))
                values.append(tw_val)
                total_values.append(values)
        return pd.DataFrame(total_values, columns=['k','t', 'optimization_method', 'value', 'trustworthiness'])

    def set_k(self,
              df: pd.DataFrame,
              epsilon: float) -> None:
        '''
        set k for each optimization method
        '''
        self.k_GD = min(df[(df.optimization_method == 'GD') & (df.value < epsilon)]['k'])
        self.k_MM = min(df[(df.optimization_method == 'MM') & (df.value < epsilon)]['k'])
        self.k_NAG = min(df[(df.optimization_method == 'NAG') & (df.value < epsilon)]['k'])

    def visualizeTermsByOptimization(self,
                                    df: pd.DataFrame,
                                    title: str,
                                    epsilon: float,
                                    saved_filename: str,
                                    fontsize: int) -> None:
        '''
        Visualize ARR and Trustworthiness by optimization method
        '''

        # color mapping for optimization methods
        color_map = {
            'GD': 'tab:blue',
            'MM': 'tab:orange',
            'NAG': 'tab:green'
        }

        # style mapping for metrics
        linestyle_map = {
            'ARR': '-',
            'TW': '--',
            # 'CON': ':'  # if needed
        }

        plt.figure(figsize=(10, 6))

        for method in self.optimization_methods:
            subset = df[df['optimization_method'] == method]

            # ARR
            plt.plot(subset['t'], subset['value'],
                    label=f"{method} ARR",
                    color=color_map[method],
                    linestyle=linestyle_map['ARR'])

            # Trustworthiness（dashed lines）
            plt.plot(subset['t'], subset['trustworthiness'],
                    label=f"{method} TW",
                    color=color_map[method],
                    linestyle=linestyle_map['TW'])

        # axis label
        plt.xlabel('t', fontsize=fontsize)
        plt.ylabel('ARR / Trustworthiness', fontsize=fontsize)
        plt.title(title, fontsize=fontsize)

        # red horizontal line for epsilon
        plt.axhline(y=epsilon, color='red', linestyle='--')

        # legend
        plt.legend(title='Method & Metric', fontsize=fontsize, title_fontsize=fontsize)
        plt.grid(True)
        plt.savefig(saved_filename, bbox_inches='tight')
        plt.show()


    def calcTermsByOptimizationForMM(self,
                                     k_max: int,
                                     momentum_coefficients: list) -> None:
        '''
        calculate terms by optimization method for MM
        '''
        optimization_method = 'MM'
        total_values = []
        for k in range(0, k_max):
            for m in momentum_coefficients:
                t = k * self.h
                self.momentum = m
                values = []
                major_val = self.calcTerms(start_t=0,
                                            end_t=self.R,
                                            t=t, 
                                            optimization_method=optimization_method,
                                            initial_value=False)
                minor_val = self.calcTerms(start_t=self.R,
                                                end_t=self.N,
                                                t=t,
                                                optimization_method=optimization_method,
                                                initial_value=False)
                values.append(k)
                values.append(t)
                values.append(m)
                values.append(optimization_method)
                values.append((1/(self.N-self.R)) * minor_val/((1/(self.N-self.R)) * minor_val + (1/self.R) * major_val))
                total_values.append(values)
        return pd.DataFrame(total_values, columns=['k','t','m', 'optimization_method', 'value'])

    def visualizeTermsByOptimizationForMM(self,
                                          df: pd.DataFrame,
                                          epsilon: float,
                                          momentum_coefficients: list,
                                          title: str,
                                          legend_title: str,
                                          saved_filename: str,) -> None:
        '''
        visualize terms by optimization method for MM
        '''
        plt.figure(figsize=(10, 6))
        for m in momentum_coefficients:
            subset = df[df['m'] == m]
            plt.plot(subset['t'], subset['value'], label=m)

        # Adding labels and legend
        plt.xlabel('t')
        plt.ylabel('Average Residual Ratio(ARR)')
        plt.title(title)
        plt.legend(title=legend_title)
        plt.axhline(y=epsilon, color='red', linestyle='--')

        plt.savefig(saved_filename, bbox_inches='tight')

        # Display the chart
        plt.show()


    def calcThreshold(self,
                      df: pd.DataFrame,
                      epsilon: float) -> list:
        '''
        calculate iteration number under some threshold(epsilon) for each optimization method
        '''
        result = df[df['value'] < epsilon].groupby('optimization_method').apply(lambda x: x.loc[x['t'].idxmin()])
        result = result.reset_index(drop=True)
        k0 = result[result.optimization_method == 'GD']['k'][0]
        k1 = result[result.optimization_method == 'MM']['k'][1]
        k2 = result[result.optimization_method == 'NAG']['k'][2]
        return [k0, k1, k2]

    def drawScatterPlot(self,
                        df: pd.DataFrame,
                        epsilon: float,
                        legend: bool,
                        fontsize: int,) -> None:
        '''
        Calculate threshold and draw scatter plot
        and draw scatter plot for each optimization method
        '''
        thresholds = self.calcThreshold(df=df, epsilon=epsilon)
        print(f"Thresholds: {thresholds}")
        k0, k1, k2 = thresholds[0], thresholds[1], thresholds[2]
        
        # Sample data and labels
        Y0 = self.getSolutionWithoptimizationMethod(k=k0, optimization_method='GD')
        Y1 = self.getSolutionWithoptimizationMethod(k=k1, optimization_method='MM')
        Y2 = self.getSolutionWithoptimizationMethod(k=k2, optimization_method='NAG')

        # Unique labels
        unique_labels = np.unique(self.y_sampled)

        # Define a function to handle the plotting
        def plot_scatter(ax, Y, labels, title, x_ticks=True, legend=legend):
            for label in labels:
                indices = self.y_sampled == label
                ax.scatter(Y[indices, 0], Y[indices, 1], label=f'{label}', alpha=0.2, s=20)
            ax.set_title(title, fontsize=fontsize)
            if not x_ticks:
                ax.set_yticks([])  # Optionally hide y-axis ticks
            if legend and 'GD' in title:
                ax.legend(fontsize=fontsize)

        # Set up figure and axes
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))  # Create 3 subplots arranged horizontally

        # Plot using the defined function
        plot_scatter(axes[0], Y0, unique_labels, f'GD (t={k0*self.h})')
        plot_scatter(axes[1], Y1, unique_labels, f'MM (t={k1*self.h})', x_ticks=False)
        plot_scatter(axes[2], Y2, unique_labels, f'NAG (t={round(k2*np.sqrt(self.h),2)})', x_ticks=False)

        plt.show()
        fig.savefig(f'scatter_plot_{k0}_{k1}_{k2}.pdf')

    def drawARIWithInitialValues(self,
                                 df: pd.DataFrame,
                                 initialization_methods: list,
                                 saved_filename: str,
                                 fontsize: int,) -> None:
        '''
        Draw ARI with initial values
        '''
        custom_palette = {
            'GD': '#ff9999',   # pink
            'MM': '#9933cc',   # purple
            'NAG': '#66cc66'   # green
        }

        # configure the size of the graph (2x2)
        _, axes = plt.subplots(2, 2, figsize=(12, 10))

        # unified range of the vertical axis (the vertical axis ranges from 0 to 1)
        y_min, y_max = 0, 0.8

        idx = 0
        for init_method in initialization_methods:
            # filter by initialization_method
            df_subset = df[df['initialization_method'] == init_method]

            ax = axes[idx // 2, idx % 2]  # fill in the subplot
            sns.lineplot(
                data=df_subset,
                x='t', 
                y='ARI', 
                hue='optimization_method',
                ax=ax,
                palette=custom_palette,
                linewidth=2.5
            )

            ax.set_ylim([y_min, y_max])
            ax.set_title(label=f'{init_method}',
                         fontsize=fontsize)
            ax.set_xlabel('t')

            if init_method in ['random', 'se']:
                ax.set_ylabel(ylabel='Adjusted Rand Index (ARI)',
                              fontsize=fontsize)
            else:
                ax.set_ylabel('')

            # Move legend to a less intrusive position for each subplot
            if idx == 0:  # Top-left (random)
                ax.legend(title='optimization method', 
                          loc='upper right',
                          fontsize=fontsize,
                          title_fontsize=fontsize)
            elif idx == 1:  # Top-right (pca)
                ax.legend(title='optimization method', 
                          loc='lower right',
                          fontsize=fontsize,
                          title_fontsize=fontsize)
            elif idx == 2:  # Bottom-left (SE) - Move to lower right to avoid overlap
                ax.legend(title='optimization method', 
                          loc='lower right',
                          fontsize=fontsize,
                          title_fontsize=fontsize)
            else:  # Bottom-right (mds)
                ax.legend(title='optimization method', 
                          loc='lower right',
                          fontsize=fontsize,
                          title_fontsize=fontsize)
            idx += 1

        plt.savefig(saved_filename, bbox_inches='tight')
        plt.tight_layout()
        plt.show()

    def from_Q_to_S_without_alpha(self, k: int):
        '''
        Calculate S from Q without alpha
        '''
        # Calculate the difference matrix (N, N, D)
        diff = self.Y[k][:, np.newaxis, :] - self.Y[k][np.newaxis, :, :]

        # calculate the sum of the squared distances of each pair
        squared_norm = np.sum(diff ** 2, axis=2)

        # calculate the denominator of the equation
        self.S[k] = (self.P - self.m * self.Q[k]) / (1 + squared_norm)


    def getSolutionPathWithEmbeddingStage(self,
                                          random_seed_for_initialization: int,
                                          optimization_method: str,
                                          total_iteration: int,):
        ## Early exaggeration stage with self.ee_iteration
        self.initialize_Y(threshold=0.1,
                            seed=random_seed_for_initialization,
                            method='random')
        self.getSolutionPath(optimization_method=optimization_method)

        ## Embedding stage
        for k in range(self.ee_iteration, total_iteration):

            ## populate Q matrix
            self.from_Y_to_Q(k=k)
            self.from_Q_to_S_without_alpha(k=k)
            self.from_S_to_Y(k=k, optimization_method='GD')

            ## Calculate Q matrix to evaluate KL divergence between P and Q
            self.cost[k] = self.calc_cost_with_KL(k=k)

    def getSolutionPathWithEmbeddingStageForALL(self,
                                                random_seed_for_initialization: int,
                                                total_iteration: int,
                                                output_gif: str,) -> None:
        ## Solution path with embedding stage using GD
        print('Start Solution Path with embedding stage using GD')
        self.ee_iteration = self.k_GD
        self.getSolutionPathWithEmbeddingStage(random_seed_for_initialization=random_seed_for_initialization,
                                               optimization_method='GD',
                                               total_iteration=total_iteration)
        self.Y_GD = self.Y.copy()

        ## Solution path with embedding stage using GD
        print('Start Solution Path with embedding stage using MM')
        self.ee_iteration = self.k_MM
        self.getSolutionPathWithEmbeddingStage(random_seed_for_initialization=random_seed_for_initialization,
                                               optimization_method='MM',
                                               total_iteration=total_iteration)
        self.Y_MM = self.Y.copy()

        ## Solution path with embedding stage using NAG
        print('Start Solution Path with embedding stage using NAG')
        self.ee_iteration = self.k_NAG
        self.getSolutionPathWithEmbeddingStage(random_seed_for_initialization=random_seed_for_initialization,
                                               optimization_method='NAG',
                                               total_iteration=total_iteration)
        self.Y_NAG = self.Y.copy()


        ## Convert to numpy array
        self.y_sampled = np.array(self.y_sampled, dtype=float)

        # Prep for Figure and Axes
        fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharex=False, sharey=False)
        titles = ["Gradient Descent (GD)", "Momentum Method (MM)", "Nesterov Accelerated GD (NAG)"]

        # Configure colormap and normalization
        cmap = plt.cm.jet  
        norm = plt.Normalize(vmin=0, vmax=9) 

        # Create scatter plots
        scatters = []
        for ax, title in zip(axes, titles):
            scat = ax.scatter([], [], c=[], cmap=cmap, norm=norm, s=20, alpha=0.2)  # 空の散布図を作成
            ax.set_title(title)
            scatters.append(scat)

        # Update functions
        def update(frame):
            # Get current data
            Y_GD_tmp = self.Y_GD[frame]
            Y_MM_tmp = self.Y_MM[frame]
            Y_NAG_tmp = self.Y_NAG[frame]

            # Get current data range
            x_min_GD, x_max_GD = Y_GD_tmp[:, 0].min(), Y_GD_tmp[:, 0].max()
            y_min_GD, y_max_GD = Y_GD_tmp[:, 1].min(), Y_GD_tmp[:, 1].max()

            x_min_MM, x_max_MM = Y_MM_tmp[:, 0].min(), Y_MM_tmp[:, 0].max()
            y_min_MM, y_max_MM = Y_MM_tmp[:, 1].min(), Y_MM_tmp[:, 1].max()

            x_min_NAG, x_max_NAG = Y_NAG_tmp[:, 0].min(), Y_NAG_tmp[:, 0].max()
            y_min_NAG, y_max_NAG = Y_NAG_tmp[:, 1].min(), Y_NAG_tmp[:, 1].max()

            # padding
            padding = 0.01

            # Set the range of the axes
            axes[0].set_xlim(x_min_GD - padding, x_max_GD + padding)
            axes[0].set_ylim(y_min_GD - padding, y_max_GD + padding)

            axes[1].set_xlim(x_min_MM - padding, x_max_MM + padding)
            axes[1].set_ylim(y_min_MM - padding, y_max_MM + padding)

            axes[2].set_xlim(x_min_NAG - padding, x_max_NAG + padding)
            axes[2].set_ylim(y_min_NAG - padding, y_max_NAG + padding)

            # Update each scatter plot
            scatters[0].set_offsets(Y_GD_tmp)
            scatters[0].set_array(self.y_sampled)

            scatters[1].set_offsets(Y_MM_tmp)
            scatters[1].set_array(self.y_sampled)

            scatters[2].set_offsets(Y_NAG_tmp)
            scatters[2].set_array(self.y_sampled)

            # display the title
            fig.suptitle(f"k = {frame}")
            return scatters

        # Create animation
        ani = animation.FuncAnimation(
            fig, update, frames=total_iteration, interval=200, blit=False  # blitをFalseに変更
        )

        # Save the animation as a GIF file
        ani.save(output_gif, writer="pillow", fps=5)

        print(f"Animation saved as {output_gif}")

    def drawScatterPlotWithEmbedding(self,
                                    df: pd.DataFrame,
                                    epsilon: float,
                                    legend: bool,
                                    final_k: int,
                                    fontsize: int,) -> None:
        '''
        Calculate threshold and draw scatter plot
        and draw scatter plot for each optimization method
        '''
        thresholds = self.calcThreshold(df=df, epsilon=epsilon)
        print(f"Thresholds: {thresholds}")
        k0, k1, k2 = thresholds[0], thresholds[1], thresholds[2]
        
        # Sample data and labels
        Y0 = self.getSolutionWithoptimizationMethod(k=k0, optimization_method='GD')
        Y1 = self.getSolutionWithoptimizationMethod(k=k1, optimization_method='MM')
        Y2 = self.getSolutionWithoptimizationMethod(k=k2, optimization_method='NAG')

        Y3 = self.getSolutionWithoptimizationMethod(k=final_k, optimization_method='GD')
        Y4 = self.getSolutionWithoptimizationMethod(k=final_k, optimization_method='MM')
        Y5 = self.getSolutionWithoptimizationMethod(k=final_k, optimization_method='NAG')

        # Unique labels
        unique_labels = np.unique(self.y_sampled)

        # Define a function to handle the plotting
        def plot_scatter(ax, Y, labels, title, x_ticks=True, legend=legend):
            for label in labels:
                indices = self.y_sampled == label
                ax.scatter(Y[indices, 0], Y[indices, 1], label=f'{label}', alpha=0.2, s=20)
            ax.set_title(title, fontsize=fontsize)
            if not x_ticks:
                ax.set_yticks([])  # Optionally hide y-axis ticks
            if legend and 'GD' in title and x_ticks:
                ax.legend(fontsize=fontsize)

        # Set up figure and axes
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))  # Create 3 subplots arranged horizontally

        # Plot using the defined function
        plot_scatter(axes[0,0], Y0, unique_labels, f'GD (k={k0}, t={k0*self.h})')
        plot_scatter(axes[0,1], Y1, unique_labels, f'MM (k={k1}, t={k1*self.h})', x_ticks=False)
        plot_scatter(axes[0,2], Y2, unique_labels, f'NAG (k={k2}, t={round(k2*np.sqrt(self.h),2)})', x_ticks=False)

        plot_scatter(axes[1,0], Y3, unique_labels, f'GD (k={final_k})', x_ticks=False)
        plot_scatter(axes[1,1], Y4, unique_labels, f'MM (k={final_k})', x_ticks=False)
        plot_scatter(axes[1,2], Y5, unique_labels, f'NAG (k={final_k})', x_ticks=False)

        plt.show()
        fig.savefig(f'scatter_plot_{k0}_{k1}_{k2}_with_embedding_{final_k}.pdf')
