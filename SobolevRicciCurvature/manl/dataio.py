import os
import pickle
import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.datasets import fetch_openml, make_swiss_roll
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, normalize
from sklearn.decomposition import TruncatedSVD
from sklearn.datasets import fetch_20newsgroups_vectorized

class dataio():
    def __init__(self):
        self.X_selected = None
        self.y_selected = None
        self.data_dir = "./data"
        self.selected_digits = None
        self.num_samples = None
        self.random_seed = 43

        ## Initial configuration
        np.random.seed(self.random_seed)
        self.set_dir()

    def set_dir(self):
        os.makedirs(self.data_dir, exist_ok=True)

    def load_mnist(self,
                   selected_digits: list = [2, 4, 6, 8],
                   samples_per_digit: int = 100):
        file_path = os.path.join(self.data_dir, "mnist_data.pkl")
        self.selected_digits = selected_digits

        if os.path.exists(file_path):
            with open(file_path, "rb") as f:
                X, y = pickle.load(f)
            print("Loaded MNIST from local file.")
        else:
            mnist = fetch_openml('mnist_784', version=1)
            X = mnist.data.to_numpy() / 255.0  # DataFrame -> Numpy & 0-1 normalization
            y = mnist.target.to_numpy().astype(int)  # DataFrame -> NumPy & convert to int
            with open(file_path, "wb") as f:
                pickle.dump((X, y), f)
            print("Downloaded and saved MNIST data.")
        
        X_list, y_list = [], []
        for digit in self.selected_digits:
            indices = np.random.choice(np.where(y == digit)[0], samples_per_digit, replace=False)
            X_list.append(X[indices])
            y_list.append(y[indices])
        
        self.X_selected = np.vstack(X_list)
        self.y_selected = np.hstack(y_list)

        self.num_samples = self.X_selected.shape[0]

    def load_kddcup(self,
                   filename: str = 'data/kddcup.data_10_percent.gz',
                   selected_labels: list = [18, 9, 11, 0, 17],
                   sample_number: int = 100):
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
        df = pd.read_csv(filename, compression='gzip', header=None, names=columns)
        # encoding
        self.df_preprocessed = self.encode_features(df)

        #  Use LabelEncoder to encode labels as numerical values
        le = LabelEncoder()
        self.df_preprocessed['label'] = le.fit_transform(self.df_preprocessed['label'])
        self.label_encoder_ = le
        self.label_id2name_ = {i: name for i, name in enumerate(le.classes_)}
        self.label_name2id_ = {name: i for i, name in enumerate(le.classes_)}

        selected_names = self.label_encoder_.inverse_transform(selected_labels)
        print(list(zip(selected_labels, selected_names)))

        df_sample = pd.concat([self.df_preprocessed[self.df_preprocessed['label'] == label].sample(n=sample_number, random_state=42, replace=False)
                                for label in selected_labels
                                ])
        print(f'original df_sample: {df_sample.shape}')

        df_sample = df_sample.drop_duplicates()
        print(f'new df_sample: {df_sample.shape}')

        ## Get the number of data points
        features = df_sample.drop(columns=['label'])

        # Scale the data to a 0-1 range using MinMaxScaler
        scaler = MinMaxScaler()
        scaled_features = scaler.fit_transform(features)

        # Convert the scaled data into a DataFrame
        self.df_scaled = pd.DataFrame(scaled_features, columns=features.columns)

        # Add the original 'label' column back
        self.df_scaled['label'] = df_sample['label'].values

        self.X_selected = np.array(self.df_scaled.drop(columns=['label']))
        self.y_selected = self.df_scaled['label']

        self.num_samples = self.X_selected.shape[0]

    def encode_features(self, df):
        '''
        Encode categorical features
        '''
        for column in df.columns:
            if column == "label":
                continue
            if df[column].dtype == type(object):  ## in case of categorical data
                le = LabelEncoder()
                df[column] = le.fit_transform(df[column])
        return df

    def load_swissroll(self,
                        n_samples: int = 1000,
                        noise: float = 0.1,
                        random_state: int = 42):
        self.X_selected, self.y_selected = make_swiss_roll(n_samples=n_samples, 
                                                        noise=noise, 
                                                        random_state=random_state)
        self.num_samples = self.X_selected.shape[0]

    def load_20newsgroups(self,
                            selected_classes: list = [0, 1, 2, 3],
                            samples_per_class: int = 300,
                            n_components: int = 100,
                            random_state: int = 0,
                        ):
        """
        selected_classes: extracted class indices (0-19)
        samples_per_class: number of samples to draw from each class
        n_components: number of dimensions after TruncatedSVD
        random_state: random seed for reproducibility
        """
        if not selected_classes:
            raise ValueError("Specify at least one class in selected_classes")

        # Load the full dataset
        X, y = fetch_20newsgroups_vectorized(return_X_y=True)

        selected_classes = np.array(sorted(set(selected_classes)))
        # Check if specified classes exist in the dataset
        all_classes = np.unique(y)
        missing = [c for c in selected_classes if c not in all_classes]
        if missing:
            raise ValueError(f"The following classes are missing: {missing}")

        # Check the number of samples per class
        rng = np.random.default_rng(random_state)
        indices_list = []
        for cls in selected_classes:
            idx_cls = np.where(y == cls)[0]
            if samples_per_class > len(idx_cls):
                raise ValueError(
                    f"The class {cls} has only {len(idx_cls)} samples, but {samples_per_class} were requested."
                )
            pick = rng.choice(idx_cls, size=samples_per_class, replace=False)
            indices_list.append(pick)

        indices = np.hstack(indices_list)
        rng.shuffle(indices)

        X_sampled = X[indices]
        y_sampled = y[indices]

        # Dimensionality reduction with TruncatedSVD
        svd = TruncatedSVD(n_components=n_components, random_state=random_state)
        X_embedded = svd.fit_transform(X_sampled)

        # Scale by singular values to adjust variance
        X_embedded = X_embedded / (svd.singular_values_ + 1e-12)

        # L2 normalization
        X_embedded = normalize(X_embedded, norm="l2")

        # Summary
        print("Selected classes:", list(selected_classes))
        print("Sampled shape:", X_embedded.shape)
        # bincount
        counts = {cls: int(np.sum(y_sampled == cls)) for cls in selected_classes}
        print("Class distribution:", counts)

        self.X_selected = X_embedded
        self.y_selected = y_sampled
        self.num_samples = X_embedded.shape[0]

    def load_scrna(self, 
                   dataset: str = "pbmc3k", 
                   samples_per_label: int = 500):
        file_path = os.path.join(self.data_dir, f"{dataset}_data.pkl")

        if os.path.exists(file_path):
            with open(file_path, "rb") as f:
                X, y = pickle.load(f)
            print(f"Loaded {dataset} from local file.")
        else:
            # Load dataset
            if dataset == "pbmc3k":
                try:
                    adata = sc.datasets.pbmc3k_processed()
                    print("Loaded pbmc3k_processed (has 'louvain').")
                except Exception:
                    adata = sc.datasets.pbmc3k()
                    print("Loaded pbmc3k (raw). Will compute clusters.")
            else:
                raise ValueError("Unsupported dataset name")

            if "louvain" not in adata.obs and "leiden" not in adata.obs:
                sc.pp.normalize_total(adata, target_sum=1e4)
                sc.pp.log1p(adata)
                sc.pp.highly_variable_genes(adata, n_top_genes=2000, flavor="seurat")
                adata = adata[:, adata.var["highly_variable"]]
                sc.pp.scale(adata, max_value=10)
                sc.tl.pca(adata, svd_solver="arpack")
                sc.pp.neighbors(adata, n_neighbors=15, n_pcs=50)
                try:
                    sc.tl.leiden(adata, key_added="leiden", resolution=1.0)
                    cluster_key = "leiden"
                except Exception:
                    sc.tl.louvain(adata, key_added="louvain", resolution=1.0)
                    cluster_key = "louvain"
            else:
                cluster_key = "leiden" if "leiden" in adata.obs else "louvain"

            # Extract data matrix and labels
            X = adata.X.toarray() if hasattr(adata.X, "toarray") else np.asarray(adata.X)
            y = adata.obs[cluster_key].to_numpy()

            with open(file_path, "wb") as f:
                pickle.dump((X, y), f)
            print(f"Saved {dataset} to {file_path}.")

        # Sampling
        labels = np.unique(y)
        X_list, y_list = [], []
        for lab in labels:
            idx = np.where(y == lab)[0]
            take = min(samples_per_label, len(idx))
            if take == 0: 
                continue
            sel = np.random.choice(idx, take, replace=False)
            X_list.append(X[sel])
            y_list.append(y[sel])

        self.X_selected = np.vstack(X_list)
        self.y_selected = np.hstack(y_list)

        # Convert string labels to integers
        le = LabelEncoder()
        self.y_selected = le.fit_transform(self.y_selected)

        self.num_samples = self.X_selected.shape[0]