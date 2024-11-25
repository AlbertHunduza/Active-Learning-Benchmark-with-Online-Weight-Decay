from abc import ABC, abstractmethod
from typing import Tuple, Literal, Union, Any, Optional
import os
from os.path import join, exists
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import pickle
import joblib
import torch.nn as nn
from torch import Tensor
from torch.nn import Module
from torch.optim import Optimizer
from torch.utils.data import Dataset, TensorDataset, DataLoader
import torchvision.transforms
from sklearn.preprocessing import MinMaxScaler
from scipy import interpolate
from extract_metafeatures import extract_metafeatures

import bohb
import bohb.configspace as cs
from bohb import BOHB



class BaseDataset(ABC):

    def __init__(self,
                 cache_folder: str,
                 config: dict,
                 pool_rng: np.random.Generator,
                 encoded: bool,
                 data_file: str,
                 device=None):

        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        self.config = config
        self.encoded = encoded
        self.pool_rng = pool_rng
        self.data_file = data_file
        self.cache_folder = cache_folder
        if encoded:
            self.encoder_model_checkpoint = config["dataset_embedded"]["encoder_checkpoint"]
            self.budget = config["dataset_embedded"]["budget"]
            self.initial_points_per_class = config["dataset_embedded"]["initial_points_per_class"]
            self.classifier_batch_size = config["dataset_embedded"]["classifier_batch_size"]
            self.class_fitting_mode = config["dataset_embedded"]["classifier_fitting_mode"]
        else:
            self.encoder_model_checkpoint = None
            self.budget = config["dataset"]["budget"]
            self.initial_points_per_class = config["dataset"]["initial_points_per_class"]
            self.classifier_batch_size = config["dataset"]["classifier_batch_size"]
            self.class_fitting_mode = config["dataset"]["classifier_fitting_mode"]
        assert self.class_fitting_mode in ["from_scratch", "finetuning", "shrinking"], "Unkown fitting mode"

        self.name = str(self.__class__).split('.')[-1][:-2]
        if encoded:
            self.name += "Encoded"

        self._load_or_download_data()  # Main call to load the data
        self.reset()  # resets the validation and seed set

        self.n_classes = self.y_test.shape[-1]
        self.x_shape = self.x_unlabeled.shape[1:]
        print(f"Loaded dataset: {self.name}")
        print(f"| Number of classes: {self.n_classes}")
        print(f"| Labeled Instances: {len(self.x_labeled)}")
        print(f"| Unlabeled Instances: {len(self.x_unlabeled)}")
        print(f"| Val Instances {len(self.x_val)}")
        print(f"| Test Instances {len(self.x_test)}")

    def reset(self):
        """
        Resets the validation split and creates a new one.
        Also chooses initial datapoints for the seed set
        """
        self._create_validation_split()
        self._create_seed_set()


    @classmethod
    def inject_config(cls, config:dict):
        """
        This method can be used to change the dataset config.
        I.e. add dropout to the classification model
        """
        pass

    @abstractmethod
    def _download_data(self, target_to_one_hot=True):
        """
        Downloads the data from web and saves it into self.cache_folder
        """
        pass

    @abstractmethod
    def load_pretext_data(self) -> tuple[Dataset, Dataset]:
        """
        Loads the raw data files for the pretext task (SimCLR)
        """
        pass

    @abstractmethod
    def get_pretext_transforms(self, config: dict) -> torchvision.transforms.Compose:
        """
        Loads the neccessary data transforms for the pretext task (SimCLR)
        """
        pass

    @abstractmethod
    def get_pretext_validation_transforms(self, config: dict) -> torchvision.transforms.Compose:
        """
        Loads the neccessary data transforms for the pretext task (SimCLR)
        """
        pass

    def _load_or_download_data(self):
        """
        Main loading function.
        Loads preprocessed data from disk or fetches it from the web
        """
        # make sure the base files are there
        if self.data_file is not None and \
           not exists(join(self.cache_folder, self.data_file)):
            print(f"No local copy of {self.name} found in {self.cache_folder}. \nDownloading Data...")
            self._download_data()
            assert hasattr(self, "x_train")
            assert hasattr(self, "y_train")
            assert hasattr(self, "x_test")
            assert hasattr(self, "y_test")
            self._save_data()

        # encode if neccessary
        if self.encoded and not exists(join(self.cache_folder, f"encoded_{self.data_file}")):
            print("Encoding the dataset...")
            x_train, y_train, x_test, y_test = self._load_data(encoded=False)
            self.n_classes = y_train.shape[-1]
            self.x_shape = x_train.shape[1:]
            self._encode(x_train, y_train, x_test, y_test)
            self._save_data()

        data = self._load_data(self.encoded)
        if data is None:
            raise ValueError(f"Dataset was not found in {self.cache_folder} and could not be downloaded")
        self.x_train, self.y_train, self.x_test, self.y_test = data
        return True

    def _encode(self, x_train, y_train, x_test, y_test):
        """
        Passes the data through the encoder model specified in the config
        """
        with torch.no_grad():
            device = "cpu"
            # device = "cuda" if torch.cuda.is_available() else "cpu"
            model = self.get_pretext_encoder(self.config)
            model.load_state_dict(torch.load(self.encoder_model_checkpoint, map_location=device))
            model = model.to(device)
            train_loader = DataLoader(TensorDataset(x_train.to(device)), shuffle=False, batch_size=512, drop_last=False)
            test_loader = DataLoader(TensorDataset(x_test.to(device)), shuffle=False, batch_size=512, drop_last=False)
            enc_train = torch.zeros((0, self.config["pretext_encoder"]["feature_dim"])).to(device)
            enc_test = torch.zeros((0, self.config["pretext_encoder"]["feature_dim"])).to(device)
            for x in tqdm(train_loader):
                x_enc = model(x[0])
                enc_train = torch.cat([enc_train, x_enc], dim=0)
            for x in tqdm(test_loader):
                x_enc = model(x[0])
                enc_test = torch.cat([enc_test, x_enc], dim=0)
            enc_train, enc_test = normalize(enc_train.numpy(), enc_test.numpy(), mode="mean_std")
            self.x_train, self.y_train, self.x_test, self.y_test = to_torch(enc_train, torch.float32, device), y_train, \
                                                                   to_torch(enc_test, torch.float32, device), y_test

    def _load_data(self, encoded) -> Union[None, Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        Loads the data from self.cache_folder and returns x_train, y_train, x_test, y_test.
        """
        if encoded:
            file = os.path.join(self.cache_folder, f"encoded_{self.data_file}")
        else:
            file = os.path.join(self.cache_folder, self.data_file)
        
        if os.path.exists(file):
            dataset = torch.load(file, map_location=torch.device("cpu"))
            
            # Extract x_train, y_train, x_test, y_test with default empty tensors for missing items
            x_train = dataset.get("x_train", torch.tensor([]))
            y_train = dataset.get("y_train", torch.tensor([]))
            x_test = dataset.get("x_test", torch.tensor([]))
            y_test = dataset.get("y_test", torch.tensor([]))
            
            return x_train, y_train, x_test, y_test

        # If loading fails, return four None values to match expected unpacking
        return None, None, None, None



    def get_classifier(self, model_rng) -> Module:
        """
        Constructs the classifier for this dataset according to the config
        :return: PyTorch Model
        """
        from classifiers.classifier import construct_model
        if self.encoded:
            model, _= construct_model(model_rng, self, self.config["classifier_embedded"])
        else:
            model, _ = construct_model(model_rng, self, self.config["classifier"])
        return model

    def get_pretext_encoder(self, config: dict, seed=1) -> nn.Module:
        """
        Constructs the encoding model for this dataset according to the config
        :return: PyTorch Model
        """
        from sim_clr.encoder import ContrastiveModel
        from classifiers.classifier import construct_model
        model_rng = torch.Generator()
        model_rng.manual_seed(seed)
        backbone, out_dim = construct_model(model_rng, self, config["pretext_encoder"], add_head=False)
        config["pretext_encoder"]["encoder_dim"] = out_dim
        model = ContrastiveModel({'backbone': backbone, 'dim': config["pretext_encoder"]["encoder_dim"]},
                                 head="mlp", features_dim=config["pretext_encoder"]["feature_dim"])
        return model
    
    @staticmethod
    def get_optimal_weight_decay(subset_size, dataset_name):
        """
        Reads and interpolates weight decay from the grid search CSV file for the optimal strategy.
        """
        # Path to the CSV file for the given dataset
        csv_path = os.path.join("grid_search_output", dataset_name, "classification", "final_results.csv")
        
        # Check if the file exists
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV file for {dataset_name} not found at {csv_path}")
        
        # Read the CSV file
        df = pd.read_csv(csv_path)
        
        # Get the available subset sizes and corresponding weight decays
        available_subset_sizes = df['subset_size'].values
        weight_decays = df['accuracy_based_best_weight_decay'].values
        
        # Handle cases where the subset size is smaller than the minimum or larger than the maximum available sizes
        if subset_size < available_subset_sizes.min():
            # Return the best weight decay for the smallest subset size
            return weight_decays[0]
        elif subset_size > available_subset_sizes.max():
            # Return the best weight decay for the largest subset size
            return weight_decays[-1]
        
        # If the exact subset size exists in the CSV, return the weight decay
        if subset_size in df['subset_size'].values:
            return float(df[df['subset_size'] == subset_size]['accuracy_based_best_weight_decay'].values[0])
        
        # Perform interpolation, ensuring no extrapolation is allowed
        f = interpolate.interp1d(available_subset_sizes, weight_decays, bounds_error=True)
        
        return float(f(subset_size))
    '''
    def _load_data(self, encoded) -> Union[None, Tuple]:
        """
        Loads the data from self.cache_folder
        Returns None on failure or a tuple(x_train, y_train, x_test, y_test)
        """
        if encoded:
            file = os.path.join(self.cache_folder, f"encoded_{self.data_file}")
        else:
            file = os.path.join(self.cache_folder, self.data_file)
        
        if not os.path.exists(file):
            raise FileNotFoundError(f"File {file} does not exist in the cache folder '{self.cache_folder}'")
        
        try:
            dataset = torch.load(file, map_location=torch.device("cpu"))
            return dataset["x_train"], dataset["y_train"], dataset["x_test"], dataset["y_test"]
        except Exception as e:
            raise ValueError(f"Failed to load data from file {file}: {e}")
    '''

    def get_regularizer_config_space(self):
        # Define the individual hyperparameters using the correct class names
        reg_weight_decay = cs.UniformHyperparameter('reg_weight_decay', lower=1e-5, upper=1e-2, log=True)
        reg_l1 = cs.UniformHyperparameter('reg_l1', lower=1e-5, upper=1e-2, log=True)
        reg_l2 = cs.UniformHyperparameter('reg_l2', lower=1e-5, upper=1e-2, log=True)
        reg_dropout = cs.CategoricalHyperparameter('reg_dropout', choices=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5])

        # Initialize the configuration space with the hyperparameters
        config_space = cs.ConfigurationSpace([reg_weight_decay, reg_l1, reg_l2, reg_dropout])

        return config_space



    def get_dataset_metafeatures(self):
        # Only take the training data for feature extraction
        x_train, y_train, _, _ = self._load_data(self.encoded)

        # Extract metafeatures from the training data
        metafeatures = extract_metafeatures(x_train, y_train)

        # Add subset_size as a feature
        metafeatures['subset_size'] = len(self.x_labeled)
        # Convert to DataFrame for compatibility with the model
        instance_df = pd.DataFrame([metafeatures])

        return instance_df


    def get_predicted_weight_decay(self, subset_size):
        # Get and process the dataset metafeatures
        metafeatures = self.get_dataset_metafeatures()
        metafeatures["subset_size"] = subset_size
        
        # Confirm it's a dictionary of 1D data
        for key, value in metafeatures.items():
            if isinstance(value, np.ndarray):
                metafeatures[key] = value.tolist()  # Convert to list if it's a numpy array

        # Ensure the dictionary is flattened and compatible with DataFrame
        metafeatures_df = pd.DataFrame(metafeatures, index=[0])
        
        # Drop non-numeric columns
        numeric_features = metafeatures_df.select_dtypes(include=[np.number])
        
        #print(f"Instance DataFrame for Prediction (numeric only): {numeric_features}")

        # Load the model
        model_path = os.path.join("metadataset", "weight_decay_model.pkl")
        with open(model_path, 'rb') as f:
            weight_decay_model = pickle.load(f)
        
        # Make prediction
        predicted_weight_decay = weight_decay_model.predict(numeric_features)[0]
        #print(f"Predicted Weight Decay: {predicted_weight_decay}")
        
        return predicted_weight_decay



    def adjust_weight_decay(self, current_size, total_budget, method, predicted_value=None):
        """
        Dynamically adjusts the weight decay based on the number of labeled instances and strategy.
        """
        min_weight_decay = float(self.config.get("min_weight_decay", 1e-6))  # Ensure float
        max_weight_decay = float(self.config.get("max_weight_decay", 0.01))  # Ensure float

        # Calculate progress ratio (0 to 1) based on current size and total budget
        progress = current_size / total_budget

        if method == "cocktail":
            decay = float(self.config["optimizer"]["weight_decay"])

        else:
            raise ValueError(f"Unknown weight decay method: {method}")
        
        # Ensure the returned value is a float
        return float(decay)


    def _construct_optimizer(self, model: Module, opt_config, current_size=None, total_budget=None, predicted_value=None) -> Optimizer:
        # Adjust weight decay dynamically based on the strategy
        weight_decay = self.adjust_weight_decay(current_size, total_budget, self.config.get("weight_decay_strategy", "linear"), predicted_value)

        opt_type = opt_config["type"].lower()
        if opt_type == "nadam":
            return torch.optim.NAdam(model.parameters(), lr=opt_config["lr"], weight_decay=weight_decay)
        elif opt_type == "adam":
            return torch.optim.Adam(model.parameters(), lr=opt_config["lr"], weight_decay=weight_decay)
        elif opt_type == "sgd":
            return torch.optim.SGD(model.parameters(), lr=opt_config["lr"], weight_decay=weight_decay)
        else:
            raise ValueError(f"Unknown optimizer type: {opt_type}")


    def get_optimizer(self, model: Module, regularizer_config=None) -> Optimizer:
        """
        Constructs the optimizer for this dataset according to the config and regularizer configuration.
        """
        if self.encoded:
            opt_config = self.config["optimizer_embedded"]
        else:
            opt_config = self.config["optimizer"]

        # Apply regularizer config
        if regularizer_config:
            opt_config = {**opt_config, **regularizer_config}

        # Remove 'type' key since it's not an argument for the optimizer
        opt_config_filtered = {k: v for k, v in opt_config.items() if k != "type"}

        opt_type = opt_config["type"].lower()
        if opt_type == "adam":
            return torch.optim.Adam(model.parameters(), **opt_config_filtered)
        elif opt_type == "sgd":
            return torch.optim.SGD(model.parameters(), **opt_config_filtered)
        else:
            raise ValueError(f"Unknown optimizer type: {opt_type}")



    def _save_data(self):
        if self.encoded:
            out_file = os.path.join(self.cache_folder, f"encoded_{self.data_file}")
        else:
            out_file = os.path.join(self.cache_folder, self.data_file)
        torch.save({
            "x_train": self.x_train,
            "y_train": self.y_train,
            "x_test": self.x_test,
            "y_test": self.y_test,
        }, out_file)
        return True

    def _create_seed_set(self):
        """
        Selects n samples per class and add them to the labeled set.
        n = config["dataset"]["initial_points_per_class"]
        """
        nClasses = self.y_train.shape[1]
        x_labeled, y_labeled = [], []

        ids = np.arange(self.x_train.shape[0], dtype=int)
        self.pool_rng.shuffle(ids)
        # np.random.shuffle(ids)
        perClassIntances = [0 for _ in range(nClasses)]
        usedIds = []
        for i in ids:
            label = torch.argmax(self.y_train[i])
            if perClassIntances[label] < self.initial_points_per_class:
                x_labeled.append(i)
                y_labeled.append(i)
                usedIds.append(i)
                perClassIntances[label] += 1
            if sum(perClassIntances) >= self.initial_points_per_class * nClasses:
                break
        unusedIds = [i for i in np.arange(self.x_train.shape[0]) if i not in usedIds]
        self.x_labeled = self.x_train[x_labeled]
        self.y_labeled = self.y_train[y_labeled]
        self.x_unlabeled = self.x_train[unusedIds]
        self.y_unlabeled = self.y_train[unusedIds]
        return True


    def _create_validation_split(self):
        if hasattr(self, "x_val"):
            # concat previous validation split and re-split
            self.x_train = torch.cat([self.x_train, self.x_val], dim=0)
            self.y_train = torch.cat([self.y_train, self.y_val], dim=0)

        ids = np.arange(self.x_train.shape[0], dtype=int)
        self.pool_rng.shuffle(ids)
        cut = int(len(ids) * self.config["dataset"]["validation_split"])
        train_ids = ids[cut:]
        val_ids = ids[:cut]
        self.x_val = self.x_train[val_ids]
        self.y_val = self.y_train[val_ids]
        self.x_train = self.x_train[train_ids]
        self.y_train = self.y_train[train_ids]


    def _convert_data_to_tensors(self):
        self.x_train = to_torch(self.x_train, torch.float32, device=self.device)
        self.y_train = to_torch(self.y_train, torch.float32, device=self.device)
        self.x_test = to_torch(self.x_test, torch.float32, device=self.device)
        self.y_test = to_torch(self.y_test, torch.float32, device=self.device)

    def to(self, device):
        """
        This mirrors the behavior of tensor.to(device), but without copying the data
        :param device: cuda or cpu
        :return: self
        """
        self.device = device
        for attr in dir(self):
            if not attr.startswith('__'):
                value = getattr(self, attr)
                if type(value) == torch.Tensor:
                    setattr(self, attr, value.to(device))
        return self

    def get_meta_data(self) -> str:
        return f"{self.name}\n" \
               f"Budget: {self.budget}\n" \
               f"Seeding points per class: {self.initial_points_per_class}\n" \
               f"Classifier Batch Size: {self.classifier_batch_size}"


class VectorDataset(Dataset[Tuple[torch.Tensor, ...]]):
    '''
    Extension of the torch BaseDataset
    Used for pretext training
    '''
    arrays: Tuple[np.ndarray, ...]

    def __init__(self, *arrays: Union[np.ndarray, torch.Tensor]) -> None:
        # assert all(arrays[0].shape[0] == tensor.shape[0] for tensor in arrays), "Size mismatch between tensors"
        self.arrays = arrays

    def __getitem__(self, index):
        return tuple(array[index] for array in self.arrays)

    def __len__(self):
        return self.arrays[0].shape[0]


##################################################################
# Data loading functions, etc.

def to_torch(x: Any, dtype: Optional[torch.dtype] = None,
             device: Union[str, int, torch.device] = "cpu", ) -> torch.Tensor:
    """
    Convert an object to torch.Tensor
    Ref: Tianshou
    """
    if isinstance(x, np.ndarray) and issubclass(
            x.dtype.type, (np.bool_, np.number)
    ):  # most often case
        x = torch.from_numpy(x).to(device)
        if dtype is not None:
            x = x.type(dtype)
        return x
    elif isinstance(x, torch.Tensor):  # second often case
        if dtype is not None:
            x = x.type(dtype)
        return x.to(device)
    else:  # fallback
        raise TypeError(f"object {x} cannot be converted to torch.")


def normalize(x_train, x_test, mode: Literal["none", "mean", "mean_std", "min_max"] = "min_max"):
    if mode == "mean":
        x_train = (x_train - np.mean(x_train, axis=0))
        x_test = (x_test - np.mean(x_test, axis=0))
    elif mode == "mean_std":
        std_train, std_test = np.std(x_train, axis=0), np.std(x_test, axis=0)
        std_train[std_train == 0.0] = 1.0  # replace 0 to avoid division by 0
        std_test[std_test == 0.0] = 1.0
        x_train = (x_train - np.mean(x_train, axis=0)) / std_train
        x_test = (x_test - np.mean(x_test, axis=0)) / std_test
    elif mode == "min_max":
        scaler = MinMaxScaler()
        x_train = scaler.fit_transform(x_train)
        x_test = scaler.fit_transform(x_test)
        x_train, x_test = np.nan_to_num(x_train), np.nan_to_num(x_test)
    else:
        raise ValueError(f"Normalization not known: {mode}")
    return x_train, x_test


def subsample_data(x, y, fraction: float, pool_rng: np.random.Generator):
    all_ids = np.arange(len(x))
    pool_rng.shuffle(all_ids)
    cutoff = int(len(all_ids) * fraction)
    ids = all_ids[:cutoff]
    new_x = x[ids]
    new_y = y[ids]
    return new_x, new_y


def convert_to_channel_first(train: Union[Tensor, np.ndarray], test: Union[Tensor, np.ndarray]):
    if isinstance(train, np.ndarray):
        train = np.moveaxis(train, -1, 1)
        test = np.moveaxis(test, -1, 1)
    else:
        train = train.permute(0, 3, 1, 2)
        test = test.permute(0, 3, 1, 2)
    return train, test


def to_one_hot(labels:np.ndarray)->np.ndarray:
    one_hot = np.zeros((len(labels), labels.max() + 1))
    one_hot[np.arange(len(labels)), labels] = 1
    return one_hot

def postprocess_torch_dataset(train, test) -> Tuple:
    x_train, y_train = train.data, np.array(train.targets)
    x_test, y_test = test.data, np.array(test.targets)
    one_hot_train = np.zeros((len(y_train), y_train.max() + 1))
    one_hot_train[np.arange(len(y_train)), y_train] = 1
    one_hot_test = np.zeros((len(y_test), y_test.max() + 1))
    one_hot_test[np.arange(len(y_test)), y_test] = 1
    return x_train, one_hot_train, x_test, one_hot_test


def postprocess_svm_data(train: tuple, test: tuple, target_to_one_hot=True) -> Tuple:
    # convert labels to int
    x_train, y_train = train[0], train[1].astype(int)
    x_test, y_test = test[0], test[1].astype(int)
    # convert inputs to numpy arrays
    x_train = x_train.toarray()
    x_test = x_test.toarray()
    # convert svm labels to onehot
    if -1 in y_train:
        # binary case
        mask = y_train == -1
        y_train[mask] += 1
        mask = y_test == -1
        y_test[mask] += 1
    elif 0 not in y_train:
        # multi label, but starts at 1
        assert len(np.unique(y_train)) == y_train.max()  # sanity check
        y_train = y_train - 1
        y_test = y_test - 1
    if target_to_one_hot:
        one_hot_train = np.zeros((len(y_train), y_train.max() + 1))
        one_hot_train[np.arange(len(y_train)), y_train] = 1
        one_hot_test = np.zeros((len(y_test), y_test.max() + 1))
        one_hot_test[np.arange(len(y_test)), y_test] = 1
        return x_train, one_hot_train, x_test, one_hot_test
    return x_train, y_train, x_test, y_test


def load_numpy_dataset(file_name: str) -> Union[None, Tuple]:
    if os.path.exists(file_name):
        try:
            with np.load(os.path.join(file_name, file_name), allow_pickle=True) as f:
                x_train, y_train = f['x_train'], f['y_train']
                x_test, y_test = f['x_test'], f['y_test']
            return x_train, y_train, x_test, y_test
        except:
            pass
    return None


class GaussianNoise(Module):
    '''
    Custom transform for augmenting vector datasets
    '''

    def __init__(self, scale=0.1, seed=1):
        super().__init__()
        self.scale = scale
        self.rng = np.random.default_rng(seed)

    def forward(self, x: Tensor):
        return x + (self.rng.normal(0, self.scale, size=x.size())).astype(np.float32)


class VectorToTensor(Module):
    """
    Custom data Transform used for compatibility in the pretext tasks
    """
    def forward(self, x: Union[list, np.ndarray]):
        return torch.Tensor(x).type(torch.float32)