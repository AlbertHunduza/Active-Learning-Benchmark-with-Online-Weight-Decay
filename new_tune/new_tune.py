'''
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import pandas as pd
import yaml
from core.helper_functions import get_dataset_by_name
from sklearn.model_selection import KFold

def set_random_seed(seed_value):
    """
    Fix random seed for reproducibility.
    """
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)  # Ensure reproducibility on CUDA if available

def get_dynamic_patience(subset_size, total_budget):
    """
    Adjust patience dynamically based on the current subset size and total budget.
    Larger subsets get more patience.
    """
    base_patience = 5  # Starting patience for small datasets
    if subset_size > total_budget * 0.5:
        return base_patience * 2
    elif subset_size > total_budget * 0.75:
        return base_patience * 3
    else:
        return base_patience

def train_and_evaluate(DatasetClass, config, cache_folder, subset_size, weight_decay, num_trials, patience=10, min_delta=0.001, num_folds=5):
    """
    Train and evaluate the model using K-Fold cross-validation and dynamic patience for early stopping.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    validation_scores = []
    losses = []

    for trial in range(num_trials):
        # Initialize the dataset class for each trial
        dataset_rng = np.random.default_rng()
        dataset = DatasetClass(cache_folder, config, dataset_rng, encoded=False)

        # Randomly sample a new subset for this trial
        random_indices = np.random.choice(len(dataset.x_train), subset_size, replace=True)
        x_train_subset = dataset.x_train[random_indices]
        y_train_subset = dataset.y_train[random_indices]

        model_rng = torch.Generator()
        model_rng.manual_seed(42 + trial)  # Set a different seed for each trial

        # Use KFold cross-validation
        kf = KFold(n_splits=num_folds, shuffle=True, random_state=42 + trial)
        for fold, (train_idx, val_idx) in enumerate(kf.split(x_train_subset)):
            print(f"Fold {fold + 1}/{num_folds}")
            
            # Split data into training and validation based on the current fold
            x_train_fold = x_train_subset[train_idx]
            y_train_fold = y_train_subset[train_idx]
            x_val_fold = x_train_subset[val_idx]
            y_val_fold = y_train_subset[val_idx]

            model = dataset.get_classifier(model_rng).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=config["optimizer"]["lr"], weight_decay=weight_decay)
            loss_fn = nn.CrossEntropyLoss()

            train_dataloader = DataLoader(TensorDataset(x_train_fold, y_train_fold), batch_size=dataset.classifier_batch_size, shuffle=True)
            val_dataloader = DataLoader(TensorDataset(x_val_fold, y_val_fold), batch_size=512)

            best_loss = float('inf')
            epochs_without_improvement = 0
            max_epochs = 500

            # Calculate dynamic patience based on subset size and total budget
            total_budget = config['dataset']['budget'] 
            patience = get_dynamic_patience(subset_size, total_budget)

            for epoch in range(max_epochs):
                model.train()
                for batch_x, batch_y in train_dataloader:
                    batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                    optimizer.zero_grad()
                    yHat = model(batch_x)
                    loss_value = loss_fn(yHat, batch_y)
                    loss_value.backward()
                    optimizer.step()

                # Validation phase
                model.eval()
                total, correct = 0, 0
                val_loss = 0.0
                with torch.no_grad():
                    for batch_x, batch_y in val_dataloader:
                        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                        yHat = model(batch_x)

                        # If batch_y is one-hot encoded, convert it to class indices
                        if len(batch_y.shape) > 1:
                            batch_y = torch.argmax(batch_y, dim=1)

                        predicted = torch.argmax(yHat, dim=1)
                        correct += (predicted == batch_y).sum().item()
                        total += batch_y.size(0)
                        val_loss += loss_fn(yHat, batch_y).item()

                accuracy = correct / total
                val_loss /= total

                # Early stopping criteria
                if val_loss < best_loss - min_delta:
                    best_loss = val_loss
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += 1

                if epochs_without_improvement >= patience:
                    print(f"Early stopping at epoch {epoch + 1}, no improvement for {patience} epochs.")
                    break

            validation_scores.append(accuracy)
            losses.append(best_loss)

    return np.mean(validation_scores), np.std(validation_scores), np.mean(losses)


def tune_weight_decay(data_folder, dataset_name, task, num_trials, initial_subset_size, budget, weight_decay_range, log_folder):
    """
    Tune weight decay over a range of values for a given dataset and log results.
    """
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # Adjusted to base path of the project
    config_file = os.path.join(base_path, 'configs', f'{dataset_name}.yaml')
    
    if not os.path.exists(config_file):
        raise FileNotFoundError(f"Configuration file not found: {config_file}")

    with open(config_file, 'r') as f:
        config = yaml.load(f, yaml.Loader)

    DatasetClass = get_dataset_by_name(dataset_name)
    all_results = []

    # Calculate the step size as one-tenth of the budget
    step_size = budget // 10

    # Only tune for multiples of step_size
    for subset_size in range(step_size, budget + 1, step_size):
        weight_decay_results = []

        for weight_decay in weight_decay_range:
            mean_accuracy, std_accuracy, avg_loss = train_and_evaluate(DatasetClass, config, data_folder, subset_size, weight_decay, num_trials)
            weight_decay_results.append({
                "weight_decay": weight_decay,
                "mean_accuracy": mean_accuracy,
                "std_accuracy": std_accuracy,
                "avg_loss": avg_loss
            })

        # Save results for this subset size to a unique CSV file
        weight_decay_results_df = pd.DataFrame(weight_decay_results)
        results_file = os.path.join(log_folder, f"weight_decay_results_subset_{subset_size}.csv")
        weight_decay_results_df.to_csv(results_file, index=False)

        # Find the weight decay with the best (lowest) loss
        best_result = min(weight_decay_results, key=lambda x: x["avg_loss"])
        avg_weight_decay = np.mean([res["weight_decay"] for res in weight_decay_results])

        all_results.append({
            "subset_size": subset_size,
            "loss": best_result["avg_loss"],
            "best_params": {"weight_decay": best_result["weight_decay"]},
            "avg_weight_decay": avg_weight_decay
        })

    # Save the final results to a CSV file
    results_df = pd.DataFrame(all_results)
    results_csv_file = os.path.join(log_folder, "final_results.csv")
    results_df.to_csv(results_csv_file, index=False)
    print(f"Final results saved to {results_csv_file}")


import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import pandas as pd
import yaml
from core.helper_functions import get_dataset_by_name
from sklearn.model_selection import KFold

def set_random_seed(seed_value):
    """
    Fix random seed for reproducibility.
    """
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)

def get_dynamic_patience(subset_size, total_budget):
    """
    Adjust patience dynamically based on the current subset size and total budget.
    Larger subsets get more patience.
    """
    base_patience = 5
    if subset_size > total_budget * 0.5:
        return base_patience * 2
    elif subset_size > total_budget * 0.75:
        return base_patience * 3
    else:
        return base_patience

def train_and_evaluate(DatasetClass, config, cache_folder, subset_size, weight_decay, num_trials, patience=10, min_delta=0.001, num_folds=5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    validation_scores = []
    losses = []

    dataset_rng = np.random.default_rng()
    dataset = DatasetClass(cache_folder, config, dataset_rng, encoded=False)

    for trial in range(num_trials):
        # Determine if we need to sample with or without replacement
        if num_trials * subset_size > len(dataset.x_train):
            # Sample with replacement if we would exhaust the dataset
            random_indices = np.random.choice(len(dataset.x_train), subset_size, replace=True)
        else:
            # Sample without replacement as long as we haven't exhausted the dataset
            random_indices = np.random.choice(len(dataset.x_train), subset_size, replace=False)

        x_train_subset = dataset.x_train[random_indices]
        y_train_subset = dataset.y_train[random_indices]

        model_rng = torch.Generator()
        model_rng.manual_seed(42 + trial)  # Different seed for each trial

        # Use KFold cross-validation
        kf = KFold(n_splits=num_folds, shuffle=True, random_state=42 + trial)
        for fold, (train_idx, val_idx) in enumerate(kf.split(x_train_subset)):
            x_train_fold = x_train_subset[train_idx]
            y_train_fold = y_train_subset[train_idx]
            x_val_fold = x_train_subset[val_idx]
            y_val_fold = y_train_subset[val_idx]

            model = dataset.get_classifier(model_rng).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=config["optimizer"]["lr"], weight_decay=weight_decay)
            loss_fn = nn.CrossEntropyLoss()

            train_dataloader = DataLoader(TensorDataset(x_train_fold, y_train_fold), batch_size=dataset.classifier_batch_size, shuffle=True)
            val_dataloader = DataLoader(TensorDataset(x_val_fold, y_val_fold), batch_size=512)

            best_loss = float('inf')
            epochs_without_improvement = 0
            max_epochs = 500

            # Calculate dynamic patience
            total_budget = config['dataset']['budget']
            patience = get_dynamic_patience(subset_size, total_budget)

            for epoch in range(max_epochs):
                model.train()
                for batch_x, batch_y in train_dataloader:
                    batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                    optimizer.zero_grad()
                    yHat = model(batch_x)
                    loss_value = loss_fn(yHat, batch_y)
                    loss_value.backward()
                    optimizer.step()

                # Validation phase
                model.eval()
                total, correct = 0, 0
                val_loss = 0.0
                with torch.no_grad():
                    for batch_x, batch_y in val_dataloader:
                        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                        yHat = model(batch_x)

                        if len(batch_y.shape) > 1:
                            batch_y = torch.argmax(batch_y, dim=1)

                        predicted = torch.argmax(yHat, dim=1)
                        correct += (predicted == batch_y).sum().item()
                        total += batch_y.size(0)
                        val_loss += loss_fn(yHat, batch_y).item()

                accuracy = correct / total
                val_loss /= total

                # Early stopping criteria
                if val_loss < best_loss - min_delta:
                    best_loss = val_loss
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += 1

                if epochs_without_improvement >= patience:
                    break

            validation_scores.append(accuracy)
            losses.append(best_loss)

    # After num_trials, return the averaged results
    return np.mean(validation_scores), np.std(validation_scores), np.mean(losses)

def tune_weight_decay(data_folder, dataset_name, task, num_trials, initial_subset_size, budget, weight_decay_range, log_folder):
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_file = os.path.join(base_path, 'configs', f'{dataset_name}.yaml')
    
    if not os.path.exists(config_file):
        raise FileNotFoundError(f"Configuration file not found: {config_file}")

    with open(config_file, 'r') as f:
        config = yaml.load(f, yaml.Loader)

    DatasetClass = get_dataset_by_name(dataset_name)
    all_results = []

    step_size = budget // 10

    for subset_size in range(step_size, budget + 1, step_size):
        weight_decay_results = []

        for weight_decay in weight_decay_range:
            mean_accuracy, std_accuracy, avg_loss = train_and_evaluate(DatasetClass, config, data_folder, subset_size, weight_decay, num_trials)
            weight_decay_results.append({
                "weight_decay": weight_decay,
                "mean_accuracy": mean_accuracy,
                "std_accuracy": std_accuracy,
                "avg_loss": avg_loss
            })

        # Convert results to a DataFrame for easier analysis
        weight_decay_results_df = pd.DataFrame(weight_decay_results)

        # Save results for the current subset size
        results_file = os.path.join(log_folder, f"weight_decay_results_subset_{subset_size}.csv")
        weight_decay_results_df.to_csv(results_file, index=False)

        # Find the weight decay with the highest average accuracy
        best_result = weight_decay_results_df.loc[weight_decay_results_df['mean_accuracy'].idxmax()]

        all_results.append({
            "subset_size": subset_size,
            "loss": best_result["avg_loss"],
            "best_params": {"weight_decay": best_result["weight_decay"]}
        })

    # Save the final results to a CSV file
    results_df = pd.DataFrame(all_results)
    results_csv_file = os.path.join(log_folder, "final_results.csv")
    results_df.to_csv(results_csv_file, index=False)
    print(f"Final results saved to {results_csv_file}")
'''

import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import pandas as pd
import yaml
from core.helper_functions import get_dataset_by_name
from sklearn.model_selection import KFold

def set_random_seed(seed_value):
    """
    Fix random seed for reproducibility.
    """
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)

def get_dynamic_patience(subset_size, total_budget):
    """
    Adjust patience dynamically based on the current subset size and total budget.
    Larger subsets get more patience.
    """
    base_patience = 5
    if subset_size > total_budget * 0.5:
        return base_patience * 2
    elif subset_size > total_budget * 0.75:
        return base_patience * 3
    else:
        return base_patience

def train_and_evaluate(DatasetClass, config, cache_folder, subset_size, weight_decay, num_trials, patience=10, min_delta=0.001, num_folds=5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    validation_scores = []
    losses = []

    dataset_rng = np.random.default_rng()
    dataset = DatasetClass(cache_folder, config, dataset_rng, encoded=False)

    for trial in range(num_trials):
        # Determine if we need to sample with or without replacement
        if num_trials * subset_size > len(dataset.x_train):
            random_indices = np.random.choice(len(dataset.x_train), subset_size, replace=True)
        else:
            random_indices = np.random.choice(len(dataset.x_train), subset_size, replace=False)

        x_train_subset = dataset.x_train[random_indices]
        y_train_subset = dataset.y_train[random_indices]

        model_rng = torch.Generator()
        model_rng.manual_seed(42 + trial)

        # Use KFold cross-validation
        kf = KFold(n_splits=num_folds, shuffle=True, random_state=42 + trial)
        for fold, (train_idx, val_idx) in enumerate(kf.split(x_train_subset)):
            x_train_fold = x_train_subset[train_idx]
            y_train_fold = y_train_subset[train_idx]
            x_val_fold = x_train_subset[val_idx]
            y_val_fold = y_train_subset[val_idx]

            model = dataset.get_classifier(model_rng).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=config["optimizer"]["lr"], weight_decay=weight_decay)
            loss_fn = nn.CrossEntropyLoss()

            train_dataloader = DataLoader(TensorDataset(x_train_fold, y_train_fold), batch_size=dataset.classifier_batch_size, shuffle=True)
            val_dataloader = DataLoader(TensorDataset(x_val_fold, y_val_fold), batch_size=512)

            best_loss = float('inf')
            epochs_without_improvement = 0
            max_epochs = 50

            total_budget = config['dataset']['budget']
            patience = get_dynamic_patience(subset_size, total_budget)

            for epoch in range(max_epochs):
                model.train()
                for batch_x, batch_y in train_dataloader:
                    batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                    optimizer.zero_grad()
                    yHat = model(batch_x)
                    loss_value = loss_fn(yHat, batch_y)
                    loss_value.backward()
                    optimizer.step()

                # Validation phase
                model.eval()
                total, correct = 0, 0
                val_loss = 0.0
                with torch.no_grad():
                    for batch_x, batch_y in val_dataloader:
                        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                        yHat = model(batch_x)

                        if len(batch_y.shape) > 1:
                            batch_y = torch.argmax(batch_y, dim=1)

                        predicted = torch.argmax(yHat, dim=1)
                        correct += (predicted == batch_y).sum().item()
                        total += batch_y.size(0)
                        val_loss += loss_fn(yHat, batch_y).item()

                accuracy = correct / total
                val_loss /= total

                if val_loss < best_loss - min_delta:
                    best_loss = val_loss
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += 1

                if epochs_without_improvement >= patience:
                    break

            validation_scores.append(accuracy)
            losses.append(best_loss)

    return np.mean(validation_scores), np.std(validation_scores), np.mean(losses)

def tune_weight_decay(data_folder, dataset_name, task, num_trials, initial_subset_size, budget, weight_decay_range, log_folder):
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_file = os.path.join(base_path, 'configs', f'{dataset_name}.yaml')
    
    if not os.path.exists(config_file):
        raise FileNotFoundError(f"Configuration file not found: {config_file}")

    with open(config_file, 'r') as f:
        config = yaml.load(f, yaml.Loader)

    DatasetClass = get_dataset_by_name(dataset_name)
    all_results = []

    step_size = budget // 10

    with tqdm(total=len(weight_decay_range) * len(range(step_size, budget + 1, step_size)), desc="Tuning weight decay") as pbar:
        for subset_size in range(step_size, budget + 1, step_size):
            weight_decay_results = []

            for weight_decay in weight_decay_range:
                mean_accuracy, std_accuracy, avg_loss = train_and_evaluate(DatasetClass, config, data_folder, subset_size, weight_decay, num_trials)
                weight_decay_results.append({
                    "weight_decay": weight_decay,
                    "mean_accuracy": mean_accuracy,
                    "std_accuracy": std_accuracy,
                    "avg_loss": avg_loss
                })
                pbar.update(1)

            weight_decay_results_df = pd.DataFrame(weight_decay_results)
            results_file = os.path.join(log_folder, f"weight_decay_results_subset_{subset_size}.csv")
            weight_decay_results_df.to_csv(results_file, index=False)

            # Find best weight decay based on accuracy
            best_accuracy_result = weight_decay_results_df.loc[weight_decay_results_df['mean_accuracy'].idxmax()]
            # Find best weight decay based on loss
            best_loss_result = weight_decay_results_df.loc[weight_decay_results_df['avg_loss'].idxmin()]

            all_results.append({
                "subset_size": subset_size,
                "loss_based_best_weight_decay": best_loss_result["weight_decay"],
                "accuracy_based_best_weight_decay": best_accuracy_result["weight_decay"],
                "best_accuracy": best_accuracy_result["mean_accuracy"],
                "best_loss": best_loss_result["avg_loss"]
            })

    results_df = pd.DataFrame(all_results)
    results_csv_file = os.path.join(log_folder, "final_results.csv")
    results_df.to_csv(results_csv_file, index=False)
    print(f"Final results saved to {results_csv_file}")