import os
import pandas as pd
import numpy as np
import argparse
import torch
from scipy.sparse import csr_matrix, vstack

def load_sparse_data(file_path):
    data = []
    rows = []
    cols = []
    labels = []

    with open(file_path, 'r') as f:
        for row_index, line in enumerate(f):
            parts = line.strip().split()
            label = int(parts[0])
            labels.append(label)

            for item in parts[1:]:
                col_index, value = item.split(":")
                rows.append(row_index)
                cols.append(int(col_index) - 1)
                data.append(float(value))

    num_features = max(cols) + 1
    sparse_matrix = csr_matrix((data, (rows, cols)), shape=(row_index + 1, num_features))
    return sparse_matrix, np.array(labels)

def load_data(dataset_name):
    train_path_txt = os.path.join("datasets", f"{dataset_name}_train.txt")
    test_path_txt = os.path.join("datasets", f"{dataset_name}_test.txt")
    train_path_pt = os.path.join("datasets", f"{dataset_name}_train.pt")
    test_path_pt = os.path.join("datasets", f"{dataset_name}_test.pt")
    al_path_pt = os.path.join("datasets", f"{dataset_name}_al.pt")
    al_embeddings_path_pt = os.path.join("datasets", f"{dataset_name}_al_embeddings.pt")

    if os.path.exists(train_path_txt) and os.path.exists(test_path_txt):
        train_data, train_labels = load_sparse_data(train_path_txt)
        test_data, test_labels = load_sparse_data(test_path_txt)
        data = vstack([train_data, test_data])
        labels = np.concatenate([train_labels, test_labels])
        return data, labels

    elif os.path.exists(train_path_pt):
        train_data = torch.load(train_path_pt)
        if os.path.exists(test_path_pt):
            test_data = torch.load(test_path_pt)
            data = torch.cat((train_data, test_data), dim=0)
        else:
            data = train_data
        return data.numpy(), None

    elif os.path.exists(al_path_pt):
        loaded_data = torch.load(al_path_pt)
        if isinstance(loaded_data, dict):
            for value in loaded_data.values():
                if isinstance(value, torch.Tensor):
                    return value.numpy(), None
        else:
            return loaded_data.numpy(), None

    elif os.path.exists(al_embeddings_path_pt):
        loaded_data = torch.load(al_embeddings_path_pt)
        if isinstance(loaded_data, dict):
            for value in loaded_data.values():
                if isinstance(value, torch.Tensor):
                    return value.numpy(), None
        else:
            return loaded_data.numpy(), None
    else:
        raise FileNotFoundError(f"No suitable data file found for {dataset_name}")

def extract_metafeatures(data, labels=None):
    metafeatures = {}

    # Number of instances
    metafeatures['num_instances'] = data.shape[0]

    # Number of features
    metafeatures['num_features'] = data.shape[1]
    
    # Number of classes and class imbalance
    if labels is not None:
        labels = labels.numpy()  # Convert the PyTorch tensor to a NumPy array
        unique_labels = np.unique(labels)
        metafeatures['num_classes'] = len(unique_labels)
        
        # Calculate class imbalance if there is more than one class
        if len(unique_labels) > 1:
            label_counts = np.bincount(labels.argmax(axis=1).astype(int))
            metafeatures['class_imbalance'] = np.min(label_counts) / np.max(label_counts)
        else:
            metafeatures['class_imbalance'] = 1.0  # For single class, imbalance is uniform
    else:
        # If no labels, set classification features to 'N/A'
        metafeatures['num_classes'] = 'N/A'
        metafeatures['class_imbalance'] = 'N/A'

    # Average Feature Correlation
    feature_data = data.toarray() if isinstance(data, csr_matrix) else data
    corr_matrix = np.corrcoef(feature_data, rowvar=False)
    corr_values = corr_matrix[np.tril_indices(data.shape[1], -1)]
    metafeatures['avg_feature_correlation'] = np.nanmean(np.abs(corr_values))
    
    # Sparsity
    if isinstance(data, csr_matrix):
        metafeatures['sparsity'] = 1 - data.count_nonzero() / (data.shape[0] * data.shape[1])
    else:
        metafeatures['sparsity'] = 1 - np.count_nonzero(data) / (data.shape[0] * data.shape[1])

    # Task type
    metafeatures['task_type'] = 'classification' if labels is not None and metafeatures['num_classes'] > 1 else 'regression'
    
    return metafeatures

def save_metafeatures(dataset_name, metafeatures):
    metafeatures_df = pd.DataFrame([metafeatures])
    metafeatures_df.insert(0, 'dataset_name', dataset_name)
    os.makedirs("metadataset", exist_ok=True)
    metafeatures_df.to_csv(os.path.join("metadataset", f"{dataset_name}_metafeatures.csv"), index=False)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract metafeatures for a given dataset")
    parser.add_argument('dataset_name', type=str, help="Name of the dataset (e.g., 'dna')")
    args = parser.parse_args()

    data, labels = load_data(args.dataset_name)
    metafeatures = extract_metafeatures(data, labels)
    save_metafeatures(args.dataset_name, metafeatures)

    print(f"Metafeatures for dataset '{args.dataset_name}' have been extracted and saved.")
