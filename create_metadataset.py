import os
import pandas as pd
import argparse

def load_metafeatures(dataset_name):
    metafeatures_path = os.path.join("metadataset", f"{dataset_name}_metafeatures.csv")
    metafeatures = pd.read_csv(metafeatures_path)
    # Drop the dataset_name and task_type columns
    metafeatures = metafeatures.drop(columns=["dataset_name", "task_type"], errors='ignore')
    return metafeatures

def load_weight_decay_tuning(dataset_name):
    tuning_path = os.path.join("gridsearch_output", dataset_name, "classification", "final_results.csv")
    tuning_results = pd.read_csv(tuning_path)
    return tuning_results

def create_metadataset(datasets):
    metadataset = pd.DataFrame()

    for dataset_name in datasets:
        # Load the metafeatures and tuning results for each dataset
        metafeatures = load_metafeatures(dataset_name)
        tuning_results = load_weight_decay_tuning(dataset_name)

        # Repeat metafeatures for each subset size
        combined_data = pd.concat([metafeatures] * len(tuning_results), ignore_index=True)
        
        # Add the tuning results (subset size and accuracy-based best weight decay)
        combined_data["subset_size"] = tuning_results["subset_size"]
        combined_data["weight_decay"] = tuning_results["accuracy_based_best_weight_decay"]
        
        # Append to the full metadataset
        metadataset = pd.concat([metadataset, combined_data], ignore_index=True)

    # Save to CSV
    os.makedirs("metadataset", exist_ok=True)
    metadataset.to_csv("metadataset/metadataset.csv", index=False)
    print("Metadataset created and saved as 'metadataset.csv' in the 'metadataset' folder.")

if __name__ == "__main__":
    # Argument parsing
    parser = argparse.ArgumentParser(description="Create a metadataset with metafeatures and weight decay tuning results.")
    parser.add_argument('datasets', nargs='+', help="List of dataset names (e.g., 'dna splice usps topv2')")
    args = parser.parse_args()

    # Create the metadataset
    create_metadataset(args.datasets)
