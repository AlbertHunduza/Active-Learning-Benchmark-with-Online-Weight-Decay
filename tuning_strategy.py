import argparse
from concurrent.futures import ProcessPoolExecutor
import os
import numpy as np
from new_tune.new_tune import tune_weight_decay

# Argument parsing
parser = argparse.ArgumentParser()
parser.add_argument("--data_folder", type=str, required=True)
parser.add_argument('--datasets', type=str, nargs='+', required=True)
parser.add_argument('--task', type=str, default="classification")
parser.add_argument('--num_trials', type=int, default=10, help="Number of trials for each weight decay")
parser.add_argument('--initial_subset_size', type=int, default=10)
parser.add_argument('--budgets', type=int, nargs='+', required=True, help="Budgets for each dataset")
parser.add_argument('--min_weight_decay', type=float, required=True, help="Minimum weight decay")
parser.add_argument('--max_weight_decay', type=float, required=True, help="Maximum weight decay")
parser.add_argument('--num_weight_decays', type=int, required=True, help="Number of weight decay values to evaluate")

args = parser.parse_args()

# Ensure datasets and budgets have matching lengths
if len(args.datasets) != len(args.budgets):
    raise ValueError("The number of datasets and budgets must match.")

def run_tuning(data_folder, dataset_name, task, num_trials, initial_subset_size, budget, min_weight_decay, max_weight_decay, num_weight_decays, base_path):
    output_folder = os.path.join(base_path, "grid_search_output")
    log_folder = os.path.join(output_folder, dataset_name, "classification")
    os.makedirs(log_folder, exist_ok=True)

    weight_decay_range = np.logspace(np.log10(min_weight_decay), np.log10(max_weight_decay), num=num_weight_decays)

    tune_weight_decay(data_folder, dataset_name, task, num_trials, initial_subset_size, budget, weight_decay_range, log_folder)

if __name__ == '__main__':
    base_path = os.path.dirname(os.path.abspath(__file__))

    with ProcessPoolExecutor() as executor:
        futures = [
            executor.submit(run_tuning, args.data_folder, dataset, args.task, args.num_trials, args.initial_subset_size, budget, args.min_weight_decay, args.max_weight_decay, args.num_weight_decays, base_path)
            for dataset, budget in zip(args.datasets, args.budgets)
        ]
        for future in futures:
            future.result()
