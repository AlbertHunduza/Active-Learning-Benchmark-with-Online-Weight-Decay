import math
import time
import experiment_util as util
import argparse
from pprint import pprint
from tqdm import tqdm
import core
import yaml
import os
import numpy as np
import torch
from core.helper_functions import *
import csv

# Argument parsing
parser = argparse.ArgumentParser()
parser.add_argument("--data_folder", type=str, required=True)
parser.add_argument("--run_id", type=int, default=1)
parser.add_argument("--agent_seed", type=int, default=1)
parser.add_argument("--pool_seed", type=int, default=1)
parser.add_argument("--model_seed", type=int, default=1)
parser.add_argument("--agent", type=str, default="entropy")
parser.add_argument("--dataset", type=str, default="mnist")
parser.add_argument("--query_size", type=int, default=20)
parser.add_argument("--encoded", type=int, default=0)
parser.add_argument("--restarts", type=int, default=1)
parser.add_argument("--experiment_postfix", type=str, default=None)

# Arguments for weight decay strategy
parser.add_argument("--weight_decay_strategy", type=str, default="linear",
                    choices=["linear", "exponential", "sigmoid", "constant", "cosine", "polynomial", "optimal", "predict"],
                    help="Weight decay adjustment strategy")
parser.add_argument("--max_weight_decay", type=float, default=0.1,
                    help="Maximum weight decay value to use")

args = parser.parse_args()
args.encoded = bool(args.encoded)

run_id = args.run_id
max_run_id = run_id + args.restarts

# Loop through multiple restarts
while run_id < max_run_id:
    with open(f"configs/{args.dataset.lower()}.yaml", 'r') as f:
        config = yaml.load(f, yaml.Loader)
    
    config["current_run_info"] = args.__dict__
    
    # Set weight decay strategy and values
    config["weight_decay_strategy"] = args.weight_decay_strategy
    config["min_weight_decay"] = config["optimizer"]["weight_decay"]  # Use optimizer's weight decay as min
    config["max_weight_decay"] = args.max_weight_decay

    # Ensure embedded optimizer settings match
    if "optimizer_embedded" in config:
        config["max_weight_decay_embedded"] = config["optimizer_embedded"]["weight_decay"]
    else:
        config["max_weight_decay_embedded"] = config["max_weight_decay"]

    # Initialize random seeds
    pool_rng = np.random.default_rng(args.pool_seed + run_id)
    model_seed = args.model_seed + run_id
    torch.random.manual_seed(args.model_seed + run_id)
    np.random.seed(args.model_seed + run_id)
    data_loader_seed = 1

    # Load agent and dataset classes
    AgentClass = get_agent_by_name(args.agent)
    DatasetClass = get_dataset_by_name(args.dataset)

    # Inject config into agent and dataset classes
    AgentClass.inject_config(config)
    DatasetClass.inject_config(config)

    # Initialize dataset and environment
    dataset = DatasetClass(args.data_folder, config, pool_rng, args.encoded)
    dataset = dataset.to(util.device)
    env = core.ALGame(dataset, pool_rng, model_seed=model_seed, data_loader_seed=data_loader_seed, device=util.device)
    agent = AgentClass(args.agent_seed, config, args.query_size)

    # Modify path to include weight decay strategy
    if args.experiment_postfix is not None:
        base_path = os.path.join("runs", dataset.name, str(args.query_size), f"{agent.name}_{args.experiment_postfix}_{args.weight_decay_strategy}")
    else:
        base_path = os.path.join("runs", dataset.name, str(args.query_size), f"{agent.name}_{args.weight_decay_strategy}")
    
    log_path = os.path.join(base_path, f"run_{run_id}")

    print(f"Starting run {run_id}")
    time.sleep(0.1)  # prevents printing issues with tqdm

    # Prepare to save results with weight decay
    results_dict = {}

    # Run the environment and agent
    with core.EnvironmentLogger(env, log_path, util.is_cluster) as env:
        done = False
        dataset.reset()
        state = env.reset()
        iterations = math.ceil(env.env.budget / args.query_size)
        iterator = tqdm(range(iterations), miniters=2)
        for i in iterator:
            action = agent.predict(*state)
            state, reward, done, truncated, info = env.step(action)

            # Capture the current weight decay
            current_weight_decay = None
            for param_group in env.env.optimizer.param_groups:  # Updated to env.env.optimizer
                current_weight_decay = param_group['weight_decay']
            
            # Save the weight decay along with the accuracy
            results_dict[i] = {
                'accuracy': env.accuracies[1][-1],
                'weight_decay': current_weight_decay
            }
            
            # Update the progress bar with accuracy and weight decay
            iterator.set_postfix({"accuracy": env.accuracies[1][-1], "weight_decay": current_weight_decay})
            if done or truncated:
                break

    # Collect and save results to a CSV file
    result_file = os.path.join(log_path, f"results_run_{run_id}.csv")
    with open(result_file, mode='w') as file:
        writer = csv.writer(file)
        # Write the header
        writer.writerow(["Iteration", "Accuracy", "Weight Decay"])
        # Write each result
        for iteration, result in results_dict.items():
            writer.writerow([iteration, result['accuracy'], result['weight_decay']])

    # Collect and save metadata
    collect_results(base_path, "run_")
    save_meta_data(log_path, agent, env, dataset, config)
    
    run_id += 1