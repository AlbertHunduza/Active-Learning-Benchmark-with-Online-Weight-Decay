#!/bin/bash
#SBATCH --job-name=active_learning_experiments
#SBATCH --output=job_output_%j.log
#SBATCH --error=job_error_%j.err
#SBATCH --mail-user=hunduza@uni-hildesheim.de
#SBATCH --partition=STUD
#SBATCH --gres=gpu:1
source activate torch-gpu

# Define datasets, agents, strategies, and other parameters
datasets=("dna""splice" "usps" "topv2")
agents=("badge" "bald" "coreset" "margin" "typiclust" "random" "shannonentropy")
strategies=("predict" "optimal" "linear" "exponential" "sigmoid" "constant" "cosine" "polynomial")
data_folder="./datasets"
query_sizes=(20 50)
restarts=50

# Loop through each combination of dataset, agent, query size, and strategy
for dataset in "${datasets[@]}"; do
    for agent in "${agents[@]}"; do
        for query_size in "${query_sizes[@]}"; do
            for strategy in "${strategies[@]}"; do
                echo "Running experiment with dataset: $dataset, agent: $agent, query size: $query_size, strategy: $strategy"
                srun python evaluate.py --data_folder "$data_folder" --agent "$agent" --dataset "$dataset" --query_size $query_size --restarts $restarts --weight_decay_strategy $strategy
            done
        done
    done
done

