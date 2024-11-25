# Active Learning Benchmark with Online Weight Decay

## Reading List
- [Munjal_Towards_Robust_and_Reproducible_Active_Learning](https://openaccess.thecvf.com/content/CVPR2022/papers/Munjal_Towards_Robust_and_Reproducible_Active_Learning_Using_Neural_Networks_CVPR_2022_paper.pdf)
- [A Comparative Survey of Deep Active Learning](https://arxiv.org/pdf/2203.13450.pdf)
- [Randomness is the Root of All Evil](https://openaccess.thecvf.com/content/WACV2023/papers/Ji_Randomness_Is_the_Root_of_All_Evil_More_Reliable_Evaluation_WACV_2023_paper.pdf)
- [A Cross-Domain Benchmark for Active Learning](https://arxiv.org/pdf/2408.00426)

## Dependencies
Python >= 3.10 

Via pip:
- torch
- torchvision
- gym
- matplotlib
- Pandas
- scikit-learn
- faiss-cpu
- nltk (additional download for nltk.word_tokenize in TopV2 needed)
- PyYAML
- batchbald-redux
- ray\[tune\] (Optional)

## Quick Start
### Download Datasets
```bash
python download_all_datasets.py --data_folder "datasets"
```

### Evaluation
```bash
python evaluate.py --data_folder "<my_folder>" --agent <name> --dataset <name> --query_size <int>
```

Available Agents:
- random
- margin
- entropy
- coreset (CoreSet Greedy)
- typiclust
- bald
- badge

Available Datasets:
- splice
- dna
- usps
- topv2

### Grid Search
```bash
python tuning_strategy.py --data_folder "datasets" --datasets DNA Splice USPS topv2 --task classification --num_trials 100 --budgets 300 400 400 200 --min_weight_decay 1e-6 --max_weight_decay 1e-1 --num_weight_decays 100 --initial_subset_size 10
```

### Metafeatures and Metadataset
```bash
# Extract metafeatures
python extract_metafeatures.py

# Create metadataset
python create_metadataset.py
```

## Weight Decay Strategies
Available strategies:
- linear
- exponential
- sigmoid
- constant (static)
- cosine
- polynomial
- optimal (grid search)
- predict (meta-learning model)
- cocktail (not selectable in the main setup, see below for instructions)

### Cocktail Strategy Setup
1. Replace files:
   - `evaluate.py`
   - `core/data.py`
   - `core/environment.py`
2. Add `regularizer_tuning.py` to the home folder
3. Run:
```bash
python evaluate.py --data_folder "%data_folder%" --agent "%%a" --dataset "%%d" --query_size %%q --restarts %restarts%
```

## Results and Outputs
- All run results: `runs/` folder
- Aggregated results: `results/` folder
- Additional plots: `results&plots/` folder

## Parallel Runs
Parallelism is controlled by:
- `run_id` (default 1)
- `restarts` (default 50)

For full parallelism:
1. Set `restarts` to 1
2. Execute 50 runs with increasing `run_ids`

Results are automatically collected after each run and stored in:
`<dataset>/<query_size>/<agent>/accuracies.csv`

## Project Structure
### Dataset
Inherit from `BaseDataset` and implement:
- `__init__()`: Set dataset hyperparameters
- `_download_data()`: Download and preprocess data
- Additional optional methods for pretext tasks and metadata

### Agent
Inherit from `BaseAgent` and implement:
- `__init__()`: Set agent hyperparameters
- `predict()`: Compute and return selection indices

### Main Workflow
The `evaluate.py` script implements the core active learning loop using a reinforcement learning-like approach.