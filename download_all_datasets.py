import requests
from requests.adapters import HTTPAdapter
from core.helper_functions import get_dataset_by_name
import argparse
import yaml
import numpy as np

# Argument parser
parser = argparse.ArgumentParser()
parser.add_argument("--data_folder", type=str, required=True)
parser.add_argument("--encode", type=int, default=0)
args = parser.parse_args()
args.encode = bool(args.encode)

# Dataset names
all_names = [
    "splice",
    "dna",
    "usps",
    "TopV2"
]

# Patch requests.get to ignore SSL verification globally
original_get = requests.get
def patched_get(*args, **kwargs):
    kwargs['verify'] = False  # Ignore SSL verification
    return original_get(*args, **kwargs)

requests.get = patched_get

# Main dataset preparation
for name in all_names:
    print("##########################################")
    print(f"downloading {name}...")
    name = name.lower()
    with open(f"configs/{name}.yaml", 'r') as f:
        config = yaml.load(f, yaml.Loader)
    pool_rng = np.random.default_rng(1)
    DatasetClass = get_dataset_by_name(name)
    DatasetClass(args.data_folder, config, pool_rng, encoded=False)
    if args.encode:
        try:
            DatasetClass(args.data_folder, config, pool_rng, encoded=True)
        except AssertionError:
            pass

print("\n")
print("> all datasets prepared")

# Restore the original requests.get after the script runs
requests.get = original_get
