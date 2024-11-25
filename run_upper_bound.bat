@echo off

rem Define the datasets and other parameters
set datasets=splice dna usps topv2
set data_folder=D:\School\Hildesheim\Thesis\Code-Base\ActiveLearning-CodeBase-Baseline\datasets
set restarts=50
set patience=20
set model_seed=1

rem Loop through each dataset and execute compute_upper_bound.py
for %%d in (%datasets%) do (
    python compute_upper_bound.py --data_folder "%data_folder%" --dataset "%%d" --restarts %restarts% --patience %patience% --model_seed %model_seed%
)
