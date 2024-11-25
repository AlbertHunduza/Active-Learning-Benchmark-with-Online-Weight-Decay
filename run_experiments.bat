@echo off

rem Define the datasets, agents, strategies, and other parameters
set datasets=usps
set agents=badge bald coreset margin typiclust random entropy
set strategies=predict optimal linear exponential sigmoid cosine polynomial constant

rem Define other parameters
set data_folder=D:\School\Hildesheim\Thesis\Code-Base\ActiveLearning-CodeBase-MetaFeatures\datasets
set query_size=50
set restarts=1

rem Loop through each dataset, agent, query size, and strategy combination
for %%d in (%datasets%) do (
    for %%a in (%agents%) do (
        for %%q in (%query_size%) do (
            for %%s in (%strategies%) do (
                python evaluate.py --data_folder "%data_folder%" --agent "%%a" --dataset "%%d" --query_size %%q --restarts %restarts% --weight_decay_strategy %%s
            )
        )
    )
)