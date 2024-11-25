import bohb.configspace as cs
from bohb import BOHB
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

def create_regularizer_configspace():
    space = cs.ConfigurationSpace()
    
    # Weight Decay
    space.add_hyperparameter(cs.UniformFloatHyperparameter('weight_decay', lower=1e-5, upper=1e-1, log=True))
    
    # Dropout
    space.add_hyperparameter(cs.UniformFloatHyperparameter('dropout', lower=0.0, upper=0.5))
    
    # L1 Regularization
    space.add_hyperparameter(cs.UniformFloatHyperparameter('l1_lambda', lower=0.0, upper=0.1))
    
    # Mixup
    space.add_hyperparameter(cs.UniformFloatHyperparameter('mixup_alpha', lower=0.0, upper=1.0))
    
    # Label Smoothing
    space.add_hyperparameter(cs.UniformFloatHyperparameter('label_smoothing', lower=0.0, upper=0.2))
    
    return space

def evaluate_regularizers(config, budget, model, optimizer, criterion, train_data, val_data, device):
    # Apply regularizers to model
    model.apply_regularizers(config)
    
    # Create data loaders
    train_loader = DataLoader(TensorDataset(*train_data), batch_size=32, shuffle=True)
    val_loader = DataLoader(TensorDataset(*val_data), batch_size=32)
    
    # Train for given budget
    for _ in range(int(budget)):
        for batch in train_loader:
            inputs, targets = [b.to(device) for b in batch]
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # Apply L1 regularization
            if config['l1_lambda'] > 0:
                l1_loss = sum(p.abs().sum() for p in model.parameters())
                loss += config['l1_lambda'] * l1_loss
            
            loss.backward()
            optimizer.step()
    
    # Evaluate on validation set
    model.eval()
    val_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in val_loader:
            inputs, targets = [b.to(device) for b in batch]
            outputs = model(inputs)
            val_loss += criterion(outputs, targets).item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    accuracy = correct / total
    return -accuracy  # BOHB minimizes, so we return negative accuracy

def tune_regularizers(model, optimizer, criterion, train_data, val_data, device, max_budget=10, min_budget=1):
    configspace = create_regularizer_configspace()
    
    def objective(config, budget):
        return evaluate_regularizers(config, budget, model, optimizer, criterion, train_data, val_data, device)
    
    opt = BOHB(configspace, objective, max_budget=max_budget, min_budget=min_budget)
    logs = opt.optimize()
    
    best_config = logs.best['config']
    return best_config