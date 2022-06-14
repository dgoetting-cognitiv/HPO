import time

import torch
import torch.nn.functional as F
import torch.optim as opt

from optimizer import Optimizer

data = {'data': torch.stack((torch.tensor([0, 0, 1], dtype=torch.float),
                             torch.tensor([0, 1, 0], dtype=torch.float),
                             torch.tensor([1, 0, 0], dtype=torch.float),
                             torch.tensor([1, 1, 1], dtype=torch.float)), dim=0),
        'input': 2,
        'output': 1}

example_config = {
    'lr': {'type': 'continuous', 'max': 0.1, 'min': 0.001, 'steps': 5},
    'optim': {'type': 'categorical', 'values': [opt.Adam, opt.RMSprop]},
    'batch_size': {'type': 'int', 'max': 4, 'min': 2, 'steps': 3},
    'architecture': {'type': 'categorical', 'values': [[10, 8, 6, 4], [8, 6, 4], [16, 4, 2], [32, 16]]},
    'dropout': {'type': 'continuous', 'max': 0.75, 'min': 0, 'steps': 5},
    'activation': {'type': 'categorical', 'values': [F.hardswish]},
    'loss': {'type': 'categorical', 'values': [torch.nn.BCELoss]}
}

optimizer = Optimizer(data, example_config, 50)
start = time.time()
loss_gr, model_gr, params_gr = optimizer.grid_search_optimize()
print(f'Finished grid search in {time.time() - start} seconds')

start = time.time()
loss_ra, model_ra, params_ra = optimizer.random_search_optimize(50)
print(f'Finished random search in {time.time() - start} seconds')

print(loss_ra)
print(params_ra)
